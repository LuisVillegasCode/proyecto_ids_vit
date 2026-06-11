# src/evaluation/evaluate_zero_day.py
import os
import torch
import logging
import argparse
import numpy as np
import shutil
import tempfile
import gc
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from src.models.vit_ablation import IDS2018Dataset, ViT_OSR, safe_collate
from src.utils.config_manager import setup_environment
from src.osr_module.mahalanobis import OpenSetShield

# ==============================================================================
# INYECCIÓN DE ENTORNO
# ==============================================================================
parser = argparse.ArgumentParser(description="Evaluador OSR Zero-Day (AUROC)")
parser.add_argument('--mode', type=str, choices=['pilot', 'prod'], required=True)
parser.add_argument('--n_min', type=int, required=True)
args, _ = parser.parse_known_args()

env = setup_environment(script_name="evaluate_zero_day", args=args)

TRAIN_DIR = env.get_path('paths', 'output', 'train_val', ensure_exists=True)
TEST_DIR = env.get_path('paths', 'output', 'hold_out_test', ensure_exists=True)
SCALER_JSON = env.get_path('paths', 'configs', 'scaler_bounds', is_file=True)
CKPT_DIR = env.get_path('paths', 'artifacts', 'checkpoints', ensure_exists=True)
RESULTS_DIR = env.get_path('paths', 'artifacts', 'results', ensure_exists=True)

def extract_latents(dataloader, model, device, split_name="train", cache_dir=None):
    """Bucle de extracción de vectores usando Memmap dinámico (Anti-Corrupción)"""
    num_samples = len(dataloader.dataset)
    latent_dim = None
    
    latents_path = os.path.join(cache_dir, f"{split_name}_latents.dat")
    labels_path = os.path.join(cache_dir, f"{split_name}_labels.dat")
    preds_path = os.path.join(cache_dir, f"{split_name}_preds.dat")
    
    latents_mm, labels_mm, preds_mm = None, None, None
    offset = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"Extrayendo {split_name}", leave=False):
            if len(inputs) == 0: continue
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            
            with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits, cls_output, _ = model(inputs)
            preds = torch.argmax(logits, dim=1)    
            
            # Inicialización dinámica en el primer lote exitoso (Anti-Hardcodeo)
            if latent_dim is None:
                latent_dim = cls_output.size(1)
                latents_mm = np.memmap(latents_path, dtype='float32', mode='w+', shape=(num_samples, latent_dim))
                labels_mm = np.memmap(labels_path, dtype='int64', mode='w+', shape=(num_samples,))
                preds_mm = np.memmap(preds_path, dtype='int64', mode='w+', shape=(num_samples,))
            
            latents_mm[offset:offset+batch_size] = cls_output.detach().cpu().numpy()
            labels_mm[offset:offset+batch_size] = labels.detach().cpu().numpy()
            preds_mm[offset:offset+batch_size] = preds.detach().cpu().numpy()
            
            offset += batch_size

    # Escudo SRE contra datasets completamente vacíos o filtrados
    if latent_dim is None:
        raise RuntimeError(f"[!] Fallo crítico: No se extrajo ningún vector válido para '{split_name}'.")
    
    # Validación Estricta: Previene corrupción silenciosa por batches omitidos
    if offset != num_samples:
        raise RuntimeError(
            f"[!] Pérdida de datos crítica en {split_name}: Se esperaban {num_samples} "
            f"muestras pero se escribieron {offset}. Revisa 'safe_collate' o el DataLoader."
        )

    latents_mm.flush()
    labels_mm.flush()
    preds_mm.flush()

    return torch.from_numpy(latents_mm), torch.from_numpy(labels_mm), torch.from_numpy(preds_mm)

def evaluate_osr(n_min, mode):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"[*] Inyectando Escudo OSR en: {device}")
    
    ckpt_path = os.path.join(CKPT_DIR, f"vit_nmin_{n_min}_checkpoint.pt")
    
    # Directorio aislado anti-colisiones
    cache_dir = tempfile.mkdtemp(prefix="osr_memmap_")
    
    try:
        # 1. PREPARACIÓN DE CONJUNTOS DE DATOS Y RED
        # El dataset Test AHORA debe contener los Zero-Days (ej. Botnet mapeado al idx 7 u 8)
        train_dataset = IDS2018Dataset(TRAIN_DIR, SCALER_JSON, n_min, mode=mode, is_osr_test=False)
        test_dataset = IDS2018Dataset(TEST_DIR, SCALER_JSON, n_min, mode=mode, is_osr_test=True)
        
        # Partición Estratificada para Calibración MAD
        logging.info("[*] Ejecutando split estratificado para validación SRE...")

        # Extraemos etiquetas base (requiere que el dataset permita acceder a labels rápido)
        # Si acceder a train_dataset.labels es muy lento, se debe pre-calcular en el constructor
        all_train_labels = train_dataset.labels
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42) # Semilla determinista (NFR1)
        
        # Generamos los índices asegurando representatividad poblacional
        for train_idx, val_idx in sss.split(np.zeros(len(all_train_labels)), all_train_labels):
            train_indices = train_idx.tolist()
            val_indices = val_idx.tolist()
        
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)
        
        cpu_count = os.cpu_count() or 2
        optimal_workers = 0 if mode == 'pilot' else max(1, cpu_count - 1)
        
        loader_kwargs = {"batch_size": 128, "shuffle": False, "num_workers": optimal_workers, "collate_fn": safe_collate}
        train_loader = DataLoader(train_subset, **loader_kwargs)
        val_loader = DataLoader(val_subset, **loader_kwargs)
        test_loader = DataLoader(test_dataset, **loader_kwargs)
        
        model = ViT_OSR(n_min=n_min).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state'])
        model.eval()

        # 2. CONSTRUCCIÓN DE PERFILES CONOCIDOS (Fase OSR FIT)
        logging.info("[*] Escaneando subespacio latente de Entrenamiento...")
        train_latents, train_labels, _ = extract_latents(train_loader, model, device, split_name="train", cache_dir=cache_dir)
        
        shield = OpenSetShield(n_components=128, lambda_mad=3.0, device=device)
        shield.fit_profiles(train_latents, train_labels, list(train_dataset.class_to_idx.values()))
        
        del train_latents, train_labels  # Liberación de RAM del host
        if torch.cuda.is_available():
            torch.cuda.empty_cache()     # Limpia memoria reservada residual en GPU

        # 3. CALIBRACIÓN DE UMBRALES (Fase OSR CALIBRATE)
        logging.info("[*] Calibrando fronteras dinámicas (MAD) usando Validación...")
        val_latents, val_labels, _ = extract_latents(val_loader, model, device, split_name="val", cache_dir=cache_dir)
        shield.calibrate_thresholds(val_latents, val_labels)
        
        del val_latents, val_labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 4. INFERENCIA EN ENTORNO ABIERTO (Fase OSR TEST)
        logging.info("[*] Detonando inferencia OSR en Conjunto de Prueba (Zero-Days Mixtos)...")
        test_latents, test_labels, test_preds = extract_latents(test_loader, model, device, split_name="test", cache_dir=cache_dir)
        
        is_anomaly, mahalanobis_distances = shield.detect_anomalies(test_latents, test_preds)
        
        # SRE: Liberación temprana de RAM antes de delegar el cálculo a Scikit-Learn
        del test_latents

        # 5. REMAPEO DINÁMICO DE META-ETIQUETAS Y MÉTRICAS (En CPU RAM)
        logging.info("[*] Calculando AUROC OOD y degradación de MCC...")
        
        # Identificar la etiqueta numérica de la Botnet (Asumimos que está añadida dinámicamente)
        # Se debe ajustar la clave exacta según cómo ingresemos los Zero Days.
        known_classes_idx = list(train_dataset.class_to_idx.values())
        
        test_labels_np = test_labels.numpy()
        distances_np = mahalanobis_distances.numpy()
        test_preds_np = test_preds.numpy()
        is_anomaly_np = is_anomaly.numpy()
        
        # Generación de la Meta-Etiqueta Binaria: 1 si es Zero-Day, 0 si es Known
        meta_y_true = np.isin(test_labels_np, known_classes_idx, invert=True).astype(int)
        
        if np.sum(meta_y_true) == 0:
            logging.error("[!] CUIDADO: No se encontraron clases Zero-Day (Botnet/Infiltration) en el dataloader de prueba.")
            return

        # Métrica OSR 1: AUROC
        auroc = roc_auc_score(meta_y_true, distances_np)
        
        # Métrica de Degración: Filtrar tráfico OOD y evaluar el MCC del modelo base + interceptación OSR
        # Si la predicción base era correcta, pero el Escudo la marca como Anomalía, cuenta como error.
        id_mask = (meta_y_true == 0)
        final_preds_id = np.where(is_anomaly_np[id_mask], -1, test_preds_np[id_mask]) # -1 representa que fue bloqueado
        mcc_id = matthews_corrcoef(test_labels_np[id_mask], final_preds_id)

        logging.info("\n" + "="*60)
        logging.info(" RESULTADOS EXPERIMENTALES: OPEN SET RECOGNITION (OSR)")
        logging.info("="*60)
        logging.info(f"[*] OOD AUROC (Capacidad de aislar Zero-Days): {auroc:.4%}")
        logging.info(f"[*] ID MCC (Degradación del Conocimiento): {mcc_id:.4f}")
        logging.info("="*60)
        
        # Persistencia de seguridad
        shield.save_profiles(os.path.join(RESULTS_DIR, f"osr_profiles_nmin_{n_min}.pt"))
        
    finally:
        # Eliminación explícita forzada (Python no permite mutar locals() directamente)
        try: del train_latents, train_labels
        except (UnboundLocalError, NameError): pass
        
        try: del val_latents, val_labels
        except (UnboundLocalError, NameError): pass
        
        try: del test_latents, test_labels, test_preds
        except (UnboundLocalError, NameError): pass
        
        try:del mahalanobis_distances, is_anomaly
        except (UnboundLocalError, NameError):pass
        
        try: del distances_np, test_labels_np, test_preds_np, is_anomaly_np
        except (UnboundLocalError, NameError): pass
        
        # Forzamos al recolector de basura de Python a soltar los archivos físicos
        gc.collect()
        
        # Limpieza absoluta de la caché
        if 'cache_dir' in locals() and os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

if __name__ == "__main__":
    evaluate_osr(args.n_min, env.mode)