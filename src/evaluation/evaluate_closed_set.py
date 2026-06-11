import os
import yaml
import torch
import logging
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef

# Reutilizamos las clases que ya blindaste en el archivo anterior
from vit_ablation import IDS2018Dataset, ViT_OSR, safe_collate

# ==============================================================================
# CONFIGURACIÓN Y TELEMETRÍA
# ==============================================================================
with open("configs/global_config.yaml", 'r') as f:
    GLOBAL_CONFIG = yaml.safe_load(f)

os.makedirs(GLOBAL_CONFIG['paths']['artifacts']['results'], exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(n_min, mode):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"[*] Evaluador iniciado en: {device}")
    
    # 1. ENRUTAMIENTO AL CONJUNTO DE PRUEBA (Hold-Out)
    test_dir = GLOBAL_CONFIG['paths']['output']['hold_out_test']
    scaler_json = GLOBAL_CONFIG['paths']['configs']['scaler_bounds']
    ckpt_path = os.path.join(GLOBAL_CONFIG['paths']['artifacts']['checkpoints'], f"vit_nmin_{n_min}_checkpoint.pt")
    
    if not os.path.exists(ckpt_path):
        logging.error(f"[!] No se encontró el checkpoint para N_min={n_min}. Entrena el modelo primero.")
        return

    # 2. PREPARACIÓN DE DATOS Y MODELO (Con Auto-Tuning de Hardware)
    dataset = IDS2018Dataset(test_dir, scaler_json, n_min, mode=mode)
    
    cpu_count = os.cpu_count() or 2
    optimal_workers = 0 if mode == 'pilot' or cpu_count <= 2 else max(1, cpu_count - 1)
    
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": 64,
        "shuffle": False, # El orden no importa en evaluación
        "num_workers": optimal_workers,
        "collate_fn": safe_collate
    }
    
    if optimal_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        
    dataloader = DataLoader(**loader_kwargs)
    
    model = ViT_OSR(n_min=n_min).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval() # Desactiva Dropout y coloca las capas en modo inferencia
    
    logging.info(f"[*] Checkpoint cargado exitosamente (Época {checkpoint['epoch']+1}). Iniciando inferencia...")

    all_preds = []
    all_labels = []
    
    # 3. BUCLE DE INFERENCIA ESTRICTA
    with torch.no_grad():
        for inputs, labels in dataloader:
            if len(inputs) == 0: continue
            
            inputs, labels = inputs.to(device), labels.to(device)
            logits, _, _ = model(inputs)
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. CÁLCULO DE MÉTRICAS METODOLÓGICAS COMPLETAS
    if len(all_labels) == 0:
        logging.error("[!] No se encontraron muestras válidas para evaluación en el dataset. Abortando inferencia.")
        return
    class_names = list(dataset.class_to_idx.keys())
    class_indices = list(dataset.class_to_idx.values()) # Protección contra clases faltantes
    
    logging.info("\n" + "="*60)
    logging.info(" REPORTE DE CLASIFICACIÓN (Métricas Base)")
    logging.info("="*60)
    
    # zero_division=0 evita warnings molestos si una clase no tiene predicciones
    report = classification_report(all_labels, all_preds, labels=class_indices, target_names=class_names, digits=4, zero_division=0)
    print(report)
    
    mcc = matthews_corrcoef(all_labels, all_preds)
    logging.info(f"[*] Coeficiente de Correlación de Matthews (MCC): {mcc:.4f}")

    # 5. CÁLCULO DE MÉTRICAS CRÍTICAS: FNR, RECALL Y FPR (Cumplimiento FR13)
    cm = confusion_matrix(all_labels, all_preds, labels=class_indices)
    
    FP = cm.sum(axis=0) - np.diag(cm) 
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    
    # Sumamos 1e-6 al denominador para evitar divisiones por cero absolutas
    FPR = FP / (FP + TN + 1e-6)
    
    # Denominador: Total de casos reales positivos para la clase
    actual_positives = FN + TP + 1e-6 # 1e-6 evita división por cero
    
    FNR = FN / actual_positives
    Recall = TP / actual_positives
    
    logging.info("\n" + "="*60)
    logging.info(" MÉTRICAS CRÍTICAS DE CIBERSEGURIDAD POR CLASE")
    logging.info("="*60)
    for idx, name in enumerate(class_names):
        logging.info(f" -> {name}: FNR = {FNR[idx]:.4%} | Recall = {Recall[idx]:.4%} | FPR = {FPR[idx]:.4%}")

    # 6. GENERACIÓN DE MATRIZ DE CONFUSIÓN VISUAL
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción del Modelo')
    plt.title(f'Matriz de Confusión ViT (N_min={n_min})\nMCC: {mcc:.4f}')
    
    cm_path = os.path.join(GLOBAL_CONFIG['paths']['artifacts']['results'], f"confusion_matrix_nmin_{n_min}.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close() # Liberar memoria RAM de Matplotlib
    logging.info(f"[*] Matriz de Confusión guardada en: {cm_path}")
    logging.info("============================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['pilot', 'prod'], required=True)
    parser.add_argument('--n_min', type=int, required=True, help="El tamaño de ventana N_min a evaluar")
    args = parser.parse_args()
    
    evaluate_model(args.n_min, args.mode)