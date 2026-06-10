# src/preprocessing/global_scaler.py
import os
import sys
import h5py
import json
import argparse
import numpy as np
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ==============================================================================
# 0. INYECCIÓN DE ENTORNO (PILOTO / PRODUCCIÓN)
# ==============================================================================
def inject_pilot_prefix(path_str: str) -> str:
    """Aísla los directorios y archivos si estamos en modo piloto."""
    if not path_str or path_str in ('/', '\\'): return path_str
    clean_path = path_str.rstrip('/\\')
    head, tail = os.path.split(clean_path)
    if tail.startswith('pilot_'): return path_str
    new_path = os.path.join(head, f"pilot_{tail}")
    if path_str.endswith(('/', '\\')): new_path += path_str[-1]
    return new_path

parser = argparse.ArgumentParser(description="Perfilador Global OSR-ViT (MinMax Scaler)")
parser.add_argument('--mode', type=str, choices=['pilot', 'prod'], required=True, 
                    help="Define si se leen los datos del entorno piloto o de producción")
args, _ = parser.parse_known_args()

try:
    with open("configs/global_config.yaml", 'r') as f:
        GLOBAL_CONFIG = yaml.safe_load(f)
        
    TRAIN_DIR = GLOBAL_CONFIG['paths']['output']['train_val']
    OUTPUT_JSON = GLOBAL_CONFIG['paths']['configs']['scaler_bounds']
except Exception as e:
    print(f"[!] FATAL ERROR: Estructura de global_config.yaml inválida o archivo faltante.\nDetalle: {e}")
    sys.exit(1)

# Aplicar aislamiento de entorno
if args.mode == 'pilot':
    TRAIN_DIR = inject_pilot_prefix(TRAIN_DIR)
    OUTPUT_JSON = inject_pilot_prefix(OUTPUT_JSON)

CHECKPOINT_FILE = OUTPUT_JSON.replace(".json", "_checkpoint.json")

# ==============================================================================
# 1. EL TRABAJADOR DE CPU (EXTRACCIÓN VECTORIZADA)
# ==============================================================================
def process_single_file(file_path):
    """
    Worker aislado. Lee tensores tridimensionales (18, 128, 3) directamente 
    y extrae mínimos y máximos mediante operaciones vectorizadas en C (NumPy).
    """
    local_min_ent, local_max_ent = float('inf'), float('-inf')
    local_min_raw, local_max_raw = float('inf'), float('-inf')
    has_data = False
    
    try:
        # swmr=True previene bloqueos de lectura concurrente
        with h5py.File(file_path, 'r', swmr=True) as hf:
            for flow_id in hf.keys():
                grp = hf[flow_id]
                
                # Leemos la nueva estructura unificada del pipeline de ingesta
                if 'rgb_e_tensor' in grp:
                    tensor = grp['rgb_e_tensor'][:]
                    
                    if (
                        tensor.size > 0 and
                        tensor.ndim == 3 and
                        tensor.shape == (18, 128, 3)
                    ):
                        # Canales 0 y 1: Raw Bytes (Forward y Backward)
                        raw_channels = tensor[..., 0:2]
                        local_min_raw = min(local_min_raw, float(np.nanmin(raw_channels)))
                        local_max_raw = max(local_max_raw, float(np.nanmax(raw_channels)))
                        
                        # Canal 2: Entropía de Shannon
                        entropy_channel = tensor[..., 2]
                        local_min_ent = min(local_min_ent, float(np.nanmin(entropy_channel)))
                        local_max_ent = max(local_max_ent, float(np.nanmax(entropy_channel)))
                        
                        has_data = True
                        
        if not has_data:
            return os.path.basename(file_path), None, None, None, None
            
        return os.path.basename(file_path), local_min_ent, local_max_ent, local_min_raw, local_max_raw
        
    except Exception as e:
        # Prevención de interrupciones por archivos parcialmente corruptos
        print(f"\n[!] Fallo silencioso en {os.path.basename(file_path)}: {repr(e)}")
        return os.path.basename(file_path), None, None, None, None

# ==============================================================================
# 2. ORQUESTADOR MLOPS
# ==============================================================================
def calculate_global_bounds():
    if not os.path.exists(TRAIN_DIR) or not os.path.isdir(TRAIN_DIR):
        print(f"[!] ERROR: El directorio de entrenamiento '{TRAIN_DIR}' no existe.")
        print("    Asegúrate de haber ejecutado el ingestion_pipeline primero en este modo.")
        sys.exit(1)

    print("=======================================================")
    print(f" INICIANDO PERFILAMIENTO GLOBAL (MODO: {args.mode.upper()})")
    print("=======================================================")
    
    state = {
        "global_min_entropy": None,
        "global_max_entropy": None,
        "global_min_raw": None,
        "global_max_raw": None,
        "processed_files": []
    }

    # Rescate de Checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            saved_state = json.load(f)
            state.update(saved_state)
        print(f"[*] Rescatando sesión interrumpida. Ya procesados: {len(state['processed_files'])}")

    train_files = []

    for root, _, files in os.walk(TRAIN_DIR):
        for f in files:
            if f.endswith('.hdf5'):
                relative_path = os.path.relpath(
                    os.path.join(root, f),
                    TRAIN_DIR
                )
                train_files.append(relative_path)

    files_to_process = [
        f for f in train_files
        if f not in state["processed_files"]
    ]
    
    if not files_to_process:
        print("[*] Todos los archivos ya han sido perfilados previamente.")
        return

    # Usamos todos los núcleos disponibles, reservando 1 para el SO
    max_cores = max(1, os.cpu_count() - 1)
    print(f"[*] Escaneando {len(files_to_process)} archivos con {max_cores} motores paralelos...")

    processed_count = 0
    
    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        futures = {
            executor.submit(
                process_single_file,
                os.path.join(TRAIN_DIR, f)
            ): f
            for f in files_to_process
        }
        
        for future in tqdm(as_completed(futures), total=len(files_to_process), desc="Perfilando tensores"):
            filename, min_ent, max_ent, min_raw, max_raw = future.result()
            
            if min_ent is not None:
                if state["global_min_entropy"] is None:
                    state["global_min_entropy"] = min_ent
                    state["global_max_entropy"] = max_ent
                    state["global_min_raw"] = min_raw
                    state["global_max_raw"] = max_raw
                else:
                    state["global_min_entropy"] = min(state["global_min_entropy"], min_ent)
                    state["global_max_entropy"] = max(state["global_max_entropy"], max_ent)
                    state["global_min_raw"] = min(state["global_min_raw"], min_raw)
                    state["global_max_raw"] = max(state["global_max_raw"], max_raw)
            
            state["processed_files"].append(filename)
            processed_count += 1
            
            # Guardado atómico del progreso cada 50 archivos
            if processed_count % 50 == 0 or processed_count == len(files_to_process):
                tmp_chk = CHECKPOINT_FILE + ".tmp"
                with open(tmp_chk, 'w') as f:
                    json.dump(state, f)
                os.rename(tmp_chk, CHECKPOINT_FILE)

    # Consolidación final de la matemática
    bounds = {
        "entropy_channel": {"min": state["global_min_entropy"], "max": state["global_max_entropy"]},
        "raw_bytes_channel": {"min": state["global_min_raw"], "max": state["global_max_raw"]}
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(bounds, f, indent=4)
        
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    print("=======================================================")
    print("[✓] PERFILAMIENTO MULTIPROCESO COMPLETADO Y BLINDADO.")
    print(f"[*] Archivo guardado en: {OUTPUT_JSON}")
    if state['global_min_entropy'] is not None:
        print(f"[*] Entropía Min: {state['global_min_entropy']:.4f} | Max: {state['global_max_entropy']:.4f}")
        print(f"[*] Raw Bytes Min: {state['global_min_raw']:.1f} | Max: {state['global_max_raw']:.1f}")
    else:
        print("[!] Advertencia: No se encontraron tensores válidos.")
    print("=======================================================")

if __name__ == "__main__":
    calculate_global_bounds()