# src/preprocessing/global_scaler.py
import os
import h5py
import json
import numpy as np
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Cargar configuración global
with open("configs/global_config.yaml", 'r') as f:
    GLOBAL_CONFIG = yaml.safe_load(f)

TRAIN_DIR = GLOBAL_CONFIG['paths']['output']['train_val']
OUTPUT_JSON = GLOBAL_CONFIG['paths']['configs']['scaler_bounds']
CHECKPOINT_FILE = OUTPUT_JSON.replace(".json", "_checkpoint.json")

def process_single_file(file_path):
    """Worker aislado para procesar un solo archivo HDF5 de forma segura y optimizada"""
    local_min_ent, local_max_ent = float('inf'), float('-inf')
    local_min_raw, local_max_raw = float('inf'), float('-inf')
    has_data = False
    
    try:
        # swmr=True es correcto para concurrencia de lectura
        with h5py.File(file_path, 'r', swmr=True) as hf:
            for flow_id in hf.keys():
                grp = hf[flow_id]
                
                # Procesamiento de entropía (Protegido contra NaNs)
                if 'blue_channel_entropy' in grp:
                    entropies = grp['blue_channel_entropy'][:]
                    if entropies.size > 0: # .size es más seguro que len() para tensores ND
                        local_min_ent = min(local_min_ent, float(np.nanmin(entropies)))
                        local_max_ent = max(local_max_ent, float(np.nanmax(entropies)))
                        has_data = True
                        
                # Procesamiento de paquetes crudos (Sin concatenación pesada)
                if 'raw_packets' in grp:
                    raw_ds = grp['raw_packets'][:]
                    if raw_ds.size > 0:
                        local_min_raw = min(local_min_raw, float(np.nanmin(raw_ds)))
                        local_max_raw = max(local_max_raw, float(np.nanmax(raw_ds)))
                        has_data = True
                        
        if not has_data:
            return os.path.basename(file_path), None, None, None, None
            
        return os.path.basename(file_path), local_min_ent, local_max_ent, local_min_raw, local_max_raw
    except Exception as e:
        # Si el archivo está corrupto, lo salta en silencio para no frenar el pool
        print(f"\n[!] Fallo silencioso en {os.path.basename(file_path)}: {repr(e)}")
        return os.path.basename(file_path), None, None, None, None

def calculate_global_bounds():
    print("=======================================================")
    print(" INICIANDO PERFILAMIENTO GLOBAL (MULTIPROCESO)")
    print("=======================================================")
    
    state = {
        "global_min_entropy": None,
        "global_max_entropy": None,
        "global_min_raw": None,
        "global_max_raw": None,
        "processed_files": []
    }

    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            saved_state = json.load(f)
            state.update(saved_state)
        print(f"[*] Rescatando sesión interrumpida. Ya procesados: {len(state['processed_files'])}")

    train_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.hdf5')]
    files_to_process = [f for f in train_files if f not in state["processed_files"]]
    
    if not files_to_process:
        print("[*] Todos los archivos ya han sido procesados previamente.")
        return

    max_cores = max(1, os.cpu_count() - 1)
    print(f"[*] Escaneando {len(files_to_process)} archivos con {max_cores} motores paralelos...")

    processed_count = 0
    
    # Ejecución masiva
    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        futures = {executor.submit(process_single_file, os.path.join(TRAIN_DIR, f)): f for f in files_to_process}
        
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
            
            # Guardar progreso en disco cada 50 archivos procesados
            if processed_count % 50 == 0 or processed_count == len(files_to_process):
                tmp_chk = CHECKPOINT_FILE + ".tmp"
                with open(tmp_chk, 'w') as f:
                    json.dump(state, f)
                os.rename(tmp_chk, CHECKPOINT_FILE)

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
    print(f"[*] Entropía Min: {state['global_min_entropy']:.4f} | Max: {state['global_max_entropy']:.4f}")
    print(f"[*] Raw Bytes Min: {state['global_min_raw']:.1f} | Max: {state['global_max_raw']:.1f}")
    print("=======================================================")

if __name__ == "__main__":
    calculate_global_bounds()