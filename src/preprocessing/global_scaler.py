import os
import h5py
import json
import numpy as np
import yaml

# Cargar configuración global
with open("configs/global_config.yaml", 'r') as f:
    GLOBAL_CONFIG = yaml.safe_load(f)

TRAIN_DIR = GLOBAL_CONFIG['paths']['output']['train_val']
OUTPUT_JSON = GLOBAL_CONFIG['paths']['configs']['scaler_bounds']
CHECKPOINT_FILE = OUTPUT_JSON.replace(".json", "_checkpoint.json")

def calculate_global_bounds():
    print("=======================================================")
    print(" INICIANDO PERFILAMIENTO GLOBAL MIN-MAX (FASE 2)")
    print("=======================================================")
    
    # 1. ESTADO BASE Y RESILIENCIA (Idempotencia)
    state = {
        "global_min_entropy": None,
        "global_max_entropy": None,
        "global_min_raw": None,
        "global_max_raw": None,
        "processed_files": []
    }

    # Cargar progreso previo si hubo una desconexión
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            saved_state = json.load(f)
            state.update(saved_state)
        print(f"[*] Rescatando sesión interrumpida. Archivos ya procesados: {len(state['processed_files'])}")

    train_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.hdf5')]
    total_files = len(train_files)
    
    # Filtrar los que ya procesamos
    files_to_process = [f for f in train_files if f not in state["processed_files"]]
    
    if len(files_to_process) < total_files:
        print(f"[*] Omitiendo {total_files - len(files_to_process)} archivos. Procesando los {len(files_to_process)} restantes...")
    else:
        print(f"[*] Escaneando {total_files} archivos de entrenamiento en modo solo lectura...")

    for idx, filename in enumerate(files_to_process, 1):
        file_path = os.path.join(TRAIN_DIR, filename)
        try:
            with h5py.File(file_path, 'r', swmr=True) as hf:
                for flow_id in hf.keys():
                    grp = hf[flow_id]
                    
                    # 1. Analizar Entropía (Canal Azul)
                    entropies = grp['blue_channel_entropy'][:]
                    if len(entropies) > 0:
                        min_ent = float(np.min(entropies))
                        max_ent = float(np.max(entropies))
                        
                        if state["global_min_entropy"] is None:
                            state["global_min_entropy"] = min_ent
                            state["global_max_entropy"] = max_ent
                        else:
                            state["global_min_entropy"] = min(state["global_min_entropy"], min_ent)
                            state["global_max_entropy"] = max(state["global_max_entropy"], max_ent)
                            
                    # 2. Analizar Bytes Crudos (Rojo/Verde) - OPTIMIZACIÓN VECTORIZADA
                    raw_ds = grp['raw_packets'][:]
                    if len(raw_ds) > 0:
                        all_bytes = np.concatenate(raw_ds) if len(raw_ds) > 1 else raw_ds[0]
                        if len(all_bytes) > 0:
                            min_raw = float(np.min(all_bytes))
                            max_raw = float(np.max(all_bytes))
                            
                            if state["global_min_raw"] is None:
                                state["global_min_raw"] = min_raw
                                state["global_max_raw"] = max_raw
                            else:
                                state["global_min_raw"] = min(state["global_min_raw"], min_raw)
                                state["global_max_raw"] = max(state["global_max_raw"], max_raw)
                            
            # Marcar archivo como completado
            state["processed_files"].append(filename)
            
            # Mejora: Guardar Checkpoint Atómico (Protección contra truncamiento por apagón)
            if idx % 10 == 0 or idx == len(files_to_process):
                tmp_chk = CHECKPOINT_FILE + ".tmp"
                with open(tmp_chk, 'w') as f:
                    json.dump(state, f)
                os.rename(tmp_chk, CHECKPOINT_FILE)
                print(f"  -> Progreso guardado: {len(state['processed_files'])}/{total_files} archivos respaldados...")

        except Exception as e:
            print(f"[!] Error crítico leyendo {filename}: {str(e)}")
            continue

    # 3. CONSOLIDACIÓN FINAL
    bounds = {
        "entropy_channel": {
            "min": state["global_min_entropy"],
            "max": state["global_max_entropy"]
        },
        "raw_bytes_channel": {
            "min": state["global_min_raw"],
            "max": state["global_max_raw"]
        }
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(bounds, f, indent=4)
        
    # Limpieza: Si terminamos con éxito, borramos el checkpoint temporal
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    print("=======================================================")
    print("[✓] PERFILAMIENTO COMPLETADO Y BLINDADO.")
    print(f"[*] Entropía Min: {state['global_min_entropy']:.4f} | Max: {state['global_max_entropy']:.4f}")
    print(f"[*] Raw Bytes Min: {state['global_min_raw']:.1f} | Max: {state['global_max_raw']:.1f}")
    print(f"[*] Guardado permanentemente en: {OUTPUT_JSON}")
    print("=======================================================")

if __name__ == "__main__":
    calculate_global_bounds()