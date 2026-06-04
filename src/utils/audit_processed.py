# src/utils/audit_processed.py
import os
import glob
import h5py
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def audit_single_file(file_path):
    """Evalúa un solo archivo HDF5. Lógica intacta de la versión secuencial."""
    filename = os.path.basename(file_path)
    is_train = "train" in file_path.lower() or "train" in filename.lower()
    
    num_flows = 0
    max_packets = 0
    corrupt_reason = None
    has_nan_inf = False
    
    try:
        with h5py.File(file_path, 'r') as f:
            flow_keys = list(f.keys())
            num_flows = len(flow_keys)
            
            if num_flows == 0:
                return (file_path, is_train, 0, 0, "Archivo vacío (sin flujos)", False)

            sample_flow = flow_keys[0]
            
            if "raw_packets" not in f[sample_flow] or "blue_channel_entropy" not in f[sample_flow]:
                return (file_path, is_train, num_flows, 0, "Faltan canales RGB-E (raw_packets / blue_channel)", False)
            
            raw_data = f[sample_flow]["raw_packets"][:]
            entropy_data = f[sample_flow]["blue_channel_entropy"][:]

            max_packets = raw_data.shape[0]

            if np.isnan(entropy_data).any() or np.isinf(entropy_data).any():
                has_nan_inf = True

    except OSError:
        corrupt_reason = "Corrupción de lectura HDF5 (OSError)"
    except Exception as e:
        corrupt_reason = f"Error inesperado: {str(e)}"

    return (file_path, is_train, num_flows, max_packets, corrupt_reason, has_nan_inf)

def audit_hdf5_dataset(directory):
    print("=======================================================")
    search_path = os.path.join(directory, "**/*.hdf5")
    hdf5_files = glob.glob(search_path, recursive=True)
    
    if not hdf5_files:
        print(f"[X] Error crítico: No se encontraron archivos .hdf5 en {directory}")
        return

    print(f"[*] Iniciando auditoría MULTIPROCESO de {len(hdf5_files)} archivos HDF5...")
    # Usar todos los núcleos disponibles de la VM menos 1 (para no congelar el sistema)
    max_cores = max(1, os.cpu_count() - 1)
    print(f"[*] Motores paralelos asignados: {max_cores}")
    print("=======================================================")

    corrupt_files = []
    nan_inf_files = []
    total_train_flows = 0
    total_test_flows = 0
    max_packets_found = 0

    # Ejecución en paralelo
    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        # Mapeamos la función a la lista de archivos
        futures = {executor.submit(audit_single_file, path): path for path in hdf5_files}
        
        for future in tqdm(as_completed(futures), total=len(hdf5_files), desc="Auditando tensores en paralelo"):
            file_path, is_train, num_flows, max_packets, corrupt_reason, has_nan_inf = future.result()
            
            # Agregamos los resultados devueltos por cada hilo
            if is_train:
                total_train_flows += num_flows
            else:
                total_test_flows += num_flows
                
            if max_packets > max_packets_found:
                max_packets_found = max_packets
                
            if corrupt_reason:
                corrupt_files.append((file_path, corrupt_reason))
                
            if has_nan_inf:
                nan_inf_files.append(file_path)

    # =======================================================
    # REPORTE FINAL DE AUDITORÍA
    # =======================================================
    print("\n=======================================================")
    print(" REPORTE FINAL DE INTEGRIDAD DE DATOS (OSR-VIT)")
    print("=======================================================")
    print(f"[✓] Archivos analizados: {len(hdf5_files)}")
    print(f"[📊] Flujos totales en Train: {total_train_flows:,}")
    print(f"[📊] Flujos totales en Test:  {total_test_flows:,}")
    print(f"[📏] Longitud máxima de paquetes detectada: {max_packets_found}")
    
    if not corrupt_files and not nan_inf_files:
        print("\n[👑] VEREDICTO: Dataset 100% SANO. Listo para el Vision Transformer.")
    else:
        print(f"\n[🚨] AMENAZAS DETECTADAS:")
        if corrupt_files:
            print(f"  -> Archivos corruptos/incompletos: {len(corrupt_files)}")
            for path, reason in corrupt_files[:5]:
                print(f"     - {os.path.basename(path)} ({reason})")
            if len(corrupt_files) > 5:
                print("     ... (y otros más)")
        if nan_inf_files:
            print(f"  -> Archivos con NaN/Inf (Peligro de Gradientes): {len(nan_inf_files)}")
            for path in nan_inf_files[:5]:
                print(f"     - {os.path.basename(path)}")
    print("=======================================================")

if __name__ == "__main__":
    import yaml

    CONFIG_PATH = "configs/global_config.yaml"
    # Ruta de respaldo (fallback) segura
    TARGET_DIR = "data/processed" 
    
    try:
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
            
        # Navegamos por la jerarquía de tu YAML: paths -> output
        if "paths" in config and "output" in config["paths"]:
            train_val_path = config["paths"]["output"].get("train_val", "")
            
            if train_val_path:
                # Utilizamos normpath y dirname para subir un nivel.
                # Convierte "data/processed/train_val/" -> "data/processed"
                TARGET_DIR = os.path.dirname(os.path.normpath(train_val_path))
                
        print(f"[*] Ruta raíz de auditoría cargada desde YAML: {TARGET_DIR}")
        
    except FileNotFoundError:
        print(f"[⚠️] Alerta: No se encontró {CONFIG_PATH}. Usando ruta por defecto: {TARGET_DIR}")
    except Exception as e:
        print(f"[⚠️] Error al leer la configuración ({str(e)}). Usando ruta por defecto: {TARGET_DIR}")

    audit_hdf5_dataset(TARGET_DIR)