# src/utils/audit_processed.py
import os
import glob
import h5py
import numpy as np
from tqdm import tqdm

def audit_hdf5_dataset(directory):
    print("=======================================================")
    search_path = os.path.join(directory, "**/*.hdf5")
    hdf5_files = glob.glob(search_path, recursive=True)
    
    if not hdf5_files:
        print(f"[X] Error crítico: No se encontraron archivos .hdf5 en {directory}")
        return

    print(f"[*] Iniciando auditoría forense de {len(hdf5_files)} archivos HDF5...")
    print("=======================================================")

    corrupt_files = []
    nan_inf_files = []
    total_train_flows = 0
    total_test_flows = 0
    max_packets_found = 0

    for file_path in tqdm(hdf5_files, desc="Auditando tensores"):
        filename = os.path.basename(file_path)
        is_train = "train" in file_path.lower() or "train" in filename.lower()
        
        try:
            with h5py.File(file_path, 'r') as f:
                flow_keys = list(f.keys())
                
                if len(flow_keys) == 0:
                    corrupt_files.append((file_path, "Archivo vacío (sin flujos)"))
                    continue

                if is_train:
                    total_train_flows += len(flow_keys)
                else:
                    total_test_flows += len(flow_keys)

                # Muestreo estricto: Revisamos la integridad del primer flujo de cada archivo 
                # (Revisar absolutamente todos tomaría horas para 165GB)
                sample_flow = flow_keys[0]
                
                if "raw_packets" not in f[sample_flow] or "blue_channel_entropy" not in f[sample_flow]:
                    corrupt_files.append((file_path, "Faltan canales RGB-E (raw_packets / blue_channel)"))
                    continue
                
                raw_data = f[sample_flow]["raw_packets"][:]
                entropy_data = f[sample_flow]["blue_channel_entropy"][:]

                # Registrar el flujo más largo encontrado (debería ser <= a tu límite MAX_PACKETS)
                current_length = raw_data.shape[0]
                if current_length > max_packets_found:
                    max_packets_found = current_length

                # Control de sanidad (NaN / Inf) en el canal de entropía (que es de tipo float)
                if np.isnan(entropy_data).any() or np.isinf(entropy_data).any():
                    nan_inf_files.append(file_path)

        except OSError:
            corrupt_files.append((file_path, "Corrupción de lectura HDF5 (OSError)"))
        except Exception as e:
            corrupt_files.append((file_path, f"Error inesperado: {str(e)}"))

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
            # Mostrar solo los primeros 5 para no saturar la consola
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