import os
import sys
import yaml
import dpkt
import socket
import hashlib
import argparse
import multiprocessing as mp
from datetime import datetime
from collections import defaultdict
import numpy as np
import h5py

# ==============================================================================
# 0. VALIDACIÓN DURA DE ENTORNO Y YAML (Respuesta al Supervisor)
# ==============================================================================
try:
    with open("configs/global_config.yaml", 'r') as f:
        GLOBAL_CONFIG = yaml.safe_load(f)
        
    YAML_PATH = GLOBAL_CONFIG['paths']['configs']['dataset_schedule']
    OUTPUT_DIR_TRAIN = GLOBAL_CONFIG['paths']['output']['train_val']
    OUTPUT_DIR_TEST = GLOBAL_CONFIG['paths']['output']['hold_out_test']
    MAX_PACKETS = GLOBAL_CONFIG['preprocessing']['max_packets_per_flow']
    MAX_BYTES = 128 # TRUNCAMIENTO TEMPRANO PARA SALVAR RAM (Metodología ViT)
    
except Exception as e:
    print(f"[!] FATAL ERROR: Estructura de global_config.yaml inválida o archivo faltante.\nDetalle: {e}")
    sys.exit(1)

# Crear directorios si no existen
os.makedirs(OUTPUT_DIR_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_DIR_TEST, exist_ok=True)
os.makedirs(GLOBAL_CONFIG['paths']['data']['dead_letters'], exist_ok=True)

# ==============================================================================
# 1. EL ORÁCULO
# ==============================================================================
def load_oracle(yaml_path):
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(f"[!] FATAL ERROR: No se pudo leer el oráculo {yaml_path}. \n{e}")
        sys.exit(1)
        
    compiled_rules = []
    for category in ['zero_day', 'closed_set']:
        if category not in config.get('attacks', {}): continue
        for attack in config['attacks'][category]:
            date_str = attack['date']
            for window in attack['time_windows']:
                start_dt = datetime.strptime(f"{date_str} {window[0]}", "%Y-%m-%d %H:%M")
                end_dt = datetime.strptime(f"{date_str} {window[1]}", "%Y-%m-%d %H:%M")
                compiled_rules.append({
                    'start_epoch': start_dt.timestamp(),
                    'end_epoch': end_dt.timestamp(),
                    'attacker_ips': set(attack['attacker_ips']), # Optimización de búsqueda
                    'victim_ips': set(attack['victim_ips']),
                    'target_folder': attack['target_folder'],
                    'label': attack['label']
                })
    return compiled_rules

# ==============================================================================
# 2. MOTOR DE REGLAS Y ENRUTAMIENTO 
# ==============================================================================
def classify_and_route(src_ip, dst_ip, timestamp, tuple_key, oracle):
    attack_label = "Benign"
    target_folder = "stratified_split"
    
    for rule in oracle:
        if rule['start_epoch'] <= timestamp <= rule['end_epoch']:
            if (src_ip in rule['attacker_ips'] and dst_ip in rule['victim_ips']) or \
               (src_ip in rule['victim_ips'] and dst_ip in rule['attacker_ips']):
                attack_label = rule['label']
                target_folder = rule['target_folder']
                break
    
    # FR3: Aislamiento determinista OOD
    if target_folder == "hold_out_test":
        return OUTPUT_DIR_TEST, attack_label
        
    # FR2.2: Partición Estratificada por Hashing
    hash_object = hashlib.md5(tuple_key.encode('utf-8'))
    hash_integer = int(hash_object.hexdigest(), 16)
    
    if (hash_integer % 10) < 9:
        return OUTPUT_DIR_TRAIN, attack_label
    else:
        return OUTPUT_DIR_TEST, attack_label

# ==============================================================================
# 3. EL TRABAJADOR DE CPU (Blindado contra OOM e Idempotencia Insegura)
# ==============================================================================
def process_pcap_chunk(pcap_file, oracle):
    filename = os.path.basename(pcap_file)
    worker_id = os.getpid()
    
    # ==========================================================
    # CORRECCIÓN DE IDEMPOTENCIA (Anti-Falsos Positivos)
    # ==========================================================
    def is_already_processed(fname):
        for directory in [OUTPUT_DIR_TRAIN, OUTPUT_DIR_TEST]:
            for existing_file in os.listdir(directory):
                # Validar sufijo exacto para evitar conflictos chunk1 vs chunk11
                if existing_file.endswith(f"_{fname}.hdf5"):
                    return True
        return False

    if is_already_processed(filename):
        print(f"[*] Worker [{worker_id}] omitiendo: {filename} (Procesamiento previo detectado).")
        return

    print(f"[*] Worker [{worker_id}] procesando: {filename}")
    
    # Manejo optimizado de RAM
    flows = defaultdict(list)
    error_summary = defaultdict(int)
     
    try:
        with open(pcap_file, 'rb') as f:
            try:
                pcap = dpkt.pcap.Reader(f)
            except ValueError as e:
                with open(os.path.join(GLOBAL_CONFIG['paths']['data']['dead_letters'], f"global_corruption.log"), "a") as err_log:
                    err_log.write(f"{datetime.now()} - {filename} Corrupción de cabecera mágica. Ignorando.\n")
                return

            while True:
                try:
                    timestamp, buf = next(pcap)
                except StopIteration:
                    break 
                except Exception as e:
                    with open(os.path.join(GLOBAL_CONFIG['paths']['data']['dead_letters'], f"truncations_worker_{worker_id}.log"), "a") as err_log:
                        err_log.write(f"{datetime.now()} - {filename} Fin abrupto (Truncado). Salvando flujos sanos previos.\n")
                    break
                    
                # LÓGICA DE NEGOCIO (FR5, FR4, Direccionalidad Bi-yectiva)
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    
                    # FR5: Enmascaramiento Espacial
                    eth.src = b'\x00\x00\x00\x00\x00\x00'
                    eth.dst = b'\x00\x00\x00\x00\x00\x00'

                    if isinstance(eth.data, dpkt.ip.IP):
                        ip = eth.data
                        src_ip_str = socket.inet_ntoa(ip.src)
                        dst_ip_str = socket.inet_ntoa(ip.dst)
                        ip.src = b'\x00\x00\x00\x00'
                        ip.dst = b'\x00\x00\x00\x00'
                    elif isinstance(eth.data, dpkt.ip6.IP6):
                        ip = eth.data
                        src_ip_str = socket.inet_ntop(socket.AF_INET6, ip.src)
                        dst_ip_str = socket.inet_ntop(socket.AF_INET6, ip.dst)
                        ip.src = b'\x00' * 16
                        ip.dst = b'\x00' * 16
                    else:
                        continue 
                    
                    if isinstance(ip.data, dpkt.tcp.TCP) or isinstance(ip.data, dpkt.udp.UDP):
                        transport = ip.data
                        
                        # Tupla Canónica (Segura para HDF5 Keys)
                        if src_ip_str < dst_ip_str:
                            canonical_tuple = f"{src_ip_str}-{dst_ip_str}-{transport.sport}-{transport.dport}-{ip.p}"
                        else:
                            canonical_tuple = f"{dst_ip_str}-{src_ip_str}-{transport.dport}-{transport.sport}-{ip.p}"
                        
                        # FR4: Limitación dura de campo receptivo (Evita sobrecarga RAM/HDF5)
                        packet_count = len(flows[canonical_tuple]) - 1
                        if packet_count >= MAX_PACKETS:
                            continue
                            
                        payload = transport.data
                        entropy = 0.0
                        if len(payload) > 0:
                            byte_counts = np.bincount(np.frombuffer(payload, dtype=np.uint8), minlength=256)
                            probabilities = byte_counts[byte_counts > 0] / len(payload)
                            entropy = -np.sum(probabilities * np.log2(probabilities))
                        
                        if len(flows[canonical_tuple]) == 0:
                            target_dir, label = classify_and_route(src_ip_str, dst_ip_str, timestamp, canonical_tuple, oracle)
                            flows[canonical_tuple].append({
                                "metadata": (target_dir, label),
                                "initiator_ip": src_ip_str
                            })
                        
                        initiator_ip = flows[canonical_tuple][0]["initiator_ip"]
                        is_forward = (src_ip_str == initiator_ip)
                        direction_flag = 1 if is_forward else 0 

                        # OPTIMIZACIÓN DE MEMORIA CRÍTICA (bytes(eth)[:MAX_BYTES])
                        raw_bytes = np.frombuffer(bytes(eth)[:MAX_BYTES], dtype=np.uint8)
                        
                        flows[canonical_tuple].append({
                            "entropy": entropy, 
                            "raw_bytes": raw_bytes,
                            "direction": direction_flag 
                        })
                        
                except Exception as e:
                    error_summary[str(e)] += 1
                    continue

    except Exception as e:
        print(f"Error crítico no controlado en {filename}: {str(e)}")

    # FR10: Dead-Letter Queue sin detener el Pipeline
    if error_summary:
        with open(os.path.join(GLOBAL_CONFIG['paths']['data']['dead_letters'], f"dlq_worker_{worker_id}.log"), "a") as dlq:
                    dlq.write(f"{datetime.now()} - {filename} - Reporte de Corrupción:\n")
                    for error_msg, count in error_summary.items():
                        dlq.write(f"  -> {count} paquetes descartados. Razón: {error_msg}\n")

    # ==========================================================
    # PERSISTENCIA ATÓMICA Y SEGURIDAD DE CLAVES (NFR7)
    # ==========================================================
    train_flows = {k: v for k, v in flows.items() if len(v) > 1 and v[0]["metadata"][0] == OUTPUT_DIR_TRAIN}
    test_flows = {k: v for k, v in flows.items() if len(v) > 1 and v[0]["metadata"][0] == OUTPUT_DIR_TEST}
    
    def write_hdf5(prefix, fname, flow_subset, target_dir):
        if not flow_subset: return
        
        tmp_name = f"{prefix}_worker_{worker_id}_{fname}.hdf5.tmp"
        tmp_path = os.path.join(target_dir, tmp_name)
        
        with h5py.File(tmp_path, 'w') as hf:
            for flow_id, packet_data in flow_subset.items():
                meta = packet_data[0]["metadata"]
                
                # Saneamiento de Clave HDF5 por seguridad
                safe_flow_id = str(flow_id).replace('/', '_').replace('\\', '_')
                grp = hf.create_group(safe_flow_id)
                grp.attrs['label'] = str(meta[1])
                
                entropies = [p["entropy"] for p in packet_data[1:]]
                grp.create_dataset('blue_channel_entropy', data=np.array(entropies, dtype=np.float32))
                
                directions = [p["direction"] for p in packet_data[1:]]
                grp.create_dataset('direction', data=np.array(directions, dtype=np.int8))
                
                dt = h5py.vlen_dtype(np.dtype('uint8'))
                raw_ds = grp.create_dataset('raw_packets', (len(packet_data)-1,), dtype=dt)
                for idx, p in enumerate(packet_data[1:]):
                    raw_ds[idx] = p["raw_bytes"]

        final_file = tmp_name.replace(".tmp", "")
        # NFR7: Renombrado Atómico (Garantiza que no queden HDF5 corruptos si el SO se apaga)
        os.rename(tmp_path, os.path.join(target_dir, final_file))

    write_hdf5("train", filename, train_flows, OUTPUT_DIR_TRAIN)
    write_hdf5("test", filename, test_flows, OUTPUT_DIR_TEST)
    
    print(f"[✓] Worker [{worker_id}] finalizó: {len(train_flows)} a Train, {len(test_flows)} a Test.")

# ==============================================================================
# ORQUESTADOR MLOps
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motor de Ingesta OSR-ViT")
    parser.add_argument('--mode', type=str, choices=['pilot', 'prod'], required=True)
    args = parser.parse_args()

    if args.mode == 'pilot':
        input_dir = GLOBAL_CONFIG['paths']['data']['pilot']
    else:
        input_dir = GLOBAL_CONFIG['paths']['data']['raw_chunks']
        
    # VALIDACIÓN DURA DE DIRECTORIO DE ENTRADA
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        print(f"[!] FATAL ERROR: El directorio de entrada '{input_dir}' no existe.")
        sys.exit(1)

    print("=======================================================")
    print(f" MOTOR DE INGESTA OSR-VIT INICIADO (MODO: {args.mode.upper()})")
    print("=======================================================")
    
    oracle_rules = load_oracle(YAML_PATH)
    pcap_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.pcap')]
    
    if not pcap_files:
        print(f"[*] ALERTA: No hay archivos PCAP detectados en {input_dir}. Finalizando con éxito pasivo.")
        sys.exit(0)
        
    # Asignación de hilos (Resiliencia si YAML pide más núcleos de los reales)
    requested_workers = GLOBAL_CONFIG['preprocessing']['multiprocessing_workers']
    max_workers = min(requested_workers, mp.cpu_count(), len(pcap_files))
    
    print(f"[*] Inicializando Pool con {max_workers} Workers Concurrentes...")
    pool = mp.Pool(processes=max_workers)
    pool.starmap(process_pcap_chunk, [(pcap, oracle_rules) for pcap in pcap_files])
    pool.close()
    pool.join()