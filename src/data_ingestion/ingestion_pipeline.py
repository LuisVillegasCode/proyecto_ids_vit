import os
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
# CONFIGURACIÓN DINÁMICA (Vía YAML)
# ==============================================================================
# Cargar el centro de mando global
with open("configs/global_config.yaml", 'r') as f:
    GLOBAL_CONFIG = yaml.safe_load(f)

YAML_PATH = GLOBAL_CONFIG['paths']['configs']['dataset_schedule']
OUTPUT_DIR_TRAIN = GLOBAL_CONFIG['paths']['output']['train_val']
OUTPUT_DIR_TEST = GLOBAL_CONFIG['paths']['output']['hold_out_test']
MAX_PACKETS = GLOBAL_CONFIG['preprocessing']['max_packets_per_flow']

# Crear directorios si no existen
os.makedirs(OUTPUT_DIR_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_DIR_TEST, exist_ok=True)
os.makedirs(GLOBAL_CONFIG['paths']['data']['dead_letters'], exist_ok=True)

# ==============================================================================
# 1. EL ORÁCULO
# ==============================================================================
def load_oracle(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    compiled_rules = []
    for category in ['zero_day', 'closed_set']:
        if category not in config['attacks']: continue
        for attack in config['attacks'][category]:
            date_str = attack['date']
            for window in attack['time_windows']:
                start_dt = datetime.strptime(f"{date_str} {window[0]}", "%Y-%m-%d %H:%M")
                end_dt = datetime.strptime(f"{date_str} {window[1]}", "%Y-%m-%d %H:%M")
                compiled_rules.append({
                    'start_epoch': start_dt.timestamp(),
                    'end_epoch': end_dt.timestamp(),
                    'attacker_ips': attack['attacker_ips'],
                    'victim_ips': attack['victim_ips'],
                    'target_folder': attack['target_folder'],
                    'label': attack['label']
                })
    return compiled_rules

# ==============================================================================
# 2. MOTOR DE REGLAS Y ENRUTAMIENTO 
# ==============================================================================
def classify_and_route(src_ip, dst_ip, timestamp, six_tuple, oracle):
    attack_label = "Benign"
    target_folder = "stratified_split"
    
    for rule in oracle:
        if rule['start_epoch'] <= timestamp <= rule['end_epoch']:
            if (src_ip in rule['attacker_ips'] and dst_ip in rule['victim_ips']) or \
               (src_ip in rule['victim_ips'] and dst_ip in rule['attacker_ips']):
                attack_label = rule['label']
                target_folder = rule['target_folder']
                break
    
    if target_folder == "hold_out_test":
        return OUTPUT_DIR_TEST, attack_label
        
    hash_object = hashlib.md5(six_tuple.encode())
    hash_integer = int(hash_object.hexdigest(), 16)
    
    if (hash_integer % 10) < 9:
        return OUTPUT_DIR_TRAIN, attack_label
    else:
        return OUTPUT_DIR_TEST, attack_label

# ==============================================================================
# 3. EL TRABAJADOR DE CPU (Refactorizado: RGB-E + Multi-Persistencia Atómica)
# ==============================================================================
def process_pcap_chunk(pcap_file, oracle):
    filename = os.path.basename(pcap_file)
    worker_id = os.getpid()
    
    # ==========================================================
    # MEJORA 1: IDEMPOTENCIA (No empezar desde cero)
    # ==========================================================
    train_exists = any(filename in f for f in os.listdir(OUTPUT_DIR_TRAIN))
    test_exists = any(filename in f for f in os.listdir(OUTPUT_DIR_TEST))
    
    if train_exists or test_exists:
        print(f"[*] Worker [{worker_id}] omitiendo: {filename} (Ya procesado).")
        return

    print(f"[*] Worker [{worker_id}] procesando: {filename}")
    flows = defaultdict(list)
    error_summary = defaultdict(int)
     
    
    try:
        with open(pcap_file, 'rb') as f:
            # ==========================================================
            # MEJORA 2: Blindaje de la Cabecera Global (Magic Number)
            # ==========================================================
            try:
                pcap = dpkt.pcap.Reader(f)
            except ValueError as e:
                with open(os.path.join(GLOBAL_CONFIG['paths']['data']['dead_letters'], f"global_corruption.log"), "a") as err_log:
                    err_log.write(f"{datetime.now()} - {filename} Corrupción total de cabecera. Ignorando archivo.\n")
                return

            # ==========================================================
            # BLINDAJE CONTRA ARCHIVOS TRUNCADOS
            # ==========================================================
            while True:
                try:
                    timestamp, buf = next(pcap)
                except StopIteration:
                    break  # Fin natural del archivo
                except Exception as e:
                    with open(os.path.join(GLOBAL_CONFIG['paths']['data']['dead_letters'], f"truncations_worker_{worker_id}.log"), "a") as err_log:
                        err_log.write(f"{datetime.now()} - {filename} truncado al final. Salvando flujos previos.\n")
                    break
                    
                # ==========================================================
                # LÓGICA DE NEGOCIO (FR5, FR4, Entropía)
                # ==========================================================
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    
                    # FR5: Enmascaramiento MAC
                    eth.src = b'\x00\x00\x00\x00\x00\x00'
                    eth.dst = b'\x00\x00\x00\x00\x00\x00'

                    # FR5: Soporte IPv4 / IPv6
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
                        six_tuple = f"{src_ip_str}-{dst_ip_str}-{transport.sport}-{transport.dport}-{ip.p}"
                        
                        if len(flows[six_tuple]) >= MAX_PACKETS:
                            continue
                            
                        payload = transport.data
                        entropy = 0.0
                        if len(payload) > 0:
                            # FR4: Cálculo de Entropía (Canal Azul)
                            byte_counts = np.bincount(np.frombuffer(payload, dtype=np.uint8), minlength=256)
                            probabilities = byte_counts[byte_counts > 0] / len(payload)
                            entropy = -np.sum(probabilities * np.log2(probabilities))
                        
                        if len(flows[six_tuple]) == 0:
                            target_dir, label = classify_and_route(src_ip_str, dst_ip_str, timestamp, six_tuple, oracle)
                            flows[six_tuple].append({"metadata": (target_dir, label)})
                        
                        flows[six_tuple].append({
                            "entropy": entropy, 
                            "raw_bytes": np.frombuffer(bytes(eth), dtype=np.uint8)
                        })
                        
                except Exception as e:
                    error_summary[str(e)] += 1
                    continue

    except Exception as e:
        print(f"Error crítico no controlado en {filename}: {str(e)}")

    # Registro de Dead-Letter Queue (FR10)
    if error_summary:
        with open(os.path.join(GLOBAL_CONFIG['paths']['data']['dead_letters'], f"dlq_worker_{worker_id}.log"), "a") as dlq:
                    dlq.write(f"{datetime.now()} - {filename} - Reporte de Corrupción:\n")
                    for error_msg, count in error_summary.items():
                        dlq.write(f"  -> {count} paquetes descartados. Razón: {error_msg}\n")

    # ==========================================================
    # PERSISTENCIA ATÓMICA DOBLE (NFR7)
    # ==========================================================
    train_flows = {k: v for k, v in flows.items() if len(v) > 1 and v[0]["metadata"][0] == OUTPUT_DIR_TRAIN}
    test_flows = {k: v for k, v in flows.items() if len(v) > 1 and v[0]["metadata"][0] == OUTPUT_DIR_TEST}
    
    def write_hdf5(prefix, filename, flow_subset, target_dir):
        if not flow_subset: return
        
        tmp_name = f"{prefix}_worker_{worker_id}_{filename}.hdf5.tmp"
        tmp_path = os.path.join(target_dir, tmp_name)
        
        with h5py.File(tmp_path, 'w') as hf:
            for flow_id, packet_data in flow_subset.items():
                meta = packet_data[0]["metadata"]
                grp = hf.create_group(flow_id)
                grp.attrs['label'] = meta[1]
                
                entropies = [p["entropy"] for p in packet_data[1:]]
                grp.create_dataset('blue_channel_entropy', data=np.array(entropies, dtype=np.float32))
                
                dt = h5py.vlen_dtype(np.dtype('uint8'))
                raw_ds = grp.create_dataset('raw_packets', (len(packet_data)-1,), dtype=dt)
                for idx, p in enumerate(packet_data[1:]):
                    raw_ds[idx] = p["raw_bytes"]

        final_file = tmp_name.replace(".tmp", "")
        os.rename(tmp_path, os.path.join(target_dir, final_file))

    write_hdf5("train", filename, train_flows, OUTPUT_DIR_TRAIN)
    write_hdf5("test", filename, test_flows, OUTPUT_DIR_TEST)
    
    print(f"[✓] Worker [{worker_id}] finalizó: {len(train_flows)} a Train, {len(test_flows)} a Test.")
# ==============================================================================
# ORQUESTADOR (Soporta CLI Flags para unificar entornos)
# ==============================================================================
if __name__ == "__main__":
    # Argument Parser para evitar scripts duplicados
    parser = argparse.ArgumentParser(description="Motor de Ingesta OSR-ViT")
    parser.add_argument('--mode', type=str, choices=['pilot', 'prod'], required=True, 
                        help="Define el entorno: 'pilot' usa data/pilot/, 'prod' usa data/raw/chunks/")
    args = parser.parse_args()

    # Enrutamiento basado en el entorno y el YAML global
    if args.mode == 'pilot':
        input_dir = GLOBAL_CONFIG['paths']['data']['pilot']
    else:
        input_dir = GLOBAL_CONFIG['paths']['data']['raw_chunks']

    print("=======================================================")
    print(f" MOTOR DE INGESTA OSR-VIT INICIADO (MODO: {args.mode.upper()})")
    print("=======================================================")
    
    oracle_rules = load_oracle(YAML_PATH)
    pcap_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.pcap')]
    
    if not pcap_files:
        print(f"ALERTA: No hay archivos PCAP en {input_dir}")
        exit(1)
        
    # Asignación de hilos controlada por el YAML (NFR2)
    max_workers = GLOBAL_CONFIG['preprocessing']['multiprocessing_workers']
    pool = mp.Pool(processes=min(max_workers, len(pcap_files)))
    pool.starmap(process_pcap_chunk, [(pcap, oracle_rules) for pcap in pcap_files])
    pool.close()
    pool.join()