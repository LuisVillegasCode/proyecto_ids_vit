import os
import sys
import yaml
import dpkt
import socket
import hashlib
import argparse
import random
import multiprocessing as mp
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import numpy as np
import h5py

def inject_pilot_prefix(path_str: str) -> str:
    if not path_str or path_str in ('/', '\\'): return path_str
    clean_path = path_str.rstrip('/\\')
    head, tail = os.path.split(clean_path)
    if tail.startswith('pilot_'): return path_str
    new_path = os.path.join(head, f"pilot_{tail}")
    if path_str.endswith(('/', '\\')): new_path += path_str[-1]
    return new_path

# ==============================================================================
# 0. CONFIGURACIÓN GLOBAL Y ARGUMENTOS
# ==============================================================================
parser = argparse.ArgumentParser(description="Motor de Ingesta OSR-ViT")
parser.add_argument('--mode', type=str, choices=['pilot', 'prod'], required=True)
args, _ = parser.parse_known_args() 

try:
    with open("configs/global_config.yaml", 'r') as f:
        GLOBAL_CONFIG = yaml.safe_load(f)
        
    YAML_PATH = GLOBAL_CONFIG['paths']['configs']['dataset_schedule']
    OUTPUT_DIR_TRAIN = GLOBAL_CONFIG['paths']['output']['train_val']
    OUTPUT_DIR_TEST = GLOBAL_CONFIG['paths']['output']['hold_out_test']
    MAX_PACKETS = GLOBAL_CONFIG['preprocessing'].get('max_packets_per_flow', 18)
    MAX_BYTES = 128 
    TELEMETRY_LOGS = GLOBAL_CONFIG['paths']['artifacts'].get('telemetry_logs', 'artifacts/logs')
    
except Exception as e:
    print(f"[!] FATAL ERROR: Estructura de global_config.yaml inválida.\nDetalle: {e}")
    sys.exit(1)

# PILAR 1: Parámetros de Máquina de Estados TCP/UDP
FLOW_TIMEOUT_SECONDS = 120.0
SWEEP_INTERVAL = 100000
TCP_FIN = 0x01
TCP_RST = 0x04

# Parámetro para prevenir OOM (Solución Problema 4)
BATCH_FLUSH_SIZE = 10000

# PILAR 4: Tasas de retención para In-Flight Undersampling
RETENTION_RATES = {
    'Benign': 0.05,
    'DoS Hulk': 0.05,
    'DDoS': 0.05,
    'DoS Slowhttptest': 0.10,
    'DoS Slowloris': 0.10,
    'DoS GoldenEye': 0.10,
    'PortScan': 0.10
}

if getattr(args, 'mode', None) == 'pilot':
    OUTPUT_DIR_TRAIN = inject_pilot_prefix(OUTPUT_DIR_TRAIN)
    OUTPUT_DIR_TEST  = inject_pilot_prefix(OUTPUT_DIR_TEST)
    TELEMETRY_LOGS   = inject_pilot_prefix(TELEMETRY_LOGS)
    
os.makedirs(OUTPUT_DIR_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_DIR_TEST, exist_ok=True)
os.makedirs(GLOBAL_CONFIG['paths']['data']['dead_letters'], exist_ok=True)
os.makedirs(TELEMETRY_LOGS, exist_ok=True)

# ==============================================================================
# PILAR 3: ORÁCULO DINÁMICO
# ==============================================================================
def load_time_aware_oracle(yaml_path):
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(f"[!] FATAL ERROR: Oráculo {yaml_path} inaccesible. \n{e}")
        sys.exit(1)
        
    rules_dict = {}
    ast_tz = timezone(timedelta(hours=-4))
    
    for category in ['zero_day', 'closed_set']:
        if category not in config.get('attacks', {}): continue
        for attack in config['attacks'][category]:
            date_str = str(attack['date']).strip()
            windows_epoch = []
            for w in attack['time_windows']:
                start_dt = datetime.strptime(f"{date_str} {w[0].strip()}", "%Y-%m-%d %H:%M").replace(tzinfo=ast_tz)
                end_dt = datetime.strptime(f"{date_str} {w[1].strip()}", "%Y-%m-%d %H:%M").replace(tzinfo=ast_tz)
                windows_epoch.append((start_dt.timestamp(), end_dt.timestamp()))
                
            for a_ip in attack['attacker_ips']:
                for v_ip in attack['victim_ips']:
                    pair = (str(a_ip).strip(), str(v_ip).strip())
                    if pair not in rules_dict:
                        rules_dict[pair] = []
                    rules_dict[pair].append({
                        'label': str(attack['label']).strip(),
                        'target_folder': str(attack['target_folder']).strip(),
                        'windows': windows_epoch
                    })
    return rules_dict

def get_packet_label_fast(src_ip, dst_ip, ts, rules_dict):
    for pair in [(src_ip, dst_ip), (dst_ip, src_ip)]:
        if pair in rules_dict:
            for attack in rules_dict[pair]:
                for w in attack['windows']:
                    if w[0] <= ts <= w[1]:
                        return attack['label'], attack['target_folder']
    return "Benign", "stratified_split"

def get_routing_split(tuple_key, target_folder):
    if target_folder == "hold_out_test":
        return OUTPUT_DIR_TEST
    hash_integer = int(hashlib.md5(tuple_key.encode('utf-8')).hexdigest(), 16)
    return OUTPUT_DIR_TRAIN if (hash_integer % 10) < 9 else OUTPUT_DIR_TEST

# ==============================================================================
# PILAR 5: ENSAMBLADOR DE TENSORES RGB-E
# ==============================================================================
def build_and_route_tensor(flow_id, state, finalized_flows, local_metrics):
    label = state['label']
    
    retention_prob = RETENTION_RATES.get(label, 1.0)
    if random.random() > retention_prob:
        local_metrics['discarded_undersampling'][label] += 1
        return

    tensor = np.zeros((MAX_PACKETS, MAX_BYTES, 3), dtype=np.float32)
    
    for i, pkt in enumerate(state['packets']):
        if i >= MAX_PACKETS: break
        
        raw = pkt['raw_bytes']
        # Solución Problema 5: Asignación segura
        length = min(len(raw), MAX_BYTES)
        
        if pkt['is_forward']:
            tensor[i, :length, 0] = raw[:length]
        else:
            tensor[i, :length, 1] = raw[:length]
            
        tensor[i, :, 2] = pkt['entropy'] 
        
    split_dir = state['target_dir']
    finalized_flows[split_dir][flow_id] = {
        'tensor': tensor,
        'label': label
    }
    local_metrics['generated_tensors'][label] += 1

# ==============================================================================
# FLUSH HDF5 (Solución Problema 4)
# ==============================================================================
def flush_tensors_hdf5(prefix, target_dir, data_dict, worker_id, filename, batch_id):
    if not data_dict: return
    # Incorporamos el batch_id para evitar sobrescrituras de lotes del mismo archivo
    tmp_path = os.path.join(target_dir, f"{prefix}_w{worker_id}_b{batch_id}_{filename}.hdf5.tmp")
    with h5py.File(tmp_path, 'w') as hf:
        for flow_id, payload in data_dict.items():
            safe_id = str(flow_id).replace('/', '_').replace('\\', '_')
            grp = hf.create_group(safe_id)
            grp.attrs['label'] = payload['label']
            grp.create_dataset('rgb_e_tensor', data=payload['tensor'], compression="lzf")
            
    os.rename(tmp_path, tmp_path.replace(".tmp", ""))
    data_dict.clear() # Liberar RAM instantáneamente

# ==============================================================================
# TRABAJADOR PRINCIPAL
# ==============================================================================
def process_pcap_chunk(pcap_file, rules_dict):
    filename = os.path.basename(pcap_file)
    worker_id = os.getpid()
    
    print(f"[*] Worker [{worker_id}] procesando: {filename}")
    
    flow_states = {}
    finalized_flows = {OUTPUT_DIR_TRAIN: {}, OUTPUT_DIR_TEST: {}}
    local_metrics = {'generated_tensors': defaultdict(int), 'discarded_undersampling': defaultdict(int)}
    
    packet_count = 0
    batch_counter = 0
    error_summary = defaultdict(int)
     
    try:
        with open(pcap_file, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            
            for timestamp, buf in pcap:
                packet_count += 1
                
                # Solución Problema 4: Flush proactivo por lotes (Control OOM)
                if packet_count % 10000 == 0:
                    total_in_ram = len(finalized_flows[OUTPUT_DIR_TRAIN]) + len(finalized_flows[OUTPUT_DIR_TEST])
                    if total_in_ram >= BATCH_FLUSH_SIZE:
                        batch_counter += 1
                        flush_tensors_hdf5("train", OUTPUT_DIR_TRAIN, finalized_flows[OUTPUT_DIR_TRAIN], worker_id, filename, batch_counter)
                        flush_tensors_hdf5("test", OUTPUT_DIR_TEST, finalized_flows[OUTPUT_DIR_TEST], worker_id, filename, batch_counter)

                # PILAR 1: Garbage Collector Global
                if packet_count % SWEEP_INTERVAL == 0:
                    expired_flows = [fid for fid, st in flow_states.items() if (timestamp - st['last_time']) > FLOW_TIMEOUT_SECONDS]
                    for fid in expired_flows:
                        if flow_states[fid]['status'] == 'CAPTURING' and len(flow_states[fid]['packets']) > 0:
                            build_and_route_tensor(fid, flow_states[fid], finalized_flows, local_metrics)
                        del flow_states[fid]

                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    if not isinstance(eth.data, dpkt.ip.IP) and not isinstance(eth.data, dpkt.ip6.IP6):
                        continue 
                    
                    ip = eth.data
                    if isinstance(ip, dpkt.ip.IP):
                        src_ip_str = socket.inet_ntoa(ip.src)
                        dst_ip_str = socket.inet_ntoa(ip.dst)
                    else:
                        src_ip_str = socket.inet_ntop(socket.AF_INET6, ip.src)
                        dst_ip_str = socket.inet_ntop(socket.AF_INET6, ip.dst)
                    
                    # Ofuscación de IP
                    eth.src = b'\x00'*6; eth.dst = b'\x00'*6
                    if isinstance(ip, dpkt.ip.IP):
                        ip.src = b'\x00'*4; ip.dst = b'\x00'*4
                    else:
                        ip.src = b'\x00'*16; ip.dst = b'\x00'*16
                    
                    proto = getattr(ip, "p", getattr(ip, "nxt", 0))
                    if not (isinstance(ip.data, dpkt.tcp.TCP) or isinstance(ip.data, dpkt.udp.UDP)):
                        continue
                        
                    transport = ip.data
                    sport, dport = transport.sport, transport.dport
                    is_teardown = False
                    
                    if isinstance(transport, dpkt.tcp.TCP):
                        if (transport.flags & TCP_FIN) or (transport.flags & TCP_RST):
                            is_teardown = True

                    canonical_tuple = f"{src_ip_str}-{dst_ip_str}-{sport}-{dport}-{proto}" if (src_ip_str, sport) <= (dst_ip_str, dport) else f"{dst_ip_str}-{src_ip_str}-{dport}-{sport}-{proto}"
                    packet_label, target_folder = get_packet_label_fast(src_ip_str, dst_ip_str, timestamp, rules_dict)

                    # PILAR 1: Timeout Inmediato
                    if canonical_tuple in flow_states:
                        if (timestamp - flow_states[canonical_tuple]['last_time']) > FLOW_TIMEOUT_SECONDS:
                            if flow_states[canonical_tuple]['status'] == 'CAPTURING' and len(flow_states[canonical_tuple]['packets']) > 0:
                                build_and_route_tensor(canonical_tuple, flow_states[canonical_tuple], finalized_flows, local_metrics)
                            del flow_states[canonical_tuple]

                    # Inicialización
                    if canonical_tuple not in flow_states:
                        flow_states[canonical_tuple] = {
                            'packets': [],
                            'last_time': timestamp,
                            'status': 'CAPTURING',
                            'label': packet_label,
                            'target_dir': get_routing_split(canonical_tuple, target_folder),
                            'initiator_ip': src_ip_str
                        }

                    state = flow_states[canonical_tuple]
                    state['last_time'] = timestamp

                    # PILAR 3 / Solución Problema 1: Promoción Dinámica de Etiquetas segura
                    if packet_label != 'Benign' and state['label'] != packet_label:
                        state['label'] = packet_label
                        # target_dir se mantiene intacto para garantizar reproducibilidad 

                    # PILAR 2: Máquina de Estados y Detección Temprana (IGNORING)
                    if state['status'] == 'CAPTURING':
                        payload = transport.data
                        entropy = 0.0
                        if len(payload) > 0:
                            byte_counts = np.bincount(np.frombuffer(payload, dtype=np.uint8), minlength=256)
                            probs = byte_counts[byte_counts > 0] / len(payload)
                            entropy = -np.sum(probs * np.log2(probs))

                        state['packets'].append({
                            'entropy': float(entropy),
                            'raw_bytes': np.frombuffer(bytes(eth)[:MAX_BYTES], dtype=np.uint8),
                            'is_forward': (src_ip_str == state['initiator_ip'])
                        })
                        
                        if len(state['packets']) == MAX_PACKETS:
                            build_and_route_tensor(canonical_tuple, state, finalized_flows, local_metrics)
                            state['status'] = 'IGNORING'
                            
                        elif is_teardown:
                            build_and_route_tensor(canonical_tuple, state, finalized_flows, local_metrics)
                            del flow_states[canonical_tuple]
                            
                    elif state['status'] == 'IGNORING':
                        if is_teardown:
                            del flow_states[canonical_tuple]

                except Exception as e:
                    error_summary[str(e)] += 1
                    continue

    except Exception as e:
        print(f"[!] Error crítico en {filename}: {str(e)}")

    # Limpieza residual
    for fid, st in flow_states.items():
        if st['status'] == 'CAPTURING' and len(st['packets']) > 0:
            build_and_route_tensor(fid, st, finalized_flows, local_metrics)

    # Flush final de cualquier tensor restante
    batch_counter += 1
    flush_tensors_hdf5("train", OUTPUT_DIR_TRAIN, finalized_flows[OUTPUT_DIR_TRAIN], worker_id, filename, batch_counter)
    flush_tensors_hdf5("test", OUTPUT_DIR_TEST, finalized_flows[OUTPUT_DIR_TEST], worker_id, filename, batch_counter)
    
    print(f"[✓] Worker [{worker_id}] procesó {filename}. Generados: {sum(local_metrics['generated_tensors'].values())}")
    return dict(local_metrics)

# ==============================================================================
# ORQUESTADOR MLOps
# ==============================================================================
if __name__ == "__main__":

    input_dir = GLOBAL_CONFIG['paths']['data']['pilot'] if args.mode == 'pilot' else GLOBAL_CONFIG['paths']['data']['raw_chunks']
        
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        print(f"[!] FATAL ERROR: Directorio de entrada '{input_dir}' no existe.")
        sys.exit(1)

    print("=======================================================")
    print(f" MOTOR OSR-VIT: RGB-E TENSOR FACTORY (MODO: {args.mode.upper()})")
    print("=======================================================")
    
    rules_dict = load_time_aware_oracle(YAML_PATH)
    pcap_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.pcap')]
    
    if not pcap_files:
        print(f"[*] ALERTA: No hay archivos PCAP en {input_dir}.")
        sys.exit(0)

    # Solución Problemas 2 y 3: Idempotencia resuelta en Orquestador O(N)
    print("[*] Evaluando integridad e idempotencia en disco...")
    processed_files_set = set()
    for directory in [OUTPUT_DIR_TRAIN, OUTPUT_DIR_TEST]:
        if os.path.exists(directory):
            processed_files_set.update(os.listdir(directory))
            
    pending_pcaps = []
    for pcap in pcap_files:
        fname = os.path.basename(pcap)
        if not any(pf.endswith(f"_{fname}.hdf5") for pf in processed_files_set):
            pending_pcaps.append(pcap)
            
    if not pending_pcaps:
        print(f"[*] ALERTA: Todos los archivos PCAP detectados ya fueron procesados previamente. Finalizando ejecución pasivamente.")
        sys.exit(0)
        
    max_workers = min(GLOBAL_CONFIG['preprocessing']['multiprocessing_workers'], mp.cpu_count(), len(pending_pcaps))
    print(f"[*] Desplegando Pool con {max_workers} Workers Concurrentes para procesar {len(pending_pcaps)} archivos pendientes...")
    
    with mp.Pool(processes=max_workers) as pool:
        resultados = pool.starmap(process_pcap_chunk, [(pcap, rules_dict) for pcap in pending_pcaps])

    # ==========================================================
    # REPORTE DE TELEMETRÍA MLOps
    # ==========================================================
    print("\n[*] Consolidando telemetría de tensores...")
    gen_counts = defaultdict(int)
    disc_counts = defaultdict(int)
    
    for res in resultados:
        if res:
            for label, count in res['generated_tensors'].items(): gen_counts[label] += count
            for label, count in res['discarded_undersampling'].items(): disc_counts[label] += count
                
    total_gen = sum(gen_counts.values())
    total_disc = sum(disc_counts.values())
    
    report_lines = [
        "=====================================================================================",
        " REPORTE FINAL: TENSORES RGB-E GENERADOS PARA ENTRENAMIENTO",
        "=====================================================================================",
        f"{'Label':<25} | {'Generados (Saved)':<18} | {'Descartados (Undersampling)':<25}",
        "-" * 75
    ]
    
    all_labels = set(gen_counts.keys()).union(set(disc_counts.keys()))
    for label in sorted(all_labels):
        report_lines.append(f"{label:<25} | {gen_counts[label]:<18,} | {disc_counts[label]:<25,}")
        
    report_lines.extend([
        "-" * 75,
        f"Total Tensores Extraídos: {total_gen + total_disc:,}",
        f"Total Redundancia Purgada: {total_disc:,}",
        f"TOTAL TENSORES EN DISCO : {total_gen:,}",
        "====================================================================================="
    ])
    
    report_text = "\n".join(report_lines)
    print(report_text)
    
    report_path = os.path.join(TELEMETRY_LOGS, f"tensor_distribution_{args.mode}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[*] Telemetría guardada en: {report_path}")