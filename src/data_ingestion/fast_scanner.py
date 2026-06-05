import os
import sys
import yaml
import dpkt
import socket
import argparse
import multiprocessing as mp
from datetime import datetime, timezone
from collections import defaultdict

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
try:
    with open("configs/global_config.yaml", 'r') as f:
        GLOBAL_CONFIG = yaml.safe_load(f)
    YAML_PATH = GLOBAL_CONFIG['paths']['configs']['dataset_schedule']
except Exception as e:
    print(f"[!] FATAL ERROR: No se pudo leer global_config.yaml\nDetalle: {e}")
    sys.exit(1)

# ==============================================================================
# LÓGICA DEL ORÁCULO (Blindado contra zonas horarias y espacios en blanco)
# ==============================================================================
def load_oracle(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
        
    compiled_rules = []
    for category in ['zero_day', 'closed_set']:
        if category not in config.get('attacks', {}): continue
        for attack in config['attacks'][category]:
            date_str = str(attack['date']).strip()
            for window in attack['time_windows']:
                # FIX: Forzamos la zona horaria a UTC
                start_dt = datetime.strptime(f"{date_str} {window[0].strip()}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                end_dt = datetime.strptime(f"{date_str} {window[1].strip()}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                
                # FIX: Limpieza estricta de espacios invisibles
                compiled_rules.append({
                    'start_epoch': start_dt.timestamp(),
                    'end_epoch': end_dt.timestamp(),
                    'attacker_ips': set(str(ip).strip() for ip in attack['attacker_ips']), 
                    'victim_ips': set(str(ip).strip() for ip in attack['victim_ips']),
                    'label': str(attack['label']).strip()
                })
    return compiled_rules

# ==============================================================================
# TRABAJADOR DE ESCANEO ULTRA RÁPIDO 
# ==============================================================================
def fast_scan_worker(pcap_file, oracle):
    worker_id = os.getpid()
    counts = defaultdict(int)
    print(f"[*] Worker [{worker_id}] escaneando a velocidad máxima: {os.path.basename(pcap_file)}")
    
    # Rango global de tiempo para descartar paquetes benignos en O(1)
    if oracle:
        global_min_time = min(rule['start_epoch'] for rule in oracle)
        global_max_time = max(rule['end_epoch'] for rule in oracle)
    else:
        global_min_time, global_max_time = float('inf'), float('-inf')

    try:
        with open(pcap_file, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            for timestamp, buf in pcap:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    if isinstance(eth.data, dpkt.ip.IP):
                        ip = eth.data
                        src_ip_str = socket.inet_ntoa(ip.src)
                        dst_ip_str = socket.inet_ntoa(ip.dst)
                    elif isinstance(eth.data, dpkt.ip6.IP6):
                        ip = eth.data
                        src_ip_str = socket.inet_ntop(socket.AF_INET6, ip.src)
                        dst_ip_str = socket.inet_ntop(socket.AF_INET6, ip.dst)
                    else:
                        continue
                    
                    # Motor de evaluación directa con Cortocircuito
                    attack_label = "Benign"
                    
                    # Solo evaluamos si el paquete cae dentro de las horas de ataque globales
                    if global_min_time <= timestamp <= global_max_time:
                        for rule in oracle:
                            if rule['start_epoch'] <= timestamp <= rule['end_epoch']:
                                if (src_ip_str in rule['attacker_ips'] and dst_ip_str in rule['victim_ips']) or \
                                   (src_ip_str in rule['victim_ips'] and dst_ip_str in rule['attacker_ips']):
                                    attack_label = rule['label']
                                    break
                                
                    counts[attack_label] += 1
                except Exception:
                    continue 
    except Exception as e:
        print(f"[!] Archivo corrupto o truncado ignorado: {os.path.basename(pcap_file)}")
        
    return dict(counts)

# ==============================================================================
# ORQUESTADOR DE DIAGNÓSTICO
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Escáner Táctico de Etiquetas OSR-ViT")
    parser.add_argument('--input_dir', type=str, required=True, help="Ruta de los PCAPs a escanear")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"[!] El directorio {args.input_dir} no existe.")
        sys.exit(1)

    print("=======================================================")
    print(" INICIANDO ESCANEO DE DIAGNÓSTICO DE ETIQUETAS")
    print("=======================================================")
    
    oracle_rules = load_oracle(YAML_PATH)
    pcap_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.pcap')]
    
    max_workers = min(mp.cpu_count(), len(pcap_files))
    pool = mp.Pool(processes=max_workers)
    resultados = pool.starmap(fast_scan_worker, [(pcap, oracle_rules) for pcap in pcap_files])
    pool.close()
    pool.join()

    print("\n[*] Consolidando resultados del escaneo de paquetes crudos...")
    global_counts = defaultdict(int)
    
    for res in resultados:
        for label, count in res.items():
            global_counts[label] += count
            
    total_packets = sum(global_counts.values())
    
    print("=====================================================================================")
    print("RADIOGRAFÍA DE DIAGNÓSTICO: DISTRIBUCIÓN DE PAQUETES (NO FLUJOS)")
    print("=====================================================================================")
    print(f"{'Label':<30} | {'Paquetes Totales':<15} | {'Porcentaje (%)':<15}")
    print("-" * 85)
    
    for label, count in sorted(global_counts.items(), key=lambda item: item[1], reverse=True):
        pct = (count / total_packets) * 100 if total_packets > 0 else 0
        print(f"{label:<30} | {count:<15} | {pct:>8.4f}%")
        
    print("-" * 85)
    print(f"Total de PAQUETES evaluados: {total_packets:,}")
    print("=====================================================================================")