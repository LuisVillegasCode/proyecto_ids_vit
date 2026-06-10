import os
import sys
import yaml
import dpkt
import socket
import argparse
import multiprocessing as mp
from datetime import datetime, timezone, timedelta
from collections import Counter

# Mapeo de prefijos del bash script a las fechas oficiales
PREFIX_TO_DATE = {
    "thu-22": "2018-02-22",
    "wed-28": "2018-02-28",
    "fri-16": "2018-02-16",
    "wed-21": "2018-02-21",
    "fri-02": "2018-03-02"
}

# Parámetros del framework OSR-ViT
FLOW_TIMEOUT_SECONDS = 120.0
MAX_PACKETS = 18
TCP_FIN = 0x01
TCP_RST = 0x04
SWEEP_INTERVAL = 500000  # Frecuencia del recolector de basura para zombis

def load_time_aware_oracle(yaml_path, target_date):
    """
    Carga el oráculo aplicando la corrección de huso horario (UTC-4 / AST)
    y precompila un diccionario O(1) para búsqueda ultrarrápida.
    """
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
        
    rules_dict = {}
    ast_tz = timezone(timedelta(hours=-4))
    
    for category in ['zero_day', 'closed_set']:
        if category not in config.get('attacks', {}): continue
            
        for attack in config['attacks'][category]:
            date_str = str(attack.get('date', '')).strip()
            if target_date and date_str != target_date:
                continue
                
            windows_epoch = []
            for w in attack.get('time_windows', []):
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
                        'windows': windows_epoch
                    })
    return rules_dict

def get_packet_label_fast(src_ip, dst_ip, ts, rules_dict):
    """Búsqueda O(1) de la etiqueta con validación temporal."""
    for pair in [(src_ip, dst_ip), (dst_ip, src_ip)]:
        if pair in rules_dict:
            for attack in rules_dict[pair]:
                for w in attack['windows']:
                    if w[0] <= ts <= w[1]:
                        return attack['label']
    return "Benign"

def session_audit_worker(pcap_file, rules_dict):
    """
    Worker optimizado con Máquina de Estados, Promoción de Etiquetas, 
    Timeout Inmediato y Recolector de Basura Global.
    """
    worker_id = os.getpid()
    print(f"[*] Worker [{worker_id}] procesando: {os.path.basename(pcap_file)}")
    
    flow_states = {}
    instance_counts = Counter()
    packet_count = 0

    try:
        with open(pcap_file, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            
            for timestamp, buf in pcap:
                packet_count += 1
                
                # --- SOLUCIÓN: GARBAGE COLLECTOR PARA ZOMBIS ---
                if packet_count % SWEEP_INTERVAL == 0:
                    dt_utc = datetime.fromtimestamp(timestamp, timezone.utc).strftime('%H:%M:%S UTC')
                    print(f"  [~] Worker [{worker_id}]: {packet_count:,} paquetes. Hora PCAP: {dt_utc} | Flujos en RAM: {len(flow_states)}")
                    
                    expired_flows = []
                    for flow_id, state in flow_states.items():
                        if (timestamp - state['last_time']) > FLOW_TIMEOUT_SECONDS:
                            expired_flows.append(flow_id)
                    
                    for flow_id in expired_flows:
                        if flow_states[flow_id]['status'] == 'CAPTURING' and flow_states[flow_id]['packets_count'] > 0:
                            instance_counts[flow_states[flow_id]['label']] += 1
                        del flow_states[flow_id]

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
                    
                    proto = getattr(ip, "p", getattr(ip, "nxt", 0))
                    sport, dport = 0, 0
                    is_teardown = False
                    
                    if isinstance(ip.data, dpkt.tcp.TCP):
                        sport, dport = ip.data.sport, ip.data.dport
                        flags = ip.data.flags
                        if (flags & TCP_FIN) or (flags & TCP_RST):
                            is_teardown = True
                    elif isinstance(ip.data, dpkt.udp.UDP):
                        sport, dport = ip.data.sport, ip.data.dport

                    endpoint_a = (src_ip_str, sport)
                    endpoint_b = (dst_ip_str, dport)
                    if endpoint_a <= endpoint_b:
                        canonical_flow = (src_ip_str, dst_ip_str, sport, dport, proto)
                    else:
                        canonical_flow = (dst_ip_str, src_ip_str, dport, sport, proto)

                    packet_label = get_packet_label_fast(src_ip_str, dst_ip_str, timestamp, rules_dict)

                    # --- SOLUCIÓN: TIMEOUT INMEDIATO ---
                    # Cortar la sesión inmediatamente si reaparece tarde
                    if canonical_flow in flow_states:
                        if (timestamp - flow_states[canonical_flow]['last_time']) > FLOW_TIMEOUT_SECONDS:
                            if flow_states[canonical_flow]['status'] == 'CAPTURING' and flow_states[canonical_flow]['packets_count'] > 0:
                                instance_counts[flow_states[canonical_flow]['label']] += 1
                            del flow_states[canonical_flow]

                    # 1. Inicialización de sesión
                    if canonical_flow not in flow_states:
                        flow_states[canonical_flow] = {
                            'packets_count': 0,
                            'last_time': timestamp,
                            'status': 'CAPTURING',
                            'label': packet_label
                        }

                    state = flow_states[canonical_flow]
                    state['last_time'] = timestamp

                    # --- SOLUCIÓN: PROMOCIÓN DINÁMICA DE ETIQUETAS ---
                    if packet_label != 'Benign' and state['label'] != packet_label:
                        state['label'] = packet_label

                    # 3. Máquina de Estados OSR-ViT
                    if state['status'] == 'CAPTURING':
                        state['packets_count'] += 1
                        
                        if state['packets_count'] == MAX_PACKETS:
                            instance_counts[state['label']] += 1
                            state['status'] = 'IGNORING'
                            
                        elif is_teardown:
                            instance_counts[state['label']] += 1
                            del flow_states[canonical_flow]
                            
                    elif state['status'] == 'IGNORING':
                        if is_teardown:
                            del flow_states[canonical_flow]

                except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, ValueError, IndexError, AttributeError):
                    continue 

    except Exception as e:
        print(f"[!] Worker [{worker_id}] ERROR CRÍTICO leyendo {pcap_file}: {e}")
        return None
        
    # Limpieza final (solucionado detalle menor)
    for state in flow_states.values():
        if state['status'] == 'CAPTURING' and state['packets_count'] > 0:
            instance_counts[state['label']] += 1

    return dict(instance_counts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulación del generador de tensores OSR-ViT (UTC-4, 120s Timeout, Truncamiento 18pkts)")
    parser.add_argument('--pcap_dir', type=str, required=True, help="Directorio con PCAPs")
    parser.add_argument('--yaml_path', type=str, default="configs/dataset_schedule.yaml", help="Ruta al oráculo")
    parser.add_argument('--prefix', type=str, required=True, help="Prefijo del día a analizar (ej. wed-21, fri-02)")
    args = parser.parse_args()

    target_date = PREFIX_TO_DATE.get(args.prefix)
    if not target_date:
        print(f"[!] ERROR: Prefijo '{args.prefix}' no reconocido.")
        sys.exit(1)

    print("=======================================================================")
    print(f" SIMULANDO GENERADOR DE TENSORES OSR-ViT | DÍA: {target_date}")
    print(" Reglas: UTC-4 | Truncamiento 18 pkts | Timeout 120s | TCP FIN/RST")
    print("=======================================================================")

    rules_dict = load_time_aware_oracle(args.yaml_path, target_date)
    pcap_files = [os.path.join(args.pcap_dir, f) for f in os.listdir(args.pcap_dir) if f.endswith('.pcap') and f.startswith(args.prefix)]

    if not pcap_files:
        print(f"[!] No se encontraron PCAPs con el prefijo {args.prefix}")
        sys.exit(1)

    max_workers = min(mp.cpu_count(), len(pcap_files))
    with mp.Pool(processes=max_workers) as pool:
        resultados = pool.starmap(session_audit_worker, [(pcap, rules_dict) for pcap in pcap_files])

    global_counts = Counter()
    for res in resultados:
        if res:
            global_counts.update(res)

    print("\n=======================================================================")
    print(" VOLUMEN FINAL ESTIMADO DE TENSORES OSR-ViT")
    print("=======================================================================")
    
    total_attack_tensors = sum(count for label, count in global_counts.items() if label != 'Benign')
    
    benign_count = global_counts.get('Benign', 0)
    print(f" [O] Tráfico Legítimo (Benign) : {benign_count:,} tensores generados")
    print("-" * 71)
    
    for label, count in sorted(global_counts.items(), key=lambda x: x[1], reverse=True):
        if label != 'Benign':
            print(f" [X] {label:<25}: {count:,} tensores generados")
            
    print("-" * 71)
    print(f" TOTAL MUESTRAS MALICIOSAS : {total_attack_tensors:,}")
    print("=======================================================================")