import os
import sys
import yaml
import dpkt
import socket
import argparse
import multiprocessing as mp
from datetime import datetime, timezone

# Mapeo de prefijos del bash script a las fechas oficiales del YAML
PREFIX_TO_DATE = {
    "thu-22": "2018-02-22",
    "wed-28": "2018-02-28",
    "fri-16": "2018-02-16",
    "wed-21": "2018-02-21",
    "fri-02": "2018-03-02"
}

def load_oracle_ips_only(yaml_path, target_date=None):
    """
    Extrae y precompila los pares de IPs del archivo de configuración para búsquedas O(1).
    Filtra por fecha si se especifica un prefijo, evitando contaminación cruzada.
    """
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(f"[!] FATAL ERROR: No se pudo leer el oráculo {yaml_path}. \n{e}")
        sys.exit(1)
        
    # Validación estructural robusta del YAML
    if not config or 'attacks' not in config:
        print("[!] FATAL ERROR: YAML malformado. Falta el nodo raíz 'attacks'.")
        sys.exit(1)

    # Precompilar pares IP en un Set para búsquedas en O(1)
    oracle_pairs = set()
    
    for category in ['zero_day', 'closed_set']:
        if category not in config['attacks']: 
            continue
            
        for attack in config['attacks'][category]:
            # Filtrado metodológico: Solo cargar IPs del día que estamos auditando
            attack_date = str(attack.get('date', '')).strip()
            if target_date and attack_date != target_date:
                continue

            # Validación por regla
            if 'attacker_ips' not in attack or 'victim_ips' not in attack:
                print(f"[!] ERROR YAML: Regla sin IPs definidas -> {attack.get('attack_name', 'Unknown')}")
                sys.exit(1)
                
            for a_ip in attack['attacker_ips']:
                for v_ip in attack['victim_ips']:
                    a_ip_clean = str(a_ip).strip()
                    v_ip_clean = str(v_ip).strip()
                    # Soporte bidireccional
                    oracle_pairs.add((a_ip_clean, v_ip_clean))
                    oracle_pairs.add((v_ip_clean, a_ip_clean))
                    
    return oracle_pairs

def exhaustive_audit_worker(pcap_file, oracle_pairs):
    """
    Worker que escanea el PCAP buscando comunicaciones físicas entre IPs documentadas.
    """
    worker_id = os.getpid()
    print(f"[*] Worker [{worker_id}] iniciando auditoría exhaustiva: {os.path.basename(pcap_file)}")
    
    # Nomenclatura empíricamente demostrable
    documented_ip_flows = set()
    packet_count = 0
    matching_packet_count = 0
    
    real_min_timestamp = float('inf')
    real_max_timestamp = float('-inf')

    try:
        with open(pcap_file, 'rb') as f:
            try:
                pcap = dpkt.pcap.Reader(f)
            except ValueError:
                print(f"[!] Worker [{worker_id}] ERROR: PCAP corrupto o formato irreconocible en {pcap_file}")
                return None

            for timestamp, buf in pcap:
                packet_count += 1
                
                # HEARTBEAT: Telemetría cada 5M de paquetes
                if packet_count % 5000000 == 0:
                    current_time_str = datetime.fromtimestamp(timestamp, timezone.utc).strftime('%H:%M:%S UTC')
                    print(f"  [~] Heartbeat Worker [{worker_id}]: {packet_count:,} paquetes procesados. Tiempo de captura actual: {current_time_str}")

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
                    
                    # Filtrado Agnóstico de Tiempo en O(1)
                    if (src_ip_str, dst_ip_str) not in oracle_pairs:
                        continue
                        
                    matching_packet_count += 1
                    
                    # Actualizar fronteras temporales empíricas
                    if timestamp < real_min_timestamp: real_min_timestamp = timestamp
                    if timestamp > real_max_timestamp: real_max_timestamp = timestamp
                    
                    # Identificar protocolo de forma robusta (soporte IPv4 e IPv6)
                    proto = getattr(ip, "p", getattr(ip, "nxt", 0))

                    # Identificar puertos si existen
                    sport, dport = 0, 0
                    if isinstance(ip.data, dpkt.tcp.TCP) or isinstance(ip.data, dpkt.udp.UDP):
                        sport = ip.data.sport
                        dport = ip.data.dport

                    # Flujo canónico más robusto (Ordenamiento por Endpoint IP + Puerto)
                    endpoint_a = (src_ip_str, sport)
                    endpoint_b = (dst_ip_str, dport)

                    if endpoint_a <= endpoint_b:
                        canonical_flow = (src_ip_str, dst_ip_str, sport, dport, proto)
                    else:
                        canonical_flow = (dst_ip_str, src_ip_str, dport, sport, proto)

                    # Registro de nuevo flujo encontrado
                    if canonical_flow not in documented_ip_flows:
                        documented_ip_flows.add(canonical_flow)
                        dt_utc = datetime.fromtimestamp(timestamp, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                        print(f"[!] FLUJO DOCUMENTADO ENCONTRADO (#{len(documented_ip_flows)}): {canonical_flow} | Epoch: {timestamp} -> {dt_utc} UTC")

                # Restricción del Except para no silenciar bugs de código
                except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, ValueError, IndexError):
                    continue 

    except Exception as e:
        print(f"[!] Worker [{worker_id}] ERROR CRÍTICO leyendo archivo {pcap_file}: {e}")
        return None
        
    return {
        "file": os.path.basename(pcap_file),
        "total_packets": packet_count,
        "matching_packets": matching_packet_count,
        "documented_ip_flows": documented_ip_flows,
        "min_ts": real_min_timestamp,
        "max_ts": real_max_timestamp
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auditoría Forense MLOps: Existencia Física de Flujos (Time-Agnostic)")
    parser.add_argument('--pcap_dir', type=str, required=True, help="Directorio con los archivos PCAP a escanear")
    parser.add_argument('--yaml_path', type=str, default="configs/dataset_schedule.yaml", help="Ruta al oráculo YAML")
    parser.add_argument('--prefix', type=str, default="", help="Prefijo del día a analizar (ej. wed-21, wed-28)")
    args = parser.parse_args()

    if not os.path.exists(args.pcap_dir):
        print(f"[!] ERROR: El directorio {args.pcap_dir} no existe.")
        sys.exit(1)

    print("=======================================================================")
    print(f" INICIANDO AUDITORÍA FORENSE: VERIFICACIÓN FÍSICA (PREFIJO: {args.prefix if args.prefix else 'TODOS'})")
    print("=======================================================================")
    
    # Determinar la fecha objetivo basada en el prefijo para evitar cargar IPs de otros días
    target_date = PREFIX_TO_DATE.get(args.prefix)
    if args.prefix and not target_date:
        print(f"[!] ADVERTENCIA: Prefijo '{args.prefix}' no mapeado a una fecha. Se cargarán todas las IPs del YAML.")
        
    # Obtenemos un set() O(1) con pares (Atacante, Víctima) exclusivos del día
    oracle_pairs = load_oracle_ips_only(args.yaml_path, target_date)
    
    if not oracle_pairs:
        print(f"[!] ERROR: No se encontraron reglas o IPs para el día {target_date} en el YAML.")
        sys.exit(1)
        
    print(f"[*] Reglas en memoria: {len(oracle_pairs)} pares de comunicación cargados.")

    pcap_files = [
        os.path.join(args.pcap_dir, f) 
        for f in os.listdir(args.pcap_dir) 
        if f.endswith('.pcap') and f.startswith(args.prefix)
    ]
    
    if not pcap_files:
        print(f"[!] No se encontraron archivos .pcap con el prefijo '{args.prefix}' en el directorio.")
        sys.exit(1)

    max_workers = min(mp.cpu_count(), len(pcap_files))
    print(f"[*] Lanzando {max_workers} workers para escanear {len(pcap_files)} archivos...\n")
    
    # Context Manager para asegurar la liberación del pool y evitar procesos zombis
    with mp.Pool(processes=max_workers) as pool:
        resultados = pool.starmap(exhaustive_audit_worker, [(pcap, oracle_pairs) for pcap in pcap_files])

    # Consolidación Global
    global_total_packets = 0
    global_matching_packets = 0
    global_documented_flows = set()
    global_min_ts = float('inf')
    global_max_ts = float('-inf')

    for res in resultados:
        if res:
            global_total_packets += res["total_packets"]
            global_matching_packets += res["matching_packets"]
            global_documented_flows.update(res["documented_ip_flows"])
            if res["min_ts"] < global_min_ts: global_min_ts = res["min_ts"]
            if res["max_ts"] > global_max_ts: global_max_ts = res["max_ts"]

    print("\n=======================================================================")
    print(" REPORTE FINAL DE AUDITORÍA FORENSE")
    print("=======================================================================")
    print(f"Total de paquetes escaneados : {global_total_packets:,}")
    print(f"Paquetes entre IPs del YAML  : {global_matching_packets:,}")
    print(f"FLUJOS FÍSICOS DOCUMENTADOS  : {len(global_documented_flows):,}")
    print("-" * 71)
    
    if len(global_documented_flows) > 0:
        dt_start = datetime.fromtimestamp(global_min_ts, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        dt_end = datetime.fromtimestamp(global_max_ts, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        print(" VENTANA TEMPORAL EMPÍRICA (UTC) EXTRAÍDA DEL PCAP:")
        print(f"  -> Primer contacto físico : {global_min_ts} ({dt_start} UTC)")
        print(f"  -> Último contacto físico : {global_max_ts} ({dt_end} UTC)")
    else:
        print(" [!] AUSENCIA TOTAL DE TRÁFICO: Las IPs documentadas no se comunicaron en este PCAP.")
    print("=======================================================================")