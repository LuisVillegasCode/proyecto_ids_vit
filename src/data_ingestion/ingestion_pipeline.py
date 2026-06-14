import os
import sys
import re
import json
import yaml
import dpkt
import socket
import hashlib
import argparse
import multiprocessing as mp
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import numpy as np
import h5py

from src.utils.config_manager import setup_environment

# ==============================================================================
# 0. CONFIGURACIÓN GLOBAL Y ARGUMENTOS
# ==============================================================================
parser = argparse.ArgumentParser(description="Motor de Ingesta OSR-ViT")
parser.add_argument('--mode', type=str, choices=['pilot', 'prod'], required=True)
args, _ = parser.parse_known_args()

env = setup_environment(script_name="ingestion_pipeline", args=args)

YAML_PATH = env.get_path('paths', 'configs', 'dataset_schedule', ensure_exists=True, is_file=True, apply_pilot=False)
OUTPUT_DIR_TRAIN = env.get_path('paths', 'output', 'train_known', ensure_exists=True)
OUTPUT_DIR_VAL = env.get_path('paths', 'output', 'val_known', ensure_exists=True)
OUTPUT_DIR_TEST = env.get_path('paths', 'output', 'test_known', ensure_exists=True)
OUTPUT_DIR_OOD = env.get_path('paths', 'output', 'test_ood', ensure_exists=True)
INGESTION_STATE_DIR = env.get_path('paths', 'output', 'ingestion_state', ensure_exists=True)
DEAD_LETTERS_DIR = env.get_path('paths', 'data', 'dead_letters', ensure_exists=True)
TELEMETRY_LOGS = env.get_path('paths', 'artifacts', 'telemetry_logs', ensure_exists=True)

MAX_PACKETS = int(env.get_value('preprocessing', 'max_packets_per_flow'))
MAX_BYTES = int(env.get_value('preprocessing', 'max_bytes_per_packet'))
DELTA_TIME_COLUMNS = int(env.get_value('preprocessing', 'delta_time_columns'))
TENSOR_WIDTH = int(env.get_value('preprocessing', 'tensor_width'))
FLOW_TIMEOUT_SECONDS = float(env.get_value('preprocessing', 'flow_timeout_seconds'))
SWEEP_INTERVAL = int(env.get_value('preprocessing', 'sweep_interval_packets'))
BATCH_FLUSH_SIZE = int(env.get_value('preprocessing', 'batch_flush_size'))
SPLIT_RATIOS = env.get_value('preprocessing', 'split_ratios')
RETENTION_RATES = env.get_value('preprocessing', 'retention_rates')

TCP_FIN = 0x01
TCP_RST = 0x04
OOD_LABELS = {'Botnet', 'Web_Attack'}
OOD_TARGET_FOLDERS = {'hold_out_test', 'test_ood'}
KNOWN_TRUNCATED_CAPTURES = {'wed-21_UCAP172.31.69.28 part 2'}
CHUNK_PATTERN = re.compile(r'^(?P<base>.+)_chunk_(?P<index>\d{5})_(?P<timestamp>\d{14})\.pcap$')

OUTPUT_DIRECTORIES = {
    'train_known': OUTPUT_DIR_TRAIN,
    'val_known': OUTPUT_DIR_VAL,
    'test_known': OUTPUT_DIR_TEST,
    'test_ood': OUTPUT_DIR_OOD,
}
SPLIT_NAME_BY_DIR = {path: name for name, path in OUTPUT_DIRECTORIES.items()}

if TENSOR_WIDTH != MAX_BYTES + DELTA_TIME_COLUMNS:
    raise ValueError("tensor_width debe ser igual a max_bytes_per_packet + delta_time_columns")
if abs(sum(float(SPLIT_RATIOS[key]) for key in ('train', 'validation', 'test')) - 1.0) > 1e-9:
    raise ValueError("Los ratios train/validation/test deben sumar 1.0")
for split_name, rates in RETENTION_RATES.items():
    for label, rate in rates.items():
        if not 0.0 <= float(rate) <= 1.0:
            raise ValueError(f"Tasa de retención inválida: {split_name}/{label}={rate}")

# ==============================================================================
# PILAR 3: ORÁCULO DINÁMICO
# ==============================================================================
def load_time_aware_oracle(yaml_path):
    try:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(f"[!] FATAL ERROR: Oráculo {yaml_path} inaccesible.\n{e}")
        sys.exit(1)

    rules_dict = {}
    ast_tz = timezone(timedelta(hours=-4))
    for category in ['zero_day', 'closed_set']:
        if category not in config.get('attacks', {}):
            continue
        for attack in config['attacks'][category]:
            date_str = str(attack['date']).strip()
            windows_epoch = []
            for window in attack['time_windows']:
                start_dt = datetime.strptime(f"{date_str} {window[0].strip()}", "%Y-%m-%d %H:%M").replace(tzinfo=ast_tz)
                end_dt = datetime.strptime(f"{date_str} {window[1].strip()}", "%Y-%m-%d %H:%M").replace(tzinfo=ast_tz)
                windows_epoch.append((start_dt.timestamp(), end_dt.timestamp()))
            for attacker_ip in attack['attacker_ips']:
                for victim_ip in attack['victim_ips']:
                    pair = (str(attacker_ip).strip(), str(victim_ip).strip())
                    rules_dict.setdefault(pair, []).append({
                        'label': str(attack['label']).strip(),
                        'attack_subtype': str(attack['attack_name']).strip(),
                        'target_folder': str(attack['target_folder']).strip(),
                        'windows': windows_epoch,
                        'priority': 2 if category == 'zero_day' else 1,
                    })
    return rules_dict


def get_packet_label_fast(src_ip, dst_ip, ts, rules_dict):
    best_match = None
    for pair in [(src_ip, dst_ip), (dst_ip, src_ip)]:
        for attack in rules_dict.get(pair, []):
            if any(start <= ts <= end for start, end in attack['windows']):
                if best_match is None or attack['priority'] > best_match['priority']:
                    best_match = attack
    if best_match is None:
        return 'Benign', 'Benign', 'stratified_split'
    return best_match['label'], best_match['attack_subtype'], best_match['target_folder']


def _stable_fraction(value):
    digest = hashlib.sha256(value.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], byteorder='big', signed=False) / float(2 ** 64)


def get_routing_split(tuple_key, target_folder):
    if target_folder in OOD_TARGET_FOLDERS:
        return OUTPUT_DIR_OOD
    value = _stable_fraction(f"split|{tuple_key}")
    train_limit = float(SPLIT_RATIOS['train'])
    validation_limit = train_limit + float(SPLIT_RATIOS['validation'])
    if value < train_limit:
        return OUTPUT_DIR_TRAIN
    return OUTPUT_DIR_VAL if value < validation_limit else OUTPUT_DIR_TEST


def _retention_rate(split_name, label):
    return float(RETENTION_RATES.get(split_name, {}).get(label, 1.0))


def _should_keep_sample(split_name, label, split_group_key):
    rate = _retention_rate(split_name, label)
    return _stable_fraction(f"retention|{split_name}|{label}|{split_group_key}") < rate


def _label_priority(label):
    if label in OOD_LABELS:
        return 2
    return 0 if label == 'Benign' else 1


def _promote_state_label(state, packet_label, attack_subtype, target_folder):
    current_priority = _label_priority(state['label'])
    new_priority = _label_priority(packet_label)
    if new_priority > current_priority:
        state['label'] = packet_label
        state['attack_subtype'] = attack_subtype
    elif new_priority == current_priority and packet_label == state['label'] and attack_subtype != 'Benign':
        state['attack_subtype'] = attack_subtype
    if packet_label in OOD_LABELS or target_folder in OOD_TARGET_FOLDERS:
        state['label'] = packet_label
        state['attack_subtype'] = attack_subtype
        state['target_dir'] = OUTPUT_DIR_OOD

# ==============================================================================
# AGRUPACIÓN, IDEMPOTENCIA Y TRAZABILIDAD DE CAPTURAS
# ==============================================================================
def _capture_id(source_capture):
    return hashlib.sha256(source_capture.encode('utf-8')).hexdigest()[:16]


def _capture_day(source_capture):
    return source_capture.split('_', 1)[0]


def group_logical_captures(pcap_files):
    grouped = {}
    for path in sorted(pcap_files):
        filename = os.path.basename(path)
        match = CHUNK_PATTERN.match(filename)
        source_capture = match.group('base') if match else os.path.splitext(filename)[0]
        item = {'path': path, 'chunk_index': int(match.group('index')) if match else None, 'chunk_timestamp': match.group('timestamp') if match else None}
        grouped.setdefault(source_capture, []).append(item)

    tasks = []
    for source_capture, items in sorted(grouped.items()):
        has_chunks = any(item['chunk_index'] is not None for item in items)
        has_original = any(item['chunk_index'] is None for item in items)
        if has_chunks and has_original:
            raise RuntimeError(f"La captura {source_capture} contiene archivo original y chunks; se evitará duplicar tráfico")
        if has_chunks:
            items.sort(key=lambda item: (item['chunk_index'], item['chunk_timestamp']))
            indices = [item['chunk_index'] for item in items]
            if len(indices) != len(set(indices)):
                raise RuntimeError(f"Índices de chunk duplicados en {source_capture}")
            expected = set(range(0, max(indices) + 1))
            missing_chunks = sorted(expected.difference(indices))
        else:
            missing_chunks = []
        tasks.append({
            'source_capture': source_capture,
            'capture_id': _capture_id(source_capture),
            'capture_day': _capture_day(source_capture),
            'source_files': [item['path'] for item in items],
            'missing_chunks': missing_chunks,
            'known_truncated': source_capture in KNOWN_TRUNCATED_CAPTURES,
        })
    return tasks


def _normalize_capture_task(pcap_file):
    if isinstance(pcap_file, dict):
        return pcap_file
    path = str(pcap_file)
    source_capture = os.path.splitext(os.path.basename(path))[0]
    return {'source_capture': source_capture, 'capture_id': _capture_id(source_capture), 'capture_day': _capture_day(source_capture), 'source_files': [path], 'missing_chunks': [], 'known_truncated': source_capture in KNOWN_TRUNCATED_CAPTURES}


def _completion_marker_path(capture_id):
    return os.path.join(INGESTION_STATE_DIR, f"{capture_id}.done.json")


def _dead_letter_path(capture_id):
    return os.path.join(DEAD_LETTERS_DIR, f"{capture_id}.errors.json")


def _write_json_atomic(path, payload):
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _cleanup_partial_outputs(capture_id):
    marker = f"_{capture_id}_b"
    for directory in OUTPUT_DIRECTORIES.values():
        for filename in os.listdir(directory):
            if marker in filename and (filename.endswith('.hdf5') or filename.endswith('.hdf5.tmp')):
                os.remove(os.path.join(directory, filename))


def _mark_capture_shards_truncated(capture_id):
    marker = f"_{capture_id}_b"
    for directory in OUTPUT_DIRECTORIES.values():
        for filename in os.listdir(directory):
            if marker not in filename or not filename.endswith('.hdf5'):
                continue
            path = os.path.join(directory, filename)
            with h5py.File(path, 'r+') as hdf5_file:
                hdf5_file.attrs['source_truncated'] = True
                for group in hdf5_file.values():
                    group.attrs['source_truncated'] = True

# ==============================================================================
# PILAR 5: ENSAMBLADOR DE TENSORES RGB-E
# ==============================================================================
def build_and_route_tensor(flow_id, state, finalized_flows, local_metrics):
    label = state['label']
    split_dir = state['target_dir']
    split_name = SPLIT_NAME_BY_DIR[split_dir]
    if not _should_keep_sample(split_name, label, state['split_group_key']):
        local_metrics['discarded_undersampling'][label] += 1
        local_metrics['discarded_by_split'][f"{split_name}|{label}"] += 1
        return

    tensor = np.zeros((MAX_PACKETS, TENSOR_WIDTH, 3), dtype=np.float32)
    for i, pkt in enumerate(state['packets'][:MAX_PACKETS]):
        raw = pkt['raw_bytes']
        length = min(len(raw), MAX_BYTES)
        channel = 0 if pkt['is_forward'] else 1
        tensor[i, :DELTA_TIME_COLUMNS, channel] = pkt['delta_time_encoded']
        tensor[i, DELTA_TIME_COLUMNS:DELTA_TIME_COLUMNS + length, channel] = raw[:length]
        tensor[i, :, 2] = pkt['entropy']
        
    if flow_id in finalized_flows[split_dir]:
        raise RuntimeError(
        f"Colisión de session_id detectada antes del flush: "
        f"{flow_id} | {state['source_capture']} | "
        f"{state['canonical_tuple']} | instancia={state['session_instance']}"
    )    

    finalized_flows[split_dir][flow_id] = {
        'tensor': tensor,
        'label': label,
        'attack_subtype': state['attack_subtype'],
        'split': split_name,
        'session_id': flow_id,
        'session_instance': state['session_instance'],
        'canonical_tuple': state['canonical_tuple'],
        'split_group_key': state['split_group_key'],
        'capture_day': state['capture_day'],
        'source_capture': state['source_capture'],
        'source_files': state['source_files'],
        'start_timestamp': state['start_time'],
        'end_timestamp': state['last_time'],
        'captured_packets': len(state['packets']),
        'observed_packets': state['observed_packets'],
        'source_truncated': state['source_truncated'],
    }
    local_metrics['generated_tensors'][label] += 1
    local_metrics['generated_by_split'][f"{split_name}|{label}"] += 1
    local_metrics['generated_by_subtype'][state['attack_subtype']] += 1

# ==============================================================================
# FLUSH HDF5 (Solución Problema 4)
# ==============================================================================
def flush_tensors_hdf5(prefix, target_dir, data_dict, worker_id, filename, batch_id):
    if not data_dict:
        return
    capture_id = filename
    final_path = os.path.join(target_dir, f"{prefix}_{capture_id}_b{batch_id:05d}.hdf5")
    tmp_path = final_path + '.tmp'
    with h5py.File(tmp_path, 'w') as hf:
        hf.attrs['capture_id'] = capture_id
        for flow_id, payload in data_dict.items():
            safe_id = str(flow_id).replace('/', '_').replace('\\', '_')
            grp = hf.create_group(safe_id)
            for attr_name in ['label', 'attack_subtype', 'split', 'session_id', 'session_instance', 'canonical_tuple', 'split_group_key', 'capture_day', 'source_capture']:
                grp.attrs[attr_name] = payload[attr_name]
            grp.attrs['source_files'] = json.dumps(payload['source_files'], ensure_ascii=False)
            grp.attrs['start_timestamp'] = payload['start_timestamp']
            grp.attrs['end_timestamp'] = payload['end_timestamp']
            grp.attrs['captured_packets'] = payload['captured_packets']
            grp.attrs['observed_packets'] = payload['observed_packets']
            grp.attrs['source_truncated'] = payload['source_truncated']
            grp.create_dataset('rgb_e_tensor', data=payload['tensor'], compression='lzf')
    os.replace(tmp_path, final_path)
    data_dict.clear()


def _flush_all(finalized_flows, worker_id, capture_id, batch_id):
    for split_name, target_dir in OUTPUT_DIRECTORIES.items():
        flush_tensors_hdf5(split_name, target_dir, finalized_flows[target_dir], worker_id, capture_id, batch_id)


def _encode_delta_time(delta_time):
    clipped = min(max(float(delta_time), 0.0), FLOW_TIMEOUT_SECONDS)
    return float(255.0 * np.log1p(clipped) / np.log1p(FLOW_TIMEOUT_SECONDS))

# ==============================================================================
# TRABAJADOR PRINCIPAL
# ==============================================================================
def process_pcap_chunk(pcap_file, rules_dict):
    task = _normalize_capture_task(pcap_file)
    source_capture = task['source_capture']
    capture_id = task['capture_id']
    capture_day = task['capture_day']
    source_files = task['source_files']
    source_file_names = [os.path.basename(path) for path in source_files]
    worker_id = os.getpid()
    marker_path = _completion_marker_path(capture_id)

    print(f"[*] Worker [{worker_id}] procesando captura lógica: {source_capture} ({len(source_files)} archivo(s))")
    _cleanup_partial_outputs(capture_id)

    flow_states = {}
    session_instance_counters = defaultdict(int)
    finalized_flows = {directory: {} for directory in OUTPUT_DIRECTORIES.values()}
    local_metrics = {
        'generated_tensors': defaultdict(int),
        'discarded_undersampling': defaultdict(int),
        'generated_by_split': defaultdict(int),
        'discarded_by_split': defaultdict(int),
        'generated_by_subtype': defaultdict(int),
    }
    packet_count = 0
    batch_counter = 0
    error_summary = defaultdict(int)
    source_truncated = bool(task['known_truncated'] or task['missing_chunks'])
    critical_error = None

    def finalize_state(canonical_tuple):
        state = flow_states.get(canonical_tuple)
        if state is not None and state['packets']:
            state['source_truncated'] = source_truncated
            build_and_route_tensor(state['session_id'], state, finalized_flows, local_metrics)
        flow_states.pop(canonical_tuple, None)

    def register_recoverable_error(error, current_file):
        nonlocal source_truncated
        source_truncated = True
        key = f"{os.path.basename(current_file)} | {type(error).__name__}: {str(error)[:160]}"
        error_summary[key] += 1
        for state in flow_states.values():
            state['source_truncated'] = True

    try:
        for source_file in source_files:
            try:
                with open(source_file, 'rb') as file:
                    pcap = dpkt.pcap.Reader(file)
                    while True:
                        try:
                            timestamp, buf = next(pcap)
                        except StopIteration:
                            break
                        except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, ValueError) as error:
                            register_recoverable_error(error, source_file)
                            break

                        packet_count += 1
                        if packet_count % 10000 == 0:
                            total_in_ram = sum(len(data) for data in finalized_flows.values())
                            if total_in_ram >= BATCH_FLUSH_SIZE:
                                batch_counter += 1
                                _flush_all(finalized_flows, worker_id, capture_id, batch_counter)

                        if packet_count % SWEEP_INTERVAL == 0:
                            expired_flows = [fid for fid, state in flow_states.items() if timestamp - state['last_time'] > FLOW_TIMEOUT_SECONDS]
                            for fid in expired_flows:
                                finalize_state(fid)

                        try:
                            eth = dpkt.ethernet.Ethernet(buf)
                            if not isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
                                continue
                            ip = eth.data
                            if isinstance(ip, dpkt.ip.IP):
                                src_ip_str = socket.inet_ntoa(ip.src)
                                dst_ip_str = socket.inet_ntoa(ip.dst)
                            else:
                                src_ip_str = socket.inet_ntop(socket.AF_INET6, ip.src)
                                dst_ip_str = socket.inet_ntop(socket.AF_INET6, ip.dst)

                            proto = getattr(ip, 'p', getattr(ip, 'nxt', 0))
                            if not isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                                continue
                            transport = ip.data
                            sport, dport = transport.sport, transport.dport
                            is_teardown = isinstance(transport, dpkt.tcp.TCP) and bool(transport.flags & (TCP_FIN | TCP_RST))

                            if (src_ip_str, sport) <= (dst_ip_str, dport):
                                canonical_tuple = f"{src_ip_str}-{dst_ip_str}-{sport}-{dport}-{proto}"
                            else:
                                canonical_tuple = f"{dst_ip_str}-{src_ip_str}-{dport}-{sport}-{proto}"

                            packet_label, attack_subtype, target_folder = get_packet_label_fast(src_ip_str, dst_ip_str, timestamp, rules_dict)

                            if canonical_tuple in flow_states and timestamp - flow_states[canonical_tuple]['last_time'] > FLOW_TIMEOUT_SECONDS:
                                finalize_state(canonical_tuple)

                            if canonical_tuple not in flow_states:
                                session_instance_counters[canonical_tuple] += 1
                                session_instance = session_instance_counters[canonical_tuple]
                                first_timestamp_us = int(round(timestamp * 1_000_000))
                                session_material = f"{source_capture}|{canonical_tuple}|{first_timestamp_us}|{session_instance}"
                                session_id = hashlib.sha256(session_material.encode('utf-8')).hexdigest()[:24]
                                split_group_key = hashlib.sha256(f"{capture_day}|{canonical_tuple}".encode('utf-8')).hexdigest()
                                
                                flow_states[canonical_tuple] = {
                                    'session_id': session_id,
                                    'session_instance': session_instance,
                                    'canonical_tuple': canonical_tuple,
                                    'split_group_key': split_group_key,
                                    'capture_day': capture_day,
                                    'source_capture': source_capture,
                                    'source_files': source_file_names,
                                    'packets': [],
                                    'start_time': timestamp,
                                    'last_time': timestamp,
                                    'last_captured_time': None,
                                    'observed_packets': 0,
                                    'status': 'CAPTURING',
                                    'label': packet_label,
                                    'attack_subtype': attack_subtype,
                                    'target_dir': get_routing_split(split_group_key, target_folder),
                                    'initiator_ip': src_ip_str,
                                    'initiator_port': sport,
                                    'source_truncated': source_truncated,
                                }

                            state = flow_states[canonical_tuple]
                            state['observed_packets'] += 1
                            state['last_time'] = timestamp
                            _promote_state_label(state, packet_label, attack_subtype, target_folder)

                            if state['status'] == 'CAPTURING':
                                delta_time = 0.0 if state['last_captured_time'] is None else max(0.0, timestamp - state['last_captured_time'])
                                payload = transport.data
                                entropy = 0.0
                                if payload:
                                    byte_counts = np.bincount(np.frombuffer(payload, dtype=np.uint8), minlength=256)
                                    probabilities = byte_counts[byte_counts > 0] / len(payload)
                                    entropy = -np.sum(probabilities * np.log2(probabilities))

                                eth.src = b'\x00' * 6
                                eth.dst = b'\x00' * 6
                                if isinstance(ip, dpkt.ip.IP):
                                    ip.src = b'\x00' * 4
                                    ip.dst = b'\x00' * 4
                                else:
                                    ip.src = b'\x00' * 16
                                    ip.dst = b'\x00' * 16

                                state['packets'].append({
                                    'entropy': float(entropy),
                                    'delta_time_encoded': _encode_delta_time(delta_time),
                                    'raw_bytes': np.frombuffer(bytes(eth)[:MAX_BYTES], dtype=np.uint8).copy(),
                                    'is_forward': src_ip_str == state['initiator_ip'] and sport == state['initiator_port'],
                                })
                                state['last_captured_time'] = timestamp
                                if len(state['packets']) >= MAX_PACKETS:
                                    state['status'] = 'IGNORING'

                            if is_teardown:
                                finalize_state(canonical_tuple)

                        except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, ValueError, IndexError, OSError) as error:
                            key = f"{os.path.basename(source_file)} | {type(error).__name__}: {str(error)[:160]}"
                            error_summary[key] += 1
                            continue
            except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, ValueError) as error:
                register_recoverable_error(error, source_file)
                continue
    except Exception as error:
        critical_error = f"{type(error).__name__}: {error}"

    if critical_error is not None:
        _write_json_atomic(_dead_letter_path(capture_id), {
            'source_capture': source_capture,
            'source_files': source_file_names,
            'source_truncated': source_truncated,
            'missing_chunks': task['missing_chunks'],
            'critical_error': critical_error,
            'packet_error_count': int(sum(error_summary.values())),
            'error_types': dict(error_summary),
            'valid_packets_processed': packet_count,
        })
        _cleanup_partial_outputs(capture_id)
        print(f"[!] Worker [{worker_id}] falló en {source_capture}: {critical_error}")
        return {'success': False, 'source_capture': source_capture, 'critical_error': critical_error, 'generated_tensors': {}, 'discarded_undersampling': {}, 'generated_by_split': {}, 'discarded_by_split': {}, 'generated_by_subtype': {}}

    for canonical_tuple in list(flow_states.keys()):
        finalize_state(canonical_tuple)

    batch_counter += 1
    _flush_all(finalized_flows, worker_id, capture_id, batch_counter)
    if source_truncated:
        _mark_capture_shards_truncated(capture_id)

    dead_letter = _dead_letter_path(capture_id)
    if source_truncated or error_summary:
        _write_json_atomic(dead_letter, {
            'source_capture': source_capture,
            'source_files': source_file_names,
            'source_truncated': source_truncated,
            'missing_chunks': task['missing_chunks'],
            'critical_error': None,
            'packet_error_count': int(sum(error_summary.values())),
            'error_types': dict(error_summary),
            'valid_packets_processed': packet_count,
        })
    elif os.path.exists(dead_letter):
        os.remove(dead_letter)

    result = {
        'success': True,
        'source_capture': source_capture,
        'source_truncated': source_truncated,
        'packets_read': packet_count,
        'packet_error_count': int(sum(error_summary.values())),
        'generated_tensors': dict(local_metrics['generated_tensors']),
        'discarded_undersampling': dict(local_metrics['discarded_undersampling']),
        'generated_by_split': dict(local_metrics['generated_by_split']),
        'discarded_by_split': dict(local_metrics['discarded_by_split']),
        'generated_by_subtype': dict(local_metrics['generated_by_subtype']),
    }
    _write_json_atomic(marker_path, result)
    print(f"[✓] Worker [{worker_id}] completó {source_capture}. Generados: {sum(result['generated_tensors'].values())}")
    return result

# ==============================================================================
# ORQUESTADOR MLOps
# ==============================================================================
if __name__ == "__main__":
    input_dir = env.get_path('paths', 'data', 'pilot' if env.mode == 'pilot' else 'raw_chunks', ensure_exists=True)
    if not os.path.isdir(input_dir):
        print(f"[!] FATAL ERROR: Directorio de entrada '{input_dir}' no existe.")
        sys.exit(1)

    print("=======================================================")
    print(f" MOTOR OSR-VIT: RGB-E TENSOR FACTORY (MODO: {env.mode.upper()})")
    print("=======================================================")

    rules_dict = load_time_aware_oracle(YAML_PATH)
    pcap_files = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir) if filename.endswith('.pcap')]
    if not pcap_files:
        print(f"[*] ALERTA: No hay archivos PCAP en {input_dir}.")
        sys.exit(0)

    capture_tasks = group_logical_captures(pcap_files)
    pending_tasks = [task for task in capture_tasks if not os.path.exists(_completion_marker_path(task['capture_id']))]
    print(f"[*] Capturas lógicas: {len(capture_tasks)} | Completadas: {len(capture_tasks) - len(pending_tasks)} | Pendientes: {len(pending_tasks)}")
    if not pending_tasks:
        print("[*] Todas las capturas poseen marcador de finalización válido.")
        sys.exit(0)

    configured_workers = int(env.get_value('preprocessing', 'multiprocessing_workers'))
    max_workers = min(configured_workers, max(1, (os.cpu_count() or 2) - 1), len(pending_tasks))
    print(f"[*] Desplegando Pool con {max_workers} workers para {len(pending_tasks)} capturas pendientes...")
    with mp.Pool(processes=max_workers) as pool:
        resultados = pool.starmap(process_pcap_chunk, [(task, rules_dict) for task in pending_tasks])

    gen_counts = defaultdict(int)
    disc_counts = defaultdict(int)
    generated_by_split = defaultdict(int)
    discarded_by_split = defaultdict(int)
    generated_by_subtype = defaultdict(int)
    failed_captures = []
    truncated_captures = 0
    total_packets = 0
    total_packet_errors = 0

    for result in resultados:
        if not result or not result.get('success', False):
            if result:
                failed_captures.append(result.get('source_capture', 'unknown'))
            continue
        truncated_captures += int(result.get('source_truncated', False))
        total_packets += int(result.get('packets_read', 0))
        total_packet_errors += int(result.get('packet_error_count', 0))
        for label, count in result['generated_tensors'].items():
            gen_counts[label] += count
        for label, count in result['discarded_undersampling'].items():
            disc_counts[label] += count
        for key, count in result['generated_by_split'].items():
            generated_by_split[key] += count
        for key, count in result['discarded_by_split'].items():
            discarded_by_split[key] += count
        for subtype, count in result['generated_by_subtype'].items():
            generated_by_subtype[subtype] += count

    report_lines = [
        "================================================================================",
        " REPORTE FINAL DE INGESTA OSR-ViT",
        "================================================================================",
        "TENSORES GUARDADOS POR SPLIT Y CLASE",
        "--------------------------------------------------------------------------------",
    ]
    for key in sorted(generated_by_split):
        split_name, label = key.split('|', 1)
        report_lines.append(f"{split_name:<18} | {label:<18} | {generated_by_split[key]:>12,}")

    report_lines.extend(["", "DESCARTADOS POR RETENCIÓN DETERMINISTA", "--------------------------------------------------------------------------------"])
    for key in sorted(discarded_by_split):
        split_name, label = key.split('|', 1)
        report_lines.append(f"{split_name:<18} | {label:<18} | {discarded_by_split[key]:>12,}")

    web_subtypes = {name: count for name, count in generated_by_subtype.items() if name in {'Brute Force -Web', 'Brute Force -XSS', 'SQL Injection'}}
    if web_subtypes:
        report_lines.extend(["", "DESGLOSE OOD WEB_ATTACK", "--------------------------------------------------------------------------------"])
        for subtype in sorted(web_subtypes):
            report_lines.append(f"{subtype:<39} | {web_subtypes[subtype]:>12,}")

    report_lines.extend([
        "", "--------------------------------------------------------------------------------",
        f"TOTAL TENSORES GUARDADOS : {sum(gen_counts.values()):,}",
        f"TOTAL TENSORES DESCARTADOS: {sum(disc_counts.values()):,}",
        f"PAQUETES VÁLIDOS LEÍDOS   : {total_packets:,}",
        f"ERRORES DE PARSEO          : {total_packet_errors:,}",
        f"CAPTURAS TRUNCADAS         : {truncated_captures:,}",
        f"CAPTURAS FALLIDAS          : {len(failed_captures):,}",
        "================================================================================",
    ])
    if failed_captures:
        report_lines.append("Ejemplos fallidos: " + ", ".join(failed_captures[:5]))

    report_text = "\n".join(report_lines)
    print(report_text)
    report_path = os.path.join(TELEMETRY_LOGS, f"tensor_distribution_{env.mode}.txt")
    tmp_report_path = report_path + '.tmp'
    with open(tmp_report_path, 'w', encoding='utf-8') as file:
        file.write(report_text)
    os.replace(tmp_report_path, report_path)
    print(f"[*] Telemetría guardada en: {report_path}")
    if failed_captures:
        sys.exit(2)
