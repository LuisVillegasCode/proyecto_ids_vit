# src/preprocessing/global_scaler.py
import os
import sys
import json
import h5py
import hashlib
import argparse
import subprocess
import numpy as np
from datetime import datetime, timezone
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from src.utils.config_manager import setup_environment

# ==============================================================================
# 0. INYECCIÓN DE ENTORNO (PILOTO / PRODUCCIÓN)
# ==============================================================================
parser = argparse.ArgumentParser(description="Perfilador Global OSR-ViT (MinMax Scaler)")
parser.add_argument('--mode', type=str, choices=['pilot', 'prod'], required=True, help="Define si se leen los datos del entorno piloto o de producción")
args, _ = parser.parse_known_args()

env = setup_environment(script_name="global_scaler", args=args)

TRAIN_DIR = env.get_path('paths', 'output', 'train_known', ensure_exists=True)
INGESTION_STATE_DIR = env.get_path('paths', 'output', 'ingestion_state', ensure_exists=True)
OUTPUT_JSON = env.get_path('paths', 'configs', 'scaler_bounds', ensure_exists=True, is_file=True)
CHECKPOINT_FILE = OUTPUT_JSON.replace(".json", "_checkpoint.json")

MAX_PACKETS = int(env.get_value('preprocessing', 'max_packets_per_flow'))
MAX_BYTES = int(env.get_value('preprocessing', 'max_bytes_per_packet'))
DELTA_TIME_COLUMNS = int(env.get_value('preprocessing', 'delta_time_columns'))
TENSOR_WIDTH = int(env.get_value('preprocessing', 'tensor_width'))
EXPECTED_SHAPE = (MAX_PACKETS, TENSOR_WIDTH, 3)
TAXONOMY = tuple(env.get_value('preprocessing', 'retention_rates', 'train_known').keys())
SOURCE_SPLIT = 'train_known'
SCHEMA_VERSION = 2

if TENSOR_WIDTH != MAX_BYTES + DELTA_TIME_COLUMNS:
    raise ValueError("tensor_width debe ser igual a max_bytes_per_packet + delta_time_columns")
if not TAXONOMY:
    raise ValueError("La taxonomía de train_known no puede estar vacía")

# ==============================================================================
# 1. UTILIDADES DE TRAZABILIDAD, MANIFIESTO Y CHECKPOINT
# ==============================================================================
def _write_json_atomic(path, payload):
    tmp_path = path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, sort_keys=True, ensure_ascii=False)
    os.replace(tmp_path, path)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata():
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], check=True, capture_output=True, text=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return 'unavailable', None


def _list_train_files():
    train_files = []
    for root, dirs, files in os.walk(TRAIN_DIR):
        dirs.sort()
        for filename in sorted(files):
            if filename.endswith('.hdf5'):
                train_files.append(os.path.relpath(os.path.join(root, filename), TRAIN_DIR))
    return train_files


def _build_manifest(train_files):
    manifest = []
    for relative_path in train_files:
        full_path = os.path.join(TRAIN_DIR, relative_path)
        try:
            with h5py.File(full_path, 'r', swmr=True) as hdf5_file:
                tensor_count = len(hdf5_file)
            manifest.append({'relative_path': relative_path, 'size_bytes': os.path.getsize(full_path), 'tensor_count': int(tensor_count)})
        except Exception as error:
            raise RuntimeError(f"No se pudo construir el manifiesto para {relative_path}: {type(error).__name__}: {error}") from error
    return manifest


def _manifest_hash(manifest):
    canonical = json.dumps(manifest, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(canonical).hexdigest()


def _load_expected_ingestion_counts():
    marker_files = sorted(
        os.path.join(INGESTION_STATE_DIR, filename)
        for filename in os.listdir(INGESTION_STATE_DIR)
        if filename.endswith('.done.json')
    )
    if not marker_files:
        raise RuntimeError(f"No existen marcadores .done.json en {INGESTION_STATE_DIR}; no puede validarse la integridad de train_known")

    counts = Counter()
    for marker_path in marker_files:
        with open(marker_path, 'r', encoding='utf-8') as file:
            marker = json.load(file)
        if marker.get('success') is not True:
            raise RuntimeError(f"Marcador de ingesta no exitoso: {marker_path}")
        for key, value in marker.get('generated_by_split', {}).items():
            split_name, separator, label = key.partition('|')
            if separator and split_name == SOURCE_SPLIT:
                counts[label] += int(value)

    unexpected = sorted(set(counts).difference(TAXONOMY))
    if unexpected:
        raise RuntimeError(f"Los marcadores contienen etiquetas no permitidas en train_known: {unexpected}")
    if any(counts.get(label, 0) <= 0 for label in TAXONOMY):
        missing = [label for label in TAXONOMY if counts.get(label, 0) <= 0]
        raise RuntimeError(f"Clases sin muestras esperadas en los marcadores: {missing}")
    return marker_files, {label: int(counts.get(label, 0)) for label in TAXONOMY}


def _checkpoint_identity(manifest_sha256, git_commit):
    return {
        'schema_version': SCHEMA_VERSION,
        'mode': env.mode,
        'source_split': SOURCE_SPLIT,
        'taxonomy': list(TAXONOMY),
        'expected_shape': list(EXPECTED_SHAPE),
        'manifest_sha256': manifest_sha256,
        'git_commit': git_commit,
        'script_sha256': _sha256_file(__file__),
        'config_sha256': _sha256_file('configs/global_config.yaml'),
    }


def _new_state(identity):
    return {
        **identity,
        'global_min_entropy': None, 'global_max_entropy': None,
        'global_min_raw': None, 'global_max_raw': None,
        'global_min_delta': None, 'global_max_delta': None,
        'global_min_packet': None, 'global_max_packet': None,
        'processed_files': [], 'valid_tensor_count': 0, 'invalid_tensor_count': 0,
        'class_counts': {label: 0 for label in TAXONOMY}, 'last_failed_files': {},
    }


def _load_or_create_state(identity, manifest_paths):
    if not os.path.exists(CHECKPOINT_FILE):
        return _new_state(identity)

    with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as file:
        state = json.load(file)

    mismatches = [key for key, value in identity.items() if state.get(key) != value]
    if mismatches:
        raise RuntimeError(f"Checkpoint incompatible con la ejecución actual. Campos distintos: {mismatches}. Archive o elimine {CHECKPOINT_FILE}")

    processed = set(state.get('processed_files', []))
    unknown = sorted(processed.difference(manifest_paths))
    if unknown:
        raise RuntimeError(f"El checkpoint contiene archivos ajenos al manifiesto actual: {unknown[:5]}")
    if int(state.get('valid_tensor_count', 0)) != sum(int(value) for value in state.get('class_counts', {}).values()):
        raise RuntimeError("Checkpoint inconsistente: valid_tensor_count no coincide con class_counts")

    state['last_failed_files'] = {}
    print(f"[*] Rescatando sesión interrumpida. Ya procesados: {len(processed)}")
    return state


def _update_bound(state, min_key, max_key, local_min, local_max):
    if state[min_key] is None:
        state[min_key], state[max_key] = float(local_min), float(local_max)
    else:
        state[min_key] = min(float(state[min_key]), float(local_min))
        state[max_key] = max(float(state[max_key]), float(local_max))


def _merge_successful_result(state, result):
    _update_bound(state, 'global_min_entropy', 'global_max_entropy', result['min_entropy'], result['max_entropy'])
    _update_bound(state, 'global_min_raw', 'global_max_raw', result['min_raw'], result['max_raw'])
    _update_bound(state, 'global_min_delta', 'global_max_delta', result['min_delta'], result['max_delta'])
    _update_bound(state, 'global_min_packet', 'global_max_packet', result['min_packet'], result['max_packet'])
    state['valid_tensor_count'] += int(result['valid_tensors'])
    for label, count in result['class_counts'].items():
        state['class_counts'][label] = int(state['class_counts'].get(label, 0)) + int(count)
    state['processed_files'].append(result['relative_path'])

# ==============================================================================
# 2. EL TRABAJADOR DE CPU (EXTRACCIÓN VECTORIZADA)
# ==============================================================================
def process_single_file(file_path):
    """Lee tensores (MAX_PACKETS, TENSOR_WIDTH, 3), valida integridad y extrae límites Min-Max."""
    relative_path = os.path.relpath(file_path, TRAIN_DIR)
    result = {
        'relative_path': relative_path, 'success': False, 'error': None, 'invalid_examples': [],
        'valid_tensors': 0, 'invalid_tensors': 0, 'class_counts': {},
        'min_entropy': float('inf'), 'max_entropy': float('-inf'),
        'min_raw': float('inf'), 'max_raw': float('-inf'),
        'min_delta': float('inf'), 'max_delta': float('-inf'),
        'min_packet': float('inf'), 'max_packet': float('-inf'),
    }
    class_counts = Counter()

    def invalidate(flow_id, reason):
        result['invalid_tensors'] += 1
        if len(result['invalid_examples']) < 5:
            result['invalid_examples'].append(f"{flow_id}: {reason}")

    try:
        with h5py.File(file_path, 'r', swmr=True) as hf:
            if len(hf) == 0:
                result['error'] = 'Archivo HDF5 sin grupos'
                return result

            for flow_id in hf.keys():
                grp = hf[flow_id]
                if 'rgb_e_tensor' not in grp:
                    invalidate(flow_id, "dataset rgb_e_tensor ausente")
                    continue

                tensor = grp['rgb_e_tensor'][:]
                if tensor.shape != EXPECTED_SHAPE:
                    invalidate(flow_id, f"forma {tensor.shape}, esperada {EXPECTED_SHAPE}")
                    continue
                if not np.isfinite(tensor).all():
                    invalidate(flow_id, "contiene NaN o infinito")
                    continue

                label = grp.attrs.get('label', '')
                if isinstance(label, bytes):
                    label = label.decode('utf-8', errors='replace')
                label = str(label)
                if label not in TAXONOMY:
                    invalidate(flow_id, f"etiqueta no permitida: {label!r}")
                    continue

                raw_channels = tensor[..., 0:2]
                delta_block = raw_channels[:, :DELTA_TIME_COLUMNS, :]
                packet_block = raw_channels[:, DELTA_TIME_COLUMNS:, :]
                entropy_channel = tensor[..., 2]

                min_raw, max_raw = float(np.nanmin(raw_channels)), float(np.nanmax(raw_channels))
                min_delta, max_delta = float(np.nanmin(delta_block)), float(np.nanmax(delta_block))
                min_packet, max_packet = float(np.nanmin(packet_block)), float(np.nanmax(packet_block))
                min_entropy, max_entropy = float(np.nanmin(entropy_channel)), float(np.nanmax(entropy_channel))

                if min_raw < -1e-6 or max_raw > 255.0 + 1e-6:
                    invalidate(flow_id, f"canales direccionales fuera de [0,255]: [{min_raw},{max_raw}]")
                    continue
                if min_entropy < -1e-6 or max_entropy > 8.0 + 1e-6:
                    invalidate(flow_id, f"entropía fuera de [0,8]: [{min_entropy},{max_entropy}]")
                    continue

                result['min_raw'], result['max_raw'] = min(result['min_raw'], min_raw), max(result['max_raw'], max_raw)
                result['min_delta'], result['max_delta'] = min(result['min_delta'], min_delta), max(result['max_delta'], max_delta)
                result['min_packet'], result['max_packet'] = min(result['min_packet'], min_packet), max(result['max_packet'], max_packet)
                result['min_entropy'], result['max_entropy'] = min(result['min_entropy'], min_entropy), max(result['max_entropy'], max_entropy)
                result['valid_tensors'] += 1
                class_counts[label] += 1

        result['class_counts'] = dict(class_counts)
        if result['invalid_tensors'] > 0:
            result['error'] = f"Se encontraron {result['invalid_tensors']} tensores inválidos"
            return result
        if result['valid_tensors'] == 0:
            result['error'] = 'No se encontraron tensores válidos'
            return result

        result['success'] = True
        return result
    except Exception as error:
        result['error'] = f"{type(error).__name__}: {error}"
        return result

# ==============================================================================
# 3. ORQUESTADOR MLOPS
# ==============================================================================
def calculate_global_bounds():
    if not os.path.isdir(TRAIN_DIR):
        print(f"[!] ERROR: El directorio de entrenamiento '{TRAIN_DIR}' no existe.")
        print("    Asegúrate de haber ejecutado ingestion_pipeline primero en este modo.")
        sys.exit(1)

    print("=======================================================")
    print(f" INICIANDO PERFILAMIENTO GLOBAL (MODO: {env.mode.upper()})")
    print("=======================================================")

    train_files = _list_train_files()
    if not train_files:
        raise RuntimeError(f"No se encontraron archivos HDF5 en {TRAIN_DIR}")

    print(f"[*] Construyendo manifiesto determinista de {len(train_files)} archivos...")
    manifest = _build_manifest(train_files)
    manifest_sha256 = _manifest_hash(manifest)
    manifest_tensor_count = sum(item['tensor_count'] for item in manifest)

    marker_files, expected_class_counts = _load_expected_ingestion_counts()
    expected_tensor_count = sum(expected_class_counts.values())
    if manifest_tensor_count != expected_tensor_count:
        raise RuntimeError(
            f"Integridad fallida antes del scaler: HDF5={manifest_tensor_count:,}, "
            f"marcadores de ingesta={expected_tensor_count:,}"
        )

    git_commit, git_dirty = _git_metadata()
    identity = _checkpoint_identity(manifest_sha256, git_commit)
    manifest_paths = {item['relative_path'] for item in manifest}
    state = _load_or_create_state(identity, manifest_paths)
    processed_files = set(state['processed_files'])
    files_to_process = [relative_path for relative_path in train_files if relative_path not in processed_files]

    configured_workers = int(env.get_value('preprocessing', 'multiprocessing_workers'))
    available_workers = max(1, (os.cpu_count() or 2) - 1)
    max_cores = min(configured_workers, available_workers, max(1, len(files_to_process)))

    if files_to_process:
        print(f"[*] Escaneando {len(files_to_process)} archivos con {max_cores} motores paralelos...")
        processed_count = 0
        failed_files = {}

        with ProcessPoolExecutor(max_workers=max_cores) as executor:
            futures = {executor.submit(process_single_file, os.path.join(TRAIN_DIR, relative_path)): relative_path for relative_path in files_to_process}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Perfilando tensores"):
                relative_path = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    failed_files[relative_path] = {'error': f"{type(error).__name__}: {error}", 'invalid_examples': []}
                    continue

                if not result['success']:
                    failed_files[relative_path] = {'error': result['error'], 'invalid_examples': result['invalid_examples']}
                    continue

                _merge_successful_result(state, result)
                processed_count += 1
                if processed_count % 50 == 0:
                    state['processed_files'] = sorted(set(state['processed_files']))
                    state['last_failed_files'] = failed_files
                    _write_json_atomic(CHECKPOINT_FILE, state)

        state['processed_files'] = sorted(set(state['processed_files']))
        state['last_failed_files'] = failed_files
        _write_json_atomic(CHECKPOINT_FILE, state)

        if failed_files:
            examples = list(failed_files.items())[:5]
            raise RuntimeError(f"Fallaron {len(failed_files)} HDF5. Ejemplos: {examples}. Consulte {CHECKPOINT_FILE}")
    else:
        print("[*] Todos los archivos del manifiesto ya estaban perfilados; se reconstruirá la salida final.")

    if set(state['processed_files']) != manifest_paths:
        pending = sorted(manifest_paths.difference(state['processed_files']))
        raise RuntimeError(f"Perfilamiento incompleto. Archivos pendientes: {pending[:5]}")

    final_manifest = _build_manifest(train_files)
    final_manifest_sha256 = _manifest_hash(final_manifest)
    if final_manifest_sha256 != manifest_sha256:
        raise RuntimeError("El manifiesto cambió durante el perfilamiento; no se generará el scaler")

    if int(state['invalid_tensor_count']) != 0:
        raise RuntimeError(f"Se registraron {state['invalid_tensor_count']} tensores inválidos")
    if int(state['valid_tensor_count']) != manifest_tensor_count:
        raise RuntimeError(f"Conteo inconsistente: procesados={state['valid_tensor_count']:,}, manifiesto={manifest_tensor_count:,}")

    actual_class_counts = {label: int(state['class_counts'].get(label, 0)) for label in TAXONOMY}
    if actual_class_counts != expected_class_counts:
        raise RuntimeError(f"Distribución inconsistente. HDF5={actual_class_counts}, marcadores={expected_class_counts}")

    bounds_to_validate = [
        ('entropy', state['global_min_entropy'], state['global_max_entropy'], 0.0, 8.0),
        ('raw', state['global_min_raw'], state['global_max_raw'], 0.0, 255.0),
        ('delta', state['global_min_delta'], state['global_max_delta'], 0.0, 255.0),
        ('packet', state['global_min_packet'], state['global_max_packet'], 0.0, 255.0),
    ]
    for name, minimum, maximum, lower, upper in bounds_to_validate:
        if minimum is None or maximum is None or not np.isfinite([minimum, maximum]).all():
            raise RuntimeError(f"Límites no finitos o ausentes para {name}")
        if minimum > maximum or minimum < lower - 1e-6 or maximum > upper + 1e-6:
            raise RuntimeError(f"Límites inválidos para {name}: [{minimum}, {maximum}]")

    created_at_utc = datetime.now(timezone.utc).isoformat()
    bounds = {
        'entropy_channel': {'min': state['global_min_entropy'], 'max': state['global_max_entropy']},
        'raw_bytes_channel': {'min': state['global_min_raw'], 'max': state['global_max_raw']},
        'delta_time_block': {'min': state['global_min_delta'], 'max': state['global_max_delta'], 'columns': [0, DELTA_TIME_COLUMNS - 1]},
        'packet_bytes_block': {'min': state['global_min_packet'], 'max': state['global_max_packet'], 'columns': [DELTA_TIME_COLUMNS, TENSOR_WIDTH - 1]},
        'metadata': {
            'schema_version': SCHEMA_VERSION,
            'created_at_utc': created_at_utc,
            'mode': env.mode,
            'source_split': SOURCE_SPLIT,
            'source_directory': TRAIN_DIR,
            'taxonomy': list(TAXONOMY),
            'git_commit': git_commit,
            'git_dirty': git_dirty,
            'script_sha256': identity['script_sha256'],
            'config_sha256': identity['config_sha256'],
            'manifest_sha256': manifest_sha256,
            'tensor_shape': list(EXPECTED_SHAPE),
            'tensor_count': int(state['valid_tensor_count']),
            'invalid_tensor_count': int(state['invalid_tensor_count']),
            'hdf5_file_count': len(manifest),
            'class_counts': actual_class_counts,
            'ingestion_marker_count': len(marker_files),
            'expected_tensor_count_from_ingestion': expected_tensor_count,
        },
        'manifest': {'sha256': manifest_sha256, 'files': manifest},
    }

    _write_json_atomic(OUTPUT_JSON, bounds)
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    print("=======================================================")
    print("[✓] PERFILAMIENTO MULTIPROCESO COMPLETADO Y VALIDADO.")
    print(f"[*] Archivo guardado en: {OUTPUT_JSON}")
    print(f"[*] Tensores: {state['valid_tensor_count']:,} | HDF5: {len(manifest):,}")
    print(f"[*] Taxonomía: {actual_class_counts}")
    print(f"[*] Manifest SHA-256: {manifest_sha256}")
    print(f"[*] Entropía Min: {state['global_min_entropy']:.4f} | Max: {state['global_max_entropy']:.4f}")
    print(f"[*] Raw Min: {state['global_min_raw']:.1f} | Max: {state['global_max_raw']:.1f}")
    print("=======================================================")


if __name__ == "__main__":
    calculate_global_bounds()
