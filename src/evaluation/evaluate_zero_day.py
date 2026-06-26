# src/evaluation/evaluate_zero_day.py
import os
import gc
import csv
import json
import math
import torch
import random
import shutil
import hashlib
import logging
import argparse
import tempfile
import subprocess
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm

from src.models.vit_ablation import (
    IDS2018Dataset,
    ViT_OSR,
    KNOWN_CLASS_TO_IDX,
    OOD_CLASS_TO_IDX,
    safe_collate,
    seed_worker,
)
from src.utils.config_manager import setup_environment
from src.osr_module.mahalanobis import OpenSetShield

# ==============================================================================
# 0. INYECCIÓN DE ENTORNO
# ==============================================================================
parser = argparse.ArgumentParser(description="Evaluador OSR Zero-Day")
parser.add_argument("--mode", type=str, choices=["pilot", "prod"], required=True)
parser.add_argument("--n_min", type=int, required=True)
args, _ = parser.parse_known_args()

env = setup_environment(script_name="evaluate_zero_day", args=args)

TRAIN_DIR = env.get_path("paths", "output", "train_known", ensure_exists=True)
VAL_DIR = env.get_path("paths", "output", "val_known", ensure_exists=True)
TEST_KNOWN_DIR = env.get_path("paths", "output", "test_known", ensure_exists=True)
TEST_OOD_DIR = env.get_path("paths", "output", "test_ood", ensure_exists=True)
SCALER_JSON = env.get_path("paths", "configs", "scaler_bounds", is_file=True)
CKPT_DIR = env.get_path("paths", "artifacts", "checkpoints", ensure_exists=True)
RESULTS_DIR = env.get_path("paths", "artifacts", "results", ensure_exists=True)
TELEMETRY_DIR = env.get_path("paths", "artifacts", "telemetry_logs", ensure_exists=True)
CACHE_SCHEMA_VERSION = 1
PROFILE_SCHEMA_VERSION = 3
RESULTS_SCHEMA_VERSION = 3
SENSITIVITY_SCHEMA_VERSION = 1

# ==============================================================================
# 1. UTILIDADES DE TRAZABILIDAD, PERSISTENCIA Y CONFIGURACIÓN
# ==============================================================================
def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _git_metadata():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], check=True, capture_output=True, text=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return "unavailable", None


def _write_json_atomic(path, payload):
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, ensure_ascii=False)
    os.replace(temporary_path, path)


def _write_text_atomic(path, text):
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        file.write(text)
    os.replace(temporary_path, path)


def _write_csv_atomic(path, fieldnames, rows):
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def _torch_save_atomic(path, payload):
    temporary_path = path + ".tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def _set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_amp(device, training_config):
    enabled = bool(training_config.get("mixed_precision", True)) and device.type == "cuda"
    requested = str(training_config.get("amp_dtype", "bfloat16")).lower()
    if not enabled:
        return False, torch.float32
    if requested == "bfloat16" and torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    if requested not in {"bfloat16", "float16"}:
        raise ValueError(f"amp_dtype no soportado: {requested}")
    if requested == "bfloat16":
        logging.warning("[!] BF16 no soportado; se utilizará FP16")
    return True, torch.float16


def _validate_checkpoint(checkpoint, checkpoint_path, n_min, scaler_sha256):
    required = {
        "model_state", "fisher_state", "class_to_idx", "tensor_shape", "model_config",
        "training_config", "experiment_hash", "hashes", "loss_variant", "n_min",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise RuntimeError(f"Checkpoint incompleto. Campos ausentes: {missing}")
    if checkpoint.get("checkpoint_kind") != "best_validation_mcc":
        raise RuntimeError("El checkpoint no es el mejor modelo seleccionado por MCC de validación")
    if int(checkpoint["n_min"]) != int(n_min):
        raise RuntimeError(f"N_min incompatible: checkpoint={checkpoint['n_min']}, solicitado={n_min}")
    if checkpoint["class_to_idx"] != KNOWN_CLASS_TO_IDX:
        raise RuntimeError(f"Mapa de clases incompatible: {checkpoint['class_to_idx']}")
    if checkpoint["tensor_shape"] != [int(n_min), 144, 3]:
        raise RuntimeError(f"Forma de entrada incompatible: {checkpoint['tensor_shape']}")
    if checkpoint["loss_variant"] != "weighted_focal_plus_progressive_fisher":
        raise RuntimeError(f"Variante de pérdida incompatible: {checkpoint['loss_variant']}")
    if not str(checkpoint["experiment_hash"]).strip():
        raise RuntimeError("experiment_hash ausente o vacío")

    expected_hashes = {
        "scaler_sha256": scaler_sha256,
        "dataset_schedule_sha256": _sha256_file("configs/dataset_schedule.yaml"),
        "script_sha256": _sha256_file("src/models/vit_ablation.py"),
    }
    mismatches = [key for key, value in expected_hashes.items() if checkpoint["hashes"].get(key) != value]
    if mismatches:
        raise RuntimeError(f"Checkpoint incompatible con los artefactos actuales. Hashes distintos: {mismatches}")
    logging.info("[✓] Checkpoint validado: %s", checkpoint_path)


def _make_loader(dataset, mode, loader_config, training_config, seed_offset):
    workers = 0 if mode == "pilot" else int(loader_config["num_workers"])
    kwargs = {
        "dataset": dataset,
        "batch_size": int(training_config["batch_size"]),
        "shuffle": False,
        "drop_last": False,
        "num_workers": workers,
        "pin_memory": bool(loader_config.get("pin_memory", True) and torch.cuda.is_available()),
        "collate_fn": safe_collate,
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(int(env.get_value("project", "seed")) + seed_offset),
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(loader_config.get("prefetch_factor", 2))
        kwargs["persistent_workers"] = bool(loader_config.get("persistent_workers", True))
    return DataLoader(**kwargs)


# ==============================================================================
# 2. CACHÉ VERSIONADA Y EXTRACCIÓN DE EMBEDDINGS
# ==============================================================================
def _cache_files(cache_dir, split_name):
    return {
        "latents": os.path.join(cache_dir, f"{split_name}_latents.dat"),
        "labels": os.path.join(cache_dir, f"{split_name}_labels.dat"),
        "preds": os.path.join(cache_dir, f"{split_name}_preds.dat"),
        "metadata": os.path.join(cache_dir, f"{split_name}_metadata.json"),
    }


def _load_cached_latents(paths, expected_identity_hash, expected_samples):
    if not os.path.isfile(paths["metadata"]):
        return None
    try:
        with open(paths["metadata"], "r", encoding="utf-8") as file:
            metadata = json.load(file)
        latent_dim = int(metadata["latent_dim"])
        if metadata.get("identity_hash") != expected_identity_hash or int(metadata.get("num_samples", -1)) != expected_samples:
            return None

        expected_sizes = {
            "latents": expected_samples * latent_dim * np.dtype("float32").itemsize,
            "labels": expected_samples * np.dtype("int64").itemsize,
            "preds": expected_samples * np.dtype("int64").itemsize,
        }
        if any(not os.path.isfile(paths[key]) or os.path.getsize(paths[key]) != size for key, size in expected_sizes.items()):
            return None

        latents_mm = np.memmap(paths["latents"], dtype="float32", mode="r+", shape=(expected_samples, latent_dim))
        labels_mm = np.memmap(paths["labels"], dtype="int64", mode="r+", shape=(expected_samples,))
        preds_mm = np.memmap(paths["preds"], dtype="int64", mode="r+", shape=(expected_samples,))
        return torch.from_numpy(latents_mm), torch.from_numpy(labels_mm), torch.from_numpy(preds_mm)
    except Exception as exc:
        logging.warning("[!] Caché inválida; se reconstruirá: %s", exc)
        return None


def extract_latents(dataloader, model, device, split_name="train", cache_dir=None, amp_enabled=False, amp_dtype=torch.float32, cache_identity=None):
    """Extrae token CLS a memmaps versionados y reutilizables."""
    if cache_dir is None:
        raise ValueError("cache_dir es obligatorio")
    os.makedirs(cache_dir, exist_ok=True)
    num_samples = len(dataloader.dataset)
    if num_samples <= 0:
        raise RuntimeError(f"Dataset vacío para {split_name}")

    cache_identity = dict(cache_identity or {})
    cache_identity.update({"schema_version": CACHE_SCHEMA_VERSION, "split_name": split_name, "num_samples": num_samples})
    identity_hash = _canonical_hash(cache_identity)
    paths = _cache_files(cache_dir, split_name)
    cached = _load_cached_latents(paths, identity_hash, num_samples)
    if cached is not None:
        logging.info("[✓] Caché de embeddings reutilizada para %s (%s muestras)", split_name, f"{num_samples:,}")
        return cached

    for key in ("latents", "labels", "preds", "metadata"):
        for candidate in (paths[key], paths[key] + ".tmp"):
            if os.path.exists(candidate):
                os.remove(candidate)

    latent_dim = None
    latents_mm = labels_mm = preds_mm = None
    offset = 0

    with torch.inference_mode():
        for inputs, labels in tqdm(dataloader, desc=f"Extrayendo {split_name}", leave=False):
            if len(inputs) == 0:
                continue
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            batch_size = inputs.size(0)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits, cls_output, _ = model(inputs, return_attention=False)
            predictions = logits.argmax(dim=1)

            if latent_dim is None:
                latent_dim = int(cls_output.size(1))
                latents_mm = np.memmap(paths["latents"] + ".tmp", dtype="float32", mode="w+", shape=(num_samples, latent_dim))
                labels_mm = np.memmap(paths["labels"] + ".tmp", dtype="int64", mode="w+", shape=(num_samples,))
                preds_mm = np.memmap(paths["preds"] + ".tmp", dtype="int64", mode="w+", shape=(num_samples,))

            latent_batch = cls_output.detach().float().cpu().numpy()
            if not np.isfinite(latent_batch).all():
                raise FloatingPointError(f"Embeddings NaN/Inf en {split_name}")
            latents_mm[offset:offset + batch_size] = latent_batch
            labels_mm[offset:offset + batch_size] = labels.detach().cpu().numpy()
            preds_mm[offset:offset + batch_size] = predictions.detach().cpu().numpy()
            offset += batch_size

    if latent_dim is None:
        raise RuntimeError(f"No se extrajo ningún embedding para {split_name}")
    if offset != num_samples:
        raise RuntimeError(f"Pérdida de datos en {split_name}: esperadas={num_samples}, escritas={offset}")

    latents_mm.flush()
    labels_mm.flush()
    preds_mm.flush()
    del latents_mm, labels_mm, preds_mm
    os.replace(paths["latents"] + ".tmp", paths["latents"])
    os.replace(paths["labels"] + ".tmp", paths["labels"])
    os.replace(paths["preds"] + ".tmp", paths["preds"])
    _write_json_atomic(paths["metadata"], {
        "identity_hash": identity_hash,
        "identity": cache_identity,
        "num_samples": num_samples,
        "latent_dim": latent_dim,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    })

    cached = _load_cached_latents(paths, identity_hash, num_samples)
    if cached is None:
        raise RuntimeError(f"No se pudo reabrir la caché recién creada para {split_name}")
    logging.info("[✓] Caché de embeddings creada para %s (%s muestras)", split_name, f"{num_samples:,}")
    return cached


def _clone_to_memmap(source, path, batch_size=50000):
    memory_map = np.memmap(path, dtype="float32", mode="w+", shape=tuple(source.shape))
    for start in range(0, source.size(0), batch_size):
        end = min(start + batch_size, source.size(0))
        memory_map[start:end] = source[start:end].numpy()
    memory_map.flush()
    return torch.from_numpy(memory_map)


def gaussianize_latents_in_place(latents, scaler, fit=False, subsample_size=100000, batch_size=50000, seed=42):
    """Ajusta exclusivamente con train_known y transforma embeddings por bloques."""
    num_samples = latents.size(0)
    if fit:
        sample_count = min(subsample_size, num_samples)
        logging.info("[*] Ajustando QuantileTransformer con %s muestras de train_known...", f"{sample_count:,}")
        if num_samples > sample_count:
            indices = np.random.default_rng(seed).choice(num_samples, size=sample_count, replace=False)
            sample_data = latents[indices].numpy()
        else:
            sample_data = latents.numpy()
        scaler.fit(sample_data)
        del sample_data

    logging.info("[*] Transformando %s embeddings al espacio Gaussiano...", f"{num_samples:,}")
    for start in tqdm(range(0, num_samples, batch_size), desc="Gaussianizando", leave=False):
        end = min(start + batch_size, num_samples)
        transformed = scaler.transform(latents[start:end].numpy()).astype(np.float32)
        if not np.isfinite(transformed).all():
            raise FloatingPointError("QuantileTransformer produjo NaN o infinito")
        latents[start:end].copy_(torch.from_numpy(transformed))


# ==============================================================================
# 3. ESCUDO CONFIGURABLE: PCA 99 %, QUANTILE, RIDGE Y MAD
# ==============================================================================
class ConfigurableOpenSetShield(OpenSetShield):
    """Extiende OpenSetShield sin modificar mahalanobis.py ni su API pública."""
    def __init__(self, variance_retained=0.99, lambda_mad=1.0, ridge_epsilon_init=1e-5, device="cuda"):
        super().__init__(n_components=1, lambda_mad=lambda_mad, device=device)
        if not 0.0 < variance_retained <= 1.0:
            raise ValueError("variance_retained debe pertenecer a (0,1]")
        if ridge_epsilon_init <= 0.0:
            raise ValueError("ridge_epsilon_init debe ser positivo")
        self.variance_retained = float(variance_retained)
        self.ridge_epsilon_init = float(ridge_epsilon_init)
        self.retained_variance_actual = None
        self.explained_variance_ratio = None
        self.class_counts = {}

    def _robust_inverse(self, cov_matrix, base_epsilon=None, max_iters=5):
        return super()._robust_inverse(cov_matrix, base_epsilon=self.ridge_epsilon_init, max_iters=max_iters)

    def fit_pca(self, train_latents, batch_size=50000):
        """Ajusta PCA exclusivamente con CLS crudos de train_known."""
        num_samples, latent_dim = train_latents.size(0), train_latents.size(1)
        if num_samples < 2:
            raise RuntimeError("Se requieren al menos dos embeddings para ajustar PCA")

        sum_latents = torch.zeros(latent_dim, dtype=torch.float64, device=self.device)
        for start in range(0, num_samples, batch_size):
            batch = train_latents[start:start + batch_size].to(self.device, dtype=torch.float64)
            sum_latents += batch.sum(dim=0)
        self.pca_mean = (sum_latents / num_samples).unsqueeze(0)

        cov_sum = torch.zeros((latent_dim, latent_dim), dtype=torch.float64, device=self.device)
        for start in range(0, num_samples, batch_size):
            batch = train_latents[start:start + batch_size].to(self.device, dtype=torch.float64)
            centered = batch - self.pca_mean
            cov_sum += centered.T @ centered
        covariance = cov_sum / (num_samples - 1)

        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        eigenvalues = torch.flip(torch.clamp(eigenvalues, min=0.0), dims=[0])
        eigenvectors = torch.flip(eigenvectors, dims=[1])
        total_variance = eigenvalues.sum()
        if not torch.isfinite(total_variance) or total_variance <= 0:
            raise RuntimeError("Varianza PCA inválida")

        cumulative = torch.cumsum(eigenvalues, dim=0) / total_variance
        target = torch.tensor(self.variance_retained, dtype=cumulative.dtype, device=cumulative.device)
        self.n_components = min(latent_dim, int(torch.searchsorted(cumulative, target).item()) + 1)
        self.pca_v = eigenvectors[:, :self.n_components]
        self.explained_variance_ratio = (eigenvalues / total_variance).detach().cpu()
        self.retained_variance_actual = float(cumulative[self.n_components - 1].item())
        if self.retained_variance_actual + 1e-12 < self.variance_retained:
            raise RuntimeError("PCA no alcanzó la varianza objetivo")
        logging.info("[*] PCA: %s/%s componentes | varianza retenida=%.6f", self.n_components, latent_dim, self.retained_variance_actual)

    def project_to_memmap(self, latents, path, batch_size=50000):
        """Proyecta CLS mediante el PCA ajustado y escribe el resultado en memmap float32."""
        if self.pca_mean is None or self.pca_v is None:
            raise RuntimeError("PCA no ajustado")
        if os.path.exists(path):
            os.remove(path)
        projected_mm = np.memmap(path, dtype="float32", mode="w+", shape=(latents.size(0), self.n_components))
        for start in range(0, latents.size(0), batch_size):
            end = min(start + batch_size, latents.size(0))
            batch = latents[start:end].to(self.device, dtype=torch.float64)
            projected = ((batch - self.pca_mean) @ self.pca_v).float().cpu().numpy()
            if not np.isfinite(projected).all():
                raise FloatingPointError("La proyección PCA produjo NaN o infinito")
            projected_mm[start:end] = projected
        projected_mm.flush()
        return torch.from_numpy(projected_mm)

    def fit_mahalanobis_profiles(self, train_latents, train_labels, class_indices, batch_size=50000):
        """Ajusta centroides y covarianzas sobre embeddings ya procesados por PCA → Quantile."""
        class_indices = [int(class_idx) for class_idx in class_indices]
        num_samples, feature_dim = train_latents.size(0), train_latents.size(1)
        if num_samples < 2:
            raise RuntimeError("Se requieren al menos dos embeddings para Mahalanobis")
        self.centroids, self.inv_covariances, self.thresholds = {}, {}, {}
        self.class_counts = {class_idx: 0 for class_idx in class_indices}
        class_sums = {class_idx: torch.zeros(feature_dim, dtype=torch.float64, device=self.device) for class_idx in class_indices}
        class_cov_sums = {class_idx: torch.zeros((feature_dim, feature_dim), dtype=torch.float64, device=self.device) for class_idx in class_indices}

        for start in range(0, num_samples, batch_size):
            batch = train_latents[start:start + batch_size].to(self.device, dtype=torch.float64)
            labels = train_labels[start:start + batch_size].to(self.device)
            for class_idx in class_indices:
                class_features = batch[labels.eq(class_idx)]
                if class_features.numel() == 0:
                    continue
                class_sums[class_idx] += class_features.sum(dim=0)
                self.class_counts[class_idx] += class_features.size(0)

        for class_idx in class_indices:
            if self.class_counts[class_idx] < 2:
                raise RuntimeError(f"Clase {class_idx} insuficiente para Mahalanobis: {self.class_counts[class_idx]} muestras")
            self.centroids[class_idx] = class_sums[class_idx] / self.class_counts[class_idx]

        for start in range(0, num_samples, batch_size):
            batch = train_latents[start:start + batch_size].to(self.device, dtype=torch.float64)
            labels = train_labels[start:start + batch_size].to(self.device)
            for class_idx in class_indices:
                class_features = batch[labels.eq(class_idx)]
                if class_features.numel() == 0:
                    continue
                centered = class_features - self.centroids[class_idx]
                class_cov_sums[class_idx] += centered.T @ centered

        for class_idx in class_indices:
            covariance_class = class_cov_sums[class_idx] / (self.class_counts[class_idx] - 1)
            self.inv_covariances[class_idx] = self._robust_inverse(covariance_class)

    def fit_profiles(self, train_latents, train_labels, class_indices, batch_size=50000):
        """Compatibilidad heredada: PCA → Mahalanobis sin Quantile. El pipeline nuevo no usa este atajo."""
        self.fit_pca(train_latents, batch_size=batch_size)
        temporary_dir = tempfile.mkdtemp(prefix="osr_fit_profiles_")
        try:
            projected = self.project_to_memmap(train_latents, os.path.join(temporary_dir, "projected.dat"), batch_size=batch_size)
            self.fit_mahalanobis_profiles(projected, train_labels, class_indices, batch_size=batch_size)
            del projected
            gc.collect()
        finally:
            shutil.rmtree(temporary_dir, ignore_errors=True)

    def calculate_distances(self, latents, labels, batch_size=50000):
        """Distancia a la clase asignada; se conserva para calibración por etiqueta real."""
        labels = labels.cpu()
        distances = torch.full((latents.size(0),), float("nan"), dtype=torch.float64, device="cpu")
        for start in range(0, latents.size(0), batch_size):
            end = min(start + batch_size, latents.size(0))
            batch = latents[start:end].to(self.device, dtype=torch.float64)
            batch_labels = labels[start:end].to(self.device)
            batch_distances = torch.full((batch.size(0),), float("nan"), dtype=torch.float64, device=self.device)
            for class_idx in sorted(self.centroids):
                mask = batch_labels.eq(class_idx)
                if not mask.any():
                    continue
                diff = batch[mask] - self.centroids[class_idx]
                distance = torch.sum((diff @ self.inv_covariances[class_idx]) * diff, dim=1)
                batch_distances[mask] = torch.clamp(distance, min=0.0)
            distances[start:end] = batch_distances.cpu()
        if not torch.isfinite(distances).all():
            raise RuntimeError("Existen etiquetas sin perfil Mahalanobis o distancias no finitas")
        return distances

    def calculate_minimum_distances(self, latents, batch_size=50000):
        """Calcula d² contra todas las clases y retorna d² mínima y k*=argmin_k d²."""
        class_indices = sorted(self.centroids)
        if not class_indices:
            raise RuntimeError("No existen perfiles Mahalanobis ajustados")
        minimum_distances = torch.empty(latents.size(0), dtype=torch.float64, device="cpu")
        winning_classes = torch.empty(latents.size(0), dtype=torch.int64, device="cpu")
        class_tensor = torch.tensor(class_indices, dtype=torch.int64, device=self.device)
        for start in range(0, latents.size(0), batch_size):
            end = min(start + batch_size, latents.size(0))
            batch = latents[start:end].to(self.device, dtype=torch.float64)
            distances_by_class = []
            for class_idx in class_indices:
                diff = batch - self.centroids[class_idx]
                distance = torch.sum((diff @ self.inv_covariances[class_idx]) * diff, dim=1)
                distances_by_class.append(torch.clamp(distance, min=0.0))
            distance_matrix = torch.stack(distances_by_class, dim=1)
            batch_minimum, batch_positions = torch.min(distance_matrix, dim=1)
            minimum_distances[start:end] = batch_minimum.cpu()
            winning_classes[start:end] = class_tensor[batch_positions].cpu()
        if not torch.isfinite(minimum_distances).all():
            raise FloatingPointError("Las distancias Mahalanobis mínimas contienen NaN o infinito")
        return minimum_distances, winning_classes


def _compute_calibration_statistics(shield, val_latents, val_labels):
    distances = shield.calculate_distances(val_latents, val_labels)
    calibration = {}
    for class_idx in sorted(shield.centroids):
        mask = val_labels.eq(class_idx)
        count = int(mask.sum().item())
        if count == 0:
            raise RuntimeError(f"Sin muestras de validación para calibrar la clase {class_idx}")
        class_distances = distances[mask]
        median = torch.median(class_distances)
        mad = torch.median(torch.abs(class_distances - median))
        if not torch.isfinite(median) or not torch.isfinite(mad):
            raise RuntimeError(f"Estadísticos MAD inválidos para clase {class_idx}")
        calibration[str(class_idx)] = {"count": count, "median": float(median.item()), "mad": float(mad.item())}
    return calibration


def _thresholds_for_lambda(calibration, lambda_mad):
    if not math.isfinite(lambda_mad) or lambda_mad <= 0:
        raise ValueError("lambda_mad debe ser positivo y finito")
    thresholds = {}
    for class_idx, values in calibration.items():
        threshold = float(values["median"] + lambda_mad * values["mad"])
        if not math.isfinite(threshold) or threshold <= 0:
            raise RuntimeError(f"Umbral inválido para clase {class_idx}: {threshold}")
        thresholds[int(class_idx)] = threshold
    return thresholds


def _apply_thresholds(shield, thresholds):
    shield.thresholds = {int(class_idx): torch.tensor(float(value), dtype=torch.float64, device="cpu") for class_idx, value in thresholds.items()}


def _calibrate_thresholds_with_stats(shield, val_latents, val_labels, lambda_mad=None):
    """Compatibilidad: calcula estadísticos, aplica lambda y retorna mediana/MAD/tau por clase."""
    lambda_value = float(shield.lambda_mad if lambda_mad is None else lambda_mad)
    calibration = _compute_calibration_statistics(shield, val_latents, val_labels)
    thresholds = _thresholds_for_lambda(calibration, lambda_value)
    _apply_thresholds(shield, thresholds)
    enriched = {}
    for class_idx, values in calibration.items():
        enriched[class_idx] = {**values, "threshold": thresholds[int(class_idx)]}
        logging.info("[*] Clase %s | n=%s | mediana=%.4f | MAD=%.4f | tau=%.4f", class_idx, f"{values['count']:,}", values["median"], values["mad"], thresholds[int(class_idx)])
    return enriched


# ==============================================================================
# 4. MÉTRICAS, SENSIBILIDAD Y REPORTES OSR
# ==============================================================================
def _normalized_scores(distances, winning_classes, thresholds):
    distances = np.asarray(distances, dtype=np.float64)
    winning_classes = np.asarray(winning_classes, dtype=np.int64)
    scores = np.empty_like(distances, dtype=np.float64)
    for class_idx in np.unique(winning_classes):
        if int(class_idx) not in thresholds:
            raise RuntimeError(f"No existe umbral para la clase geométrica {class_idx}")
        threshold = float(thresholds[int(class_idx)])
        if not np.isfinite(threshold) or threshold <= 0:
            raise RuntimeError(f"Umbral inválido para la clase geométrica {class_idx}: {threshold}")
        mask = winning_classes == class_idx
        scores[mask] = distances[mask] / threshold
    if not np.isfinite(scores).all():
        raise FloatingPointError("Scores OSR normalizados contienen NaN o infinito")
    return scores


def _binary_metrics(meta_true, meta_pred, scores):
    confusion = confusion_matrix(meta_true, meta_pred, labels=[0, 1])
    tn, fp, fn, tp = confusion.ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(meta_true, meta_pred, average="binary", zero_division=0)
    fpr_curve, tpr_curve, thresholds_curve = roc_curve(meta_true, scores)
    valid_95 = np.flatnonzero(tpr_curve >= 0.95)
    fpr_at_95 = float(np.min(fpr_curve[valid_95])) if valid_95.size else None
    if fpr_curve.size > 5000:
        selected = np.unique(np.linspace(0, fpr_curve.size - 1, 5000, dtype=np.int64))
        fpr_curve, tpr_curve, thresholds_curve = fpr_curve[selected], tpr_curve[selected], thresholds_curve[selected]
    thresholds_serializable = [float(value) if np.isfinite(value) else None for value in thresholds_curve]
    return {
        "ood_auroc": float(roc_auc_score(meta_true, scores)), "ood_aupr": float(average_precision_score(meta_true, scores)),
        "binary_mcc": float(matthews_corrcoef(meta_true, meta_pred)), "balanced_accuracy": float(balanced_accuracy_score(meta_true, meta_pred)),
        "ood_precision": float(precision), "ood_recall": float(recall), "ood_fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "known_false_rejection_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0, "ood_f1": float(f1), "fpr_at_95_tpr": fpr_at_95,
        "confusion_matrix": confusion.tolist(), "roc_curve": {"fpr": fpr_curve.tolist(), "tpr": tpr_curve.tolist(), "thresholds": thresholds_serializable},
    }


def _ood_breakdown(ood_labels, ood_anomalies, ood_scores):
    breakdown = {}
    for name, class_idx in OOD_CLASS_TO_IDX.items():
        mask = ood_labels == class_idx
        count = int(mask.sum())
        if count == 0:
            raise RuntimeError(f"Clase OOD ausente en test_ood: {name}")
        class_scores = ood_scores[mask]
        breakdown[name] = {
            "support": count, "detected_as_ood": int(ood_anomalies[mask].sum()), "ood_recall": float(ood_anomalies[mask].mean()),
            "score_mean": float(class_scores.mean()), "score_median": float(np.median(class_scores)),
            "score_min": float(class_scores.min()), "score_max": float(class_scores.max()),
        }
    return breakdown


def _read_ood_subtypes(dataset):
    subtypes, current_filename, current_file = [], None, None
    try:
        for filename, flow_id, _ in dataset.index:
            if filename != current_filename:
                if current_file is not None:
                    current_file.close()
                current_filename = filename
                current_file = h5py.File(os.path.join(dataset.data_dir, filename), "r", swmr=True)
            subtype = current_file[flow_id].attrs.get("attack_subtype", "Unknown")
            if isinstance(subtype, bytes):
                subtype = subtype.decode("utf-8", errors="replace")
            subtypes.append(str(subtype))
    finally:
        if current_file is not None:
            current_file.close()
    if len(subtypes) != len(dataset):
        raise RuntimeError("El número de subtipos OOD no coincide con el índice")
    return np.asarray(subtypes, dtype=object)


def _ood_subtype_breakdown(subtypes, anomalies, scores):
    breakdown = {}
    for subtype in sorted(set(subtypes.tolist())):
        mask = subtypes == subtype
        class_scores = scores[mask]
        breakdown[subtype] = {
            "support": int(mask.sum()), "detected_as_ood": int(anomalies[mask].sum()), "ood_recall": float(anomalies[mask].mean()),
            "score_mean": float(class_scores.mean()), "score_median": float(np.median(class_scores)),
        }
    return breakdown


def _resolve_osr_settings(osr_config):
    pca_variance = float(osr_config.get("pca_variance_retained", 0.99))
    if not 0.0 < pca_variance <= 1.0:
        raise RuntimeError(
            f"pca_variance_retained debe estar en (0, 1]; valor configurado={pca_variance}"
        )
    candidates = [float(value) for value in osr_config.get("lambda_candidates", [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0])]
    if not candidates or any(not math.isfinite(value) or value <= 0 for value in candidates):
        raise ValueError("lambda_candidates debe contener valores positivos y finitos")
    candidates = sorted(set(candidates))
    legacy_lambda_used = "selected_lambda_mad" not in osr_config and "mad_multiplier_lambda" in osr_config
    selected_lambda = osr_config.get("selected_lambda_mad", osr_config.get("mad_multiplier_lambda"))
    if selected_lambda is not None:
        selected_lambda = float(selected_lambda)
        if not math.isfinite(selected_lambda) or selected_lambda <= 0:
            raise ValueError("selected_lambda_mad debe ser null o un número positivo y finito")
        if not any(math.isclose(selected_lambda, value, rel_tol=0.0, abs_tol=1e-12) for value in candidates):
            raise ValueError("selected_lambda_mad debe pertenecer a lambda_candidates")
    max_frr = float(osr_config.get("max_val_known_frr", 0.10))
    if not 0.0 <= max_frr <= 1.0:
        raise ValueError("max_val_known_frr debe pertenecer a [0,1]")
    quantile_count = int(osr_config.get("quantile_count", 10000))
    quantile_subsample = int(osr_config.get("quantile_subsample_size", 100000))
    if quantile_count < 2 or quantile_subsample < 2:
        raise ValueError("quantile_count y quantile_subsample_size deben ser >= 2")
    return {
        "pca_variance_retained": pca_variance, "lambda_candidates": candidates, "selected_lambda_mad": selected_lambda,
        "max_val_known_frr": max_frr, "ridge_epsilon_init": float(osr_config.get("ridge_epsilon_init", 1e-5)),
        "quantile_count": quantile_count, "quantile_subsample_size": quantile_subsample, "legacy_lambda_key_used": legacy_lambda_used,
    }


def _evaluate_lambda_sensitivity(shield, val_latents, val_labels, calibration, candidates, max_frr):
    minimum_distances, winning_classes = shield.calculate_minimum_distances(val_latents)
    distances_np = minimum_distances.numpy()
    winning_np = winning_classes.numpy().astype(np.int64, copy=False)
    labels_np = val_labels.numpy().astype(np.int64, copy=False)
    names_by_idx = {idx: name for name, idx in KNOWN_CLASS_TO_IDX.items()}
    results = []
    for lambda_mad in candidates:
        thresholds = _thresholds_for_lambda(calibration, lambda_mad)
        scores = _normalized_scores(distances_np, winning_np, thresholds)
        rejected = scores >= 1.0
        frr_by_class = {}
        for class_idx in sorted(shield.centroids):
            mask = labels_np == class_idx
            if not mask.any():
                raise RuntimeError(f"Clase {class_idx} ausente en val_known")
            frr_by_class[names_by_idx[class_idx]] = float(rejected[mask].mean())
        global_frr = float(rejected.mean())
        results.append({
            "lambda_mad": float(lambda_mad), "known_total": int(labels_np.size), "known_rejected": int(rejected.sum()),
            "known_accepted": int((~rejected).sum()), "global_frr": global_frr, "eligible_frr": bool(global_frr <= max_frr),
            "geometric_class_accuracy": float((winning_np == labels_np).mean()), "frr_by_class": frr_by_class,
            "thresholds_by_class": {names_by_idx[class_idx]: float(value) for class_idx, value in thresholds.items()},
        })
    return results


def _build_sensitivity_text_report(payload):
    lines = [
        "=" * 100, "SENSIBILIDAD LAMBDA MAD SOBRE val_known", "=" * 100,
        f"Fecha UTC             : {payload['created_at_utc']}", f"N_min                 : {payload['n_min']}",
        f"PCA objetivo          : {payload['pca_variance_target']:.4f}", f"PCA real              : {payload['pca_variance_actual']:.6f}",
        f"Componentes PCA       : {payload['pca_components']}", f"FRR máximo orientativo: {payload['max_val_known_frr']:.4f}",
        "", f"{'Lambda':>8} | {'FRR global':>12} | {'Elegible':>8} | {'Rechazados':>11} | {'Exactitud k*':>12}", "-" * 100,
    ]
    for row in payload["lambda_results"]:
        lines.append(f"{row['lambda_mad']:>8.4f} | {row['global_frr']:>12.6f} | {str(row['eligible_frr']):>8} | {row['known_rejected']:>11,} | {row['geometric_class_accuracy']:>12.6f}")
        lines.append("  FRR por clase: " + ", ".join(f"{name}={value:.6f}" for name, value in row["frr_by_class"].items()))
    lines.extend(["", "La selección final debe realizarse únicamente con estos resultados de val_known.", "=" * 100])
    return "\n".join(lines) + "\n"


def _write_sensitivity_artifacts(payload, n_min):
    json_path = os.path.join(RESULTS_DIR, f"osr_lambda_sensitivity_nmin_{n_min}.json")
    text_path = os.path.join(RESULTS_DIR, f"osr_lambda_sensitivity_nmin_{n_min}.txt")
    csv_path = os.path.join(RESULTS_DIR, f"osr_lambda_sensitivity_nmin_{n_min}.csv")
    csv_rows = []
    for row in payload["lambda_results"]:
        csv_rows.append({
            "lambda_mad": row["lambda_mad"], "global_frr": row["global_frr"], "eligible_frr": row["eligible_frr"],
            "known_total": row["known_total"], "known_rejected": row["known_rejected"], "known_accepted": row["known_accepted"],
            "geometric_class_accuracy": row["geometric_class_accuracy"],
            "frr_by_class_json": json.dumps(row["frr_by_class"], sort_keys=True),
            "thresholds_by_class_json": json.dumps(row["thresholds_by_class"], sort_keys=True),
        })
    fields = ["lambda_mad", "global_frr", "eligible_frr", "known_total", "known_rejected", "known_accepted", "geometric_class_accuracy", "frr_by_class_json", "thresholds_by_class_json"]
    _write_json_atomic(json_path, payload)
    _write_text_atomic(text_path, _build_sensitivity_text_report(payload))
    _write_csv_atomic(csv_path, fields, csv_rows)
    logging.info("[*] Sensibilidad lambda: %s | %s | %s", csv_path, json_path, text_path)
    return {"csv": csv_path, "json": json_path, "txt": text_path}


def _build_text_report(results):
    metrics = results["metrics"]
    binary = metrics["binary_detection"]
    degradation = metrics["known_degradation"]
    osr = results["osr_configuration"]
    lines = [
        "=" * 88, "EVALUACIÓN OPEN-SET / ZERO-DAY OSR-ViT", "=" * 88,
        f"Fecha UTC                    : {results['created_at_utc']}", f"N_min                        : {results['n_min']}",
        f"Checkpoint                   : {results['checkpoint']['path']}", f"Mejor época                  : {results['checkpoint']['best_epoch']}",
        f"MCC validación               : {results['checkpoint']['best_val_mcc']:.6f}", f"Train / Val / Test ID / OOD  : {results['sample_counts']}",
        f"Pipeline OSR                 : {osr['pipeline_order']}", f"Componentes PCA              : {osr['pca_components']}",
        f"Varianza PCA retenida        : {osr['pca_variance_actual']:.6f}", f"Lambda MAD seleccionado      : {osr['selected_lambda_mad']:.6f}",
        f"FRR val_known seleccionado   : {osr['selected_lambda_val_frr']:.6f}", "", "MÉTRICAS PRINCIPALES OOD", "-" * 88,
        f"AUROC OOD normalizado        : {binary['ood_auroc']:.6f}", f"AUROC Mahalanobis bruto      : {metrics['ood_auroc_raw_mahalanobis']:.6f}",
        f"AUPR OOD                     : {binary['ood_aupr']:.6f}", f"MCC binario Known/OOD        : {binary['binary_mcc']:.6f}",
        f"Balanced Accuracy            : {binary['balanced_accuracy']:.6f}", f"OOD Precision                : {binary['ood_precision']:.6f}",
        f"OOD Recall                   : {binary['ood_recall']:.6f}", f"OOD FNR                      : {binary['ood_fnr']:.6f}",
        f"Known False Rejection Rate   : {binary['known_false_rejection_rate']:.6f}", f"OOD F1                       : {binary['ood_f1']:.6f}",
        f"FPR@95TPR                    : {binary['fpr_at_95_tpr']}", "", "DEGRADACIÓN DEL CONOCIMIENTO CERRADO", "-" * 88,
        f"MCC ID antes del escudo      : {degradation['mcc_before_shield']:.6f}", f"MCC ID después del escudo    : {degradation['mcc_after_shield']:.6f}",
        f"Delta MCC                    : {degradation['mcc_delta']:.6f}", f"Concordancia ViT / k*        : {degradation['vit_geometric_agreement']:.6f}",
        "", "DETECCIÓN POR CLASE OOD", "-" * 88,
    ]
    for name, values in metrics["per_ood_class"].items():
        lines.append(f"{name:<12} | Support={values['support']:,} | Detectados={values['detected_as_ood']:,} | Recall={values['ood_recall']:.6f} | ScoreMediana={values['score_median']:.6f}")
    if metrics.get("per_ood_subtype"):
        lines.extend(["", "DETECCIÓN POR SUBTIPO OOD", "-" * 88])
        for name, values in metrics["per_ood_subtype"].items():
            lines.append(f"{name:<24} | Support={values['support']:,} | Detectados={values['detected_as_ood']:,} | Recall={values['ood_recall']:.6f} | ScoreMediana={values['score_median']:.6f}")
    lines.extend(["", "UMBRALES MAD POR CLASE CONOCIDA", "-" * 88])
    for class_name, values in results["calibration_by_class"].items():
        lines.append(f"{class_name:<8} | n={values['count']:,} | mediana={values['median']:.6f} | MAD={values['mad']:.6f} | tau={values['threshold']:.6f}")
    lines.append("=" * 88)
    return "\n".join(lines) + "\n"


# ==============================================================================
# 5. ORQUESTADOR OSR
# ==============================================================================
def evaluate_osr(n_min, mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(env.get_value("project", "seed"))
    loader_config = env.get_value("data_loader")
    training_config = env.get_value("training")
    model_config_global = env.get_value("vit_model")
    osr_config = env.get_value("osr_shield")
    settings = _resolve_osr_settings(osr_config)
    tensor_width = int(env.get_value("preprocessing", "tensor_width"))
    _set_global_seed(seed)

    if int(n_min) != 7:
        raise RuntimeError(f"La evaluación actual fue definida para N_min=7; valor solicitado={n_min}")
    if tensor_width != 144:
        raise RuntimeError(f"tensor_width incompatible: {tensor_width}")
    if settings["legacy_lambda_key_used"]:
        logging.warning("[!] Compatibilidad legacy: se usó mad_multiplier_lambda como selected_lambda_mad")

    checkpoint_path = os.path.join(CKPT_DIR, f"vit_nmin_{n_min}_checkpoint.pt")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No se encontró el mejor checkpoint: {checkpoint_path}")

    checkpoint_sha256 = _sha256_file(checkpoint_path)
    scaler_sha256 = _sha256_file(SCALER_JSON)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    _validate_checkpoint(checkpoint, checkpoint_path, n_min, scaler_sha256)
    model_config = checkpoint["model_config"]
    if dict(model_config) != dict(model_config_global):
        raise RuntimeError("La configuración ViT actual no coincide con la guardada en el checkpoint")

    train_dataset = IDS2018Dataset(TRAIN_DIR, SCALER_JSON, n_min, max_bytes=tensor_width, mode=mode, split_name="train_known")
    val_dataset = IDS2018Dataset(VAL_DIR, SCALER_JSON, n_min, max_bytes=tensor_width, mode=mode, split_name="val_known")
    final_evaluation = settings["selected_lambda_mad"] is not None
    known_dataset = IDS2018Dataset(TEST_KNOWN_DIR, SCALER_JSON, n_min, max_bytes=tensor_width, mode=mode, split_name="test_known") if final_evaluation else None
    ood_dataset = IDS2018Dataset(TEST_OOD_DIR, SCALER_JSON, n_min, max_bytes=tensor_width, mode=mode, is_osr_test=True, split_name="test_ood") if final_evaluation else None

    known_datasets = {"train_known": train_dataset, "val_known": val_dataset}
    if final_evaluation:
        known_datasets["test_known"] = known_dataset
    for split_name, dataset in known_datasets.items():
        counts = {name: int(dataset.class_counts.get(idx, 0)) for name, idx in KNOWN_CLASS_TO_IDX.items()}
        missing = [name for name, count in counts.items() if count == 0]
        if missing:
            raise RuntimeError(f"Clases conocidas ausentes en {split_name}: {missing}")
    if final_evaluation:
        ood_counts = {name: int(ood_dataset.class_counts.get(idx, 0)) for name, idx in OOD_CLASS_TO_IDX.items()}
        missing_ood = [name for name, count in ood_counts.items() if count == 0]
        if missing_ood:
            raise RuntimeError(f"Clases OOD ausentes en test_ood: {missing_ood}")

    train_loader = _make_loader(train_dataset, mode, loader_config, training_config, 10)
    val_loader = _make_loader(val_dataset, mode, loader_config, training_config, 11)
    known_loader = _make_loader(known_dataset, mode, loader_config, training_config, 12) if final_evaluation else None
    ood_loader = _make_loader(ood_dataset, mode, loader_config, training_config, 13) if final_evaluation else None

    model = ViT_OSR(n_min=n_min, max_bytes=tensor_width, patch_size=(1, int(model_config["patch_size"])), embed_dim=int(model_config["embed_dim"]), depth=int(model_config["depth"]), num_heads=int(model_config["num_heads"]), num_classes=len(KNOWN_CLASS_TO_IDX)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    amp_enabled, amp_dtype = _resolve_amp(device, checkpoint["training_config"])
    logging.info("[*] Evaluador OSR en %s | AMP=%s | dtype=%s", device, amp_enabled, amp_dtype)
    base_cache_identity = {
        "checkpoint_sha256": checkpoint_sha256, "scaler_sha256": scaler_sha256, "n_min": int(n_min), "tensor_width": tensor_width,
        "class_to_idx": KNOWN_CLASS_TO_IDX, "model_config": model_config, "amp_enabled": amp_enabled, "amp_dtype": str(amp_dtype),
    }
    cache_key = _canonical_hash(base_cache_identity)[:16]
    cache_dir = os.path.join(TELEMETRY_DIR, f"osr_embedding_cache_nmin_{n_min}_{cache_key}")
    os.makedirs(cache_dir, exist_ok=True)
    transient_dir = tempfile.mkdtemp(prefix=f"osr_transformed_nmin_{n_min}_", dir=TELEMETRY_DIR)

    train_latents = train_labels = train_preds = None
    val_latents = val_labels = val_preds = None
    known_latents = known_labels = known_preds = None
    ood_latents = ood_labels = ood_preds = None
    train_gaussian = val_gaussian = known_gaussian = ood_gaussian = None

    try:
        split_specs = [("train_known", train_loader, train_dataset), ("val_known", val_loader, val_dataset)]
        if final_evaluation:
            split_specs.extend([("test_known", known_loader, known_dataset), ("test_ood", ood_loader, ood_dataset)])
        extracted = {}
        for split_name, dataloader, dataset in split_specs:
            identity = {**base_cache_identity, "dataset_manifest_sha256": dataset.dataset_manifest_sha256, "dataset_size": len(dataset), "split_name": split_name}
            extracted[split_name] = extract_latents(dataloader, model, device, split_name=split_name, cache_dir=cache_dir, amp_enabled=amp_enabled, amp_dtype=amp_dtype, cache_identity=identity)

        train_latents, train_labels, train_preds = extracted["train_known"]
        val_latents, val_labels, val_preds = extracted["val_known"]
        if final_evaluation:
            known_latents, known_labels, known_preds = extracted["test_known"]
            ood_latents, ood_labels, ood_preds = extracted["test_ood"]

        shield = ConfigurableOpenSetShield(variance_retained=settings["pca_variance_retained"], lambda_mad=settings["selected_lambda_mad"] or settings["lambda_candidates"][0], ridge_epsilon_init=settings["ridge_epsilon_init"], device=device)
        logging.info("[*] Ajustando PCA exclusivamente con CLS crudos de train_known...")
        shield.fit_pca(train_latents)

        train_gaussian = shield.project_to_memmap(train_latents, os.path.join(transient_dir, "train_pca_quantile.dat"))
        val_gaussian = shield.project_to_memmap(val_latents, os.path.join(transient_dir, "val_pca_quantile.dat"))
        if final_evaluation:
            known_gaussian = shield.project_to_memmap(known_latents, os.path.join(transient_dir, "known_pca_quantile.dat"))
            ood_gaussian = shield.project_to_memmap(ood_latents, os.path.join(transient_dir, "ood_pca_quantile.dat"))

        sample_size = min(settings["quantile_subsample_size"], train_gaussian.size(0))
        quantile_transformer = QuantileTransformer(n_quantiles=min(settings["quantile_count"], sample_size), output_distribution="normal", random_state=seed, subsample=sample_size, copy=False)
        gaussianize_latents_in_place(train_gaussian, quantile_transformer, fit=True, subsample_size=sample_size, seed=seed)
        gaussianize_latents_in_place(val_gaussian, quantile_transformer, fit=False, seed=seed)
        if final_evaluation:
            gaussianize_latents_in_place(known_gaussian, quantile_transformer, fit=False, seed=seed)
            gaussianize_latents_in_place(ood_gaussian, quantile_transformer, fit=False, seed=seed)

        logging.info("[*] Ajustando perfiles Mahalanobis sobre train_known transformado...")
        shield.fit_mahalanobis_profiles(train_gaussian, train_labels, list(KNOWN_CLASS_TO_IDX.values()))
        logging.info("[*] Calculando mediana y MAD exclusivamente con val_known...")
        calibration_base = _compute_calibration_statistics(shield, val_gaussian, val_labels)
        lambda_results = _evaluate_lambda_sensitivity(shield, val_gaussian, val_labels, calibration_base, settings["lambda_candidates"], settings["max_val_known_frr"])

        created_at_utc = datetime.now(timezone.utc).isoformat()
        sensitivity_payload = {
            "schema_version": SENSITIVITY_SCHEMA_VERSION, "created_at_utc": created_at_utc, "mode": mode, "n_min": int(n_min),
            "selection_split": "val_known", "test_splits_used": False, "pca_variance_target": float(shield.variance_retained),
            "pca_variance_actual": float(shield.retained_variance_actual), "pca_components": int(shield.n_components),
            "max_val_known_frr": settings["max_val_known_frr"], "lambda_candidates": settings["lambda_candidates"], "lambda_results": lambda_results,
            "pipeline_order": "PCA -> QuantileTransformer -> Mahalanobis", "class_selection": "argmin_all_class_mahalanobis_distances",
            "rejection_rule": "score >= 1", "checkpoint_sha256": checkpoint_sha256, "dataset_manifest_val_known": val_dataset.dataset_manifest_sha256,
        }
        sensitivity_paths = _write_sensitivity_artifacts(sensitivity_payload, n_min)
        if not final_evaluation:
            logging.info("[✓] Sensibilidad completada. Configure selected_lambda_mad para ejecutar la evaluación final")
            return sensitivity_payload

        selected_lambda = float(settings["selected_lambda_mad"])
        selected_row = next(row for row in lambda_results if math.isclose(row["lambda_mad"], selected_lambda, rel_tol=0.0, abs_tol=1e-12))
        selected_thresholds = _thresholds_for_lambda(calibration_base, selected_lambda)
        _apply_thresholds(shield, selected_thresholds)
        shield.lambda_mad = selected_lambda
        if selected_row["global_frr"] > settings["max_val_known_frr"]:
            logging.warning("[!] Lambda %.4f supera el FRR orientativo en val_known: %.6f > %.6f", selected_lambda, selected_row["global_frr"], settings["max_val_known_frr"])

        known_distances, known_winners = shield.calculate_minimum_distances(known_gaussian)
        ood_distances, ood_winners = shield.calculate_minimum_distances(ood_gaussian)
        known_labels_np = known_labels.numpy().astype(np.int64, copy=False)
        known_preds_np = known_preds.numpy().astype(np.int64, copy=False)
        ood_labels_np = ood_labels.numpy().astype(np.int64, copy=False)
        ood_preds_np = ood_preds.numpy().astype(np.int64, copy=False)
        known_winners_np = known_winners.numpy().astype(np.int64, copy=False)
        ood_winners_np = ood_winners.numpy().astype(np.int64, copy=False)
        known_distances_np = known_distances.numpy()
        ood_distances_np = ood_distances.numpy()
        thresholds_float = {int(key): float(value) for key, value in selected_thresholds.items()}
        known_scores = _normalized_scores(known_distances_np, known_winners_np, thresholds_float)
        ood_scores = _normalized_scores(ood_distances_np, ood_winners_np, thresholds_float)
        known_anomaly_np = known_scores >= 1.0
        ood_anomaly_np = ood_scores >= 1.0

        meta_true = np.concatenate([np.zeros(len(known_scores), dtype=np.int64), np.ones(len(ood_scores), dtype=np.int64)])
        meta_pred = np.concatenate([known_anomaly_np.astype(np.int64), ood_anomaly_np.astype(np.int64)])
        normalized_scores = np.concatenate([known_scores, ood_scores])
        raw_distances = np.concatenate([known_distances_np, ood_distances_np])
        if np.unique(meta_true).size != 2:
            raise RuntimeError("AUROC requiere muestras Known y OOD")

        binary_metrics = _binary_metrics(meta_true, meta_pred, normalized_scores)
        raw_auroc = float(roc_auc_score(meta_true, raw_distances))
        mcc_before = float(matthews_corrcoef(known_labels_np, known_preds_np))
        post_shield_predictions = np.where(known_anomaly_np, -1, known_preds_np)
        mcc_after = float(matthews_corrcoef(known_labels_np, post_shield_predictions))
        per_ood_class = _ood_breakdown(ood_labels_np, ood_anomaly_np, ood_scores)
        ood_subtypes = _read_ood_subtypes(ood_dataset)
        per_ood_subtype = _ood_subtype_breakdown(ood_subtypes, ood_anomaly_np, ood_scores)

        names_by_idx = {idx: name for name, idx in KNOWN_CLASS_TO_IDX.items()}
        calibration_by_class = {}
        for class_idx, values in calibration_base.items():
            calibration_by_class[names_by_idx[int(class_idx)]] = {**values, "threshold": thresholds_float[int(class_idx)]}
        git_commit, git_dirty = _git_metadata()
        sample_counts = {"train_known": len(train_dataset), "val_known": len(val_dataset), "test_known": len(known_dataset), "test_ood": len(ood_dataset)}
        metrics = {
            "binary_detection": binary_metrics, "ood_auroc_raw_mahalanobis": raw_auroc,
            "known_degradation": {
                "mcc_before_shield": mcc_before, "mcc_after_shield": mcc_after, "mcc_delta": mcc_after - mcc_before,
                "known_rejected": int(known_anomaly_np.sum()), "known_total": int(len(known_anomaly_np)),
                "vit_geometric_agreement": float((known_preds_np == known_winners_np).mean()),
                "geometric_class_accuracy": float((known_labels_np == known_winners_np).mean()),
            },
            "per_ood_class": per_ood_class, "per_ood_subtype": per_ood_subtype,
        }
        results = {
            "schema_version": RESULTS_SCHEMA_VERSION, "created_at_utc": created_at_utc, "mode": mode, "n_min": int(n_min),
            "input_shape": [int(n_min), tensor_width, 3], "class_to_idx": KNOWN_CLASS_TO_IDX, "ood_class_to_idx": OOD_CLASS_TO_IDX,
            "sample_counts": sample_counts,
            "dataset_manifests": {
                "train_known": train_dataset.dataset_manifest_sha256, "val_known": val_dataset.dataset_manifest_sha256,
                "test_known": known_dataset.dataset_manifest_sha256, "test_ood": ood_dataset.dataset_manifest_sha256,
            },
            "checkpoint": {
                "path": checkpoint_path, "sha256": checkpoint_sha256, "experiment_hash": checkpoint["experiment_hash"],
                "checkpoint_kind": checkpoint["checkpoint_kind"], "best_epoch": int(checkpoint.get("best_epoch", checkpoint["epoch"])) + 1,
                "best_val_mcc": float(checkpoint.get("best_val_mcc", 0.0)), "best_val_loss": float(checkpoint.get("best_val_loss", 0.0)),
            },
            "scaler_sha256": scaler_sha256, "scaler_manifest_sha256": train_dataset.scaler_manifest_sha256,
            "git_commit": git_commit, "git_dirty": git_dirty, "model_config": model_config,
            "osr_configuration": {
                "pipeline_order": "PCA -> QuantileTransformer -> Mahalanobis", "pca_fit_split": "train_known",
                "quantile_fit_split": "train_known_projected", "profile_fit_split": "train_known_transformed",
                "lambda_selection_split": "val_known", "final_test_split": "test_known + test_ood",
                "quantile_subsample_size": sample_size, "quantile_count": int(quantile_transformer.n_quantiles_),
                "pca_variance_target": float(shield.variance_retained), "pca_variance_actual": float(shield.retained_variance_actual),
                "pca_components": int(shield.n_components), "ridge_epsilon_init": float(shield.ridge_epsilon_init),
                "lambda_candidates": settings["lambda_candidates"], "selected_lambda_mad": selected_lambda,
                "max_val_known_frr": settings["max_val_known_frr"], "selected_lambda_val_frr": float(selected_row["global_frr"]),
                "class_selection": "argmin_all_class_mahalanobis_distances", "continuous_score": "minimum_mahalanobis_distance / winning_class_mad_threshold",
                "rejection_rule": "score >= 1",
            },
            "calibration_by_class": calibration_by_class, "lambda_sensitivity_paths": sensitivity_paths,
            "selected_lambda_validation": selected_row, "metrics": metrics, "embedding_cache_dir": cache_dir,
        }

        json_path = os.path.join(RESULTS_DIR, f"open_set_nmin_{n_min}.json")
        text_path = os.path.join(RESULTS_DIR, f"open_set_nmin_{n_min}.txt")
        roc_path = os.path.join(RESULTS_DIR, f"open_set_roc_nmin_{n_min}.png")
        confusion_path = os.path.join(RESULTS_DIR, f"open_set_confusion_nmin_{n_min}.png")
        profiles_path = os.path.join(RESULTS_DIR, f"osr_profiles_nmin_{n_min}.pt")
        profile_payload = {
            "schema_version": PROFILE_SCHEMA_VERSION, "created_at_utc": created_at_utc, "quantile_transformer": quantile_transformer,
            "pca_mean": shield.pca_mean.detach().cpu(), "pca_v": shield.pca_v.detach().cpu(),
            "explained_variance_ratio": shield.explained_variance_ratio, "retained_variance_actual": shield.retained_variance_actual,
            "centroids": {key: value.detach().cpu() for key, value in shield.centroids.items()},
            "inv_covariances": {key: value.detach().cpu() for key, value in shield.inv_covariances.items()},
            "thresholds": thresholds_float, "calibration_by_class": calibration_by_class, "class_counts_train": shield.class_counts,
            "class_to_idx": KNOWN_CLASS_TO_IDX, "configuration": results["osr_configuration"], "checkpoint": results["checkpoint"],
            "dataset_manifests": results["dataset_manifests"], "scaler_sha256": scaler_sha256,
        }
        _torch_save_atomic(profiles_path, profile_payload)
        _write_json_atomic(json_path, results)
        _write_text_atomic(text_path, _build_text_report(results))

        roc_data = binary_metrics["roc_curve"]
        plt.figure(figsize=(8, 6))
        plt.plot(roc_data["fpr"], roc_data["tpr"], label=f"OSR AUROC={binary_metrics['ood_auroc']:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Azar")
        plt.xlabel("False Positive Rate (Known rechazado)")
        plt.ylabel("True Positive Rate (OOD detectado)")
        plt.title(f"Curva ROC Open-Set OSR-ViT (N_min={n_min})")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(roc_path, dpi=300)
        plt.close()

        binary_confusion = np.asarray(binary_metrics["confusion_matrix"], dtype=np.int64)
        plt.figure(figsize=(7, 6))
        sns.heatmap(binary_confusion, annot=True, fmt="d", cmap="Blues", xticklabels=["Known", "OOD"], yticklabels=["Known", "OOD"])
        plt.ylabel("Etiqueta real")
        plt.xlabel("Decisión del escudo")
        plt.title(f"Matriz binaria OSR (N_min={n_min})\nMCC={binary_metrics['binary_mcc']:.4f}")
        plt.tight_layout()
        plt.savefig(confusion_path, dpi=300)
        plt.close()

        logging.info("[✓] Evaluación OSR completada")
        logging.info("[*] AUROC=%.6f | AUPR=%.6f | MCC binario=%.6f | OOD Recall=%.6f | Known FRR=%.6f", binary_metrics["ood_auroc"], binary_metrics["ood_aupr"], binary_metrics["binary_mcc"], binary_metrics["ood_recall"], binary_metrics["known_false_rejection_rate"])
        logging.info("[*] MCC ID antes=%.6f | después=%.6f | delta=%.6f", mcc_before, mcc_after, mcc_after - mcc_before)
        logging.info("[*] JSON: %s | TXT: %s | ROC: %s | CM: %s | Perfiles: %s", json_path, text_path, roc_path, confusion_path, profiles_path)
        return results

    finally:
        del train_gaussian, val_gaussian, known_gaussian, ood_gaussian
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if os.path.isdir(transient_dir):
            shutil.rmtree(transient_dir, ignore_errors=True)


if __name__ == "__main__":
    evaluate_osr(args.n_min, env.mode)