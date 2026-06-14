import argparse
import hashlib
import json
import logging
import math
import os
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from src.utils.config_manager import setup_environment

# ==============================================================================
# 0. INYECCIÓN DE ENTORNO Y CONFIGURACIÓN GLOBAL
# ==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=["pilot", "prod"], required=True)
env = None

KNOWN_CLASS_TO_IDX = {"Benign": 0, "DoS": 1, "DDoS": 2}
OOD_CLASS_TO_IDX = {"Botnet": 3, "Web_Attack": 4}
INDEX_SCHEMA_VERSION = 2

def _ensure_environment(mode: str):
    """Inicializa el entorno únicamente cuando se ejecuta el entrenamiento."""
    global env
    if env is None:
        env = setup_environment(script_name="phase3_ablation", args=argparse.Namespace(mode=mode))
    return env


def safe_collate(batch):
    """Mantiene tolerancia en piloto; en producción IDS2018Dataset propaga el error."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.utils.data.dataloader.default_collate(batch)


def seed_worker(worker_id):
    """Deriva semillas reproducibles para Python y NumPy desde PyTorch."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def _atomic_torch_save(payload, path):
    temporary_path = path + ".tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def _atomic_json_save(payload, path):
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, ensure_ascii=False)
    os.replace(temporary_path, path)


def _canonical_split_name(data_dir):
    name = os.path.basename(os.path.normpath(data_dir))
    return name[6:] if name.startswith("pilot_") else name


# ==============================================================================
# 1. EL PUENTE I/O: DATASET, ÍNDICE VERSIONADO Y SAMPLER
# ==============================================================================
class IDS2018Dataset(Dataset):
    def __init__(self, data_dir, scaler_json, n_min, max_bytes=None, mode="prod", is_osr_test=False, split_name=None):
        self.data_dir = data_dir
        self.n_min = int(n_min)
        self.mode = mode
        self.is_osr_test = bool(is_osr_test)
        self.strict = mode == "prod"
        self.split_name = split_name or _canonical_split_name(data_dir)
        self.worker_file_cache = {}
        self.worker_cache_limit = 20

        with open(scaler_json, "r", encoding="utf-8") as file:
            self.scaler_bounds = json.load(file)

        self.scaler_sha256 = _sha256_file(scaler_json)
        self.scaler_metadata = self.scaler_bounds.get("metadata", {})
        tensor_shape = self.scaler_metadata.get("tensor_shape")
        if not isinstance(tensor_shape, list) or len(tensor_shape) != 3:
            raise RuntimeError("scaler_bounds.json no contiene metadata.tensor_shape válida")

        self.expected_tensor_shape = tuple(int(value) for value in tensor_shape)
        self.tensor_width = self.expected_tensor_shape[1]
        self.max_bytes = self.tensor_width
        if max_bytes is not None and int(max_bytes) != self.tensor_width:
            raise ValueError(f"Ancho solicitado={max_bytes}, pero el scaler exige {self.tensor_width}")
        if not 1 <= self.n_min <= self.expected_tensor_shape[0]:
            raise ValueError(f"n_min debe pertenecer a [1,{self.expected_tensor_shape[0]}]")

        self.min_e = float(self.scaler_bounds["entropy_channel"]["min"])
        self.max_e = float(self.scaler_bounds["entropy_channel"]["max"])
        self.min_r = float(self.scaler_bounds["raw_bytes_channel"]["min"])
        self.max_r = float(self.scaler_bounds["raw_bytes_channel"]["max"])

        scaler_taxonomy = self.scaler_metadata.get("taxonomy")
        if scaler_taxonomy != list(KNOWN_CLASS_TO_IDX):
            raise RuntimeError(f"Taxonomía del scaler incompatible: {scaler_taxonomy}")
        if self.scaler_metadata.get("source_split") != "train_known":
            raise RuntimeError("El scaler no fue ajustado exclusivamente con train_known")
        if tuple(self.expected_tensor_shape) != (18, 144, 3):
            raise RuntimeError(f"Forma del scaler incompatible con el experimento: {self.expected_tensor_shape}")

        self.known_class_to_idx = dict(KNOWN_CLASS_TO_IDX)
        self.class_to_idx = dict(KNOWN_CLASS_TO_IDX)
        if self.is_osr_test:
            self.class_to_idx.update(OOD_CLASS_TO_IDX)

        if self.split_name in {"train_known", "val_known", "test_known"}:
            self.allowed_labels = set(KNOWN_CLASS_TO_IDX)
        elif self.split_name == "test_ood":
            self.allowed_labels = set(OOD_CLASS_TO_IDX)
        else:
            self.allowed_labels = set(self.class_to_idx)

        self.dataset_manifest, self.dataset_manifest_sha256 = self._build_quick_manifest()
        self.scaler_manifest_sha256 = str(self.scaler_metadata.get("manifest_sha256", ""))
        self._validate_train_manifest_against_scaler()
        self.index, self.class_counts = self._build_or_load_index()
        self.class_counts = self._recompute_class_counts(self.index)
        self.labels = np.asarray([item[2] for item in self.index], dtype=np.int64)

    def _build_quick_manifest(self):
        files = sorted(filename for filename in os.listdir(self.data_dir) if filename.endswith(".hdf5"))
        if not files:
            raise RuntimeError(f"No se encontraron HDF5 en {self.data_dir}")
        manifest = []
        for filename in files:
            path = os.path.join(self.data_dir, filename)
            stat = os.stat(path)
            manifest.append({"relative_path": filename, "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
        return manifest, _canonical_hash(manifest)

    def _validate_train_manifest_against_scaler(self):
        if self.split_name != "train_known":
            return
        scaler_files = self.scaler_bounds.get("manifest", {}).get("files")
        if not isinstance(scaler_files, list):
            raise RuntimeError("El scaler no contiene el manifiesto de train_known")
        current = [{"relative_path": item["relative_path"], "size_bytes": item["size_bytes"]} for item in self.dataset_manifest]
        expected = [{"relative_path": item["relative_path"], "size_bytes": int(item["size_bytes"])} for item in scaler_files]
        if current != expected:
            raise RuntimeError("Los HDF5 de train_known cambiaron después de calcular el scaler")

    def _index_identity(self):
        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "mode": self.mode,
            "split": self.split_name,
            "n_min": self.n_min,
            "class_to_idx": self.class_to_idx,
            "allowed_labels": sorted(self.allowed_labels),
            "expected_tensor_shape": list(self.expected_tensor_shape),
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "scaler_sha256": self.scaler_sha256,
            "scaler_manifest_sha256": self.scaler_manifest_sha256,
            "is_osr_test": self.is_osr_test,
        }

    def _recompute_class_counts(self, index):
        class_counts = {idx: 0 for idx in self.class_to_idx.values()}
        for _, _, class_idx in index:
            if class_idx in class_counts:
                class_counts[class_idx] += 1
        return class_counts

    def _build_or_load_index(self):
        index_file = os.path.join(self.data_dir, f"dataset_index_v{INDEX_SCHEMA_VERSION}_{self.split_name}_nmin{self.n_min}.pt")
        identity = self._index_identity()

        if os.path.exists(index_file):
            try:
                payload = torch.load(index_file, map_location="cpu")
                mismatches = [key for key, value in identity.items() if not isinstance(payload, dict) or payload.get(key) != value]
                if not mismatches:
                    logging.info("[*] Cargando índice versionado desde %s", index_file)
                    return payload["index"], payload["class_counts"]
                logging.info("[*] Índice invalidado automáticamente. Campos distintos: %s", mismatches)
            except Exception as exc:
                logging.warning("[!] Índice incompatible o corrupto; se reconstruirá: %s", exc)

        logging.info("[*] Construyendo índice HDF5 versionado para %s...", self.split_name)
        index = []
        errors = []
        files = [item["relative_path"] for item in self.dataset_manifest]

        for filename in tqdm(files, desc=f"Indexando {self.split_name}"):
            path = os.path.join(self.data_dir, filename)
            try:
                with h5py.File(path, "r", swmr=True) as hdf5_file:
                    for flow_id in hdf5_file.keys():
                        group = hdf5_file[flow_id]
                        label_name = group.attrs.get("label", "")
                        split_attr = group.attrs.get("split", "")
                        if isinstance(label_name, bytes):
                            label_name = label_name.decode("utf-8", errors="replace")
                        if isinstance(split_attr, bytes):
                            split_attr = split_attr.decode("utf-8", errors="replace")
                        label_name, split_attr = str(label_name), str(split_attr)

                        if split_attr != self.split_name:
                            raise RuntimeError(f"{filename}/{flow_id}: split={split_attr}, esperado={self.split_name}")
                        if label_name not in self.allowed_labels or label_name not in self.class_to_idx:
                            raise RuntimeError(f"{filename}/{flow_id}: etiqueta no permitida {label_name!r}")
                        if "rgb_e_tensor" not in group:
                            raise RuntimeError(f"{filename}/{flow_id}: rgb_e_tensor ausente")
                        if tuple(group["rgb_e_tensor"].shape) != self.expected_tensor_shape:
                            raise RuntimeError(f"{filename}/{flow_id}: forma {group['rgb_e_tensor'].shape}, esperada {self.expected_tensor_shape}")

                        captured_packets = int(group.attrs.get("captured_packets", -1))
                        if captured_packets < 0:
                            raise RuntimeError(f"{filename}/{flow_id}: captured_packets ausente")
                        if captured_packets >= self.n_min:
                            index.append((filename, flow_id, self.class_to_idx[label_name]))
            except Exception as exc:
                if self.strict:
                    raise RuntimeError(f"Error indexando {filename}: {exc}") from exc
                errors.append(f"{filename}: {exc}")
                logging.error("[!] Error indexando %s: %s", filename, exc)

        if self.mode == "pilot" and len(index) > 1000:
            by_class = {idx: [] for idx in self.class_to_idx.values()}
            for item in index:
                by_class[item[2]].append(item)
            selected = []
            present = [idx for idx, items in by_class.items() if items]
            quota = max(1, 1000 // max(1, len(present)))
            for idx in present:
                selected.extend(by_class[idx][:quota])
            if len(selected) < 1000:
                selected_ids = {(item[0], item[1]) for item in selected}
                selected.extend(item for item in index if (item[0], item[1]) not in selected_ids and len(selected) < 1000)
            index = selected[:1000]

        if not index:
            raise RuntimeError(f"El índice de {self.split_name} quedó vacío para N_min={self.n_min}")

        class_counts = self._recompute_class_counts(index)
        payload = {
            **identity,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "index": index,
            "class_counts": class_counts,
            "tensor_count": len(index),
            "indexing_errors": errors,
        }
        _atomic_torch_save(payload, index_file)
        logging.info("[✓] Índice guardado: %s | muestras=%s", index_file, f"{len(index):,}")
        return index, class_counts

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        filename, flow_id, label = self.index[idx]
        path = os.path.join(self.data_dir, filename)

        try:
            if path not in self.worker_file_cache:
                if len(self.worker_file_cache) >= self.worker_cache_limit:
                    oldest_path = next(iter(self.worker_file_cache))
                    self.worker_file_cache.pop(oldest_path).close()
                self.worker_file_cache[path] = h5py.File(path, "r", swmr=True)

            group = self.worker_file_cache[path][flow_id]
            tensor_np = group["rgb_e_tensor"][:]
            if tuple(tensor_np.shape) != self.expected_tensor_shape:
                raise ValueError(f"Tensor inválido en {filename}/{flow_id}: {tensor_np.shape}")

            tensor_np = tensor_np[:self.n_min, :self.tensor_width, :]
            image = np.transpose(tensor_np, (2, 0, 1)).astype(np.float32, copy=False)

            if self.max_r > self.min_r:
                image[0] = (image[0] - self.min_r) / (self.max_r - self.min_r)
                image[1] = (image[1] - self.min_r) / (self.max_r - self.min_r)
            if self.max_e > self.min_e:
                image[2] = (image[2] - self.min_e) / (self.max_e - self.min_e)

            return torch.from_numpy(np.ascontiguousarray(image)), torch.tensor(label, dtype=torch.long)

        except Exception as exc:
            message = f"HDF5 Read Error en {filename}, flow {flow_id}: {exc}"
            if self.strict:
                raise RuntimeError(message) from exc
            logging.error("[!] %s", message)
            return None

    def close(self):
        for file in self.worker_file_cache.values():
            try:
                file.close()
            except Exception:
                pass
        self.worker_file_cache.clear()

    def __del__(self):
        self.close()


class StratifiedBatchSampler(Sampler):
    """Muestreo estratificado sin reemplazo: conserva la distribución natural y fuerza presencia de clases activas."""
    def __init__(self, labels, batch_size, seed, drop_last=False):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.class_indices = {int(label): np.flatnonzero(self.labels == label) for label in np.unique(self.labels)}
        if self.batch_size < len(self.class_indices):
            raise ValueError("batch_size debe ser mayor o igual al número de clases presentes")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        if self.drop_last:
            return len(self.labels) // self.batch_size
        return math.ceil(len(self.labels) / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        pools = {label: rng.permutation(indices).tolist() for label, indices in self.class_indices.items()}
        positions = {label: 0 for label in pools}
        remaining = {label: len(indices) for label, indices in pools.items()}
        total_remaining = sum(remaining.values())

        while total_remaining > 0:
            current_size = min(self.batch_size, total_remaining)
            if self.drop_last and current_size < self.batch_size:
                break

            batch = []
            active = [label for label, count in remaining.items() if count > 0]
            if current_size >= len(active):
                for label in active:
                    batch.append(pools[label][positions[label]])
                    positions[label] += 1
                    remaining[label] -= 1
                    total_remaining -= 1
            else:
                active = sorted(active, key=lambda label: (-remaining[label], label))[:current_size]
                for label in active:
                    batch.append(pools[label][positions[label]])
                    positions[label] += 1
                    remaining[label] -= 1
                    total_remaining -= 1

            while len(batch) < current_size:
                active = [label for label, count in remaining.items() if count > 0]
                weights = np.asarray([remaining[label] for label in active], dtype=np.float64)
                label = int(rng.choice(active, p=weights / weights.sum()))
                batch.append(pools[label][positions[label]])
                positions[label] += 1
                remaining[label] -= 1
                total_remaining -= 1

            rng.shuffle(batch)
            yield batch


# ==============================================================================
# 2. ARQUITECTURA: VISION TRANSFORMER OSR Y XAI
# ==============================================================================
class TransparentTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, embed_dim), nn.Dropout(dropout))

    def forward(self, x, return_attention=True):
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.attn(norm_x, norm_x, norm_x, need_weights=return_attention)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class ViT_OSR(nn.Module):
    def __init__(self, n_min, max_bytes=144, patch_size=(1, 16), embed_dim=768, depth=12, num_heads=12, num_classes=3):
        super().__init__()
        self.patch_h, self.patch_w = patch_size
        if n_min % self.patch_h != 0 or max_bytes % self.patch_w != 0:
            raise ValueError("Las dimensiones de entrada deben ser divisibles por patch_size")
        num_patches = (n_min // self.patch_h) * (max_bytes // self.patch_w)
        patch_dim = 3 * self.patch_h * self.patch_w

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.patch_h, p2=self.patch_w),
            nn.LayerNorm(patch_dim), nn.Linear(patch_dim, embed_dim), nn.LayerNorm(embed_dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.layers = nn.ModuleList([TransparentTransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    def forward(self, image, return_attention=True):
        x = self.to_patch_embedding(image)
        batch_size, num_tokens, _ = x.shape
        x = torch.cat((self.cls_token.expand(batch_size, -1, -1), x), dim=1)
        x = x + self.pos_embedding[:, :num_tokens + 1]
        attention_maps = []
        for layer in self.layers:
            x, attn_weights = layer(x, return_attention=return_attention)
            if return_attention:
                attention_maps.append(attn_weights)
        cls_output = x[:, 0]
        return self.mlp_head(cls_output), cls_output, attention_maps


class FocalLoss(nn.Module):
    """Weighted Focal Loss utilizada como pérdida principal de clasificación."""
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction debe ser 'mean', 'sum' o 'none'")
        if weight is None:
            self.weight = None
        else:
            self.register_buffer("weight", weight)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_unweighted = F.cross_entropy(inputs, targets, reduction="none")
        probability_true_class = torch.exp(-ce_unweighted)
        ce_weighted = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        focal_loss = ((1.0 - probability_true_class) ** self.gamma) * ce_weighted
        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class RobustFisherLoss(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int, device, momentum: float = 0.5, eps: float = 1e-6):
        super().__init__()
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum debe pertenecer al intervalo [0, 1]")
        if eps <= 0.0:
            raise ValueError("eps debe ser estrictamente positivo")

        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("centroids", torch.zeros(num_classes, feat_dim, device=device))
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool, device=device))

    def forward(self, features, labels, logits):
        predictions = logits.detach().argmax(dim=1)
        correct_mask = predictions.eq(labels)
        intra_sum = features.new_tensor(0.0)
        valid_samples = 0
        differentiable_centroids = []
        differentiable_counts = []

        for class_idx in range(self.num_classes):
            class_features = features[labels.eq(class_idx) & correct_mask]
            if class_features.size(0) == 0:
                continue
            batch_centroid = class_features.mean(dim=0)
            differentiable_centroids.append(batch_centroid)
            differentiable_counts.append(class_features.size(0))

            with torch.no_grad():
                if not self.initialized[class_idx]:
                    self.centroids[class_idx].copy_(batch_centroid.detach())
                    self.initialized[class_idx] = True
                else:
                    self.centroids[class_idx].mul_(self.momentum).add_(batch_centroid.detach(), alpha=1.0 - self.momentum)

            distances = torch.sum((class_features - self.centroids[class_idx].detach()) ** 2, dim=1)
            intra_sum = intra_sum + distances.sum()
            valid_samples += class_features.size(0)

        if valid_samples == 0:
            return features.sum() * 0.0

        intra_loss = intra_sum / valid_samples
        if len(differentiable_centroids) < 2:
            return intra_loss

        batch_centroids = torch.stack(differentiable_centroids)
        counts = features.new_tensor(differentiable_counts, dtype=features.dtype)
        weights = counts / counts.sum()
        global_centroid = torch.sum(weights.unsqueeze(1) * batch_centroids, dim=0)
        inter_loss = torch.sum(weights * torch.sum((batch_centroids - global_centroid) ** 2, dim=1))
        return intra_loss / (inter_loss + self.eps)


def get_lambda(current_epoch, total_epochs):
    """Peso progresivo definido por la metodología: 0.05 * exp(-5p)."""
    progress = current_epoch / max(total_epochs - 1, 1)
    return 0.05 * math.exp(-5.0 * progress)


# ==============================================================================
# 3. MÉTRICAS, VALIDACIÓN Y GEOMETRÍA
# ==============================================================================
def _update_confusion(confusion, labels, predictions, num_classes):
    encoded = labels.detach().to(torch.int64) * num_classes + predictions.detach().to(torch.int64)
    confusion += torch.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes).cpu().numpy()


def _metrics_from_confusion(confusion, class_to_idx):
    confusion = np.asarray(confusion, dtype=np.int64)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    numerator = correct * total - float(np.dot(support, predicted))
    denominator = math.sqrt(max(0.0, (total ** 2 - float(np.dot(predicted, predicted))) * (total ** 2 - float(np.dot(support, support)))))
    mcc = numerator / denominator if denominator > 0.0 else 0.0
    recall = np.divide(np.diag(confusion), support, out=np.zeros_like(support, dtype=np.float64), where=support > 0)

    names_by_idx = {idx: name for name, idx in class_to_idx.items()}
    recall_by_class = {names_by_idx[idx]: float(recall[idx]) for idx in range(len(names_by_idx))}
    fnr_by_class = {name: float(1.0 - value) for name, value in recall_by_class.items()}
    support_by_class = {names_by_idx[idx]: int(support[idx]) for idx in range(len(names_by_idx))}
    present = recall[support > 0]
    return {
        "mcc": float(mcc),
        "recall_macro": float(present.mean()) if present.size else 0.0,
        "recall_by_class": recall_by_class,
        "fnr_by_class": fnr_by_class,
        "support_by_class": support_by_class,
        "confusion_matrix": confusion.tolist(),
    }


def _geometry_from_stream(counts, sums, squared_norm_sums, eps=1e-6):
    valid = counts > 0
    if int(valid.sum().item()) < 2:
        return {"intra_class_scatter": 0.0, "inter_class_scatter": 0.0, "intra_inter_ratio": 0.0, "class_counts": counts.cpu().tolist()}

    counts_f = counts.float()
    centroids = torch.zeros_like(sums)
    centroids[valid] = sums[valid] / counts_f[valid].unsqueeze(1)
    total_count = counts_f[valid].sum()
    global_centroid = sums[valid].sum(dim=0) / total_count

    intra_total = torch.clamp(torch.sum(squared_norm_sums[valid] - torch.sum(sums[valid] * sums[valid], dim=1) / counts_f[valid]), min=0.0)
    inter_total = torch.clamp(torch.sum(counts_f[valid] * torch.sum((centroids[valid] - global_centroid) ** 2, dim=1)), min=0.0)
    intra = intra_total / total_count
    inter = inter_total / total_count
    return {
        "intra_class_scatter": float(intra.item()),
        "inter_class_scatter": float(inter.item()),
        "intra_inter_ratio": float((intra / (inter + eps)).item()),
        "class_counts": [int(value) for value in counts.cpu().tolist()],
    }


def _evaluate_validation(model, dataloader, criterion, device, amp_enabled, amp_dtype, num_classes, feat_dim, class_to_idx):
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    running_loss = 0.0
    sample_count = 0
    geometry_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    geometry_sums = torch.zeros(num_classes, feat_dim, dtype=torch.float32, device=device)
    geometry_squared_norms = torch.zeros(num_classes, dtype=torch.float32, device=device)

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validación", leave=False):
            if len(inputs) == 0:
                continue
            inputs = inputs.to(device, non_blocking=amp_enabled)
            labels = labels.to(device, non_blocking=amp_enabled)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits, features, _ = model(inputs, return_attention=False)
                loss = criterion(logits, labels)

            predictions = logits.argmax(dim=1)
            _update_confusion(confusion, labels, predictions, num_classes)
            batch_size = labels.size(0)
            running_loss += float(loss.item()) * batch_size
            sample_count += batch_size

            features = features.float()
            for class_idx in range(num_classes):
                class_features = features[labels.eq(class_idx)]
                if class_features.numel() == 0:
                    continue
                geometry_counts[class_idx] += class_features.size(0)
                geometry_sums[class_idx] += class_features.sum(dim=0)
                geometry_squared_norms[class_idx] += torch.sum(class_features * class_features)

    if sample_count == 0:
        raise RuntimeError("No se procesó ninguna muestra válida de val_known")

    metrics = _metrics_from_confusion(confusion, class_to_idx)
    metrics["loss"] = running_loss / sample_count
    metrics["geometry"] = _geometry_from_stream(geometry_counts, geometry_sums, geometry_squared_norms)
    return metrics


# ==============================================================================
# 4. ORQUESTADOR: ENTRENAMIENTO FOCAL + FISHER
# ==============================================================================
def _resolve_amp(device, train_conf):
    enabled = bool(train_conf.get("mixed_precision", True)) and device.type == "cuda"
    requested = str(train_conf.get("amp_dtype", "bfloat16")).lower()
    if not enabled:
        return False, torch.float32, False
    if requested == "bfloat16" and torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, False
    if requested not in {"bfloat16", "float16"}:
        raise ValueError(f"amp_dtype no soportado: {requested}")
    if requested == "bfloat16":
        logging.warning("[!] BF16 no soportado; se utilizará FP16")
    return True, torch.float16, True


def _create_grad_scaler(device, enabled=True):
    """Compatibilidad con APIs nuevas y anteriores de PyTorch."""
    enabled = bool(enabled and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _restore_rng_state(checkpoint, device, train_generator=None, val_generator=None):
    if "rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state"].cpu())
    if device.type == "cuda" and checkpoint.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    if train_generator is not None and checkpoint.get("train_generator_state") is not None:
        train_generator.set_state(checkpoint["train_generator_state"])
    if val_generator is not None and checkpoint.get("val_generator_state") is not None:
        val_generator.set_state(checkpoint["val_generator_state"])


def _checkpoint_identity(n_min, tensor_width, class_to_idx, sampler_config, train_dataset, val_dataset, vit_conf, train_conf, loader_conf):
    git_commit, git_dirty = _git_metadata()
    hashes = {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "script_sha256": _sha256_file(__file__),
        "global_config_sha256": _sha256_file("configs/global_config.yaml"),
        "dataset_schedule_sha256": _sha256_file("configs/dataset_schedule.yaml"),
        "scaler_sha256": train_dataset.scaler_sha256,
        "scaler_manifest_sha256": train_dataset.scaler_manifest_sha256,
        "train_index_manifest_sha256": train_dataset.dataset_manifest_sha256,
        "val_index_manifest_sha256": val_dataset.dataset_manifest_sha256,
    }
    identity = {
        "n_min": int(n_min),
        "tensor_shape": [int(n_min), int(tensor_width), 3],
        "class_to_idx": class_to_idx,
        "sampler_config": sampler_config,
        "model_config": dict(vit_conf),
        "training_config": dict(train_conf),
        "data_loader_config": dict(loader_conf),
        "loss_variant": "weighted_focal_plus_progressive_fisher",
        "lambda_formula": "0.05 * exp(-5p)",
        "hashes": {key: value for key, value in hashes.items() if key != "git_dirty"},
    }
    return identity, hashes, _canonical_hash(identity)


def train_ablation_study(mode):
    global env
    env = _ensure_environment(mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_conf = env.get_value("training")
    vit_conf = env.get_value("vit_model")
    loader_conf = env.get_value("data_loader")
    seed = int(env.get_value("project", "seed"))
    _set_global_seed(seed)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    amp_enabled, amp_dtype, use_grad_scaler = _resolve_amp(device, train_conf)
    logging.info("[*] Acelerador: %s | AMP: %s | dtype: %s", device, amp_enabled, amp_dtype)

    n_min_candidates = [9]
    epochs_per_ablation = 5 if mode == "pilot" else int(train_conf["epochs"])
    batch_size = 32 if mode == "pilot" else int(train_conf["batch_size"])
    learning_rate = float(train_conf["learning_rate"])
    checkpoint_frequency = int(train_conf.get("checkpoint_frequency", 5))
    tensor_width = int(env.get_value("preprocessing", "tensor_width"))

    scaler_json = env.get_path("paths", "configs", "scaler_bounds", is_file=True)
    train_dir = env.get_path("paths", "output", "train_known", ensure_exists=True)
    val_dir = env.get_path("paths", "output", "val_known", ensure_exists=True)
    checkpoint_dir = env.get_path("paths", "artifacts", "checkpoints", ensure_exists=True)
    telemetry_dir = env.get_path("paths", "artifacts", "telemetry_logs", ensure_exists=True)

    for n_min in n_min_candidates:
        logging.info("\n%s\n[*] ENTRENAMIENTO FOCAL + FISHER | N_min=%s\n%s", "=" * 60, n_min, "=" * 60)

        train_dataset = IDS2018Dataset(train_dir, scaler_json, n_min, max_bytes=tensor_width, mode=mode, split_name="train_known")
        val_dataset = IDS2018Dataset(val_dir, scaler_json, n_min, max_bytes=tensor_width, mode=mode, split_name="val_known")
        num_classes = len(KNOWN_CLASS_TO_IDX)

        train_counts = [train_dataset.class_counts.get(idx, 0) for idx in range(num_classes)]
        val_counts = [val_dataset.class_counts.get(idx, 0) for idx in range(num_classes)]
        missing_train = [name for name, idx in KNOWN_CLASS_TO_IDX.items() if train_counts[idx] == 0]
        missing_val = [name for name, idx in KNOWN_CLASS_TO_IDX.items() if val_counts[idx] == 0]
        if missing_train or missing_val:
            message = f"Clases ausentes | train={missing_train} | val={missing_val}"
            if mode == "prod":
                raise RuntimeError(message)
            logging.warning("[!] %s", message)

        total_samples = len(train_dataset)
        present_class_count = sum(count > 0 for count in train_counts)
        class_weights = torch.tensor([0.0 if count == 0 else total_samples / (present_class_count * count) for count in train_counts], dtype=torch.float32, device=device)

        train_sampler = StratifiedBatchSampler(train_dataset.labels, batch_size=batch_size, seed=seed, drop_last=False)
        sampler_config = {
            "name": "stratified_without_replacement",
            "batch_size": batch_size,
            "seed": seed,
            "preserves_natural_epoch_distribution": True,
            "minimum_one_sample_per_active_class": True,
            "replacement": False,
        }

        configured_workers = 0 if mode == "pilot" else int(loader_conf["num_workers"])
        loader_common = {
            "num_workers": configured_workers,
            "pin_memory": bool(loader_conf.get("pin_memory", True) and device.type == "cuda"),
            "collate_fn": safe_collate,
            "worker_init_fn": seed_worker,
        }
        if configured_workers > 0:
            loader_common["prefetch_factor"] = int(loader_conf.get("prefetch_factor", 2))
            loader_common["persistent_workers"] = bool(loader_conf.get("persistent_workers", True))

        train_generator = torch.Generator().manual_seed(seed)
        val_generator = torch.Generator().manual_seed(seed + 1)
        train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, generator=train_generator, **loader_common)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, generator=val_generator, **loader_common)

        raw_model = ViT_OSR(
            n_min=n_min, max_bytes=tensor_width, patch_size=(1, int(vit_conf["patch_size"])),
            embed_dim=int(vit_conf["embed_dim"]), depth=int(vit_conf["depth"]),
            num_heads=int(vit_conf["num_heads"]), num_classes=num_classes,
        ).to(device)

        model = raw_model
        if bool(train_conf.get("compile_model", False)) and hasattr(torch, "compile"):
            model = torch.compile(raw_model)

        total_parameters = sum(parameter.numel() for parameter in raw_model.parameters() if parameter.requires_grad)
        logging.info("[*] Train=%s | Val=%s | clases train=%s | clases val=%s", f"{len(train_dataset):,}", f"{len(val_dataset):,}", train_counts, val_counts)
        logging.info("[*] ViT: %sd, %s capas, %s cabezas, ancho=%s | parámetros=%s", vit_conf["embed_dim"], vit_conf["depth"], vit_conf["num_heads"], tensor_width, f"{total_parameters:,}")

        criterion_primary = FocalLoss(weight=class_weights, gamma=2.0).to(device)
        criterion_fisher = RobustFisherLoss(num_classes=num_classes, feat_dim=int(vit_conf["embed_dim"]), device=device).to(device)
        optimizer = torch.optim.AdamW(raw_model.parameters(), lr=learning_rate, weight_decay=0.01)
        grad_scaler = _create_grad_scaler(device, enabled=use_grad_scaler)

        experiment_identity, hashes, experiment_hash = _checkpoint_identity(
            n_min, tensor_width, KNOWN_CLASS_TO_IDX, sampler_config, train_dataset, val_dataset,
            vit_conf, train_conf, loader_conf,
        )

        latest_path = os.path.join(checkpoint_dir, f"vit_nmin_{n_min}_latest.pt")
        best_path = os.path.join(checkpoint_dir, f"vit_nmin_{n_min}_checkpoint.pt")
        history_path = os.path.join(telemetry_dir, f"training_history_nmin_{n_min}_{mode}.json")
        start_epoch, best_val_mcc, best_val_loss, best_epoch, history = 0, float("-inf"), float("inf"), -1, []

        if os.path.exists(latest_path):
            checkpoint = torch.load(latest_path, map_location=device)
            if checkpoint.get("experiment_hash") != experiment_hash:
                raise RuntimeError("El checkpoint existente no corresponde al dataset, scaler, configuración o código actuales")
            if "fisher_state" not in checkpoint:
                raise RuntimeError("El checkpoint existente no contiene fisher_state")
            raw_model.load_state_dict(checkpoint["model_state"])
            criterion_fisher.load_state_dict(checkpoint["fisher_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            grad_scaler.load_state_dict(checkpoint.get("scaler_state", {}))
            _restore_rng_state(checkpoint, device, train_generator, val_generator)
            start_epoch = int(checkpoint["epoch"]) + 1
            best_val_mcc = float(checkpoint.get("best_val_mcc", float("-inf")))
            best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
            best_epoch = int(checkpoint.get("best_epoch", -1))
            history = list(checkpoint.get("history", []))
            logging.info("[*] Reanudando desde la época %s", start_epoch + 1)

        if start_epoch >= epochs_per_ablation:
            logging.info("[*] Entrenamiento N_min=%s ya completado. Mejor época=%s, MCC=%.4f", n_min, best_epoch + 1, best_val_mcc)
            continue

        for epoch in range(start_epoch, epochs_per_ablation):
            epoch_start = time.perf_counter()
            train_sampler.set_epoch(epoch)
            model.train()
            criterion_fisher.train()
            running_total_loss = 0.0
            running_primary_loss = 0.0
            running_fisher_loss = 0.0
            processed_steps = 0
            train_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
            lambda_fisher = get_lambda(epoch, epochs_per_ablation)

            progress_bar = tqdm(train_loader, desc=f"Época {epoch + 1}/{epochs_per_ablation}")
            for inputs, labels in progress_bar:
                if len(inputs) == 0:
                    continue
                inputs = inputs.to(device, non_blocking=amp_enabled)
                labels = labels.to(device, non_blocking=amp_enabled)
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    logits, features, _ = model(inputs, return_attention=False)
                    primary_loss = criterion_primary(logits, labels)

                fisher_loss = criterion_fisher(features.float(), labels, logits.float())
                total_loss = primary_loss.float() + lambda_fisher * fisher_loss
                if not torch.isfinite(total_loss):
                    raise FloatingPointError(f"Loss NaN/Inf en época {epoch + 1}: focal={primary_loss.item()}, fisher={fisher_loss.item()}, lambda={lambda_fisher}")

                grad_scaler.scale(total_loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                processed_steps += 1
                running_total_loss += float(total_loss.item())
                running_primary_loss += float(primary_loss.item())
                running_fisher_loss += float(fisher_loss.item())
                predictions = logits.detach().argmax(dim=1)
                _update_confusion(train_confusion, labels, predictions, num_classes)
                progress_bar.set_postfix(total=f"{total_loss.item():.4f}", focal=f"{primary_loss.item():.4f}", fisher=f"{fisher_loss.item():.4f}", lambda_f=f"{lambda_fisher:.6f}")

            if processed_steps == 0:
                raise RuntimeError("No se procesó ningún lote válido durante la época")

            train_metrics = _metrics_from_confusion(train_confusion, KNOWN_CLASS_TO_IDX)
            train_metrics.update({
                "total_loss": running_total_loss / processed_steps,
                "focal_loss": running_primary_loss / processed_steps,
                "fisher_loss": running_fisher_loss / processed_steps,
            })
            val_metrics = _evaluate_validation(
                model, val_loader, criterion_primary, device, amp_enabled, amp_dtype,
                num_classes, int(vit_conf["embed_dim"]), KNOWN_CLASS_TO_IDX,
            )

            epoch_duration = time.perf_counter() - epoch_start
            epoch_record = {
                "epoch": epoch + 1,
                "lambda_fisher": lambda_fisher,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "duration_seconds": epoch_duration,
                "train": train_metrics,
                "validation": val_metrics,
            }
            history.append(epoch_record)
            _atomic_json_save(history, history_path)

            improved = val_metrics["mcc"] > best_val_mcc or (math.isclose(val_metrics["mcc"], best_val_mcc, rel_tol=0.0, abs_tol=1e-12) and val_metrics["loss"] < best_val_loss)
            if improved:
                best_val_mcc, best_val_loss, best_epoch = val_metrics["mcc"], val_metrics["loss"], epoch

            checkpoint_payload = {
                "epoch": epoch,
                "model_state": raw_model.state_dict(),
                "fisher_state": criterion_fisher.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": grad_scaler.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
                "train_generator_state": train_generator.get_state(),
                "val_generator_state": val_generator.get_state(),
                "best_val_mcc": best_val_mcc,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "geometry_metrics": val_metrics["geometry"],
                "history": history,
                "class_to_idx": KNOWN_CLASS_TO_IDX,
                "class_weights": class_weights.detach().cpu(),
                "sampler_config": sampler_config,
                "seed": seed,
                "tensor_shape": [n_min, tensor_width, 3],
                "model_config": dict(vit_conf),
                "training_config": dict(train_conf),
                "data_loader_config": dict(loader_conf),
                "experiment_identity": experiment_identity,
                "experiment_hash": experiment_hash,
                "hashes": hashes,
                "loss_variant": "weighted_focal_plus_progressive_fisher",
                "lambda_formula": "0.05 * exp(-5p)",
                "n_min": n_min,
            }
            _atomic_torch_save(checkpoint_payload, latest_path)

            if improved:
                best_payload = dict(checkpoint_payload)
                best_payload["checkpoint_kind"] = "best_validation_mcc"
                _atomic_torch_save(best_payload, best_path)
                logging.info("[✓] Nuevo mejor checkpoint: época=%s | Val MCC=%.4f | Val loss=%.4f", epoch + 1, best_val_mcc, best_val_loss)

            if (epoch + 1) % checkpoint_frequency == 0:
                historical_path = os.path.join(checkpoint_dir, f"vit_nmin_{n_min}_epoch_{epoch + 1}.pt")
                shutil.copyfile(latest_path, historical_path)

            logging.info(
                "[N_min=%s] Época %s | Train Total=%.4f Focal=%.4f Fisher=%.4f MCC=%.4f | "
                "Val Loss=%.4f MCC=%.4f RecallMacro=%.4f | Intra=%.4f Inter=%.4f Ratio=%.4f | %.1fs",
                n_min, epoch + 1, train_metrics["total_loss"], train_metrics["focal_loss"],
                train_metrics["fisher_loss"], train_metrics["mcc"], val_metrics["loss"],
                val_metrics["mcc"], val_metrics["recall_macro"],
                val_metrics["geometry"]["intra_class_scatter"],
                val_metrics["geometry"]["inter_class_scatter"],
                val_metrics["geometry"]["intra_inter_ratio"], epoch_duration,
            )
            logging.info("[VAL] Recall=%s | FNR=%s | Support=%s", val_metrics["recall_by_class"], val_metrics["fnr_by_class"], val_metrics["support_by_class"])

        logging.info("[✓] Entrenamiento completado. Mejor época=%s | Val MCC=%.4f | checkpoint=%s", best_epoch + 1, best_val_mcc, best_path)


if __name__ == "__main__":
    args, _ = parser.parse_known_args()
    env = setup_environment(script_name="phase3_ablation", args=args)
    train_ablation_study(env.mode)