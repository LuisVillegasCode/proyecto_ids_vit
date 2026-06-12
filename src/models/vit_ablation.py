import argparse
import json
import logging
import math
import os
import random
import shutil

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from sklearn.metrics import matthews_corrcoef
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.utils.config_manager import setup_environment

# ==============================================================================
# 0. INYECCIÓN DE ENTORNO Y CONFIGURACIÓN GLOBAL
# ==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=["pilot", "prod"], required=True)
env = None

def _ensure_environment(mode: str):
    """
    Inicializa el entorno únicamente cuando se ejecuta el entrenamiento.
    Mantiene train_ablation_study(mode) compatible.
    """
    global env

    if env is None:
        runtime_args = argparse.Namespace(mode=mode)
        env = setup_environment(script_name="phase3_ablation",args=runtime_args)

    return env


def safe_collate(batch):
    """Descarta muestras inválidas sin provocar el fallo completo del DataLoader."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.utils.data.dataloader.default_collate(batch)


# ==============================================================================
# 1. EL PUENTE I/O: DATALOADER OPTIMIZADO (FR4.1 + NFR2)
# ==============================================================================
class IDS2018Dataset(Dataset):
    def __init__(
        self,
        data_dir,
        scaler_json,
        n_min,
        max_bytes=128,
        mode="prod",
        is_osr_test=False,
    ):
        self.data_dir = data_dir
        self.n_min = n_min
        self.max_bytes = max_bytes
        self.mode = mode
        self.worker_file_cache = {}

        with open(scaler_json, "r", encoding="utf-8") as file:
            bounds = json.load(file)
            self.min_e = bounds["entropy_channel"]["min"]
            self.max_e = bounds["entropy_channel"]["max"]
            self.min_r = bounds["raw_bytes_channel"]["min"]
            self.max_r = bounds["raw_bytes_channel"]["max"]

        self.class_to_idx = {
            "Benign": 0,
            "BruteForce": 1,
            "DoS": 2,
            "DDoS": 3,
            "Brute_Force_Web": 4,
            "Brute_Force_XSS": 5,
            "SQL_Injection": 6,
        }

        if is_osr_test:
            self.class_to_idx["Botnet"] = 7
            self.class_to_idx["Infiltration"] = 8

        self.index, self.class_counts = self._build_or_load_index()
        
        # Recalcular siempre desde el índice realmente utilizado.
        self.class_counts = {
            class_id: 0
            for class_id in self.class_to_idx.values()
        }

        for _, _, class_id in self.index:
            self.class_counts[class_id] += 1
        
        self.labels = np.asarray([item[2] for item in self.index], dtype=np.int64)

    def _recompute_class_counts(self, index):
        """Garantiza que los recuentos correspondan al índice realmente utilizado."""
        class_counts = {idx: 0 for idx in self.class_to_idx.values()}
        for _, _, class_idx in index:
            if class_idx in class_counts:
                class_counts[class_idx] += 1
        return class_counts

    def _build_or_load_index(self):
        index_file = os.path.join(self.data_dir, f"dataset_index_{self.mode}.pt")

        if os.path.exists(index_file):
            logging.info("[*] Cargando índice cacheado desde %s", index_file)
            cached_index, _ = torch.load(index_file)
            index = cached_index[:1000] if self.mode == "pilot" else cached_index
            class_counts = self._recompute_class_counts(index)
            return index, class_counts

        logging.info("[*] Construyendo índice maestro HDF5... (solo la primera vez)")
        files = sorted(
            filename
            for filename in os.listdir(self.data_dir)
            if filename.endswith(".hdf5")
        )
        if self.mode == "pilot":
            files = files[:2]

        index = []
        for filename in tqdm(files, desc="Indexando"):
            path = os.path.join(self.data_dir, filename)
            try:
                with h5py.File(path, "r", swmr=True) as hdf5_file:
                    for flow_id in hdf5_file.keys():
                        label_name = hdf5_file[flow_id].attrs.get("label", "Benign")
                        if label_name in self.class_to_idx:
                            index.append(
                                (filename, flow_id, self.class_to_idx[label_name])
                            )
            except Exception as exc:
                logging.error("[!] Error indexando %s: %s", filename, exc)

        if self.mode == "pilot":
            index = index[:1000]

        class_counts = self._recompute_class_counts(index)
        torch.save((index, class_counts), index_file)
        return index, class_counts

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        filename, flow_id, label = self.index[idx]
        path = os.path.join(self.data_dir, filename)

        try:
            if path not in self.worker_file_cache:
                if len(self.worker_file_cache) > 20:
                    oldest_path = next(iter(self.worker_file_cache))
                    self.worker_file_cache[oldest_path].close()
                    del self.worker_file_cache[oldest_path]
                self.worker_file_cache[path] = h5py.File(path, "r", swmr=True)

            hdf5_file = self.worker_file_cache[path]
            group = hdf5_file[flow_id]
            tensor_np = group["rgb_e_tensor"][:]
            tensor_np = tensor_np[: self.n_min, : self.max_bytes, :]

            if tensor_np.ndim != 3 or tensor_np.shape[-1] != 3:
                raise ValueError(
                    f"Tensor inválido en {filename}/{flow_id}: {tensor_np.shape}"
                )

            image = np.transpose(tensor_np, (2, 0, 1))

            if self.max_r > self.min_r:
                image[0] = (image[0] - self.min_r) / (self.max_r - self.min_r)
                image[1] = (image[1] - self.min_r) / (self.max_r - self.min_r)
            if self.max_e > self.min_e:
                image[2] = (image[2] - self.min_e) / (self.max_e - self.min_e)

            return (
                torch.tensor(image, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long),
            )

        except Exception as exc:
            logging.error(
                "[!] HDF5 Read Error en %s, flow %s: %s",
                filename,
                flow_id,
                exc,
            )
            return None


# ==============================================================================
# 2. ARQUITECTURA: VISION TRANSFORMER OSR Y XAI (FR6, FR13)
# ==============================================================================
class TransparentTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attention=True):
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.attn(
            norm_x,
            norm_x,
            norm_x,
            need_weights=return_attention,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class ViT_OSR(nn.Module):
    def __init__(
        self,
        n_min,
        max_bytes=128,
        patch_size=(1, 16),
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=7,
    ):
        super().__init__()
        self.patch_h, self.patch_w = patch_size
        num_patches = (n_min // self.patch_h) * (max_bytes // self.patch_w)
        patch_dim = 3 * self.patch_h * self.patch_w

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=self.patch_h,
                p2=self.patch_w,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.layers = nn.ModuleList(
            [
                TransparentTransformerBlock(embed_dim, num_heads)
                for _ in range(depth)
            ]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, image, return_attention=True):
        x = self.to_patch_embedding(image)
        batch_size, num_tokens, _ = x.shape
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, : num_tokens + 1]

        attention_maps = []
        for layer in self.layers:
            x, attn_weights = layer(x, return_attention=return_attention)
            if return_attention:
                attention_maps.append(attn_weights)

        cls_output = x[:, 0]
        logits = self.mlp_head(cls_output)
        return logits, cls_output, attention_maps


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
        ce_weighted = F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            reduction="none",
        )
        focal_loss = ((1.0 - probability_true_class) ** self.gamma) * ce_weighted

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class RobustFisherLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        device,
        momentum: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()

        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum debe pertenecer al intervalo [0, 1]")
        if eps <= 0.0:
            raise ValueError("eps debe ser estrictamente positivo")

        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.eps = eps

        self.register_buffer(
            "centroids",
            torch.zeros(num_classes, feat_dim, device=device),
        )
        self.register_buffer(
            "initialized",
            torch.zeros(num_classes, dtype=torch.bool, device=device),
        )

    def forward(self, features, labels, logits):
        predictions = logits.detach().argmax(dim=1)
        correct_mask = predictions.eq(labels)

        intra_sum = features.new_tensor(0.0)
        valid_samples = 0
        differentiable_centroids = []
        differentiable_counts = []

        for class_idx in range(self.num_classes):
            class_mask = labels.eq(class_idx) & correct_mask
            class_features = features[class_mask]

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
                    self.centroids[class_idx].mul_(self.momentum).add_(
                        batch_centroid.detach(),
                        alpha=1.0 - self.momentum,
                    )

            distances = torch.sum(
                (
                    class_features
                    - self.centroids[class_idx].detach()
                )
                ** 2,
                dim=1,
            )
            intra_sum = intra_sum + distances.sum()
            valid_samples += class_features.size(0)

        if valid_samples == 0:
            return features.sum() * 0.0

        intra_loss = intra_sum / valid_samples

        if len(differentiable_centroids) < 2:
            return intra_loss

        batch_centroids = torch.stack(differentiable_centroids)
        counts = features.new_tensor(
            differentiable_counts,
            dtype=features.dtype,
        )
        weights = counts / counts.sum()

        global_centroid = torch.sum(
            weights.unsqueeze(1) * batch_centroids,
            dim=0,
        )
        centroid_distances = torch.sum(
            (batch_centroids - global_centroid) ** 2,
            dim=1,
        )
        inter_loss = torch.sum(weights * centroid_distances)

        return intra_loss / (inter_loss + self.eps)


def get_lambda(current_epoch, total_epochs):
    """Peso progresivo definido por la referencia: 0.05 * exp(-5p)."""
    progress = current_epoch / max(total_epochs - 1, 1)
    return 0.05 * math.exp(-5.0 * progress)


# ==============================================================================
# 3. ORQUESTADOR: ESTUDIO DE ABLACIÓN Y ENTRENAMIENTO (FR7, NFR5, FR11)
# ==============================================================================
def _create_grad_scaler(device):
    """Compatibilidad con APIs nuevas y anteriores de PyTorch."""
    enabled = device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _restore_rng_state(checkpoint, device):
    if "rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state"].cpu())
    cuda_rng_state = checkpoint.get("cuda_rng_state_all")
    if device.type == "cuda" and cuda_rng_state is not None:
        torch.cuda.set_rng_state_all(cuda_rng_state)
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])


def train_ablation_study(mode):
    global env
    env = _ensure_environment(mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    logging.info("[*] Acelerador detectado: %s", device)

    n_min_candidates = [9]
    train_conf = env.get_value("training")
    vit_conf = env.get_value("vit_model")
    config_workers = env.get_value("preprocessing", "multiprocessing_workers")

    epochs_per_ablation = 5 if mode == "pilot" else train_conf["epochs"]
    batch_size = 32 if mode == "pilot" else train_conf["batch_size"]
    learning_rate = train_conf["learning_rate"]
    checkpoint_frequency = train_conf.get("checkpoint_frequency", 5)

    scaler_json = env.get_path(
        "paths",
        "configs",
        "scaler_bounds",
        is_file=True,
    )
    train_dir = env.get_path(
        "paths",
        "output",
        "train_val",
        ensure_exists=True,
    )
    checkpoint_dir = env.get_path(
        "paths",
        "artifacts",
        "checkpoints",
        ensure_exists=True,
    )

    for n_min in n_min_candidates:
        logging.info(
            "\n%s\n[*] INICIANDO ABLACIÓN PARA N_min = %s\n%s",
            "=" * 60,
            n_min,
            "=" * 60,
        )

        dataset = IDS2018Dataset(
            train_dir,
            scaler_json,
            n_min,
            mode=mode,
        )
        if len(dataset) == 0:
            raise RuntimeError("El conjunto de entrenamiento está vacío")

        num_classes = len(dataset.class_to_idx)
        counts = [dataset.class_counts.get(idx, 0) for idx in range(num_classes)]
        missing_classes = [idx for idx, count in enumerate(counts) if count == 0]

        if missing_classes:
            missing_names = [
                name
                for name, idx in dataset.class_to_idx.items()
                if idx in missing_classes
            ]
            message = f"Clases sin muestras en entrenamiento: {missing_names}"
            if mode == "prod":
                raise RuntimeError(message)
            logging.warning("[!] %s. Se continuará únicamente por ser modo piloto.", message)

        total_samples = len(dataset)
        present_class_count = sum(count > 0 for count in counts)
        class_weights = torch.tensor(
            [
                0.0
                if count == 0
                else total_samples / (present_class_count * count)
                for count in counts
            ],
            dtype=torch.float32,
            device=device,
        )

        optimal_workers = 0 if mode == "pilot" else config_workers
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": optimal_workers,
            "pin_memory": amp_enabled,
            "collate_fn": safe_collate,
        }
        if optimal_workers > 0:
            loader_kwargs["prefetch_factor"] = 2

        dataloader = DataLoader(**loader_kwargs)

        model = ViT_OSR(
            n_min=n_min,
            patch_size=(1, vit_conf["patch_size"]),
            embed_dim=vit_conf["embed_dim"],
            depth=vit_conf["depth"],
            num_heads=vit_conf["num_heads"],
            num_classes=num_classes,
        ).to(device)

        total_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        logging.info(
            "[*] ViT Arquitectura: %sd, %s layers, %s heads",
            vit_conf["embed_dim"],
            vit_conf["depth"],
            vit_conf["num_heads"],
        )
        logging.info("[*] Parámetros entrenables: %s", f"{total_parameters:,}")
        logging.info(
            "[*] Objetivo: Weighted Focal Loss + Progressive Fisher Regularization"
        )

        criterion_primary = FocalLoss(
            weight=class_weights,
            gamma=2.0,
        ).to(device)
        criterion_fisher = RobustFisherLoss(
            num_classes=num_classes,
            feat_dim=vit_conf["embed_dim"],
            device=device,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )
        grad_scaler = _create_grad_scaler(device)

        start_epoch = 0
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"vit_nmin_{n_min}_checkpoint.pt",
        )

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)

            if "fisher_state" not in checkpoint:
                raise RuntimeError(
                    "El checkpoint existente fue creado sin estado Fisher y no es "
                    "compatible con este objetivo de entrenamiento. Renómbralo o "
                    "elimínalo antes de comenzar el entrenamiento Fisher."
                )

            model.load_state_dict(checkpoint["model_state"])
            criterion_fisher.load_state_dict(checkpoint["fisher_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            grad_scaler.load_state_dict(checkpoint["scaler_state"])
            _restore_rng_state(checkpoint, device)

            start_epoch = checkpoint["epoch"] + 1
            logging.info(
                "[*] Rescatando entrenamiento desde la época %s",
                start_epoch,
            )

        if start_epoch >= epochs_per_ablation:
            logging.info(
                "[*] Ablación N_min=%s ya completada. Saltando.",
                n_min,
            )
            continue

        for epoch in range(start_epoch, epochs_per_ablation):
            model.train()
            criterion_fisher.train()

            running_total_loss = 0.0
            running_primary_loss = 0.0
            running_fisher_loss = 0.0
            processed_steps = 0
            all_predictions = []
            all_labels = []

            lambda_fisher = get_lambda(epoch, epochs_per_ablation)
            progress_bar = tqdm(
                dataloader,
                desc=f"Época {epoch + 1}/{epochs_per_ablation}",
            )

            for inputs, labels in progress_bar:
                if len(inputs) == 0:
                    continue

                inputs = inputs.to(
                    device,
                    non_blocking=amp_enabled,
                )
                labels = labels.to(
                    device,
                    non_blocking=amp_enabled,
                )
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=device.type,
                    enabled=amp_enabled,
                ):
                    logits, features, _ = model(
                        inputs,
                        return_attention=False,
                    )
                    primary_loss = criterion_primary(logits, labels)

                # Fisher queda fuera de AMP y recibe entradas float32.
                fisher_loss = criterion_fisher(
                    features.float(),
                    labels,
                    logits.float(),
                )
                total_loss = primary_loss.float() + (
                    lambda_fisher * fisher_loss
                )

                if not torch.isfinite(total_loss):
                    raise FloatingPointError(
                        f"Loss NaN/Inf en época {epoch + 1}, "
                        f"primary={primary_loss.item()}, "
                        f"fisher={fisher_loss.item()}, "
                        f"lambda={lambda_fisher}"
                    )

                grad_scaler.scale(total_loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                processed_steps += 1
                running_total_loss += total_loss.item()
                running_primary_loss += primary_loss.item()
                running_fisher_loss += fisher_loss.item()

                predictions = logits.detach().argmax(dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())

                progress_bar.set_postfix(
                    total=f"{total_loss.item():.4f}",
                    focal=f"{primary_loss.item():.4f}",
                    fisher=f"{fisher_loss.item():.4f}",
                    lambda_f=f"{lambda_fisher:.6f}",
                )

            if processed_steps == 0:
                raise RuntimeError(
                    "No se procesó ningún lote válido durante la época"
                )

            epoch_total_loss = running_total_loss / processed_steps
            epoch_primary_loss = running_primary_loss / processed_steps
            epoch_fisher_loss = running_fisher_loss / processed_steps
            mcc = (
                matthews_corrcoef(all_labels, all_predictions)
                if all_labels
                else 0.0
            )

            logging.info(
                "[N_min=%s] Época %s | Total: %.4f | Focal: %.4f | "
                "Fisher: %.4f | lambda: %.6f | MCC Train: %.4f",
                n_min,
                epoch + 1,
                epoch_total_loss,
                epoch_primary_loss,
                epoch_fisher_loss,
                lambda_fisher,
                mcc,
            )

            temporary_checkpoint = checkpoint_path + ".tmp"
            checkpoint_payload = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "fisher_state": criterion_fisher.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": grad_scaler.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all() if amp_enabled else None
                ),
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
                "mcc": mcc,
                "loss_variant": (
                    "weighted_focal_plus_progressive_fisher"
                ),
                "lambda_formula": "0.05 * exp(-5p)",
                "n_min": n_min,
            }
            torch.save(checkpoint_payload, temporary_checkpoint)
            os.replace(temporary_checkpoint, checkpoint_path)

            if (epoch + 1) % checkpoint_frequency == 0:
                historical_path = os.path.join(
                    checkpoint_dir,
                    f"vit_nmin_{n_min}_epoch_{epoch + 1}.pt",
                )
                shutil.copyfile(checkpoint_path, historical_path)
                logging.info(
                    "  -> Checkpoint histórico guardado: %s",
                    historical_path,
                )


if __name__ == "__main__":
    args, _ = parser.parse_known_args()
    env = setup_environment(
        script_name="phase3_ablation",
        args=args
    )
    train_ablation_study(env.mode)