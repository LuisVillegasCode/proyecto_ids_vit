# src/validation/validate_fisher_pipeline.py

import argparse
import copy
import logging
import math
import os
import random
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader, Subset

from src.models.vit_ablation import (
    FocalLoss,
    IDS2018Dataset,
    RobustFisherLoss,
    ViT_OSR,
    get_lambda,
    safe_collate,
)
from src.utils.config_manager import setup_environment


parser = argparse.ArgumentParser(
    description="Pruebas aisladas de integración Fisher para ViT-OSR"
)
parser.add_argument("--mode", choices=["pilot", "prod"], required=True)
parser.add_argument("--n_min", type=int, default=9)
parser.add_argument(
    "--test",
    choices=["all", "synthetic", "checkpoint", "single_batch", "short_run"],
    default="all",
)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--samples_per_class", type=int, default=4)
parser.add_argument("--short_epochs", type=int, default=3)
parser.add_argument("--train_per_class", type=int, default=32)
parser.add_argument("--val_per_class", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

env = setup_environment(script_name="validate_fisher_pipeline", args=args)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_grad_scaler(device: torch.device):
    enabled = device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def validate_gradients(model: torch.nn.Module) -> float:
    squared_norm = 0.0
    params_with_grad = 0

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue

        params_with_grad += 1

        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(
                f"Gradiente NaN/Inf detectado en el parámetro '{name}'."
            )

        norm = parameter.grad.detach().float().norm(2).item()
        squared_norm += norm * norm

    if params_with_grad == 0:
        raise RuntimeError("Ningún parámetro del modelo recibió gradientes.")

    return math.sqrt(squared_norm)


def actual_class_counts(labels: np.ndarray, num_classes: int) -> Dict[int, int]:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    return {class_id: int(counts[class_id]) for class_id in range(num_classes)}


def make_class_weights(
    labels: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    present = counts > 0

    weights = np.zeros(num_classes, dtype=np.float32)
    if present.any():
        total = counts[present].sum()
        num_present = int(present.sum())
        weights[present] = total / (num_present * counts[present])

    return torch.tensor(weights, dtype=torch.float32, device=device)


def choose_balanced_indices(
    labels: np.ndarray,
    samples_per_class: int,
    seed: int,
) -> List[int]:
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected: List[int] = []

    for class_id in np.unique(labels):
        candidates = np.flatnonzero(labels == class_id)
        amount = min(samples_per_class, len(candidates))
        if amount > 0:
            chosen = rng.choice(candidates, size=amount, replace=False)
            selected.extend(chosen.tolist())

    rng.shuffle(selected)

    selected_classes = np.unique(labels[selected]) if selected else np.array([])
    if len(selected_classes) < 2:
        raise RuntimeError(
            "La prueba requiere al menos dos clases presentes en el subconjunto."
        )

    return selected


def choose_balanced_train_val_indices(
    labels: np.ndarray,
    train_per_class: int,
    val_per_class: int,
    seed: int,
) -> Tuple[List[int], List[int]]:
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)

    train_indices: List[int] = []
    val_indices: List[int] = []

    for class_id in np.unique(labels):
        candidates = np.flatnonzero(labels == class_id)
        rng.shuffle(candidates)

        train_amount = min(train_per_class, len(candidates))
        remaining = max(0, len(candidates) - train_amount)
        val_amount = min(val_per_class, remaining)

        train_indices.extend(candidates[:train_amount].tolist())
        val_indices.extend(
            candidates[train_amount:train_amount + val_amount].tolist()
        )

    if len(np.unique(labels[train_indices])) < 2:
        raise RuntimeError("El short run necesita al menos dos clases en train.")

    if not val_indices or len(np.unique(labels[val_indices])) < 2:
        raise RuntimeError("El short run necesita al menos dos clases en validación.")

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def build_model(
    n_min: int,
    vit_conf: dict,
    num_classes: int,
    device: torch.device,
) -> ViT_OSR:
    return ViT_OSR(
        n_min=n_min,
        patch_size=(1, vit_conf["patch_size"]),
        embed_dim=vit_conf["embed_dim"],
        depth=vit_conf["depth"],
        num_heads=vit_conf["num_heads"],
        num_classes=num_classes,
    ).to(device)


def load_dataset(n_min: int, mode: str) -> Tuple[IDS2018Dataset, dict]:
    scaler_json = env.get_path(
        "paths", "configs", "scaler_bounds", is_file=True
    )
    train_dir = env.get_path(
        "paths", "output", "train_val", ensure_exists=True
    )
    vit_conf = env.get_value("vit_model")

    dataset = IDS2018Dataset(
        train_dir,
        scaler_json,
        n_min,
        mode=mode,
    )

    if len(dataset) == 0:
        raise RuntimeError("El dataset de entrenamiento está vacío.")

    counts = actual_class_counts(
        dataset.labels,
        len(dataset.class_to_idx),
    )
    logging.info("[*] Distribución real disponible para pruebas: %s", counts)

    if sum(value > 0 for value in counts.values()) < 2:
        raise RuntimeError("Se necesitan al menos dos clases para probar Fisher.")

    return dataset, vit_conf


def compute_geometry(
    features: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor | None = None,
    only_correct: bool = True,
    eps: float = 1e-6,
) -> Dict[str, float]:
    labels = labels.long()

    if logits is not None and only_correct:
        predictions = torch.argmax(logits, dim=1)
        usable = predictions.eq(labels)
    else:
        usable = torch.ones_like(labels, dtype=torch.bool)

    centroids = []
    counts = []
    intra_sum = features.new_tensor(0.0)
    valid_samples = 0

    for class_id in torch.unique(labels):
        mask = labels.eq(class_id) & usable
        class_features = features[mask]
        if class_features.size(0) == 0:
            continue

        centroid = class_features.mean(dim=0)
        centroids.append(centroid)
        counts.append(class_features.size(0))

        intra_sum = intra_sum + torch.sum(
            (class_features - centroid) ** 2
        )
        valid_samples += class_features.size(0)

    if valid_samples == 0:
        return {
            "valid_samples": 0.0,
            "valid_classes": 0.0,
            "intra": 0.0,
            "inter": 0.0,
            "ratio": 0.0,
        }

    intra = intra_sum / valid_samples

    if len(centroids) < 2:
        return {
            "valid_samples": float(valid_samples),
            "valid_classes": float(len(centroids)),
            "intra": float(intra.detach().item()),
            "inter": 0.0,
            "ratio": float(intra.detach().item()),
        }

    stacked = torch.stack(centroids)
    count_tensor = features.new_tensor(counts, dtype=features.dtype)
    weights = count_tensor / count_tensor.sum()
    global_centroid = torch.sum(weights.unsqueeze(1) * stacked, dim=0)
    inter = torch.sum(
        weights * torch.sum((stacked - global_centroid) ** 2, dim=1)
    )
    ratio = intra / (inter + eps)

    return {
        "valid_samples": float(valid_samples),
        "valid_classes": float(len(centroids)),
        "intra": float(intra.detach().item()),
        "inter": float(inter.detach().item()),
        "ratio": float(ratio.detach().item()),
    }


def run_synthetic_test(device: torch.device) -> None:
    logging.info("[*] TEST 1/4: Fisher sintético y flujo de gradientes")

    num_classes = 3
    feat_dim = 16
    samples_per_class = 6

    features = torch.randn(
        num_classes * samples_per_class,
        feat_dim,
        device=device,
        requires_grad=True,
    )
    labels = torch.arange(num_classes, device=device).repeat_interleave(
        samples_per_class
    )

    logits = torch.full(
        (labels.size(0), num_classes),
        -5.0,
        device=device,
    )
    logits[torch.arange(labels.size(0), device=device), labels] = 5.0

    criterion = RobustFisherLoss(
        num_classes=num_classes,
        feat_dim=feat_dim,
        device=device,
    )

    loss = criterion(features, labels, logits)

    if not torch.isfinite(loss):
        raise FloatingPointError("Fisher sintético devolvió NaN/Inf.")

    loss.backward()

    if features.grad is None:
        raise RuntimeError("Fisher no generó gradiente sobre features.")

    if not torch.isfinite(features.grad).all():
        raise FloatingPointError("Fisher produjo gradientes NaN/Inf.")

    if features.grad.abs().sum().item() == 0:
        raise RuntimeError("El gradiente Fisher es exactamente cero.")

    if int(criterion.initialized.sum().item()) != num_classes:
        raise RuntimeError("No se inicializaron todos los centroides sintéticos.")

    preserved_centroid = criterion.centroids[2].detach().clone()

    features_second = torch.randn(
        8,
        feat_dim,
        device=device,
        requires_grad=True,
    )
    labels_second = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1],
        device=device,
    )
    logits_second = torch.full((8, num_classes), -5.0, device=device)
    logits_second[torch.arange(8, device=device), labels_second] = 5.0

    _ = criterion(features_second, labels_second, logits_second)

    if not torch.equal(preserved_centroid, criterion.centroids[2]):
        raise RuntimeError(
            "El centroide de una clase ausente cambió; falló el rescate EMA=1.0."
        )

    logging.info(
        "[✓] Fisher sintético aprobado | Loss=%.6f | GradSum=%.6f",
        loss.item(),
        features.grad.abs().sum().item(),
    )


def run_checkpoint_roundtrip_test(device: torch.device) -> None:
    logging.info("[*] TEST 2/4: persistencia y restauración de fisher_state")

    model = torch.nn.Linear(8, 3).to(device)
    criterion = RobustFisherLoss(
        num_classes=3,
        feat_dim=8,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = make_grad_scaler(device)

    inputs = torch.randn(12, 8, device=device)
    labels = torch.arange(3, device=device).repeat_interleave(4)
    logits = model(inputs)
    features = inputs.requires_grad_(True)

    # Logits controlados para garantizar que las tres clases actualicen su EMA.
    logits_for_fisher = torch.full((labels.size(0), 3), -5.0, device=device)
    logits_for_fisher[torch.arange(labels.size(0), device=device), labels] = 5.0

    fisher_loss = criterion(features, labels, logits_for_fisher)
    total_loss = torch.nn.functional.cross_entropy(logits, labels) + 0.01 * fisher_loss
    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()

    with tempfile.TemporaryDirectory(prefix="fisher_ckpt_test_") as tmp_dir:
        path = os.path.join(tmp_dir, "checkpoint.pt")
        torch.save(
            {
                "epoch": 0,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "fisher_state": criterion.state_dict(),
            },
            path,
        )

        restored_model = torch.nn.Linear(8, 3).to(device)
        restored_fisher = RobustFisherLoss(
            num_classes=3,
            feat_dim=8,
            device=device,
        )
        restored_optimizer = torch.optim.AdamW(
            restored_model.parameters(),
            lr=1e-3,
        )
        restored_scaler = make_grad_scaler(device)

        checkpoint = torch.load(path, map_location=device)
        restored_model.load_state_dict(checkpoint["model_state"])
        restored_optimizer.load_state_dict(checkpoint["optimizer_state"])
        restored_scaler.load_state_dict(checkpoint["scaler_state"])
        restored_fisher.load_state_dict(checkpoint["fisher_state"])

        if not torch.equal(
            criterion.initialized,
            restored_fisher.initialized,
        ):
            raise RuntimeError("initialized no se restauró correctamente.")

        if not torch.allclose(
            criterion.centroids,
            restored_fisher.centroids,
        ):
            raise RuntimeError("Los centroides EMA no se restauraron correctamente.")

    logging.info("[✓] Persistencia de fisher_state aprobada")


def train_single_batch_variant(
    variant: str,
    initial_state: dict,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    vit_conf: dict,
    n_min: int,
    num_classes: int,
    device: torch.device,
    steps: int,
    learning_rate: float,
) -> Dict[str, float]:
    set_seed(args.seed)

    if variant not in {"focal", "focal_fisher"}:
        raise ValueError(
            f"Variante no reconocida: {variant!r}. "
            "Debe ser 'focal' o 'focal_fisher'."
        )

    if steps < 1:
        raise ValueError("steps debe ser mayor o igual que 1.")

    if learning_rate <= 0:
        raise ValueError("learning_rate debe ser mayor que 0.")
    
    model = build_model(
        n_min,
        vit_conf,
        num_classes,
        device,
    )
    model.load_state_dict(initial_state)

    class_weights = make_class_weights(
        labels.detach().cpu().numpy(),
        num_classes,
        device,
    )

    focal = FocalLoss(weight=class_weights, gamma=2.0)
    fisher = RobustFisherLoss(
        num_classes=num_classes,
        feat_dim=vit_conf["embed_dim"],
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )
    
    model.eval()

    with torch.no_grad():
        initial_logits, _, _ = model(
            inputs,
            return_attention=False,
        )
        initial_focal = float(
            focal(initial_logits.float(), labels).item()
        )
    
        if not torch.isfinite(initial_logits).all():
            raise FloatingPointError(
            f"{variant}: los logits iniciales contienen NaN/Inf."
        )

        if not math.isfinite(initial_focal):
            raise FloatingPointError(
                f"{variant}: la Focal Loss inicial contiene NaN/Inf."
            )
            
    model.train()
    last_metrics: Dict[str, float] = {}

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits, features, _ = model(inputs, return_attention=False,)
        focal_loss = focal(logits.float(), labels)

        if variant == "focal_fisher":
            fisher_loss = fisher(
            features.float(),
            labels,
            logits.float(),
            )
            lambda_value = 5e-2 * math.exp(
                -5.0 * (step / max(steps - 1, 1))
            )
        else:
            fisher_loss = features.new_tensor(0.0)
            lambda_value = 0.0    

        total_loss = focal_loss.float() + lambda_value * fisher_loss

        # Validaciones antes del backward
        if not torch.isfinite(inputs).all():
            raise FloatingPointError(
                f"{variant}: los inputs contienen NaN/Inf."
            )

        if not torch.isfinite(logits).all():
            raise FloatingPointError(
                f"{variant}: los logits contienen NaN/Inf en el paso {step + 1}."
            )

        if not torch.isfinite(features).all():
            raise FloatingPointError(
                f"{variant}: los embeddings contienen NaN/Inf en el paso {step + 1}."
            )
            
        if not torch.isfinite(focal_loss):
            raise FloatingPointError(
            f"{variant}: Focal Loss produjo NaN/Inf en el paso {step + 1}."
        )

        if not torch.isfinite(fisher_loss):
            raise FloatingPointError(
                f"{variant}: Fisher Loss produjo NaN/Inf en el paso {step + 1}."
            )

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"{variant}: loss NaN/Inf en el paso {step + 1}. "
                f"Focal={focal_loss.item()}, Fisher={fisher_loss.item()}"
            )

        total_loss.backward()

        grad_norm = validate_gradients(model)

        # Protección adicional ante explosión de gradientes
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
            error_if_nonfinite=True,
        )

        optimizer.step()

        predictions = torch.argmax(logits, dim=1)
        mcc = matthews_corrcoef(
            labels.detach().cpu().numpy(),
            predictions.detach().cpu().numpy(),
        )

        geometry = compute_geometry(
            features.detach().float(),
            labels,
            logits.detach().float(),
            only_correct=True,
        )

        last_metrics = {
            "focal": float(focal_loss.detach().item()),
            "fisher": float(fisher_loss.detach().item()),
            "total": float(total_loss.detach().item()),
            "mcc": float(mcc),
            "grad_norm": float(grad_norm),
            **geometry,
        }

        if step == 0 or (step + 1) % 10 == 0:
            logging.info(
                "[%s %d/%d] Focal=%.6f Fisher=%.6f Total=%.6f "
                "MCC=%.4f Grad=%.4f Intra=%.6f Inter=%.6f Ratio=%.6f",
                variant,
                step + 1,
                steps,
                last_metrics["focal"],
                last_metrics["fisher"],
                last_metrics["total"],
                last_metrics["mcc"],
                last_metrics["grad_norm"],
                last_metrics["intra"],
                last_metrics["inter"],
                last_metrics["ratio"],
            )
    
    model.eval()

    with torch.no_grad():
        final_logits, final_features, _ = model(
            inputs,
            return_attention=False,
        )
        final_focal = focal(final_logits.float(), labels)
        
        if not torch.isfinite(final_logits).all():
            raise FloatingPointError(
            f"{variant}: los logits finales contienen NaN/Inf."
        )

        if not torch.isfinite(final_features).all():
            raise FloatingPointError(
                f"{variant}: los embeddings finales contienen NaN/Inf."
            )

        if not torch.isfinite(final_focal):
            raise FloatingPointError(
                f"{variant}: la Focal Loss final produjo NaN/Inf."
            )
        
        final_predictions = torch.argmax(final_logits, dim=1)

        final_mcc = matthews_corrcoef(
            labels.detach().cpu().numpy(),
            final_predictions.detach().cpu().numpy(),
        )

        final_geometry = compute_geometry(
            final_features.float(),
            labels,
            final_logits.float(),
            only_correct=True,
        )
        
    last_metrics["last_train_fisher"] = last_metrics["fisher"]
    last_metrics["last_train_total"] = last_metrics["total"]
    last_metrics["last_train_grad_norm"] = last_metrics["grad_norm"]

    last_metrics["focal"] = float(final_focal.item())
    last_metrics["mcc"] = float(final_mcc)
    last_metrics.update(final_geometry)

    if last_metrics["focal"] >= initial_focal:
        raise RuntimeError(f"{variant}: Focal Loss no disminuyó.")

    if last_metrics["mcc"] < 0.90:
        raise RuntimeError(
            f"{variant}: MCC final insuficiente ({last_metrics['mcc']:.4f})."
        )

    logging.info(
        "[✓] %s aprobó single-batch | Focal %.6f -> %.6f | MCC=%.4f",
        variant,
        initial_focal,
        last_metrics["focal"],
        last_metrics["mcc"],
    )
    return last_metrics


def run_single_batch_test(
    dataset: IDS2018Dataset,
    vit_conf: dict,
    device: torch.device,
) -> None:
    logging.info("[*] TEST 3/4: sobreajuste de un lote balanceado")

    indices = choose_balanced_indices(
        dataset.labels,
        args.samples_per_class,
        args.seed,
    )
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=len(indices),
        shuffle=False,
        num_workers=0,
        collate_fn=safe_collate,
    )

    inputs, labels = next(iter(loader))
    if len(inputs) == 0:
        raise RuntimeError("El lote balanceado quedó vacío.")

    if torch.unique(labels).numel() < 2:
        raise RuntimeError("El lote balanceado contiene menos de dos clases.")

    inputs = inputs.to(device)
    labels = labels.to(device)

    set_seed(args.seed)
    base_model = build_model(
        args.n_min,
        vit_conf,
        len(dataset.class_to_idx),
        device,
    )
    initial_state = copy.deepcopy(base_model.state_dict())
    del base_model

    learning_rate = env.get_value("training")["learning_rate"]

    baseline = train_single_batch_variant(
        "focal",
        initial_state,
        inputs,
        labels,
        vit_conf,
        args.n_min,
        len(dataset.class_to_idx),
        device,
        args.steps,
        learning_rate,
    )
    fisher = train_single_batch_variant(
        "focal_fisher",
        initial_state,
        inputs,
        labels,
        vit_conf,
        args.n_min,
        len(dataset.class_to_idx),
        device,
        args.steps,
        learning_rate,
    )

    logging.info(
        "[*] Comparación single-batch | "
        "Focal ratio=%.6f | Fisher ratio=%.6f | "
        "Focal MCC=%.4f | Fisher MCC=%.4f",
        baseline["ratio"],
        fisher["ratio"],
        baseline["mcc"],
        fisher["mcc"],
    )


@torch.no_grad()
def evaluate_subset(
    model: ViT_OSR,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_features: List[torch.Tensor] = []

    for inputs, labels in loader:
        if len(inputs) == 0:
            continue

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda"),
        ):
            logits, features, _ = model(inputs)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        all_features.append(features.float().cpu())

    if not all_labels:
        raise RuntimeError("Validación corta sin lotes válidos.")

    labels_np = np.concatenate(all_labels)
    preds_np = np.concatenate(all_preds)
    features_cpu = torch.cat(all_features)
    labels_cpu = torch.from_numpy(labels_np)

    geometry = compute_geometry(
        features_cpu,
        labels_cpu,
        logits=None,
        only_correct=False,
    )

    return {
        "mcc": float(matthews_corrcoef(labels_np, preds_np)),
        **geometry,
    }


def train_short_variant(
    variant: str,
    initial_state: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_labels_np: np.ndarray,
    vit_conf: dict,
    dataset: IDS2018Dataset,
    device: torch.device,
) -> Dict[str, float]:
    set_seed(args.seed)

    model = build_model(
        args.n_min,
        vit_conf,
        len(dataset.class_to_idx),
        device,
    )
    model.load_state_dict(initial_state)

    class_weights = make_class_weights(
        train_labels_np,
        len(dataset.class_to_idx),
        device,
    )

    focal = FocalLoss(weight=class_weights, gamma=2.0)
    fisher = RobustFisherLoss(
        num_classes=len(dataset.class_to_idx),
        feat_dim=vit_conf["embed_dim"],
        device=device,
    )

    learning_rate = env.get_value("training")["learning_rate"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    scaler = make_grad_scaler(device)

    for epoch in range(args.short_epochs):
        model.train()
        processed = 0
        running = 0.0
        lambda_value = (
            get_lambda(epoch, args.short_epochs)
            if variant == "focal_fisher"
            else 0.0
        )

        for inputs, labels in train_loader:
            if len(inputs) == 0:
                continue

            processed += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda"),
            ):
                logits, features, _ = model(inputs)
                focal_loss = focal(logits, labels)

            if variant == "focal_fisher":
                with torch.autocast(
                    device_type=device.type,
                    enabled=False,
                ):
                    fisher_loss = fisher(
                        features.float(),
                        labels,
                        logits.float(),
                    )
            else:
                fisher_loss = features.new_tensor(0.0)

            total_loss = focal_loss.float() + lambda_value * fisher_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"{variant}: NaN/Inf en short run, época {epoch + 1}."
                )

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            validate_gradients(model)
            scaler.step(optimizer)
            scaler.update()
            running += total_loss.item()

        if processed == 0:
            raise RuntimeError(f"{variant}: época corta sin lotes válidos.")

        logging.info(
            "[%s] short epoch %d/%d | Loss=%.6f | Lambda=%.8f",
            variant,
            epoch + 1,
            args.short_epochs,
            running / processed,
            lambda_value,
        )

    return evaluate_subset(model, val_loader, device)


def run_short_ablation(
    dataset: IDS2018Dataset,
    vit_conf: dict,
    device: torch.device,
) -> None:
    logging.info("[*] TEST 4/4: miniablación corta Focal vs Focal+Fisher")

    train_indices, val_indices = choose_balanced_train_val_indices(
        dataset.labels,
        args.train_per_class,
        args.val_per_class,
        args.seed,
    )

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=min(32, len(train_subset)),
        # Orden fijo para que ambas variantes vean exactamente los mismos lotes.
        shuffle=False,
        num_workers=0,
        collate_fn=safe_collate,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=min(64, len(val_subset)),
        shuffle=False,
        num_workers=0,
        collate_fn=safe_collate,
    )

    train_labels_np = dataset.labels[np.asarray(train_indices)]

    set_seed(args.seed)
    base_model = build_model(
        args.n_min,
        vit_conf,
        len(dataset.class_to_idx),
        device,
    )
    initial_state = copy.deepcopy(base_model.state_dict())
    del base_model

    focal_metrics = train_short_variant(
        "focal",
        initial_state,
        train_loader,
        val_loader,
        train_labels_np,
        vit_conf,
        dataset,
        device,
    )
    fisher_metrics = train_short_variant(
        "focal_fisher",
        initial_state,
        train_loader,
        val_loader,
        train_labels_np,
        vit_conf,
        dataset,
        device,
    )

    logging.info(
        "[*] RESULTADO MINIABLACIÓN\n"
        "    Focal        -> MCC=%.4f Intra=%.6f Inter=%.6f Ratio=%.6f\n"
        "    Focal+Fisher -> MCC=%.4f Intra=%.6f Inter=%.6f Ratio=%.6f",
        focal_metrics["mcc"],
        focal_metrics["intra"],
        focal_metrics["inter"],
        focal_metrics["ratio"],
        fisher_metrics["mcc"],
        fisher_metrics["intra"],
        fisher_metrics["inter"],
        fisher_metrics["ratio"],
    )

    if fisher_metrics["ratio"] >= focal_metrics["ratio"]:
        logging.warning(
            "[!] Fisher no mejoró la relación intra/inter en la miniablación."
        )

    if fisher_metrics["mcc"] + 0.05 < focal_metrics["mcc"]:
        logging.warning(
            "[!] Fisher degradó el MCC de validación en más de 0.05."
        )


def main() -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("[*] Dispositivo de validación: %s", device)

    requested = args.test

    if requested in {"all", "synthetic"}:
        run_synthetic_test(device)

    if requested in {"all", "checkpoint"}:
        run_checkpoint_roundtrip_test(device)

    if requested in {"all", "single_batch", "short_run"}:
        dataset, vit_conf = load_dataset(args.n_min, args.mode)

        if requested in {"all", "single_batch"}:
            run_single_batch_test(dataset, vit_conf, device)

        if requested in {"all", "short_run"}:
            run_short_ablation(dataset, vit_conf, device)

    logging.info("[✓] Pruebas solicitadas completadas.")


if __name__ == "__main__":
    main()
