import os
import json
import math
import torch
import random
import hashlib
import logging
import argparse
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.vit_ablation import IDS2018Dataset, ViT_OSR, KNOWN_CLASS_TO_IDX, safe_collate, seed_worker
from src.utils.config_manager import setup_environment

# ==============================================================================
# 0. INYECCIÓN DE ENTORNO Y ARGUMENTOS
# ==============================================================================
parser = argparse.ArgumentParser(description="Evaluador Closed-Set OSR-ViT")
parser.add_argument("--mode", type=str, choices=["pilot", "prod"], required=True)
parser.add_argument("--n_min", type=int, required=True, help="Tamaño de ventana N_min a evaluar")
args, _ = parser.parse_known_args()

env = setup_environment(script_name="closed_set_evaluation", args=args)

TEST_DIR = env.get_path("paths", "output", "test_known", ensure_exists=True)
SCALER_JSON = env.get_path("paths", "configs", "scaler_bounds", is_file=True)
CKPT_DIR = env.get_path("paths", "artifacts", "checkpoints", ensure_exists=True)
RESULTS_DIR = env.get_path("paths", "artifacts", "results", ensure_exists=True)

# ==============================================================================
# 1. UTILIDADES DE TRAZABILIDAD Y MÉTRICAS
# ==============================================================================
def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _update_confusion(confusion, labels, predictions, num_classes):
    encoded = labels.detach().to(torch.int64) * num_classes + predictions.detach().to(torch.int64)
    confusion += torch.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes).cpu().numpy()


def _safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float64), where=denominator != 0)


def _calculate_metrics(confusion, class_to_idx):
    confusion = np.asarray(confusion, dtype=np.int64)
    total = int(confusion.sum())
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0).astype(np.float64) - tp
    fn = confusion.sum(axis=1).astype(np.float64) - tp
    tn = float(total) - (tp + fp + fn)
    support = confusion.sum(axis=1).astype(np.int64)
    predicted = confusion.sum(axis=0).astype(np.int64)

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    fnr = _safe_divide(fn, tp + fn)
    fpr = _safe_divide(fp, fp + tn)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    accuracy = float(tp.sum() / total) if total else 0.0

    correct = float(tp.sum())
    numerator = correct * total - float(np.dot(support, predicted))
    denominator = math.sqrt(max(0.0, (total ** 2 - float(np.dot(predicted, predicted))) * (total ** 2 - float(np.dot(support, support)))))
    mcc = numerator / denominator if denominator > 0.0 else 0.0

    names_by_idx = {idx: name for name, idx in class_to_idx.items()}
    per_class = {}
    for idx in range(len(names_by_idx)):
        name = names_by_idx[idx]
        per_class[name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "fnr": float(fnr[idx]),
            "fpr": float(fpr[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
            "predicted": int(predicted[idx]),
        }

    weights = support.astype(np.float64)
    weighted_f1 = float(np.average(f1, weights=weights)) if weights.sum() else 0.0
    return {
        "mcc": float(mcc),
        "accuracy": accuracy,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "total_samples": total,
    }


def _build_text_report(results):
    metrics = results["metrics"]
    lines = [
        "=" * 78,
        "EVALUACIÓN CLOSED-SET OSR-ViT",
        "=" * 78,
        f"Fecha UTC               : {results['created_at_utc']}",
        f"Split                    : {results['source_split']}",
        f"N_min                    : {results['n_min']}",
        f"Forma de entrada         : {results['input_shape']}",
        f"Checkpoint               : {results['checkpoint']['path']}",
        f"SHA-256 checkpoint       : {results['checkpoint']['sha256']}",
        f"Mejor época              : {results['checkpoint']['best_epoch']}",
        f"MCC de validación        : {results['checkpoint']['best_val_mcc']:.6f}",
        f"Muestras evaluadas       : {metrics['total_samples']:,}",
        "",
        "MÉTRICAS GLOBALES",
        "-" * 78,
        f"MCC                      : {metrics['mcc']:.6f}",
        f"Accuracy                 : {metrics['accuracy']:.6f}",
        f"Macro Precision          : {metrics['macro_precision']:.6f}",
        f"Macro Recall             : {metrics['macro_recall']:.6f}",
        f"Macro F1                 : {metrics['macro_f1']:.6f}",
        f"Weighted F1              : {metrics['weighted_f1']:.6f}",
        "",
        "MÉTRICAS POR CLASE",
        "-" * 78,
    ]

    for name, values in metrics["per_class"].items():
        lines.append(
            f"{name:<10} | Precision={values['precision']:.6f} | Recall={values['recall']:.6f} | "
            f"FNR={values['fnr']:.6f} | FPR={values['fpr']:.6f} | F1={values['f1']:.6f} | "
            f"Support={values['support']:,}"
        )

    lines.extend(["", "MATRIZ DE CONFUSIÓN", "-" * 78])
    for row in metrics["confusion_matrix"]:
        lines.append(" ".join(f"{value:>10,}" for value in row))
    lines.append("=" * 78)
    return "\n".join(lines) + "\n"


def _validate_checkpoint(checkpoint, checkpoint_path, n_min, scaler_sha256):
    required = {
        "model_state", "fisher_state", "class_to_idx", "tensor_shape", "model_config",
        "training_config", "experiment_hash", "hashes", "loss_variant", "n_min",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise RuntimeError(f"Checkpoint incompleto. Campos ausentes: {missing}")

    if checkpoint.get("checkpoint_kind") != "best_validation_mcc":
        raise RuntimeError("El archivo encontrado no es el mejor checkpoint seleccionado por MCC de validación")
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

    hashes = checkpoint["hashes"]
    expected_hashes = {
        "scaler_sha256": scaler_sha256,
        "global_config_sha256": _sha256_file("configs/global_config.yaml"),
        "dataset_schedule_sha256": _sha256_file("configs/dataset_schedule.yaml"),
        "script_sha256": _sha256_file("src/models/vit_ablation.py"),
    }
    mismatches = [key for key, value in expected_hashes.items() if hashes.get(key) != value]
    if mismatches:
        raise RuntimeError(f"Checkpoint incompatible con los artefactos actuales. Hashes distintos: {mismatches}")

    logging.info("[✓] Checkpoint validado: %s", checkpoint_path)


# ==============================================================================
# 2. EVALUACIÓN CLOSED-SET
# ==============================================================================
def evaluate_model(n_min, mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(env.get_value("project", "seed"))
    loader_config = env.get_value("data_loader")
    training_config = env.get_value("training")
    tensor_width = int(env.get_value("preprocessing", "tensor_width"))
    _set_global_seed(seed)

    if tensor_width != 144:
        raise RuntimeError(f"tensor_width incompatible: {tensor_width}")

    checkpoint_path = os.path.join(CKPT_DIR, f"vit_nmin_{n_min}_checkpoint.pt")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No se encontró el mejor checkpoint: {checkpoint_path}")

    checkpoint_sha256 = _sha256_file(checkpoint_path)
    scaler_sha256 = _sha256_file(SCALER_JSON)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    _validate_checkpoint(checkpoint, checkpoint_path, n_min, scaler_sha256)

    dataset = IDS2018Dataset(TEST_DIR, SCALER_JSON, n_min, max_bytes=tensor_width, mode=mode, split_name="test_known")
    if dataset.class_to_idx != KNOWN_CLASS_TO_IDX:
        raise RuntimeError(f"Mapa de clases del Dataset incompatible: {dataset.class_to_idx}")

    class_counts = {name: int(dataset.class_counts.get(idx, 0)) for name, idx in KNOWN_CLASS_TO_IDX.items()}
    missing_classes = [name for name, count in class_counts.items() if count == 0]
    if missing_classes:
        raise RuntimeError(f"Clases ausentes en test_known para N_min={n_min}: {missing_classes}")

    workers = 0 if mode == "pilot" else int(loader_config["num_workers"])
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": int(training_config["batch_size"]),
        "shuffle": False,
        "drop_last": False,
        "num_workers": workers,
        "pin_memory": bool(loader_config.get("pin_memory", True) and device.type == "cuda"),
        "collate_fn": safe_collate,
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(seed + 2),
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = int(loader_config.get("prefetch_factor", 2))
        loader_kwargs["persistent_workers"] = bool(loader_config.get("persistent_workers", True))
    dataloader = DataLoader(**loader_kwargs)

    model_config = checkpoint["model_config"]
    model = ViT_OSR(
        n_min=n_min,
        max_bytes=tensor_width,
        patch_size=(1, int(model_config["patch_size"])),
        embed_dim=int(model_config["embed_dim"]),
        depth=int(model_config["depth"]),
        num_heads=int(model_config["num_heads"]),
        num_classes=len(KNOWN_CLASS_TO_IDX),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    amp_enabled, amp_dtype = _resolve_amp(device, checkpoint["training_config"])
    logging.info("[*] Evaluador iniciado en %s | AMP=%s | dtype=%s", device, amp_enabled, amp_dtype)
    logging.info("[*] test_known=%s muestras | distribución=%s", f"{len(dataset):,}", class_counts)

    num_classes = len(KNOWN_CLASS_TO_IDX)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.inference_mode():
        for inputs, labels in tqdm(dataloader, total=len(dataloader), desc="Evaluando test_known"):
            if len(inputs) == 0:
                continue
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits, _, _ = model(inputs, return_attention=False)
            predictions = logits.argmax(dim=1)
            _update_confusion(confusion, labels, predictions, num_classes)

    metrics = _calculate_metrics(confusion, KNOWN_CLASS_TO_IDX)
    if metrics["total_samples"] != len(dataset):
        raise RuntimeError(f"Conteo evaluado inconsistente: inferencia={metrics['total_samples']}, índice={len(dataset)}")
    if metrics["total_samples"] != sum(class_counts.values()):
        raise RuntimeError("El soporte de la evaluación no coincide con class_counts")

    git_commit, git_dirty = _git_metadata()
    created_at_utc = datetime.now(timezone.utc).isoformat()
    results = {
        "schema_version": 1,
        "created_at_utc": created_at_utc,
        "mode": mode,
        "source_split": "test_known",
        "n_min": int(n_min),
        "input_shape": [int(n_min), tensor_width, 3],
        "class_to_idx": KNOWN_CLASS_TO_IDX,
        "dataset_class_counts": class_counts,
        "dataset_manifest_sha256": dataset.dataset_manifest_sha256,
        "scaler_sha256": scaler_sha256,
        "scaler_manifest_sha256": dataset.scaler_manifest_sha256,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "model_config": model_config,
        "checkpoint": {
            "path": checkpoint_path,
            "sha256": checkpoint_sha256,
            "experiment_hash": checkpoint["experiment_hash"],
            "checkpoint_kind": checkpoint["checkpoint_kind"],
            "best_epoch": int(checkpoint.get("best_epoch", checkpoint["epoch"])) + 1,
            "best_val_mcc": float(checkpoint.get("best_val_mcc", 0.0)),
            "best_val_loss": float(checkpoint.get("best_val_loss", 0.0)),
        },
        "metrics": metrics,
    }

    json_path = os.path.join(RESULTS_DIR, f"closed_set_nmin_{n_min}.json")
    text_path = os.path.join(RESULTS_DIR, f"closed_set_nmin_{n_min}.txt")
    matrix_path = os.path.join(RESULTS_DIR, f"confusion_matrix_nmin_{n_min}.png")
    _write_json_atomic(json_path, results)
    _write_text_atomic(text_path, _build_text_report(results))

    class_names = list(KNOWN_CLASS_TO_IDX)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("Etiqueta real")
    plt.xlabel("Predicción del modelo")
    plt.title(f"Matriz de confusión ViT Closed-Set (N_min={n_min})\nMCC: {metrics['mcc']:.4f}")
    plt.tight_layout()
    plt.savefig(matrix_path, dpi=300)
    plt.close()

    logging.info("[✓] Evaluación closed-set completada")
    logging.info("[*] MCC=%.6f | Accuracy=%.6f | Macro F1=%.6f | Weighted F1=%.6f", metrics["mcc"], metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"])
    for name, values in metrics["per_class"].items():
        logging.info(" -> %s: Precision=%.4f Recall=%.4f FNR=%.4f FPR=%.4f F1=%.4f Support=%s", name, values["precision"], values["recall"], values["fnr"], values["fpr"], values["f1"], f"{values['support']:,}")
    logging.info("[*] JSON: %s", json_path)
    logging.info("[*] TXT : %s", text_path)
    logging.info("[*] PNG : %s", matrix_path)
    return results


if __name__ == "__main__":
    evaluate_model(args.n_min, env.mode)
