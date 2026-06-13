import os
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import h5py


SOURCE_DIR = Path("data/processed/train_val")
TARGET_DIR = Path("data/processed/pilot_train_val")
OUTPUT_FILE = TARGET_DIR / "fisher_validation_pilot.hdf5"

TARGET_CLASSES = [
    "Benign",
    "DoS",
    "DDoS",
]

SAMPLES_PER_CLASS = 64
RANDOM_SEED = 42
TENSOR_DATASET_NAME = "rgb_e_tensor"


def normalize_label(value) -> str:
    """Normaliza etiquetas HDF5 almacenadas como str o bytes."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()

    return str(value).strip()


def clean_target_directory(overwrite: bool) -> None:
    """
    Evita mezclar el piloto nuevo con HDF5 o índices cacheados anteriores.
    """
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    existing_files = list(TARGET_DIR.glob("*.hdf5"))
    existing_cache = list(TARGET_DIR.glob("dataset_index_*.pt"))

    if (existing_files or existing_cache) and not overwrite:
        raise RuntimeError(
            f"El directorio {TARGET_DIR} contiene datos piloto anteriores. "
            "Ejecuta nuevamente usando --overwrite."
        )

    if overwrite:
        for path in existing_files + existing_cache:
            path.unlink()
            print(f"[-] Eliminado: {path}")


def select_balanced_samples():
    """
    Selecciona exactamente SAMPLES_PER_CLASS grupos por clase sin cargar
    los tensores completos en memoria.
    """
    rng = random.Random(RANDOM_SEED)

    source_files = sorted(SOURCE_DIR.glob("*.hdf5"))
    rng.shuffle(source_files)

    if not source_files:
        raise RuntimeError(
            f"No se encontraron archivos .hdf5 en {SOURCE_DIR}"
        )

    selected = {
        label: []
        for label in TARGET_CLASSES
    }

    print(
        f"[*] Buscando {SAMPLES_PER_CLASS} muestras por clase "
        f"en {len(source_files)} archivos..."
    )

    for file_path in source_files:
        if all(
            len(selected[label]) >= SAMPLES_PER_CLASS
            for label in TARGET_CLASSES
        ):
            break

        try:
            with h5py.File(file_path, "r") as hf:
                flow_ids = list(hf.keys())
                rng.shuffle(flow_ids)

                for flow_id in flow_ids:
                    node = hf[flow_id]

                    if not isinstance(node, h5py.Group):
                        continue

                    label = normalize_label(
                        node.attrs.get("label", "Benign")
                    )

                    if label not in selected:
                        continue

                    if len(selected[label]) >= SAMPLES_PER_CLASS:
                        continue

                    if TENSOR_DATASET_NAME not in node:
                        continue

                    selected[label].append(
                        (file_path, flow_id)
                    )

        except (OSError, KeyError, ValueError) as error:
            print(
                f"[!] No se pudo inspeccionar {file_path.name}: {error}"
            )

    missing = {
        label: SAMPLES_PER_CLASS - len(samples)
        for label, samples in selected.items()
        if len(samples) < SAMPLES_PER_CLASS
    }

    if missing:
        details = ", ".join(
            f"{label}: faltan {amount}"
            for label, amount in missing.items()
        )
        raise RuntimeError(
            f"No fue posible completar el piloto equilibrado. {details}"
        )

    return selected


def write_balanced_hdf5(selected) -> None:
    """
    Copia únicamente los grupos seleccionados y conserva su estructura,
    datasets, compresión y atributos originales.
    """
    temporary_file = Path(str(OUTPUT_FILE) + ".tmp")

    # Intercalar las clases evita que un recorte posterior tome primero
    # todas las muestras de una sola clase.
    ordered_samples = []

    for sample_index in range(SAMPLES_PER_CLASS):
        for label in TARGET_CLASSES:
            source_path, flow_id = selected[label][sample_index]
            ordered_samples.append(
                (label, source_path, flow_id)
            )

    # Agrupar por archivo evita abrir el mismo HDF5 muchas veces.
    samples_by_file = defaultdict(list)

    for output_index, (label, source_path, flow_id) in enumerate(
        ordered_samples
    ):
        samples_by_file[source_path].append(
            (output_index, label, flow_id)
        )

    try:
        with h5py.File(temporary_file, "w") as target_hf:
            target_hf.attrs["purpose"] = "Fisher validation pilot"
            target_hf.attrs["samples_per_class"] = SAMPLES_PER_CLASS
            target_hf.attrs["random_seed"] = RANDOM_SEED

            for source_path, samples in samples_by_file.items():
                with h5py.File(source_path, "r") as source_hf:
                    for output_index, label, flow_id in samples:
                        target_id = f"sample_{output_index:06d}"

                        source_hf.copy(
                            source_hf[flow_id],
                            target_hf,
                            name=target_id,
                        )

                        # Trazabilidad adicional.
                        target_hf[target_id].attrs["source_file"] = (
                            source_path.name
                        )
                        target_hf[target_id].attrs["source_flow_id"] = (
                            str(flow_id)
                        )

        os.replace(temporary_file, OUTPUT_FILE)

    except Exception:
        if temporary_file.exists():
            temporary_file.unlink()
        raise


def verify_output() -> None:
    """Comprueba físicamente la distribución del HDF5 generado."""
    counts = Counter()

    with h5py.File(OUTPUT_FILE, "r") as hf:
        for flow_id in hf.keys():
            label = normalize_label(
                hf[flow_id].attrs.get("label", "Benign")
            )
            counts[label] += 1

            if TENSOR_DATASET_NAME not in hf[flow_id]:
                raise RuntimeError(
                    f"La muestra {flow_id} no contiene "
                    f"'{TENSOR_DATASET_NAME}'."
                )

    expected_total = len(TARGET_CLASSES) * SAMPLES_PER_CLASS

    if sum(counts.values()) != expected_total:
        raise RuntimeError(
            "El número final de muestras no coincide con lo esperado."
        )

    for label in TARGET_CLASSES:
        if counts[label] != SAMPLES_PER_CLASS:
            raise RuntimeError(
                f"Distribución incorrecta para {label}: "
                f"{counts[label]} muestras."
            )

    print("\n[✓] Dataset piloto Fisher construido correctamente")
    print(f"[✓] Archivo: {OUTPUT_FILE}")
    print(f"[✓] Total: {expected_total} muestras")

    for label in TARGET_CLASSES:
        print(f"    {label:<20}: {counts[label]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construye un piloto equilibrado para validar Fisher Loss"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Elimina HDF5 e índices piloto anteriores.",
    )
    args = parser.parse_args()

    if not SOURCE_DIR.is_dir():
        raise RuntimeError(
            f"El directorio fuente no existe: {SOURCE_DIR}"
        )

    clean_target_directory(args.overwrite)
    selected = select_balanced_samples()
    write_balanced_hdf5(selected)
    verify_output()


if __name__ == "__main__":
    main()