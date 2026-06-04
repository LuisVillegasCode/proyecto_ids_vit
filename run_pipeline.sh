#!/bin/bash

set -euo pipefail

# ========================================================
# Instalación de dependencias (Sin sudo, ejecutado como root en Docker)
# ========================================================
apt-get update
apt-get install -y p7zip-full

# ========================================================
# Preparación de directorios
# ========================================================
mkdir -p data/raw/chunks
mkdir -p data/downloads

# ========================================================
# Configuración
# ========================================================
declare -A archivos=(
    ["Thursday-22-02-2018"]="thu-22"
    ["Wednesday-28-02-2018"]="wed-28"
    ["Friday-16-02-2018"]="fri-16"
    ["Wednesday-21-02-2018"]="wed-21"
    ["Friday-02-03-2018"]="fri-02"
)

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ========================================================
# Procesamiento por día
# ========================================================
for dia in "${!archivos[@]}"; do
    prefijo="${archivos[$dia]}"

    echo "========================================================"
    echo "[*] INICIANDO LÍNEA DE ENSAMBLAJE PARA: $dia ($prefijo)"
    echo "========================================================"

    # A. Descarga desde S3 (Usa el awscli de tu Docker)
    echo "[*] Descargando ZIP..."
    aws s3 cp "s3://cse-cic-ids2018/Original Network Traffic and Log data/${dia}/pcap.zip" "data/downloads/${dia}.zip" --no-sign-request

    # B. Extracción
    echo "[*] Descomprimiendo..."
    7z x "data/downloads/${dia}.zip" "-odata/downloads/${dia}/" -y

    # C. Liberar espacio eliminando ZIP
    rm -f "data/downloads/${dia}.zip"

    # D. Renombrado y movimiento seguro
    echo "[*] Estandarizando nombres y extensiones..."
    find "data/downloads/${dia}" -type f -print0 | while IFS= read -r -d '' file; do
        filename="$(basename "$file")"
        clean_name="${filename%.pcap}"
        mv -- "$file" "data/raw/chunks/${prefijo}_${clean_name}.pcap"
    done
    rm -rf "data/downloads/${dia}"

    # E. Fragmentar archivos grandes (Usa el editcap de tu Docker)
    echo "[*] Buscando archivos > 1GB..."
    find data/raw/chunks -type f -name "${prefijo}_*.pcap" -size +1G -print0 | while IFS= read -r -d '' bigfile; do
        echo "    -> Fragmentando: $bigfile"
        editcap -c 500000 "$bigfile" "${bigfile%.pcap}_chunk.pcap"
        rm -f -- "$bigfile"
    done

    # F. Ingesta a Tensores Bidireccionales
    # (Ya estamos en Docker, ejecutamos directamente)
    echo "[*] Ejecutando pipeline Python..."
    python src/data_ingestion/ingestion_pipeline.py --mode prod

    # G. Limpieza final de PCAPs del día
    echo "[*] Eliminando PCAPs procesados..."
    find data/raw/chunks -maxdepth 1 -type f -name "${prefijo}_*.pcap" -delete

    echo "[✓] DÍA $dia COMPLETADO CON ÉXITO."
    echo
done

echo "========================================================"
echo "[🚀] TODA LA INGESTA HA FINALIZADO."
echo "========================================================"