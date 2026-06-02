#!/bin/bash

# ==============================================================================
# Script de Fragmentación de Tráfico
# Divide archivos .pcap monolíticos masivos en fragmentos estáticos precalculados
# para permitir una ingesta concurrente (multiprocessing) segura.
# ==============================================================================

# Definición de rutas según la arquitectura del SRS
INPUT_DIR="./data/raw"
OUTPUT_DIR="./data/raw/chunks"

# Volumen estático definido para evadir RAM Overflow
PACKET_COUNT=500000 

echo "======================================================="
echo " INICIANDO FRAGMENTACIÓN FÍSICA DE PCAPS"
echo "======================================================="

# Crear el directorio de salida si no existe
mkdir -p "$OUTPUT_DIR"

# Verificar si hay archivos .pcap en la carpeta raw
shopt -s nullglob
pcap_files=("$INPUT_DIR"/*.pcap)

if [ ${#pcap_files[@]} -eq 0 ]; then
    echo "ALERTA: No se encontraron archivos .pcap en $INPUT_DIR."
    echo "Asegúrate de ejecutar la ingesta desde AWS S3 primero."
    exit 1
fi

# Procesamiento iterativo
for pcap_file in "${pcap_files[@]}"; do
    filename=$(basename -- "$pcap_file")
    name="${filename%.*}"
    
    echo "[*] Procesando archivo masivo: $filename"
    echo "    -> Generando particiones de $PACKET_COUNT paquetes..."
    
    # Ejecución de la herramienta C nativa (editcap)
    editcap -c $PACKET_COUNT "$pcap_file" "$OUTPUT_DIR/${name}_chunk.pcap"
    
    if [ $? -eq 0 ]; then
        echo "[✓] Fragmentación exitosa para: $filename"
    else
        echo "[X] Error crítico al fragmentar: $filename"
        exit 1
    fi
done

echo "======================================================="
echo " PROCESO COMPLETADO."
echo " Los fragmentos están listos en: $OUTPUT_DIR"
echo "======================================================="