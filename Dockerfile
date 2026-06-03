# Capa Base: Ubuntu 22.04 + PyTorch + Integración nativa con CUDA
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Capa OS: Evitamos que la instalación
ENV DEBIAN_FRONTEND=noninteractive

# Capa de Herramientas de Sistema: AWS (ingesta) y wireshark-common (que contiene editcap)
RUN apt-get update && apt-get install -y \
    awscli \
    wireshark-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Establecemos el directorio de trabajo del orquestador
WORKDIR /app

# Capa Python: Dependencias analíticas y de Deep Learning exigidas en el SRS
# Nota: scikit-learn se usará para el PCA latente en el módulo OSR (FR8)
# Nota: fvcore se usará para el cálculo de FLOPs (FR13)
RUN pip install --no-cache-dir \
    dpkt \
    pandas \
    numpy \
    h5py \
    einops \
    scikit-learn \
    mlflow \
    dvc \
    fvcore

# Por defecto, al iniciar el contenedor abriremos una terminal bash
CMD ["/bin/bash"]