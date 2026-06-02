# Capa Base: Ubuntu 22.04 + PyTorch + Integración nativa con CUDA
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Capa OS: Evitamos que la instalación de tshark congele el contenedor pidiendo confirmación
ENV DEBIAN_FRONTEND=noninteractive

# Capa de Herramientas de Sistema: AWS (ingesta) y tshark
RUN apt-get update && apt-get install -y \
    awscli \
    tshark \
    git \
    && rm -rf /var/lib/apt/lists/*

# Establecemos el directorio de trabajo del orquestador
WORKDIR /app

# Capa Python: Dependencias analíticas y de Deep Learning exigidas en el SRS
# Nota: Instalamos scikit-learn para usar IPCA en el conjunto piloto
RUN pip install --no-cache-dir \
    dpkt \
    pandas \
    numpy \
    h5py \
    einops \
    scikit-learn \
    mlflow \
    dvc

# Por defecto, al iniciar el contenedor abriremos una terminal bash
CMD ["/bin/bash"]