# OSR-ViT: detección de amenazas no conocidas en tráfico de red

Este repositorio contiene una investigación académica orientada a detección de intrusiones en redes mediante aprendizaje profundo y reconocimiento de conjunto abierto.

La propuesta transforma los primeros paquetes de una sesión de red en una representación multicanal RGB-E y utiliza un Vision Transformer para obtener una representación latente del tráfico. A partir de esos embeddings se construyen perfiles por clase y se aplica una etapa de rechazo basada en distancia de Mahalanobis para identificar muestras fuera de distribución.

## Objetivo

El proyecto busca reducir la dependencia de los IDS basados únicamente en clasificación cerrada, donde toda muestra debe asignarse a una clase conocida. El enfoque open-set permite rechazar tráfico que no se ajusta a los patrones aprendidos y tratarlo como potencialmente desconocido.

## Flujo general

```text
Sesión de red
    |
    v
Primeros paquetes
    |
    v
Representación RGB-E
    |
    v
Vision Transformer
    |
    v
Embedding latente
    |
    v
Perfiles por clase
    |
    v
Distancia de Mahalanobis
    |
    +--> Clase conocida
    |
    +--> Muestra fuera de distribución
```

## Componentes principales

- Ingesta y preparación de tráfico de red.
- Construcción de representaciones RGB-E a partir de sesiones.
- Entrenamiento y evaluación del Vision Transformer.
- Extracción de embeddings desde el token CLS.
- Evaluación closed-set.
- Evaluación open-set.
- Ajuste de perfiles de Mahalanobis por clase.
- Calibración de umbrales de rechazo.
- Registro de métricas y artefactos experimentales.

## Dataset

La evaluación se realizó sobre CSE-CIC-IDS2018.

Para los experimentos open-set se trabajó con clases conocidas y clases reservadas como tráfico fuera de distribución. Esta separación permite medir no solo la capacidad de clasificación, sino también la capacidad de rechazo ante amenazas no observadas durante el entrenamiento.

## Resultados principales

En evaluación closed-set se obtuvieron los siguientes resultados:

- Accuracy: 0.9995
- Macro F1: 0.9989
- MCC: 0.9986

En evaluación open-set:

- AUROC: 0.8280
- AUPR: 0.7474
- Recall OOD: 0.6836
- F1 OOD: 0.7511
- False Rejection Rate sobre clases conocidas: 0.1933

Los resultados closed-set y open-set deben interpretarse por separado. Una alta exactitud sobre clases conocidas no implica una capacidad equivalente para detectar amenazas no observadas.

## Estructura relevante

```text
configs/
src/
├── data_ingestion/
├── evaluation/
│   ├── evaluate_closed_set.py
│   └── evaluate_zero_day.py
├── models/
│   └── vit_ablation.py
├── osr_module/
│   └── mahalanobis.py
├── preprocessing/
├── utils/
└── notebooks/
```

## Dependencias

El proyecto utiliza, entre otras:

- PyTorch
- NumPy
- pandas
- scikit-learn
- PyYAML
- Matplotlib
- Seaborn
- Pillow
- einops
- h5py
- dpkt
- MLflow
- DVC
- fvcore

Las dependencias de Python están declaradas en `requirements.txt`. El repositorio también incluye un `Dockerfile` para reproducir el entorno experimental.

## Limitaciones

El enfoque utiliza una ventana fija de paquetes por sesión, por lo que conexiones muy cortas o comportamiento malicioso tardío pueden no quedar representados por completo.

Los resultados dependen de la partición experimental definida sobre CSE-CIC-IDS2018 y no deben interpretarse como desempeño general frente a cualquier amenaza zero-day.

El punto de operación open-set presenta un compromiso entre sensibilidad ante tráfico desconocido y rechazo incorrecto de tráfico conocido.

## Alcance

Este proyecto tiene fines académicos y de investigación. No constituye por sí solo un IDS listo para producción ni sustituye controles operativos como SIEM, EDR, NDR o monitoreo de red continuo.

## Autoría

Proyecto desarrollado en el contexto del curso de Inteligencia Artificial II de la Universidad Nacional de Ingeniería.
