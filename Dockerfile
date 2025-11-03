# Dockerfile para proyecto Kedro MTG
FROM python:3.11-slim

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos de dependencias
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar el proyecto completo
COPY . .

# Instalar el proyecto en modo desarrollo
RUN pip install -e .

# Crear directorios necesarios para datos
RUN mkdir -p data/01_raw data/02_intermediate data/03_primary \
    data/04_feature data/05_model_input data/06_models \
    data/07_model_output data/08_reporting

# Exponer puerto para kedro-viz (opcional)
EXPOSE 4141

# Comando por defecto: ejecutar el pipeline completo
CMD ["kedro", "run", "--pipeline=__default__"]

