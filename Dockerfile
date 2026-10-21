# Dockerfile para la aplicación principal (webapp + pipeline de IA)

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Evitar que Python escriba archivos .pyc y forzar stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Instalar Python 3.10, PIP y dependencias del sistema requeridas para FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Crear alias para python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements primero para aprovechar la caché de Docker
COPY requirements.txt .

# ─── Estrategia de instalación para resolver conflictos de numpy ───
# El problema: TTS 0.22.0 pide numpy==1.22.0, pero audio-separator y scipy
# modernos necesitan numpy>=1.25 para numpy._typing. La solución es usar
# numpy 1.26.4 (la última versión 1.x) que tiene _typing y es compatible
# con ambas librerías.

# 1. Instalar numpy 1.26.4 primero (compatible con scipy moderno Y con TTS en la práctica)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy==1.26.4"

# 2. Instalar TTS (ignorará su requisito estricto de numpy==1.22.0 porque ya está instalado)
RUN pip install --no-cache-dir --no-deps TTS && \
    pip install --no-cache-dir $(pip show TTS 2>/dev/null | grep Requires | sed 's/Requires: //' | tr ',' '\n' | grep -v numpy | tr '\n' ' ')

# 3. Instalar audio-separator y scipy compatibles
RUN pip install --no-cache-dir "scipy>=1.11.0,<1.14.0" "audio-separator"

# 4. Instalar torch y onnxruntime y transformers compatible con TTS 0.22.0
RUN pip install --no-cache-dir torch onnxruntime torchcodec "transformers==4.42.4"

# 5. Instalar el resto de requirements.txt (sin forzar reinstalación)
RUN pip install --no-cache-dir -r requirements.txt

# 6. PASO FINAL CRÍTICO: Forzar numpy 1.26.4 y pandas compatible por si algo lo ha pisado
RUN pip install --no-cache-dir --force-reinstall "numpy==1.26.4" "pandas==1.5.3"

# Copiar todo el código de la aplicación
COPY . .

# Exponer el puerto de la webapp
EXPOSE 8080

# Lanzar el servidor Uvicorn
CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8080"]
