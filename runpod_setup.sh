#!/bin/bash
set -e

echo "=========================================="
echo "🚀 Configurando DoblAI en RunPod GPU..."
echo "=========================================="

# 1. Instalar Ollama si no está presente
if ! command -v ollama &> /dev/null; then
    echo "📦 Instalando Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Arrancar servicio Ollama en segundo plano
echo "🤖 Iniciando servidor de Ollama..."
ollama serve > /dev/null 2>&1 &
sleep 3

# Descargar modelo de traducción
echo "📥 Descargando modelo de traducción qwen2.5:3b..."
ollama pull qwen2.5:3b

# 2. Instalar dependencias del sistema requeridas (ffmpeg)
echo "🎬 Verificando ffmpeg..."
apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null

# 3. Instalar librerías de Python requeridas para Python 3.12 en RunPod
echo "🐍 Instalando librerías de IA..."
pip install --no-cache-dir setuptools wheel
pip install --no-cache-dir "numpy<2.0.0,>=1.26.0"
pip install --no-cache-dir pandas scipy "transformers==4.42.4"
pip install --no-cache-dir torchcodec || true
pip install --no-cache-dir TTS
pip install --no-cache-dir -r requirements.txt

# Asegurar que las carpetas de trabajo existan
mkdir -p input output work

# 4. Lanzar la webapp
echo "=========================================="
echo "🎉 ¡DoblAI listo! Iniciando servidor web..."
echo "🌐 Accede al puerto 8080 desde tu navegador"
echo "=========================================="
export OLLAMA_URL="http://localhost:11434"
export COQUI_TOS_AGREED="1"
uvicorn webapp.app:app --host 0.0.0.0 --port 8080
