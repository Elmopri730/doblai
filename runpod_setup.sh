#!/bin/bash

echo "=========================================="
echo "🚀 Configurando DoblAI en RunPod GPU..."
echo "=========================================="

# 1. Instalar herramientas de compilación
pip install --no-cache-dir setuptools wheel cython

# 2. Instalar Ollama si no está presente
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

# 3. Instalar dependencias del sistema requeridas (ffmpeg, git)
echo "🎬 Verificando ffmpeg y herramientas del sistema..."
apt-get update -qq && apt-get install -y -qq ffmpeg git > /dev/null

# 4. Instalar librerías base de Python compatibles
echo "🐍 Instalando librerías base de IA..."
pip install --no-cache-dir "numpy<2.0.0,>=1.26.0"
pip install --no-cache-dir pandas scipy "transformers==4.42.4"
pip install --no-cache-dir torchcodec || true

# 5. Instalar Coqui TTS con parche para Python 3.12 si no está instalado
echo "🎙️ Verificando Coqui TTS..."
if ! python -c "import TTS" &> /dev/null; then
    rm -rf /tmp/TTS
    git clone --depth 1 https://github.com/coqui-ai/TTS.git /tmp/TTS
    sed -i 's/3.12/3.15/g' /tmp/TTS/setup.py
    pip install --no-cache-dir --no-deps --no-build-isolation -e /tmp/TTS
fi

# Instalar siempre las dependencias de inferencia de TTS
echo "📦 Instalando dependencias de inferencia..."
pip install --no-cache-dir coqpit pysbd anyascii unidecode num2words spacy soundfile librosa scikit-learn

# 6. Instalar el resto de dependencias de DoblAI
echo "📦 Instalando dependencias de la aplicación..."
pip install --no-cache-dir -r requirements.txt

# Asegurar que las carpetas de trabajo existan
mkdir -p input output work

# 7. Probar que los módulos cargan correctamente
echo "🔍 Verificando instalación..."
python -c "import torch; import torchaudio; import TTS; print('✅ Todos los módulos cargan correctamente!')"

# 8. Instalar y arrancar túnel seguro Cloudflare para acceso público inmediato
echo "🌐 Configurando enlace web público..."
if ! command -v cloudflared &> /dev/null; then
    curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared
    chmod +x /usr/local/bin/cloudflared
fi

# Lanzar uvicorn en segundo plano
export OLLAMA_URL="http://localhost:11434"
export COQUI_TOS_AGREED="1"
uvicorn webapp.app:app --host 0.0.0.0 --port 8080 > /tmp/uvicorn.log 2>&1 &
sleep 2

# Lanzar túnel Cloudflare
rm -f /tmp/tunnel.log
cloudflared tunnel --url http://127.0.0.1:8080 > /tmp/tunnel.log 2>&1 &
sleep 5

TUNNEL_URL=$(grep -o 'https://[-0-9a-z]*\.trycloudflare\.com' /tmp/tunnel.log | head -n 1)

echo "=========================================================="
echo "🎉 ¡DoblAI ESTÁ FUNCIONANDO EN LA RTX 4090!"
echo "👉 ENTRA DESDE TU NAVEGADOR A ESTE ENLACE:"
echo "   $TUNNEL_URL"
echo "=========================================================="

# Mantener la terminal activa mostrando logs
tail -f /tmp/uvicorn.log
