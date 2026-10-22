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
pip install --no-cache-dir torchcodec onnxruntime-gpu || true

# 5. Instalar Coqui TTS compatible con Python 3.12 (paquete coqui-tts)
echo "🎙️ Verificando Coqui TTS..."
if ! python -c "from TTS.api import TTS" &> /dev/null; then
    pip uninstall -y TTS trainer coqpit 2>/dev/null || true
    pip install --no-cache-dir coqui-tts
fi

# 6. Instalar el resto de dependencias de DoblAI
echo "📦 Instalando dependencias de la aplicación..."
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir "av<19.0.0"
pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
pip install --no-cache-dir onnxruntime-gpu

# Asegurar que las carpetas de trabajo existan
mkdir -p input output work

# 7. Probar que los módulos cargan correctamente
echo "🔍 Verificando instalación y aceleración GPU..."
python -c "import torch; from TTS.api import TTS; print('Torch CUDA:', torch.cuda.is_available()); print('✅ Todos los módulos cargan correctamente!')"

# 8. Arrancar túnel Cloudflare y servidor web en primer plano
echo "🌐 Configurando enlace web público..."
pkill -9 -f "uvicorn webapp.app:app" 2>/dev/null || true
pkill -9 -f "cloudflared tunnel" 2>/dev/null || true
sleep 1

if ! command -v cloudflared &> /dev/null; then
    curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared
    chmod +x /usr/local/bin/cloudflared
fi

# Lanzar túnel Cloudflare en segundo plano con nohup
rm -f /tmp/tunnel.log
nohup cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8080 > /tmp/tunnel.log 2>&1 &

echo "⏳ Conectando túnel público..."
TUNNEL_URL=""
for i in $(seq 1 25); do
    TUNNEL_URL=$(grep -o 'https://[-0-9a-z]*\.trycloudflare\.com' /tmp/tunnel.log 2>/dev/null | head -n 1 || true)
    if [ -n "$TUNNEL_URL" ]; then
        break
    fi
    sleep 1
done

echo "=========================================================="
echo "🎉 ¡DoblAI ESTÁ FUNCIONANDO EN LA RTX 4090!"
echo "👉 ENTRA DESDE TU NAVEGADOR A ESTE ENLACE:"
echo "   $TUNNEL_URL"
echo "=========================================================="
echo ""
echo "🚀 Servidor activo y escuchando peticiones:"

export OLLAMA_URL="http://localhost:11434"
export COQUI_TOS_AGREED="1"
exec uvicorn webapp.app:app --host 0.0.0.0 --port 8080
