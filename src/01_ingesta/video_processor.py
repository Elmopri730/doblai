import os
import subprocess
import json
import logging
from pathlib import Path
import hashlib
import shutil

logger = logging.getLogger("dubbing.ingestion")

class VideoProcessor:
    """
    Clase para procesar la etapa 01 (Ingesta de vídeo).
    Valida el archivo, extrae metadatos y separa el audio raw sin pérdida.
    """
    def __init__(self, config: dict):
        self.config = config
        self.work_dir = Path(config.get("pipeline", {}).get("work_dir", "./work"))
        self.output_dir = Path(config.get("pipeline", {}).get("output_dir", "./output"))

    def get_video_id(self, video_path: Path) -> str:
        """Genera un ID único para el vídeo basado en su nombre y un hash rápido."""
        hasher = hashlib.md5()
        hasher.update(str(video_path.name).encode('utf-8'))
        hasher.update(str(os.path.getsize(video_path)).encode('utf-8'))
        fast_hash = hasher.hexdigest()[:8]
        # Sanitizar nombre del archivo
        clean_name = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in video_path.stem])
        return f"{clean_name}_{fast_hash}"

    def get_metadata(self, video_path: Path) -> dict:
        """Obtiene los metadatos del vídeo original usando ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path)
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.error(f"Error al ejecutar ffprobe en {video_path}: {e}")
            raise RuntimeError(f"El archivo {video_path.name} no es un archivo de vídeo válido o ffprobe falló.")

    def run(self, video_path: Path) -> dict:
        """
        Ejecuta la etapa de ingesta.
        Crea la estructura de carpetas, extrae el audio y guarda metadatos.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Archivo de vídeo no encontrado: {video_path}")

        video_id = self.get_video_id(video_path)
        stage_dir = self.work_dir / video_id / "01_ingesta"
        stage_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[{video_id}] Iniciando etapa 01 (Ingesta de vídeo)...")

        # 1. Obtener metadatos
        metadata = self.get_metadata(video_path)
        
        # Encontrar datos interesantes de los streams
        video_stream = next((s for s in metadata.get("streams", []) if s.get("codec_type") == "video"), {})
        audio_stream = next((s for s in metadata.get("streams", []) if s.get("codec_type") == "audio"), {})
        
        duration = float(metadata.get("format", {}).get("duration", 0))
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        fps = video_stream.get("r_frame_rate", "0/0")
        
        video_info = {
            "video_id": video_id,
            "original_filename": video_path.name,
            "duration_seconds": duration,
            "resolution": f"{width}x{height}",
            "fps": fps,
            "video_codec": video_stream.get("codec_name"),
            "audio_codec": audio_stream.get("codec_name"),
            "original_metadata": metadata
        }

        # Guardar metadata.json
        info_path = stage_dir / "video_info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(video_info, f, indent=2, ensure_ascii=False)
        logger.info(f"[{video_id}] Metadatos guardados en {info_path.name}")

        # 2. Extraer audio raw sin pérdida (WAV PCM 16-bit, 44.1kHz, estéreo/mono según origen)
        # Si tiene pista de audio, la extraemos. Si no, lanzamos error.
        if not audio_stream:
            raise RuntimeError(f"El vídeo {video_path.name} no contiene pista de audio.")

        raw_audio_path = stage_dir / "vocals_raw.wav"
        
        # Comando para extraer audio WAV
        # -y sobreescribe
        # -vn deshabilita video
        # -acodec pcm_s16le formato sin pérdida
        # -ar 44100 frecuencia de muestreo recomendada
        audio_cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            str(raw_audio_path)
        ]
        
        logger.info(f"[{video_id}] Extrayendo pista de audio a WAV PCM (44.1kHz)...")
        subprocess.run(audio_cmd, capture_output=True, check=True)
        logger.info(f"[{video_id}] Audio extraído correctamente: {raw_audio_path.name}")

        # 3. Normalizar vídeo (Generar video_normalized.mp4)
        # Si ya es MP4, se copia directamente en milisegundos sin re-encode.
        # Si es otro contenedor (MKV, MOV, etc.), se remuxea con -c:v copy en 1 segundo.
        normalized_video_path = stage_dir / "video_normalized.mp4"
        
        if video_path.suffix.lower() == ".mp4":
            logger.info(f"[{video_id}] Copiando vídeo directo (contenedor MP4)...")
            shutil.copy2(video_path, normalized_video_path)
        else:
            logger.info(f"[{video_id}] Remuxeando vídeo a contenedor MP4 (stream copy sin pérdida)...")
            video_cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-c:v", "copy",
                "-an",
                str(normalized_video_path)
            ]
            try:
                subprocess.run(video_cmd, capture_output=True, check=True)
                logger.info(f"[{video_id}] Remux completado: {normalized_video_path.name}")
            except subprocess.CalledProcessError:
                logger.warning(f"[{video_id}] Remux falló, transcodificando rápido con NVENC/ultrafast...")
                # Fallback con GPU NVENC o CPU ultrafast
                video_fallback = [
                    "ffmpeg", "-y",
                    "-i", str(video_path),
                    "-c:v", "h264_nvenc",
                    "-preset", "p1",
                    "-an",
                    str(normalized_video_path)
                ]
                try:
                    subprocess.run(video_fallback, capture_output=True, check=True)
                except Exception:
                    subprocess.run([
                        "ffmpeg", "-y", "-i", str(video_path),
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-an",
                        str(normalized_video_path)
                    ], capture_output=True, check=True)
        logger.info(f"[{video_id}] Vídeo listo: {normalized_video_path.name}")

        # 4. Extraer subtítulos embebidos si existen (Opcional, no detiene el pipeline si falla)
        sub_streams = [s for s in metadata.get("streams", []) if s.get("codec_type") == "subtitle"]
        if sub_streams:
            subs_dir = stage_dir / "subtitles_extracted"
            subs_dir.mkdir(exist_ok=True)
            logger.info(f"[{video_id}] Se detectaron {len(sub_streams)} pistas de subtítulos. Extrayendo...")
            for idx, stream in enumerate(sub_streams):
                lang = stream.get("tags", {}).get("language", f"track_{idx}")
                sub_output = subs_dir / f"sub_{lang}.srt"
                # Extraer pista de subtítulos a SRT
                sub_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(video_path),
                    "-map", f"0:s:{idx}",
                    str(sub_output)
                ]
                try:
                    subprocess.run(sub_cmd, capture_output=True, check=True)
                    logger.info(f"[{video_id}] Subtítulo extraído: {sub_output.name}")
                except subprocess.CalledProcessError:
                    logger.warning(f"[{video_id}] No se pudo extraer la pista de subtítulo {idx} ({lang}).")

        logger.info(f"[{video_id}] Etapa 01 (Ingesta) completada con éxito.")
        return {
            "video_id": video_id,
            "stage_dir": str(stage_dir),
            "video_info": video_info,
            "vocals_raw": str(raw_audio_path),
            "video_normalized": str(normalized_video_path)
        }
