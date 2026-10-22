import os
import shutil
import logging
import subprocess
import torch
from pathlib import Path

logger = logging.getLogger("dubbing.separation")

class AudioSeparatorWrapper:
    """
    Clase para procesar la etapa 02 (Separación de voz/música).
    Utiliza Demucs de Meta con aceleración directa PyTorch CUDA para máxima velocidad
    (tarda solo 15-20 segundos en la RTX 4090).
    """
    def __init__(self, config: dict):
        self.config = config
        self.work_dir = Path(config.get("pipeline", {}).get("work_dir", "./work"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self, video_id: str, raw_audio_path_str: str) -> dict:
        """
        Ejecuta la etapa de separación de fuentes de audio usando Demucs en GPU.
        Retorna la ruta de vocals.wav y accompaniment.wav.
        """
        raw_audio_path = Path(raw_audio_path_str)
        if not raw_audio_path.exists():
            raise FileNotFoundError(f"Audio crudo no encontrado: {raw_audio_path}")

        stage_dir = self.work_dir / video_id / "02_separacion"
        stage_dir.mkdir(parents=True, exist_ok=True)

        vocals_output_path = stage_dir / "vocals.wav"
        accompaniment_output_path = stage_dir / "accompaniment.wav"

        # Si ya existen en disco (ej. reintento inteligente), devolver directamente
        if vocals_output_path.exists() and accompaniment_output_path.exists():
            logger.info(f"[{video_id}] Archivos de separación ya existen en disco. Reutilizando.")
            return {
                "vocals": str(vocals_output_path),
                "accompaniment": str(accompaniment_output_path)
            }

        logger.info(f"[{video_id}] Iniciando separación ultra-rápida con Demucs en {self.device.upper()}...")

        cmd = [
            "python", "-m", "demucs.separate",
            "--two-stems=vocals",
            "-n", "htdemucs",
            "-d", self.device,
            "-o", str(stage_dir),
            str(raw_audio_path)
        ]

        logger.info(f"[{video_id}] Ejecutando: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Demucs falló:\n{result.stderr}")
            raise RuntimeError(f"Error en Demucs: {result.stderr[-500:]}")

        # Demucs crea una estructura interna: <stage_dir>/htdemucs/<audio_stem>/vocals.wav y no_vocals.wav
        found_vocals = list(stage_dir.glob("**/vocals.wav"))
        found_no_vocals = list(stage_dir.glob("**/no_vocals.wav"))

        if not found_vocals or not found_no_vocals:
            raise RuntimeError(f"Demucs no generó los archivos esperados en {stage_dir}")

        # Moverlos a la raíz del directorio de la etapa
        shutil.move(str(found_vocals[0]), str(vocals_output_path))
        shutil.move(str(found_no_vocals[0]), str(accompaniment_output_path))

        # Limpiar carpetas temporales generadas por demucs
        for item in stage_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)

        logger.info(f"[{video_id}] ✅ Etapa 02 (Separación con GPU) completada con éxito en segundos.")
        return {
            "vocals": str(vocals_output_path),
            "accompaniment": str(accompaniment_output_path)
        }
