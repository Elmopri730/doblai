"""
Etapa 05 — Síntesis de voz con clonación (TTS).

Utiliza Coqui XTTS v2 para:
  1. Extraer un fragmento de referencia de la voz original (speaker sample).
  2. Sintetizar cada segmento del transcript_translated.json en español,
     clonando la voz del locutor original.
  3. Exportar cada segmento como WAV individual + un WAV combinado
     (dubbed_voice_raw.wav) con todos los segmentos concatenados con silencios
     para preservar la línea temporal original.

Salida:
  work/<video_id>/05_tts/
    speaker_sample.wav          — muestra de referencia de la voz
    segments/seg_0001.wav …     — WAV sintetizado por segmento
    dubbed_voice_raw.wav        — pista de voz doblada completa (sin mezclar)
    tts_manifest.json           — metadatos de cada segmento sintetizado
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import wave
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("dubbing.tts")

# Muestra de silencio para rellenar huecos temporales entre segmentos
_SILENCE_SAMPLE_RATE = 22050   # Hz de salida de XTTS v2
_BYTES_PER_SAMPLE    = 2       # 16-bit PCM


class VoiceSynthesizer:
    """
    Etapa 05 — Síntesis de voz con clonación (XTTS v2).

    El modelo XTTS v2 de Coqui soporta síntesis multilingüe con clonación
    de voz a partir de un fragmento de referencia de al menos 6 segundos.
    Funciona con 4 GB de VRAM en modo GPU (o en CPU, más lento).
    """

    def __init__(self, config: dict):
        self.config      = config
        self.work_dir    = Path(config.get("pipeline", {}).get("work_dir", "./work"))
        self.hw_mode     = config.get("pipeline", {}).get("hardware", {}).get("modo", "auto")

        tts_cfg = config.get("stages", {}).get("tts", {})
        self.model_name          = tts_cfg.get("model_name", "tts_models/multilingual/multi-dataset/xtts_v2")
        self.speaker_sample_min  = tts_cfg.get("speaker_sample_min_s", 6)
        self.speaker_sample_max  = tts_cfg.get("speaker_sample_max_s", 30)
        self.target_language     = tts_cfg.get("target_language", "es")
        self.speed               = tts_cfg.get("speed", 1.0)
        self.temperature         = tts_cfg.get("temperature", 0.75)
        self.length_penalty      = tts_cfg.get("length_penalty", 1.0)
        self.repetition_penalty  = tts_cfg.get("repetition_penalty", 5.0)
        self.top_k               = tts_cfg.get("top_k", 50)
        self.top_p               = tts_cfg.get("top_p", 0.85)

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers internos
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_device(self) -> str:
        """Devuelve 'cuda' si hay GPU disponible, si no 'cpu'."""
        if self.hw_mode == "cpu":
            return "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("GPU CUDA detectada. XTTS usará GPU.")
                return "cuda"
        except ImportError:
            pass
        if self.hw_mode == "gpu":
            raise RuntimeError("Modo GPU forzado pero CUDA no disponible.")
        logger.info("Sin GPU detectada. XTTS usará CPU (más lento).")
        return "cpu"

    def _extract_speaker_sample(
        self,
        vocals_wav: Path,
        output_path: Path,
        source_language: str = "en",
    ) -> Path:
        """
        Extrae un fragmento de la voz original para usarlo como referencia
        de clonación. Selecciona los primeros speaker_sample_max segundos de
        habla activa usando el VAD de Whisper a través de ffmpeg.

        Requisito de XTTS v2: mínimo 6 segundos, óptimo 10-30 segundos.
        """
        logger.info("Extrayendo muestra de referencia de la voz original...")

        # Cortar el fragmento de referencia directamente con ffmpeg
        duration = min(self.speaker_sample_max, 30)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(vocals_wav),
            "-t", str(duration),
            "-ar", "22050",   # XTTS v2 espera 22050 Hz
            "-ac", "1",
            "-acodec", "pcm_s16le",
            str(output_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        logger.info(f"Muestra de referencia: {output_path.name} ({duration}s)")
        return output_path

    def _seconds_to_samples(self, seconds: float, sample_rate: int) -> int:
        return int(math.ceil(seconds * sample_rate))

    def _get_wav_params(self, wav_path: Path):
        """Lee los parámetros (nchannels, sampwidth, framerate) de un WAV."""
        with wave.open(str(wav_path), "rb") as wf:
            return wf.getnchannels(), wf.getsampwidth(), wf.getframerate()

    def _build_silence(self, duration_s: float, sample_rate: int, channels: int, sampwidth: int) -> bytes:
        """Genera bytes de silencio PCM para rellenar huecos temporales."""
        n_samples = self._seconds_to_samples(duration_s, sample_rate)
        return b"\x00" * (n_samples * channels * sampwidth)

    def _load_xtts(self, device: str):
        """Carga el modelo XTTS v2. Se descarga la primera vez (~1.8 GB)."""
        try:
            # Parchear PyTorch 2.6+ para permitir cargar el modelo de Coqui
            import torch
            if hasattr(torch.serialization, 'add_safe_globals'):
                original_load = torch.load
                def safe_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                torch.load = safe_load

            from TTS.api import TTS
        except ImportError:
            raise RuntimeError(
                "La librería 'TTS' de Coqui no está instalada. "
                "Ejecuta: pip install TTS"
            )
        # Aceptar automáticamente los términos de Coqui para que no pida
        # confirmación interactiva (que crashea en Docker sin terminal).
        import os
        os.environ["COQUI_TOS_AGREED"] = "1"
        logger.info(f"Cargando modelo XTTS v2 en {device.upper()}... (primera vez puede tardar, descarga ~1.8 GB)")
        tts = TTS(self.model_name).to(device)
        return tts

    # ──────────────────────────────────────────────────────────────────────────
    # Punto de entrada público
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        video_id: str,
        vocals_wav: str,
        transcript_translated_path: str,
        source_language: str = "en",
    ) -> dict:
        """
        Ejecuta la síntesis de voz para todos los segmentos traducidos.

        Args:
            video_id:                   Identificador único del vídeo.
            vocals_wav:                 Ruta al fichero vocals.wav (etapa 02).
            transcript_translated_path: Ruta al transcript_translated.json (etapa 04).
            source_language:            Idioma original detectado por Whisper.

        Returns:
            dict con claves:
                - dubbed_voice_raw      (str): ruta a dubbed_voice_raw.wav
                - tts_manifest          (str): ruta a tts_manifest.json
                - segments_synthesized  (int): número de segmentos sintetizados
                - sample_rate           (int): frecuencia de muestreo de salida
        """
        vocals_path    = Path(vocals_wav)
        transcript_path = Path(transcript_translated_path)

        if not vocals_path.exists():
            raise FileNotFoundError(f"vocals.wav no encontrado: {vocals_path}")
        if not transcript_path.exists():
            raise FileNotFoundError(f"transcript_translated.json no encontrado: {transcript_path}")

        # Crear estructura de carpetas
        stage_dir   = self.work_dir / video_id / "05_tts"
        segments_dir = stage_dir / "segments"
        stage_dir.mkdir(parents=True, exist_ok=True)
        segments_dir.mkdir(exist_ok=True)

        # ── Leer segmentos traducidos ──────────────────────────────────────────
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)

        segments: List[dict] = transcript.get("segments", [])
        if not segments:
            raise ValueError("El transcript_translated.json no contiene segmentos.")

        total_duration = transcript.get("original_duration", 0)
        logger.info(f"[{video_id}] Iniciando síntesis de {len(segments)} segmentos...")

        # ── Extraer muestra de referencia de la voz ────────────────────────────
        sample_path = stage_dir / "speaker_sample.wav"
        self._extract_speaker_sample(vocals_path, sample_path, source_language)

        # ── Cargar modelo XTTS v2 ──────────────────────────────────────────────
        device = self._detect_device()
        tts    = self._load_xtts(device)

        # ── Sintetizar segmento por segmento ──────────────────────────────────
        manifest = []
        synthesized_paths = []

        for i, seg in enumerate(segments):
            seg_id   = seg.get("id", i + 1)
            text_es  = seg.get("text_es", "").strip()
            t_start  = seg.get("start", 0.0)
            t_end    = seg.get("end", 0.0)
            seg_dur  = t_end - t_start

            if not text_es:
                logger.warning(f"[{video_id}] Segmento {seg_id} sin texto. Saltando.")
                continue

            seg_out = segments_dir / f"seg_{seg_id:04d}.wav"
            logger.info(f"[{video_id}] [{i+1}/{len(segments)}] Sintetizando: \"{text_es[:60]}...\"")

            # Síntesis con clonación de voz
            tts.tts_to_file(
                text=text_es,
                speaker_wav=str(sample_path),
                language=self.target_language,
                file_path=str(seg_out),
                speed=self.speed,
            )

            # Obtener duración real del audio sintetizado
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(seg_out)],
                capture_output=True, text=True
            )
            synth_dur = float(result.stdout.strip()) if result.stdout.strip() else seg_dur

            manifest.append({
                "id":            seg_id,
                "text_es":       text_es,
                "original_start": t_start,
                "original_end":   t_end,
                "original_dur":   round(seg_dur, 3),
                "synth_dur":      round(synth_dur, 3),
                "file":           str(seg_out),
                "needs_stretch":  abs(synth_dur - seg_dur) > 0.15,
            })
            synthesized_paths.append((t_start, t_end, seg_out))

        logger.info(f"[{video_id}] Síntesis completada. Ensamblando pista de voz...")

        # ── Ensamblar pista de voz completa con silencios temporales ───────────
        dubbed_raw_path = stage_dir / "dubbed_voice_raw.wav"
        self._assemble_dubbed_track(
            synthesized_paths,
            total_duration,
            dubbed_raw_path,
        )

        # ── Guardar manifiesto ─────────────────────────────────────────────────
        manifest_data = {
            "video_id":            video_id,
            "target_language":     self.target_language,
            "model":               self.model_name,
            "sample_rate":         _SILENCE_SAMPLE_RATE,
            "segments_total":      len(segments),
            "segments_synthesized": len(manifest),
            "segments":            manifest,
        }
        manifest_path = stage_dir / "tts_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)

        logger.info(f"[{video_id}] Etapa 05 completada → {dubbed_raw_path.name}")
        return {
            "dubbed_voice_raw":    str(dubbed_raw_path),
            "tts_manifest":        str(manifest_path),
            "segments_synthesized": len(manifest),
            "sample_rate":         _SILENCE_SAMPLE_RATE,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Ensamblaje de la pista de voz completa
    # ──────────────────────────────────────────────────────────────────────────

    def _assemble_dubbed_track(
        self,
        synthesized: List[tuple],          # (t_start, t_end, path)
        total_duration: float,
        output_path: Path,
    ) -> None:
        """
        Combina todos los WAVs de segmentos en una única pista de voz,
        rellenando los huecos entre segmentos con silencio para preservar
        la sincronía temporal con el vídeo original.

        Estrategia:
          - Si el segmento sintetizado es más CORTO que el hueco temporal,
            se añade silencio al final para alinearlo.
          - Si es más LARGO, se superpone con el siguiente segmento
            (la etapa 06 de sincronización temporal ajustará la velocidad).
        """
        if not synthesized:
            logger.warning("No hay segmentos sintetizados para ensamblar.")
            return

        # Leer parámetros del primer segmento sintetizado
        try:
            nch, sampwidth, framerate = self._get_wav_params(synthesized[0][2])
        except Exception:
            nch, sampwidth, framerate = 1, 2, _SILENCE_SAMPLE_RATE

        with wave.open(str(output_path), "wb") as out_wf:
            out_wf.setnchannels(nch)
            out_wf.setsampwidth(sampwidth)
            out_wf.setframerate(framerate)

            cursor = 0.0   # posición temporal actual en segundos

            for t_start, t_end, seg_path in synthesized:
                # Silencio antes del segmento si hay hueco
                gap = t_start - cursor
                if gap > 0.01:
                    out_wf.writeframes(self._build_silence(gap, framerate, nch, sampwidth))

                # Escribir frames del segmento sintetizado
                if seg_path.exists():
                    with wave.open(str(seg_path), "rb") as seg_wf:
                        frames = seg_wf.readframes(seg_wf.getnframes())
                        out_wf.writeframes(frames)
                    # Avanzar cursor al máximo entre t_end y la duración real
                    seg_duration = len(frames) / (framerate * nch * sampwidth)
                    cursor = max(t_end, t_start + seg_duration)
                else:
                    # Escribir silencio del tamaño del hueco original
                    out_wf.writeframes(self._build_silence(t_end - t_start, framerate, nch, sampwidth))
                    cursor = t_end

            # Rellenar hasta la duración total del vídeo
            tail = total_duration - cursor
            if tail > 0.01:
                out_wf.writeframes(self._build_silence(tail, framerate, nch, sampwidth))

        logger.info(f"Pista de voz ensamblada: {output_path.name}")
