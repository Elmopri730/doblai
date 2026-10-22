import json
import logging
import subprocess
import importlib
from pathlib import Path

logger = logging.getLogger("dubbing.transcripcion")


class Transcriber:
    """
    Etapa 03 — Transcripción de voz a texto.

    Utiliza faster-whisper (CTranslate2, cuantización INT8) para convertir
    el audio de voz aislada a segmentos de texto con timestamps de precisión
    a nivel de palabra.

    Salida: work/<video_id>/03_transcripcion/transcript_raw.json
    """

    WHISPER_MODEL_MAP = {
        "whisper-large-v3":         "large-v3",
        "whisper-large-v3-turbo":   "large-v3-turbo",
        "whisper-medium":           "medium",
        "whisper-small":            "small",
        "whisper-base":             "base",
        "whisper-tiny":             "tiny",
        "large-v3":                 "large-v3",
        "large-v3-turbo":           "large-v3-turbo",
        "medium":                   "medium",
        "small":                    "small",
        "base":                     "base",
        "tiny":                     "tiny",
    }

    def __init__(self, config: dict):
        self.config = config
        self.work_dir = Path(config.get("pipeline", {}).get("work_dir", "./work"))
        self.hardware_mode = config.get("pipeline", {}).get("hardware", {}).get("modo", "auto")

        # Nombre del modelo tal como viene en config.yaml (ej. "whisper-large-v3")
        raw_model = (
            config.get("stages", {})
                  .get("transcription", {})
                  .get("model", "large-v3-turbo")
        )
        # Normalizar si tiene prefijo whisper-
        normalized = self.WHISPER_MODEL_MAP.get(raw_model)
        if not normalized:
            normalized = raw_model.lower().replace("whisper-", "")
        self.model_name = normalized

        # Idioma de transcripción (None = auto-detección de Whisper)
        lang_cfg = (
            config.get("stages", {})
                  .get("transcription", {})
                  .get("language", "auto")
        )
        self.language = None if lang_cfg in ("auto", "AUTO", "") else lang_cfg

        # Tipo de cómputo según disponibilidad de VRAM
        self.compute_type, self.device = self._detect_compute()

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _detect_compute(self):
        """
        Selecciona dispositivo y tipo de cómputo automáticamente.
        Con GPU: INT8 (óptimo para 4 GB VRAM).
        Sin GPU: INT8 en CPU (más lento pero funcional).
        """
        if self.hardware_mode == "cpu":
            logger.info("Modo CPU forzado por configuración.")
            return "int8", "cpu"

        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                logger.info(f"GPU detectada con {vram_gb:.1f} GB VRAM. Usando INT8 en CUDA.")
                return "int8", "cuda"
        except ImportError:
            pass

        if self.hardware_mode == "gpu":
            raise RuntimeError(
                "Se solicitó modo GPU pero no se detectó CUDA disponible. "
                "Instala los drivers de NVIDIA o cambia hardware.modo a 'auto' o 'cpu'."
            )

        logger.info("No se detectó GPU CUDA. Usando CPU (int8).")
        return "int8", "cpu"

    @staticmethod
    def _resample_to_16k(input_wav: Path, output_wav: Path) -> None:
        """
        Convierte cualquier WAV a 16 kHz mono (formato óptimo para Whisper).
        Usa FFmpeg para no introducir dependencias extra de Python.
        """
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_wav),
            "-ar", "16000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            str(output_wav),
        ]
        subprocess.run(cmd, capture_output=True, check=True)

    # ------------------------------------------------------------------
    # Punto de entrada público
    # ------------------------------------------------------------------

    def run(self, video_id: str, vocals_wav: str) -> dict:
        """
        Ejecuta la transcripción de la pista de voz aislada.

        Args:
            video_id:   Identificador único del vídeo (generado en etapa 01).
            vocals_wav: Ruta al fichero vocals.wav producido por etapa 02.

        Returns:
            dict con claves:
                - transcript_raw  (str): ruta a transcript_raw.json
                - language        (str): idioma detectado por Whisper
                - segments_count  (int): número de segmentos transcritos
        """
        vocals_path = Path(vocals_wav)
        if not vocals_path.exists():
            raise FileNotFoundError(f"Audio de voz no encontrado: {vocals_path}")

        stage_dir = self.work_dir / video_id / "03_transcripcion"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # 1. Resample a 16kHz si hace falta
        wav_16k = stage_dir / "vocals_16k.wav"
        logger.info(f"[{video_id}] Convirtiendo audio a 16kHz mono para Whisper...")
        self._resample_to_16k(vocals_path, wav_16k)

        # 2. Cargar modelo (se descarga automáticamente la primera vez)
        logger.info(
            f"[{video_id}] Cargando modelo Whisper '{self.model_name}' "
            f"en {self.device.upper()} con compute_type={self.compute_type}..."
        )
        fw = importlib.import_module("faster_whisper")
        WhisperModel = fw.WhisperModel

        model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

        # 3. Transcribir
        logger.info(f"[{video_id}] Iniciando transcripción...")
        segments_iter, info = model.transcribe(
            str(wav_16k),
            language=self.language,
            word_timestamps=True,   # timestamps a nivel de palabra
            vad_filter=True,        # filtra silencios con VAD
            vad_parameters={
                "min_silence_duration_ms": 400,
                "speech_pad_ms": 200,
            },
            beam_size=5,
        )

        # 4. Materializar los segmentos a lista serializable
        segments = []
        for seg in segments_iter:
            words = []
            if seg.words:
                for w in seg.words:
                    words.append({
                        "word":        w.word.strip(),
                        "start":       round(w.start, 3),
                        "end":         round(w.end, 3),
                        "probability": round(w.probability, 4),
                    })
            segments.append({
                "id":               seg.id,
                "start":            round(seg.start, 3),
                "end":              round(seg.end, 3),
                "text":             seg.text.strip(),
                "avg_logprob":      round(seg.avg_logprob, 4),
                "no_speech_prob":   round(seg.no_speech_prob, 4),
                "words":            words,
            })

        detected_lang = info.language

        # 5. Guardar transcript_raw.json
        transcript_data = {
            "video_id":        video_id,
            "model":           self.model_name,
            "language":        detected_lang,
            "language_prob":   round(info.language_probability, 4),
            "duration":        round(info.duration, 3),
            "segments_count":  len(segments),
            "segments":        segments,
        }

        out_path = stage_dir / "transcript_raw.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"[{video_id}] Transcripción completada: {len(segments)} segmentos "
            f"en idioma '{detected_lang}' ({info.language_probability:.0%} confianza)."
        )
        logger.info(f"[{video_id}] Transcripción guardada en: {out_path}")

        return {
            "transcript_raw":   str(out_path),
            "language":         detected_lang,
            "segments_count":   len(segments),
        }
