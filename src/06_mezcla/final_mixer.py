"""
Etapa 06 — Mezcla final de audio y exportación del vídeo doblado.

Combina:
  1. La pista de voz doblada (dubbed_voice_raw.wav, etapa 05).
  2. La música/ambiente de fondo (accompaniment.wav, etapa 02),
     reducida de volumen para que la voz protagonice.
  3. El vídeo normalizado (video_normalized.mp4, etapa 01).

Y genera el fichero final de salida:
  output/<video_id>/<nombre_original>_doblado_es.mp4

También exporta los subtítulos en los formatos configurados (.srt, .vtt, .ass).
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import List

logger = logging.getLogger("dubbing.mezcla")


# ── Conversores de timestamps ──────────────────────────────────────────────────

def _secs_to_srt(seconds: float) -> str:
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def _secs_to_vtt(seconds: float) -> str:
    return _secs_to_srt(seconds).replace(",", ".")

def _secs_to_ass(seconds: float) -> str:
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = seconds % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


class FinalMixer:
    """
    Etapa 06 — Mezcla de audio y exportación del vídeo doblado.

    Genera el vídeo final y los ficheros de subtítulos usando solo FFmpeg,
    sin dependencias adicionales de Python para el procesamiento de vídeo.
    """

    ACCOMPANIMENT_DUCK_DB = -6   # dB de reducción de la música de fondo

    def __init__(self, config: dict):
        self.config     = config
        self.work_dir   = Path(config.get("pipeline", {}).get("work_dir",   "./work"))
        self.output_dir = Path(config.get("pipeline", {}).get("output_dir", "./output"))
        self.sub_formats: List[str] = (
            config.get("pipeline", {}).get("formatos_subtitulos", ["srt", "vtt"])
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Generación de subtítulos
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_srt(self, segments: List[dict], output_path: Path) -> None:
        lines = []
        for i, seg in enumerate(segments, 1):
            lines.append(str(i))
            lines.append(f"{_secs_to_srt(seg['original_start'])} --> {_secs_to_srt(seg['original_end'])}")
            lines.append(seg.get("text_es", ""))
            lines.append("")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"SRT generado: {output_path.name}")

    def _generate_vtt(self, segments: List[dict], output_path: Path) -> None:
        lines = ["WEBVTT", ""]
        for seg in segments:
            lines.append(f"{_secs_to_vtt(seg['original_start'])} --> {_secs_to_vtt(seg['original_end'])}")
            lines.append(seg.get("text_es", ""))
            lines.append("")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"VTT generado: {output_path.name}")

    def _generate_ass(self, segments: List[dict], output_path: Path) -> None:
        header = (
            "[Script Info]\nTitle: Doblaje ES\nScriptType: v4.00+\n\n"
            "[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,Bold,Italic,"
            "Underline,Alignment,BorderStyle,Outline,Shadow\n"
            "Style: Default,Arial,22,&H00FFFFFF,0,0,0,2,1,2,0\n\n"
            "[Events]\nFormat: Layer,Start,End,Style,Name,Text\n"
        )
        events = []
        for seg in segments:
            text = seg.get("text_es", "").replace("\n", "\\N")
            events.append(
                f"Dialogue: 0,{_secs_to_ass(seg['original_start'])},"
                f"{_secs_to_ass(seg['original_end'])},Default,,{text}"
            )
        output_path.write_text(header + "\n".join(events), encoding="utf-8")
        logger.info(f"ASS generado: {output_path.name}")

    def _export_subtitles(
        self,
        segments: List[dict],
        output_stem: Path,
    ) -> List[str]:
        """Genera todos los formatos de subtítulos configurados."""
        exported = []
        for fmt in self.sub_formats:
            out = output_stem.with_suffix(f".{fmt}")
            if fmt == "srt":
                self._generate_srt(segments, out)
            elif fmt == "vtt":
                self._generate_vtt(segments, out)
            elif fmt == "ass":
                self._generate_ass(segments, out)
            exported.append(str(out))
        return exported

    # ──────────────────────────────────────────────────────────────────────────
    # Mezcla de audio con FFmpeg
    # ──────────────────────────────────────────────────────────────────────────

    def _mix_audio(
        self,
        dubbed_voice: Path,
        accompaniment: Path,
        output_audio: Path,
    ) -> None:
        """
        Mezcla la voz doblada con la música de fondo usando FFmpeg.
        La música se atenúa -6 dB para que la voz destaque.
        Ambas pistas se normalizan a -14 LUFS (estándar de streaming).
        """
        cmd = [
            "ffmpeg", "-y",
            "-i", str(dubbed_voice),
            "-i", str(accompaniment),
            "-filter_complex",
            (
                # Reducir música de fondo 6 dB y mezclar con la voz
                f"[1:a]volume={self.ACCOMPANIMENT_DUCK_DB}dB[bg];"
                "[0:a][bg]amix=inputs=2:duration=longest:normalize=0[mix];"
                # Normalización LUFS (EBU R128)
                "[mix]loudnorm=I=-14:TP=-1.5:LRA=11[out]"
            ),
            "-map", "[out]",
            "-acodec", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            str(output_audio),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg falló al mezclar el audio:\n{result.stderr[-800:]}"
            )
        logger.info(f"Audio mezclado: {output_audio.name}")

    def _mux_video(
        self,
        video_normalized: Path,
        mixed_audio: Path,
        output_video: Path,
    ) -> None:
        """
        Combina el vídeo original con el nuevo audio mezclado.
        Copia el stream de vídeo directamente (sin re-encode) para
        preservar la calidad y ahorrar tiempo de procesado.
        """
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_normalized),
            "-i", str(mixed_audio),
            "-map", "0:v:0",     # Stream de vídeo original
            "-map", "1:a:0",     # Nuevo audio mezclado
            "-c:v", "copy",      # Sin re-encode de vídeo (rápido)
            "-c:a", "copy",
            "-shortest",
            str(output_video),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg falló al muxear el vídeo:\n{result.stderr[-800:]}"
            )
        logger.info(f"Vídeo final exportado: {output_video.name}")

    # ──────────────────────────────────────────────────────────────────────────
    # Punto de entrada público
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        video_id: str,
        video_normalized: str,
        accompaniment_wav: str,
        dubbed_voice_raw: str,
        tts_manifest_path: str,
        original_filename: str,
    ) -> dict:
        """
        Ejecuta la mezcla final y exporta el vídeo doblado.

        Args:
            video_id:            Identificador único del vídeo.
            video_normalized:    Ruta al video_normalized.mp4 (etapa 01).
            accompaniment_wav:   Ruta al accompaniment.wav (etapa 02).
            dubbed_voice_raw:    Ruta al dubbed_voice_raw.wav (etapa 05).
            tts_manifest_path:   Ruta al tts_manifest.json (etapa 05).
            original_filename:   Nombre original del fichero de entrada.

        Returns:
            dict con claves:
                - output_video   (str): ruta al vídeo final doblado
                - subtitles      (list[str]): rutas a los ficheros de subtítulos
        """
        # Verificar entradas
        for path_str, label in [
            (video_normalized,  "video_normalized.mp4"),
            (accompaniment_wav, "accompaniment.wav"),
            (dubbed_voice_raw,  "dubbed_voice_raw.wav"),
            (tts_manifest_path, "tts_manifest.json"),
        ]:
            if not Path(path_str).exists():
                raise FileNotFoundError(f"{label} no encontrado: {path_str}")

        # Cargar manifiesto TTS para extraer segmentos y generar subtítulos
        with open(tts_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        segments = manifest.get("segments", [])

        # Directorio de trabajo de la etapa
        stage_dir = self.work_dir / video_id / "06_mezcla"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Directorio de salida final del vídeo
        video_out_dir = self.output_dir / video_id
        video_out_dir.mkdir(parents=True, exist_ok=True)

        # Nombre de fichero de salida limpio
        clean_stem = re.sub(r"[^\w\-]", "_", Path(original_filename).stem)
        output_stem = video_out_dir / f"{clean_stem}_doblado_es"

        logger.info(f"[{video_id}] Mezclando audio (voz doblada + música de fondo)...")

        # 1. Mezclar audio: voz doblada + música de fondo
        mixed_audio = stage_dir / "mixed_audio.aac"
        self._mix_audio(
            Path(dubbed_voice_raw),
            Path(accompaniment_wav),
            mixed_audio,
        )

        # 2. Muxear: vídeo original + nuevo audio mezclado
        output_video = output_stem.with_suffix(".mp4")
        logger.info(f"[{video_id}] Combinando vídeo y audio mezclado...")
        self._mux_video(Path(video_normalized), mixed_audio, output_video)

        # 3. Exportar subtítulos
        logger.info(f"[{video_id}] Generando subtítulos en formatos: {self.sub_formats}...")
        subtitle_files = self._export_subtitles(segments, output_stem)

        logger.info(
            f"[{video_id}] ✅ Etapa 06 completada.\n"
            f"    Vídeo final:   {output_video}\n"
            f"    Subtítulos:    {subtitle_files}"
        )

        return {
            "output_video": str(output_video),
            "subtitles":    subtitle_files,
        }
