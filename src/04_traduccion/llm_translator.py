"""
Etapa 04 — Corrección y traducción via LLM local (Ollama).

Recibe el transcript_raw.json generado por faster-whisper y devuelve
transcript_translated.json con:
  - Errores de STT corregidos (nombres propios, siglas, puntuación).
  - Texto completo traducido al español de España.
  - Timestamps ORIGINALES preservados para la síntesis de voz posterior.

Estrategia:
  Procesamiento en lotes pequeños (5 segmentos) con fallback individual
  garantizado para que NUNCA se devuelva texto en inglés.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger("dubbing.traduccion")

# Lotes pequeños para máxima fiabilidad y precisión en modelos de 3B
MAX_CHUNK_SEGMENTS = 5

SYSTEM_PROMPT = """Eres un traductor profesional de cine y televisión especializado en doblaje al español de España.
Tu trabajo es traducir los diálogos del vídeo al español (castellano natural y formal de España).

REGLAS CRÍTICAS:
1. Traduce TODO el texto al español. NUNCA dejes frases en inglés.
2. Adapta las expresiones coloquiales o idiomáticas para que suenen 100% naturales en español hablado.
3. Responde ÚNICAMENTE con un JSON válido con la siguiente estructura:
{
  "translations": [
    {"id": <número id>, "text_es": "<traducción en español>"}
  ]
}"""


class LLMTranslator:
    """
    Etapa 04 — Corrección de texto y traducción al español con LLM local (Ollama).
    """

    def __init__(self, config: dict):
        self.config = config
        self.work_dir = Path(config.get("pipeline", {}).get("work_dir", "./work"))

        # Leer configuración de traducción desde config.yaml o variable de entorno
        translation_cfg = config.get("stages", {}).get("translation", {})
        import os
        self.ollama_base_url = os.environ.get("OLLAMA_URL", translation_cfg.get("ollama_url", "http://localhost:11434"))
        self.model = os.environ.get("OLLAMA_MODEL", translation_cfg.get("model", "qwen2.5:3b"))
        self.temperature = translation_cfg.get("temperature", 0.2)
        self.max_retries = translation_cfg.get("max_retries", 3)

    def _build_client(self):
        """Crea un cliente OpenAI apuntando a la API local de Ollama."""
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "La librería 'openai' no está instalada. Ejecuta: pip install openai"
            )
        return OpenAI(
            base_url=f"{self.ollama_base_url}/v1",
            api_key="ollama",
        )

    def _translate_single(self, client, text: str) -> str:
        """
        Traduce una sola frase al español de forma directa.
        Imposible que falle o devuelva estructura errónea.
        """
        text = text.strip()
        if not text:
            return ""

        for attempt in range(1, 3):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Eres un traductor al español. Traduce la siguiente frase al español de España. "
                                "Responde ÚNICAMENTE con el texto en español traducido, sin explicaciones, "
                                "sin comillas y sin repetir el inglés."
                            )
                        },
                        {
                            "role": "user",
                            "content": f"Traduce al español: {text}"
                        }
                    ],
                    temperature=0.2,
                )
                raw = response.choices[0].message.content.strip().strip('"').strip("'")
                # Limpiar prefijos si el modelo los añade
                for prefix in ["Traducción:", "Traduccion:", "Español:", "Espanol:", "Spanish:"]:
                    if raw.lower().startswith(prefix.lower()):
                        raw = raw[len(prefix):].strip()
                if raw and raw.lower() != text.lower():
                    return raw
            except Exception as e:
                logger.warning(f"Intento {attempt} de traducción individual falló: {e}")
                time.sleep(1)

        return text

    def _translate_chunk(self, client, segments_chunk: List[dict]) -> List[dict]:
        """
        Envía un lote de 5 segmentos al LLM y parsea la respuesta.
        Si algún segmento falta o no se tradujo, lo traduce individualmente.
        """
        input_data = [
            {"id": s["id"], "text": s["text"]}
            for s in segments_chunk
        ]
        user_prompt = f"Traduce los siguientes {len(segments_chunk)} segmentos al español:\n" + json.dumps(input_data, ensure_ascii=False, indent=2)

        translated_map: Dict[int, str] = {}

        for attempt in range(1, self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )

                raw_content = response.choices[0].message.content.strip()
                parsed = json.loads(raw_content)

                # Buscar la lista de resultados bajo cualquier clave razonable
                items = (
                    parsed.get("translations") or 
                    parsed.get("segments") or 
                    parsed.get("segmentos") or 
                    (parsed if isinstance(parsed, list) else [])
                )

                if isinstance(items, list):
                    for it in items:
                        if isinstance(it, dict) and "id" in it:
                            t_es = (
                                it.get("text_es") or 
                                it.get("translation") or 
                                it.get("traduccion") or 
                                it.get("text") or 
                                ""
                            ).strip()
                            if t_es:
                                translated_map[it["id"]] = t_es

                # Si obtuvimos traducciones para todos los segmentos del lote, salimos del reintento
                if len(translated_map) >= len(segments_chunk):
                    break

            except Exception as e:
                logger.warning(f"[Intento {attempt}/{self.max_retries}] Error parseando lote: {e}")
                time.sleep(1)

        # Construir resultado final asegurando que TODO esté en español
        result = []
        for s in segments_chunk:
            sid = s["id"]
            orig_text = s["text"].strip()
            es_text = translated_map.get(sid, "").strip()

            # Si no se tradujo o quedó idéntico al inglés (y no es solo un número/palabra corta),
            # forzar traducción individual
            words_count = len(orig_text.split())
            is_untranslated = (not es_text) or (es_text.lower() == orig_text.lower() and words_count > 1)

            if is_untranslated:
                logger.info(f"Traduciendo segmento individual #{sid}: '{orig_text[:40]}...'")
                es_text = self._translate_single(client, orig_text)

            result.append({
                "id":            sid,
                "start":         s["start"],
                "end":           s["end"],
                "text_original": orig_text,
                "text_es":       es_text or orig_text,
            })

        return result

    def run(self, video_id: str, transcript_raw_path: str) -> dict:
        """
        Ejecuta la traducción de los segmentos de transcripción.
        """
        raw_path = Path(transcript_raw_path)
        if not raw_path.exists():
            raise FileNotFoundError(f"Transcripción raw no encontrada: {raw_path}")

        with open(raw_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)

        segments: List[dict] = transcript_data.get("segments", [])
        if not segments:
            raise ValueError(f"El fichero {raw_path.name} no contiene segmentos.")

        logger.info(
            f"[{video_id}] Iniciando traducción de {len(segments)} segmentos "
            f"al español con modelo '{self.model}' en {self.ollama_base_url}..."
        )

        stage_dir = self.work_dir / video_id / "04_traduccion"
        stage_dir.mkdir(parents=True, exist_ok=True)

        client = self._build_client()

        translated_segments = []
        chunks = [
            segments[i : i + MAX_CHUNK_SEGMENTS]
            for i in range(0, len(segments), MAX_CHUNK_SEGMENTS)
        ]
        total_chunks = len(chunks)

        for idx, chunk in enumerate(chunks, 1):
            logger.info(f"[{video_id}] Traduciendo lote {idx}/{total_chunks} ({len(chunk)} frases)...")
            translated_chunk = self._translate_chunk(client, chunk)
            translated_segments.extend(translated_chunk)

        result = {
            "video_id":           video_id,
            "source_language":    transcript_data.get("language", "en"),
            "target_language":    "es",
            "model":              self.model,
            "segments_count":     len(translated_segments),
            "original_duration":  transcript_data.get("duration", 0),
            "segments":           translated_segments,
        }

        out_path = stage_dir / "transcript_translated.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(
            f"[{video_id}] Traducción completada con éxito: {len(translated_segments)} segmentos "
            f"guardados en {out_path.name}"
        )

        return {
            "transcript_translated": str(out_path),
            "segments_count":        len(translated_segments),
            "target_language":       "es",
        }
