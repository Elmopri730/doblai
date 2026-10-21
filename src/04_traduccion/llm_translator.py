"""
Etapa 04 — Corrección y traducción via LLM local (Ollama).

Recibe el transcript_raw.json generado por faster-whisper y devuelve
transcript_translated.json con:
  - Errores de STT corregidos (nombres propios, siglas, puntuación).
  - Texto completo traducido al español de España.
  - Timestamps ORIGINALES preservados para la síntesis de voz posterior.

Estrategia de procesamiento por lotes (chunking):
  Whisper puede generar cientos de segmentos. Enviamos al LLM grupos de
  MAX_CHUNK_SEGMENTS segmentos para no superar la ventana de contexto de
  los modelos pequeños (3B) y garantizar respuestas JSON estables.
"""

import json
import logging
import time
from pathlib import Path
from typing import List

logger = logging.getLogger("dubbing.traduccion")

# Número máximo de segmentos por llamada al LLM
MAX_CHUNK_SEGMENTS = 20

# System prompt que instruye al LLM — NO modificar sin justificación técnica.
SYSTEM_PROMPT = """Eres un corrector y traductor profesional especializado en doblaje audiovisual.

Tu tarea es procesar una lista de segmentos de transcripción automática (STT) en formato JSON.
Para cada segmento debes:
1. CORREGIR los errores típicos de STT: palabras mal transcritas, nombres propios, siglas,
   puntuación incorrecta, repeticiones y fragmentaciones. NO alteres el significado.
2. TRADUCIR el texto corregido al español de España (castellano formal pero natural),
   adaptando las expresiones idiomáticas para que suenen bien en doblaje.
3. CONSERVAR exactamente los campos "id", "start" y "end" sin modificarlos.

Responde ÚNICAMENTE con un objeto JSON válido con la siguiente estructura, sin texto adicional:
{
  "segments": [
    {
      "id": <número igual al original>,
      "start": <número igual al original>,
      "end": <número igual al original>,
      "text_original": "<texto corregido en idioma original>",
      "text_es": "<traducción al español>"
    }
  ]
}"""


class LLMTranslator:
    """
    Etapa 04 — Corrección de texto y traducción al español con LLM local (Ollama).

    Utiliza la API compatible con OpenAI de Ollama para enviar los segmentos
    en lotes y recibir JSON estructurado.
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

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _build_client(self):
        """Crea un cliente OpenAI apuntando a la API local de Ollama."""
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "La librería 'openai' no está instalada. "
                "Ejecuta: pip install openai"
            )
        return OpenAI(
            base_url=f"{self.ollama_base_url}/v1",
            api_key="ollama",  # Ollama no requiere API key real
        )

    def _translate_chunk(self, client, segments_chunk: List[dict]) -> List[dict]:
        """
        Envía un lote de segmentos al LLM y devuelve los segmentos traducidos.
        Reintenta hasta max_retries veces si la respuesta no es JSON válido.
        """
        # Preparar input minimalista para el LLM (solo id, start, end, text)
        slim_segments = [
            {"id": s["id"], "start": s["start"], "end": s["end"], "text": s["text"]}
            for s in segments_chunk
        ]
        user_content = json.dumps({"segments": slim_segments}, ensure_ascii=False)

        for attempt in range(1, self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_content},
                    ],
                    temperature=self.temperature,
                    # Forzar respuesta JSON con Ollama (equivalente a response_format)
                    response_format={"type": "json_object"},
                )

                raw_content = response.choices[0].message.content.strip()

                # Parsear JSON de respuesta
                parsed = json.loads(raw_content)
                translated_segs = parsed.get("segments", [])

                if len(translated_segs) != len(segments_chunk):
                    raise ValueError(
                        f"El LLM devolvió {len(translated_segs)} segmentos "
                        f"pero se enviaron {len(segments_chunk)}."
                    )

                return translated_segs

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(
                    f"[Intento {attempt}/{self.max_retries}] "
                    f"Respuesta inválida del LLM: {e}. Reintentando..."
                )
                if attempt == self.max_retries:
                    # Fallback: devolver segmentos con texto original sin traducir
                    logger.error(
                        "No se pudo traducir el lote tras todos los reintentos. "
                        "Se conserva el texto original sin traducción."
                    )
                    return [
                        {
                            "id":            s["id"],
                            "start":         s["start"],
                            "end":           s["end"],
                            "text_original": s["text"],
                            "text_es":       s["text"],  # fallback = original
                        }
                        for s in segments_chunk
                    ]
                time.sleep(1)

    # ------------------------------------------------------------------
    # Punto de entrada público
    # ------------------------------------------------------------------

    def run(self, video_id: str, transcript_raw_path: str) -> dict:
        """
        Ejecuta la corrección y traducción de los segmentos de transcripción.

        Args:
            video_id:             Identificador único del vídeo.
            transcript_raw_path:  Ruta al transcript_raw.json (salida etapa 03).

        Returns:
            dict con claves:
                - transcript_translated (str): ruta a transcript_translated.json
                - segments_count        (int): número de segmentos procesados
                - target_language       (str): idioma de destino usado
        """
        raw_path = Path(transcript_raw_path)
        if not raw_path.exists():
            raise FileNotFoundError(f"Transcripción raw no encontrada: {raw_path}")

        # Cargar transcript_raw
        with open(raw_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)

        segments: List[dict] = transcript_data.get("segments", [])
        if not segments:
            raise ValueError(f"El fichero {raw_path.name} no contiene segmentos.")

        logger.info(
            f"[{video_id}] Iniciando traducción de {len(segments)} segmentos "
            f"con modelo '{self.model}' en {self.ollama_base_url}..."
        )

        # Crear directorio de salida
        stage_dir = self.work_dir / video_id / "04_traduccion"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Construir cliente Ollama
        client = self._build_client()

        # Procesar en chunks para no saturar la ventana de contexto del 3B
        translated_segments = []
        chunks = [
            segments[i : i + MAX_CHUNK_SEGMENTS]
            for i in range(0, len(segments), MAX_CHUNK_SEGMENTS)
        ]
        total_chunks = len(chunks)

        for idx, chunk in enumerate(chunks, 1):
            logger.info(f"[{video_id}] Traduciendo lote {idx}/{total_chunks} ({len(chunk)} segmentos)...")
            translated_chunk = self._translate_chunk(client, chunk)
            translated_segments.extend(translated_chunk)

        # Ensamblar resultado final
        result = {
            "video_id":           video_id,
            "source_language":    transcript_data.get("language", "unknown"),
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
            f"[{video_id}] Traducción completada: {len(translated_segments)} segmentos "
            f"guardados en {out_path.name}"
        )

        return {
            "transcript_translated": str(out_path),
            "segments_count":        len(translated_segments),
            "target_language":       "es",
        }
