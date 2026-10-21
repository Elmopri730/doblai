"""
webapp/app.py — Backend FastAPI para el Sistema de Doblaje Automático.

Expone:
  GET  /                              → Sirve la interfaz HTML
  POST /api/upload                    → Sube vídeo(s) e inicia el pipeline
  GET  /api/jobs                      → Lista todos los trabajos con estado
  GET  /api/jobs/{job_id}/events      → Server-Sent Events (SSE) de progreso en tiempo real
  GET  /api/jobs/{job_id}/download/{file_type}  → Descarga de resultado
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
import uuid
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── Ajustar sys.path para importar los módulos del pipeline ──────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("dubbing.webapp")

# ── Configuración ─────────────────────────────────────────────────────────────
CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


PIPELINE_CONFIG = load_config()
WORK_DIR   = Path(PIPELINE_CONFIG.get("pipeline", {}).get("work_dir",  "./work"))
OUTPUT_DIR = Path(PIPELINE_CONFIG.get("pipeline", {}).get("output_dir", "./output"))
UPLOAD_DIR = ROOT / "input"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Estado en memoria de los trabajos ─────────────────────────────────────────

STAGES_INFO: List[dict] = [
    {"key": "01_ingesta",       "label": "Ingesta y análisis del vídeo"},
    {"key": "02_separacion",    "label": "Separación de voz y música"},
    {"key": "03_transcripcion", "label": "Transcripción de voz a texto"},
    {"key": "04_traduccion",    "label": "Corrección y traducción al español"},
    {"key": "05_tts",           "label": "Síntesis de voz en español (TTS)"},
    {"key": "06_mezcla",        "label": "Mezcla de audio y vídeo final"},
]

TOTAL_STAGES = len(STAGES_INFO)

jobs: Dict[str, dict] = {}   # job_id → estado del trabajo
events_queues: Dict[str, asyncio.Queue] = {}   # job_id → cola SSE por cliente


def new_job(filename: str, video_path: Path) -> str:
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "id":          job_id,
        "filename":    filename,
        "status":      "queued",       # queued | running | done | error
        "current_stage_idx": -1,
        "current_stage":    "",
        "progress_pct":     0,
        "message":     "En cola...",
        "created_at":  time.time(),
        "outputs":     {},
        "error":       None,
        "video_path":  str(video_path),
    }
    events_queues[job_id] = asyncio.Queue()
    return job_id


def emit(job_id: str, payload: dict):
    """Emite un evento SSE poniendo el payload en la cola del job."""
    if job_id in events_queues:
        try:
            events_queues[job_id].put_nowait(payload)
        except asyncio.QueueFull:
            pass
    # También actualizar estado global
    jobs[job_id].update(payload)


def run_pipeline_thread(job_id: str):
    """
    Función que corre el pipeline en un thread de fondo.
    INTELIGENTE: Si se reintenta un trabajo, detecta qué etapas ya tienen
    archivos generados en disco y las salta, retomando desde la que falló.
    """
    import importlib
    VideoProcessor        = importlib.import_module("src.01_ingesta.video_processor").VideoProcessor
    AudioSeparatorWrapper = importlib.import_module("src.02_separacion.audio_separator").AudioSeparatorWrapper
    Transcriber           = importlib.import_module("src.03_transcripcion.transcriber").Transcriber
    LLMTranslator         = importlib.import_module("src.04_traduccion.llm_translator").LLMTranslator

    config = PIPELINE_CONFIG
    video_path = Path(jobs[job_id]["video_path"])
    outputs = {}

    # Intentar recuperar el video_id previo (si es un retry)
    video_id = jobs[job_id].get("video_id")

    def update_stage(idx: int, message: str):
        stage = STAGES_INFO[idx]
        pct = int(((idx) / TOTAL_STAGES) * 100)
        emit(job_id, {
            "status":           "running",
            "current_stage_idx": idx,
            "current_stage":     stage["label"],
            "progress_pct":      pct,
            "message":           message,
        })

    def stage_done(idx: int):
        pct = int(((idx + 1) / TOTAL_STAGES) * 100)
        emit(job_id, {
            "current_stage_idx": idx,
            "progress_pct":      pct,
        })

    def _check_stage_files(vid: str, stage: str, required_files: list) -> bool:
        """Comprueba si todos los archivos requeridos de una etapa existen en disco."""
        if not vid:
            return False
        for f in required_files:
            p = WORK_DIR / vid / stage / f
            if not p.exists():
                return False
        return True

    try:
        # ── Etapa 01: Ingesta ────────────────────────────────────────────────
        stage01_done = video_id and _check_stage_files(
            video_id, "01_ingesta", ["vocals_raw.wav", "video_normalized.mp4"]
        )
        if stage01_done:
            update_stage(0, "⏩ Ingesta ya completada, saltando...")
            outputs["01_ingesta"] = {
                "video_id": video_id,
                "vocals_raw": str(WORK_DIR / video_id / "01_ingesta" / "vocals_raw.wav"),
                "video_normalized": str(WORK_DIR / video_id / "01_ingesta" / "video_normalized.mp4"),
            }
            logger.info(f"[{job_id}] Etapa 01 ya existe en disco, saltando.")
        else:
            update_stage(0, "Analizando el vídeo y extrayendo audio...")
            processor = VideoProcessor(config)
            res = processor.run(video_path)
            outputs["01_ingesta"] = res
            video_id = res["video_id"]
            jobs[job_id]["video_id"] = video_id
        stage_done(0)

        # ── Etapa 02: Separación ─────────────────────────────────────────────
        stage02_done = video_id and _check_stage_files(
            video_id, "02_separacion", ["vocals.wav", "accompaniment.wav"]
        )
        if stage02_done:
            update_stage(1, "⏩ Separación ya completada, saltando...")
            outputs["02_separacion"] = {
                "vocals": str(WORK_DIR / video_id / "02_separacion" / "vocals.wav"),
                "accompaniment": str(WORK_DIR / video_id / "02_separacion" / "accompaniment.wav"),
            }
            logger.info(f"[{job_id}] Etapa 02 ya existe en disco, saltando.")
        else:
            update_stage(1, "Separando la voz de la música de fondo...")
            separator = AudioSeparatorWrapper(config)
            res = separator.run(video_id, outputs["01_ingesta"]["vocals_raw"])
            outputs["02_separacion"] = res
        stage_done(1)

        # ── Etapa 03: Transcripción ──────────────────────────────────────────
        stage03_done = video_id and _check_stage_files(
            video_id, "03_transcripcion", ["transcript_raw.json"]
        )
        if stage03_done:
            update_stage(2, "⏩ Transcripción ya completada, saltando...")
            import json as _json
            raw_path = WORK_DIR / video_id / "03_transcripcion" / "transcript_raw.json"
            with open(raw_path, "r", encoding="utf-8") as _f:
                raw_data = _json.load(_f)
            outputs["03_transcripcion"] = {
                "transcript_raw": str(raw_path),
                "language": raw_data.get("language", "en"),
            }
            logger.info(f"[{job_id}] Etapa 03 ya existe en disco, saltando.")
        else:
            update_stage(2, "Transcribiendo la voz a texto con Whisper...")
            transcriber = Transcriber(config)
            res = transcriber.run(video_id, outputs["02_separacion"]["vocals"])
            outputs["03_transcripcion"] = res
        stage_done(2)

        # ── Etapa 04: Traducción ─────────────────────────────────────────────
        stage04_done = video_id and _check_stage_files(
            video_id, "04_traduccion", ["transcript_translated.json"]
        )
        if stage04_done:
            update_stage(3, "⏩ Traducción ya completada, saltando...")
            outputs["04_traduccion"] = {
                "transcript_translated": str(WORK_DIR / video_id / "04_traduccion" / "transcript_translated.json"),
            }
            logger.info(f"[{job_id}] Etapa 04 ya existe en disco, saltando.")
        else:
            update_stage(3, "Traduciendo al español con IA local...")
            translator = LLMTranslator(config)
            res = translator.run(video_id, outputs["03_transcripcion"]["transcript_raw"])
            outputs["04_traduccion"] = res
        stage_done(3)

        # ── Etapa 05: Síntesis de voz (TTS) ────────────────────────────────────
        update_stage(4, "Sintetizando la voz en español con clonación...")
        VoiceSynthesizer = importlib.import_module("src.05_tts.voice_synthesizer").VoiceSynthesizer
        synthesizer = VoiceSynthesizer(config)
        vocals_path = outputs.get("02_separacion", {}).get(
            "vocals",
            str(WORK_DIR / video_id / "02_separacion" / "vocals.wav")
        )
        translated_path = outputs.get("04_traduccion", {}).get(
            "transcript_translated",
            str(WORK_DIR / video_id / "04_traduccion" / "transcript_translated.json")
        )
        source_lang = outputs.get("03_transcripcion", {}).get("language", "en")
        res = synthesizer.run(video_id, vocals_path, translated_path, source_lang)
        outputs["05_tts"] = res
        stage_done(4)

        # ── Etapa 06: Mezcla final ────────────────────────────────────────────
        update_stage(5, "Mezclando voz, música y exportando vídeo final...")
        FinalMixer = importlib.import_module("src.06_mezcla.final_mixer").FinalMixer
        mixer = FinalMixer(config)
        res = mixer.run(
            video_id,
            outputs.get("01_ingesta", {}).get("video_normalized",
                str(WORK_DIR / video_id / "01_ingesta" / "video_normalized.mp4")),
            outputs.get("02_separacion", {}).get("accompaniment",
                str(WORK_DIR / video_id / "02_separacion" / "accompaniment.wav")),
            outputs["05_tts"]["dubbed_voice_raw"],
            outputs["05_tts"]["tts_manifest"],
            jobs[job_id]["filename"],
        )
        outputs["06_mezcla"] = res
        stage_done(5)

        # ── Finalizado ───────────────────────────────────────────────────────
        emit(job_id, {
            "status":       "done",
            "progress_pct": 100,
            "message":      "¡Doblaje completado! Vídeo listo para descargar.",
            "outputs":      {
                "output_video":            outputs["06_mezcla"].get("output_video"),
                "subtitles":               outputs["06_mezcla"].get("subtitles", []),
                "transcript_translated":   outputs["04_traduccion"].get("transcript_translated"),
            },
        })

    except Exception as e:
        logger.error(f"[{job_id}] Error en el pipeline: {e}", exc_info=True)
        emit(job_id, {
            "status":  "error",
            "message": f"Error: {str(e)}",
            "error":   str(e),
        })


# ── App FastAPI ────────────────────────────────────────────────────────────────
app = FastAPI(title="Sistema de Doblaje Automático", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos (CSS, JS adicionales si los hay)
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Sirve la interfaz web principal."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=503, detail="UI no encontrada.")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Recibe un fichero de vídeo, lo guarda y lanza el pipeline en background."""
    allowed = {".mp4", ".mkv", ".mov", ".avi"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Formato '{ext}' no soportado. Usa MP4, MKV, MOV o AVI.")

    # Guardar fichero
    safe_name = f"{uuid.uuid4().hex}{ext}"
    dest = UPLOAD_DIR / safe_name
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)

    # Registrar trabajo y arrancar thread
    job_id = new_job(file.filename, dest)
    thread = threading.Thread(target=run_pipeline_thread, args=(job_id,), daemon=True)
    thread.start()

    return {"job_id": job_id, "filename": file.filename}


@app.get("/api/jobs")
async def list_jobs():
    """Devuelve la lista de todos los trabajos y su estado actual."""
    return list(reversed(list(jobs.values())))


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    """
    Server-Sent Events: envía actualizaciones de progreso en tiempo real
    al navegador del cliente sin necesidad de polling.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado.")

    queue = events_queues.get(job_id)

    async def event_generator():
        # Enviar estado inicial inmediatamente
        yield f"data: {json.dumps(jobs[job_id])}\n\n"

        while True:
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=25)
                yield f"data: {json.dumps(payload)}\n\n"
                # Si el job terminó, cerramos el stream
                if payload.get("status") in ("done", "error"):
                    break
            except asyncio.TimeoutError:
                # Heartbeat para mantener la conexión viva
                yield ": heartbeat\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/jobs/{job_id}/download/video")
async def download_video(job_id: str):
    """Descarga el vídeo final doblado en MP4."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado.")

    video_path = jobs[job_id].get("outputs", {}).get("output_video")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="El vídeo doblado aún no está disponible.")

    stem = Path(jobs[job_id].get("filename", "video")).stem
    return FileResponse(
        path=video_path,
        filename=f"{stem}_doblado_es.mp4",
        media_type="video/mp4",
    )


@app.get("/api/jobs/{job_id}/download/subtitle/{fmt}")
async def download_subtitle(job_id: str, fmt: str):
    """Descarga los subtítulos en el formato indicado (srt, vtt, ass)."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado.")
    if fmt not in ("srt", "vtt", "ass"):
        raise HTTPException(status_code=400, detail="Formato no válido.")

    subtitles: list = jobs[job_id].get("outputs", {}).get("subtitles", [])
    match = next((s for s in subtitles if s.endswith(f".{fmt}")), None)
    if not match or not Path(match).exists():
        raise HTTPException(status_code=404, detail=f"Subtítulos .{fmt} no disponibles.")

    stem = Path(jobs[job_id].get("filename", "video")).stem
    return FileResponse(
        path=match,
        filename=f"{stem}_es.{fmt}",
        media_type="text/plain",
    )


@app.get("/api/jobs/{job_id}/download/transcript")
async def download_transcript(job_id: str):
    """Descarga el transcript traducido en JSON."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado.")

    transcript_path = jobs[job_id].get("outputs", {}).get("transcript_translated")
    if not transcript_path or not Path(transcript_path).exists():
        raise HTTPException(status_code=404, detail="El fichero de transcripción aún no está disponible.")

    return FileResponse(
        path=transcript_path,
        filename=f"doblaje_{job_id[:8]}_es.json",
        media_type="application/json",
    )


@app.post("/api/jobs/{job_id}/retry")
async def retry_job(job_id: str):
    """Reintenta un trabajo que ha fallado, reseteando su estado."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado.")
    
    job = jobs[job_id]
    if job["status"] not in ("error", "done"):
        raise HTTPException(status_code=400, detail="Solo se pueden reintentar trabajos fallidos o finalizados.")

    # Resetear estado pero mantener outputs y video_path (el pipeline reutilizará lo ya hecho)
    job["status"] = "queued"
    job["current_stage_idx"] = -1
    job["current_stage"] = ""
    job["progress_pct"] = 0
    job["message"] = "Reintentando..."
    job["error"] = None
    
    # Asegurar que hay una cola de eventos fresca
    events_queues[job_id] = asyncio.Queue()
    
    # Volver a lanzar el thread
    thread = threading.Thread(target=run_pipeline_thread, args=(job_id,), daemon=True)
    thread.start()
    
    return {"status": "ok"}


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Elimina el trabajo de la UI. Si no estaba completado, borra también los archivos del PC."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado.")
    
    job = jobs[job_id]
    
    # Lógica de borrado físico si no estaba "done"
    if job["status"] != "done":
        # Borrar vídeo original subido
        if job.get("video_path"):
            try:
                Path(job["video_path"]).unlink(missing_ok=True)
            except Exception:
                pass
        
        # Borrar carpeta de trabajo temporal
        if job.get("video_id"):
            work_folder = WORK_DIR / job["video_id"]
            if work_folder.exists():
                try:
                    shutil.rmtree(work_folder, ignore_errors=True)
                except Exception:
                    pass
    
    # Borrar de memoria
    if job_id in jobs:
        del jobs[job_id]
    if job_id in events_queues:
        del events_queues[job_id]
        
    return {"status": "deleted"}
