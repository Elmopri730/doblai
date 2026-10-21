import os
import sys
import argparse
import yaml
import logging
import time
from datetime import datetime
from pathlib import Path
import json

# Añadir directorio actual al path
sys.path.append(str(Path(__file__).parent))

from src.config_schema import GlobalConfig
import importlib
VideoProcessor = importlib.import_module("src.01_ingesta.video_processor").VideoProcessor
AudioSeparatorWrapper = importlib.import_module("src.02_separacion.audio_separator").AudioSeparatorWrapper
Transcriber = importlib.import_module("src.03_transcripcion.transcriber").Transcriber
LLMTranslator = importlib.import_module("src.04_traduccion.llm_translator").LLMTranslator
VoiceSynthesizer = importlib.import_module("src.05_tts.voice_synthesizer").VoiceSynthesizer
FinalMixer = importlib.import_module("src.06_mezcla.final_mixer").FinalMixer

# Formateador para JSON Lines
class JsonLinesFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "video_id": getattr(record, "video_id", "system"),
            "stage": getattr(record, "stage", "orchestrator"),
            "level": record.levelname,
            "message": record.getMessage()
        }
        duration = getattr(record, "duration_s", None)
        if duration is not None:
            log_entry["duration_s"] = duration
        return json.dumps(log_entry, ensure_ascii=False)

def setup_logger(log_level_str="INFO"):
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    
    # Logger raíz del pipeline
    root_logger = logging.getLogger("dubbing")
    root_logger.setLevel(log_level)
    
    # Limpiar manejadores existentes
    root_logger.handlers = []

    # Manejador de consola (Legible por humanos)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", 
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    return root_logger

class PipelineOrchestrator:
    def __init__(self, config_path: str, stages_filter: str = None, reprocess: bool = False):
        self.config_path = Path(config_path)
        self.reprocess = reprocess
        
        # Cargar y validar configuración
        self.config = self.load_config()
        self.work_dir = Path(self.config.pipeline.work_dir)
        self.output_dir = Path(self.config.pipeline.output_dir)
        self.logs_dir = Path("./logs")

        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Configurar filtros de etapas
        if stages_filter:
            self.stages_to_run = [s.strip() for s in stages_filter.split(",")]
        else:
            self.stages_to_run = [
                "01_ingesta", "02_separacion", "03_transcripcion",
                "04_traduccion", "05_tts", "06_mezcla"
            ]

        # Mapear nombres de etapas a sus índices
        self.stages_map = {
            "01_ingesta":       "01_ingesta",
            "02_separacion":    "02_separacion",
            "03_transcripcion": "03_transcripcion",
            "04_traduccion":    "04_traduccion",
            "05_tts":           "05_tts",
            "06_mezcla":        "06_mezcla",
        }

    def load_config(self) -> GlobalConfig:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
            
        # Pydantic valida y autocompleta con valores por defecto si faltan campos
        return GlobalConfig(**yaml_data)

    def run_stage_with_logs(self, stage_name: str, video_id: str, run_func, *args, **kwargs):
        """Ejecuta una función de etapa y gestiona los logs JSON de forma aislada para esa etapa."""
        logger = logging.getLogger(f"dubbing.{stage_name}")
        stage_log_dir = self.logs_dir / video_id
        stage_log_dir.mkdir(parents=True, exist_ok=True)
        stage_log_file = stage_log_dir / f"{stage_name}.log"

        # Crear handler de archivo en formato JSON Lines
        file_handler = logging.FileHandler(stage_log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        json_formatter = JsonLinesFormatter()
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)

        # Guardar metadatos del registro
        extra_attrs = {"video_id": video_id, "stage": stage_name}
        
        # Filtro adaptador de logging para inyectar atributos extra
        class ExtraFilter(logging.Filter):
            def filter(self, record):
                for k, v in extra_attrs.items():
                    setattr(record, k, v)
                return True
        
        extra_filter = ExtraFilter()
        logger.addFilter(extra_filter)

        start_time = time.time()
        result = None
        retries = self.config.pipeline.procesamiento_lote.reintentos_por_etapa
        
        for attempt in range(1, retries + 1):
            try:
                logger.info(f"Iniciando intento {attempt} de la etapa {stage_name}...")
                result = run_func(*args, **kwargs)
                break
            except Exception as e:
                logger.error(f"Error en intento {attempt}: {e}", exc_info=True)
                if attempt == retries:
                    # Remover handlers antes de lanzar el error
                    logger.removeHandler(file_handler)
                    logger.removeFilter(extra_filter)
                    file_handler.close()
                    raise e
                time.sleep(2)  # Pequeño delay entre reintentos

        duration = time.time() - start_time
        logger.info(f"Etapa {stage_name} completada en {duration:.2f} segundos.", extra={"duration_s": duration})

        # Limpiar handlers
        logger.removeHandler(file_handler)
        logger.removeFilter(extra_filter)
        file_handler.close()

        return result

    def process_single_video(self, video_path_str: str) -> bool:
        video_path = Path(video_path_str)
        if not video_path.exists():
            print(f"Error: Archivo de vídeo no encontrado en {video_path}")
            return False

        # Instanciar el procesador de ingesta temporalmente para obtener el ID del vídeo
        # y no duplicar lógica
        config_dict = self.config.model_dump()
        processor = VideoProcessor(config_dict)
        video_id = processor.get_video_id(video_path)

        print(f"\n========================================================")
        print(f"PROCESANDO VÍDEO: {video_path.name}")
        print(f"VIDEO ID: {video_id}")
        print(f"========================================================\n")

        outputs = {}
        try:
            # ETAPA 01: INGESTA
            if "01_ingesta" in self.stages_to_run:
                ingesta_out_dir = self.work_dir / video_id / "01_ingesta"
                vocals_raw_path = ingesta_out_dir / "vocals_raw.wav"
                
                # Si ya existe y no se fuerza reprocesamiento, se salta (idempotencia)
                if vocals_raw_path.exists() and not self.reprocess:
                    print(f"[!] Etapa 01_ingesta ya completada. Saltando debido a idempotencia...")
                    # Simular la salida
                    outputs["01_ingesta"] = {
                        "video_id": video_id,
                        "vocals_raw": str(vocals_raw_path),
                        "video_normalized": str(ingesta_out_dir / "video_normalized.mp4")
                    }
                else:
                    res = self.run_stage_with_logs(
                        "01_ingesta", 
                        video_id, 
                        processor.run, 
                        video_path
                    )
                    outputs["01_ingesta"] = res
            else:
                # Si se salta la ingesta pero se requiere para etapas siguientes,
                # cargamos rutas por defecto
                outputs["01_ingesta"] = {
                    "video_id": video_id,
                    "vocals_raw": str(self.work_dir / video_id / "01_ingesta" / "vocals_raw.wav"),
                    "video_normalized": str(self.work_dir / video_id / "01_ingesta" / "video_normalized.mp4")
                }

            # ETAPA 02: SEPARACIÓN DE VOZ/MÚSICA
            if "02_separacion" in self.stages_to_run:
                separacion_out_dir = self.work_dir / video_id / "02_separacion"
                vocals_path = separacion_out_dir / "vocals.wav"
                
                if vocals_path.exists() and not self.reprocess:
                    print(f"[!] Etapa 02_separacion ya completada. Saltando debido a idempotencia...")
                    outputs["02_separacion"] = {
                        "vocals": str(vocals_path),
                        "accompaniment": str(separacion_out_dir / "accompaniment.wav")
                    }
                else:
                    separator = AudioSeparatorWrapper(config_dict)
                    vocals_raw = outputs["01_ingesta"]["vocals_raw"]
                    
                    res = self.run_stage_with_logs(
                        "02_separacion",
                        video_id,
                        separator.run,
                        video_id,
                        vocals_raw
                    )
                    outputs["02_separacion"] = res

            # ETAPA 03: TRANSCRIPCIÓN
            if "03_transcripcion" in self.stages_to_run:
                transcripcion_out = self.work_dir / video_id / "03_transcripcion" / "transcript_raw.json"

                if transcripcion_out.exists() and not self.reprocess:
                    print(f"[!] Etapa 03_transcripcion ya completada. Saltando (idempotencia)...")
                    outputs["03_transcripcion"] = {
                        "transcript_raw":  str(transcripcion_out),
                        "language":        "unknown",
                        "segments_count":  0,
                    }
                else:
                    transcriber = Transcriber(config_dict)
                    vocals_path = outputs.get("02_separacion", {}).get(
                        "vocals",
                        str(self.work_dir / video_id / "02_separacion" / "vocals.wav")
                    )
                    res = self.run_stage_with_logs(
                        "03_transcripcion",
                        video_id,
                        transcriber.run,
                        video_id,
                        vocals_path,
                    )
                    outputs["03_transcripcion"] = res

            # ETAPA 04: CORRECCIÓN Y TRADUCCIÓN
            if "04_traduccion" in self.stages_to_run:
                traduccion_out = self.work_dir / video_id / "04_traduccion" / "transcript_translated.json"

                if traduccion_out.exists() and not self.reprocess:
                    print(f"[!] Etapa 04_traduccion ya completada. Saltando (idempotencia)...")
                    outputs["04_traduccion"] = {
                        "transcript_translated": str(traduccion_out),
                        "segments_count":         0,
                        "target_language":         "es",
                    }
                else:
                    translator = LLMTranslator(config_dict)
                    raw_transcript = outputs.get("03_transcripcion", {}).get(
                        "transcript_raw",
                        str(self.work_dir / video_id / "03_transcripcion" / "transcript_raw.json")
                    )
                    res = self.run_stage_with_logs(
                        "04_traduccion",
                        video_id,
                        translator.run,
                        video_id,
                        raw_transcript,
                    )
                    outputs["04_traduccion"] = res

            # ETAPA 05: SÍNTESIS DE VOZ (TTS + CLONACIÓN)
            if "05_tts" in self.stages_to_run:
                tts_out = self.work_dir / video_id / "05_tts" / "dubbed_voice_raw.wav"

                if tts_out.exists() and not self.reprocess:
                    print(f"[!] Etapa 05_tts ya completada. Saltando (idempotencia)...")
                    outputs["05_tts"] = {
                        "dubbed_voice_raw": str(tts_out),
                        "tts_manifest": str(self.work_dir / video_id / "05_tts" / "tts_manifest.json"),
                        "segments_synthesized": 0,
                    }
                else:
                    synthesizer = VoiceSynthesizer(config_dict)
                    vocals_path = outputs.get("02_separacion", {}).get(
                        "vocals",
                        str(self.work_dir / video_id / "02_separacion" / "vocals.wav")
                    )
                    translated_path = outputs.get("04_traduccion", {}).get(
                        "transcript_translated",
                        str(self.work_dir / video_id / "04_traduccion" / "transcript_translated.json")
                    )
                    source_lang = outputs.get("03_transcripcion", {}).get("language", "en")
                    res = self.run_stage_with_logs(
                        "05_tts",
                        video_id,
                        synthesizer.run,
                        video_id,
                        vocals_path,
                        translated_path,
                        source_lang,
                    )
                    outputs["05_tts"] = res

            # ETAPA 06: MEZCLA FINAL Y EXPORTACIÓN
            if "06_mezcla" in self.stages_to_run:
                output_video_pattern = self.output_dir / video_id
                already_done = list(output_video_pattern.glob("*_doblado_es.mp4")) if output_video_pattern.exists() else []

                if already_done and not self.reprocess:
                    print(f"[!] Etapa 06_mezcla ya completada. Saltando (idempotencia)...")
                    outputs["06_mezcla"] = {
                        "output_video": str(already_done[0]),
                        "subtitles": [],
                    }
                else:
                    mixer = FinalMixer(config_dict)
                    res = self.run_stage_with_logs(
                        "06_mezcla",
                        video_id,
                        mixer.run,
                        video_id,
                        outputs.get("01_ingesta", {}).get("video_normalized",
                            str(self.work_dir / video_id / "01_ingesta" / "video_normalized.mp4")),
                        outputs.get("02_separacion", {}).get("accompaniment",
                            str(self.work_dir / video_id / "02_separacion" / "accompaniment.wav")),
                        outputs.get("05_tts", {}).get("dubbed_voice_raw",
                            str(self.work_dir / video_id / "05_tts" / "dubbed_voice_raw.wav")),
                        outputs.get("05_tts", {}).get("tts_manifest",
                            str(self.work_dir / video_id / "05_tts" / "tts_manifest.json")),
                        video_path.name,
                    )
                    outputs["06_mezcla"] = res

            print(f"\n[+] Procesamiento del vídeo {video_path.name} completado con éxito.")
            print(f"    - Transcripción raw en:   {outputs.get('03_transcripcion', {}).get('transcript_raw')}")
            print(f"    - Traducción ES en:        {outputs.get('04_traduccion', {}).get('transcript_translated')}")
            print(f"    - Voz doblada en:          {outputs.get('05_tts', {}).get('dubbed_voice_raw')}")
            print(f"    - Vídeo final en:           {outputs.get('06_mezcla', {}).get('output_video')}")
            print(f"    - Subtítulos:              {outputs.get('06_mezcla', {}).get('subtitles')}\n")
            return True

        except Exception as e:
            print(f"\n[❌] ERROR CRÍTICO procesando el vídeo {video_path.name}: {e}\n", file=sys.stderr)
            return False

    def process_batch(self, batch_dir_str: str):
        batch_dir = Path(batch_dir_str)
        if not batch_dir.exists() or not batch_dir.is_dir():
            print(f"Error: Directorio de lotes inválido en {batch_dir}")
            return

        print(f"\n========================================================")
        print(f"INICIANDO PROCESAMIENTO POR LOTES: {batch_dir}")
        print(f"========================================================\n")

        # Buscar todos los archivos de video soportados
        extensions = [".mp4", ".mkv", ".mov", ".avi"]
        video_files = [p for p in batch_dir.iterdir() if p.suffix.lower() in extensions]

        if not video_files:
            print("No se encontraron vídeos para procesar en el directorio.")
            return

        print(f"Se detectaron {len(video_files)} vídeos para procesar.")
        
        batch_report = {
            "start_time": datetime.now().isoformat(),
            "total_videos": len(video_files),
            "completed": [],
            "failed": []
        }

        for idx, video_path in enumerate(video_files, 1):
            print(f"\n[Vídeo {idx} de {len(video_files)}] Iniciando...")
            success = self.process_single_video(str(video_path))
            
            # Tolerancia a fallos: un video fallido no detiene el lote (RF-13.3)
            report_entry = {
                "filename": video_path.name,
                "timestamp": datetime.now().isoformat()
            }
            if success:
                batch_report["completed"].append(report_entry)
            else:
                batch_report["failed"].append(report_entry)

        batch_report["end_time"] = datetime.now().isoformat()
        
        # Escribir reporte de lote
        report_path = self.logs_dir / "batch_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(batch_report, f, indent=2, ensure_ascii=False)

        print(f"\n========================================================")
        print(f"PROCESAMIENTO POR LOTES FINALIZADO")
        print(f"Completados con éxito: {len(batch_report['completed'])}")
        print(f"Fallidos: {len(batch_report['failed'])}")
        print(f"Reporte guardado en: {report_path}")
        print(f"========================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Automatizado de Doblaje de Vídeo")
    parser.add_argument("--input", type=str, help="Ruta al archivo de vídeo individual a procesar")
    parser.add_argument("--batch", type=str, help="Ruta al directorio con múltiples vídeos a procesar")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Ruta al archivo de configuración config.yaml")
    parser.add_argument("--stages", type=str, help="Filtro de etapas separadas por coma a ejecutar (ej. '01_ingesta,02_separacion')")
    parser.add_argument("--reprocess", action="store_true", help="Fuerza el reprocesamiento de las etapas aunque ya existan salidas")

    args = parser.parse_args()

    # Inicializar logs del sistema
    setup_logger("INFO")

    if not args.input and not args.batch:
        parser.print_help()
        sys.exit(1)

    orchestrator = PipelineOrchestrator(
        config_path=args.config,
        stages_filter=args.stages,
        reprocess=args.reprocess
    )

    if args.input:
        orchestrator.process_single_video(args.input)
    elif args.batch:
        orchestrator.process_batch(args.batch)
