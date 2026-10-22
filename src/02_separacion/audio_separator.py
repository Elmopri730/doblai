import os
import shutil
import logging
from pathlib import Path
from audio_separator.separator import Separator

logger = logging.getLogger("dubbing.separation")

class AudioSeparatorWrapper:
    """
    Clase para procesar la etapa 02 (Separación de voz/música).
    Usa la biblioteca audio-separator para aislar la voz de la música de fondo.
    """
    def __init__(self, config: dict):
        self.config = config
        self.work_dir = Path(config.get("pipeline", {}).get("work_dir", "./work"))
        
        # Parámetros del hardware
        self.hardware_config = config.get("pipeline", {}).get("hardware", {})
        self.modo = self.hardware_config.get("modo", "auto").lower()
        self.vram_minima = self.hardware_config.get("vram_minima_gb", 8)

        # Parámetros de la etapa
        self.stage_config = config.get("stages", {}).get("separation", {})
        raw_model = self.stage_config.get("model", "UVR-MDX-NET-Inst_HQ_3.onnx")
        if raw_model in ["htdemucs", "demucs", ""]:
            raw_model = "UVR-MDX-NET-Inst_HQ_3.onnx"
        self.model_name = raw_model

    def detect_onnx_provider(self):
        """Detecta los proveedores de ejecución de ONNX disponibles y retorna el adecuado."""
        try:
            import onnxruntime as ort
            if hasattr(ort, "get_available_providers"):
                available_providers = ort.get_available_providers()
                logger.info(f"Proveedores ONNX Runtime disponibles: {available_providers}")
                if self.modo == "gpu" or self.modo == "auto":
                    if "CUDAExecutionProvider" in available_providers:
                        logger.info("CUDAExecutionProvider seleccionado.")
                        return "CUDAExecutionProvider"
                if "CPUExecutionProvider" in available_providers:
                    return "CPUExecutionProvider"
        except Exception as e:
            logger.warning(f"Aviso al detectar proveedores ONNX: {e}")
        return None

    def run(self, video_id: str, raw_audio_path_str: str) -> dict:
        """
        Ejecuta la etapa de separación de fuentes de audio.
        Retorna la ruta de vocals.wav y accompaniment.wav.
        """
        raw_audio_path = Path(raw_audio_path_str)
        if not raw_audio_path.exists():
            raise FileNotFoundError(f"Audio crudo no encontrado: {raw_audio_path}")

        stage_dir = self.work_dir / video_id / "02_separacion"
        stage_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[{video_id}] Iniciando etapa 02 (Separación de voz/música)...")

        # Inicializar el Separador de audio
        separator = Separator(output_dir=str(stage_dir))
        provider = self.detect_onnx_provider()
        if provider:
            separator.onnx_execution_providers = [provider]
        
        logger.info(f"[{video_id}] Cargando modelo de separación {self.model_name}...")
        separator.load_model(model_filename=self.model_name)
        
        logger.info(f"[{video_id}] Procesando separación de audio (esto puede tomar un momento)...")
        # Realizar la separación
        output_files = separator.separate(str(raw_audio_path))
        logger.info(f"[{video_id}] Separación terminada. Archivos generados: {output_files}")

        # Mapear los resultados
        # La biblioteca devuelve una tupla/lista de archivos generados.
        # Generalmente, uno contiene la voz y otro el instrumental.
        vocals_output_path = stage_dir / "vocals.wav"
        accompaniment_output_path = stage_dir / "accompaniment.wav"

        vocals_found = False
        accompaniment_found = False

        for file_path_str in output_files:
            file_path = stage_dir / file_path_str
            # El nombre suele contener "(vocals)" o "(instrumental)" o palabras clave
            name_lower = file_path.name.lower()
            
            # Clasificar y mover los archivos a su nombre final bajo el contrato
            if "(vocals)" in name_lower or "(voice)" in name_lower:
                shutil.move(str(file_path), str(vocals_output_path))
                vocals_found = True
            elif "(instrumental)" in name_lower or "_(instrumental)" in name_lower or "no_vocals" in name_lower:
                shutil.move(str(file_path), str(accompaniment_output_path))
                accompaniment_found = True
            else:
                # Fallback por si la nomenclatura varía
                pass

        # Si por alguna razón la clasificación por nombre falla, hacemos mapeo por descarte
        if not vocals_found or not accompaniment_found:
            logger.warning(f"[{video_id}] No se pudo determinar con certeza cuál es la pista de voz o instrumental analizando los nombres. Mapeando por orden por defecto.")
            remaining_files = [stage_dir / f for f in os.listdir(stage_dir) if f.endswith(".wav") and f not in ("vocals.wav", "accompaniment.wav")]
            
            if len(remaining_files) >= 2:
                # UVR-MDX-NET-Inst_HQ_3 genera:
                # Archivo 1: Instrumental (Primary)
                # Archivo 2: Vocals (Secondary)
                # Así que ordenamos alfabéticamente o por orden de retorno.
                # Si el primer archivo de salida es el instrumental:
                inst_candidate = stage_dir / output_files[0]
                vocals_candidate = stage_dir / output_files[1]
                
                if inst_candidate.exists() and vocals_candidate.exists():
                    shutil.move(str(inst_candidate), str(accompaniment_output_path))
                    shutil.move(str(vocals_candidate), str(vocals_output_path))
                    vocals_found = True
                    accompaniment_found = True

        if not vocals_found or not accompaniment_found:
            raise RuntimeError(f"[{video_id}] Error: no se pudieron generar o identificar los archivos vocals.wav y accompaniment.wav.")

        logger.info(f"[{video_id}] Pista de voz guardada en: {vocals_output_path.name}")
        logger.info(f"[{video_id}] Pista de fondo/música guardada en: {accompaniment_output_path.name}")
        logger.info(f"[{video_id}] Etapa 02 (Separación) completada con éxito.")

        return {
            "vocals": str(vocals_output_path),
            "accompaniment": str(accompaniment_output_path)
        }
