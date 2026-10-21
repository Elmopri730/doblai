from pydantic import BaseModel, Field
from typing import List, Optional

class HardwareConfig(BaseModel):
    modo: str = Field(default="auto", description="auto | gpu | cpu")
    vram_minima_gb: int = Field(default=8, description="VRAM mínima requerida")

class ProcesamientoLoteConfig(BaseModel):
    max_paralelo: int = Field(default=2, description="Límite de paralelismo para lotes")
    reintentos_por_etapa: int = Field(default=2, description="Número de reintentos por etapa")

class GpuAlquiladaConfig(BaseModel):
    proveedor: str = Field(default="runpod", description="runpod | vast.ai | lambda_labs | paperspace | coreweave")
    activar_solo_si_local_insuficiente: bool = Field(default=True)

class PipelineConfig(BaseModel):
    idioma_destino: str = Field(default="es")
    lip_sync: bool = Field(default=False)
    formatos_subtitulos: List[str] = Field(default=["srt", "vtt", "ass"])
    input_dir: str = Field(default="./input")
    work_dir: str = Field(default="./work")
    output_dir: str = Field(default="./output")
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    procesamiento_lote: ProcesamientoLoteConfig = Field(default_factory=ProcesamientoLoteConfig)
    gpu_alquilada: GpuAlquiladaConfig = Field(default_factory=GpuAlquiladaConfig)

class SeparationStageConfig(BaseModel):
    model: str = Field(default="UVR-MDX-NET-Inst_HQ_3.onnx")

class TranscriptionStageConfig(BaseModel):
    model: str = Field(default="whisper-large-v3")
    language: str = Field(default="auto")

class TtsStageConfig(BaseModel):
    engine: str = Field(default="coqui-xtts", description="Motor TTS: coqui-xtts")
    model_name: str = Field(default="tts_models/multilingual/multi-dataset/xtts_v2", description="Modelo XTTS v2 de Coqui")
    speaker_sample_min_s: int = Field(default=6, description="Duración mínima en segundos del sample de referencia de la voz")
    speaker_sample_max_s: int = Field(default=30, description="Duración máxima del sample de referencia")
    target_language: str = Field(default="es", description="Idioma de destino de la síntesis")
    speed: float = Field(default=1.0, description="Velocidad de la síntesis (1.0 = normal)")
    temperature: float = Field(default=0.75, description="Creatividad del modelo TTS")
    length_penalty: float = Field(default=1.0)
    repetition_penalty: float = Field(default=5.0)
    top_k: int = Field(default=50)
    top_p: float = Field(default=0.85)

class TranslationStageConfig(BaseModel):
    model: str = Field(default="qwen2.5:3b", description="Modelo Ollama para corrección y traducción")
    ollama_url: str = Field(default="http://localhost:11434", description="URL base de la instancia de Ollama")
    temperature: float = Field(default=0.2, description="Temperatura del LLM (0=determinista, 1=creativo)")
    max_retries: int = Field(default=3, description="Reintentos si la respuesta JSON es inválida")

class StagesConfig(BaseModel):
    separation: SeparationStageConfig = Field(default_factory=SeparationStageConfig)
    transcription: TranscriptionStageConfig = Field(default_factory=TranscriptionStageConfig)
    translation: TranslationStageConfig = Field(default_factory=TranslationStageConfig)
    tts: TtsStageConfig = Field(default_factory=TtsStageConfig)

class GlobalConfig(BaseModel):
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    stages: StagesConfig = Field(default_factory=StagesConfig)
