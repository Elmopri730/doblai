# CLAUDE.md — Sistema Automatizado de Doblaje de Vídeo con IA

> **Lee este archivo completo antes de cualquier acción.** Es la fuente de verdad del proyecto.

---

## 1. Resumen del proyecto

Sistema de doblaje automático de vídeo impulsado por IA con español como idioma de destino.
Transforma un vídeo en cualquier idioma en un doblaje al español de alta calidad mediante un pipeline
modular y desacoplado (extracción, separación, transcripción, corrección, traducción, adaptación, clonación, TTS, sincronización, lip sync opcional, mezcla y subtítulos), ejecutable localmente o en GPU alquilada sin APIs de pago.

---

## 2. Principios no negociables

- **100 % local y open source.** Ninguna dependencia de APIs de pago (p. ej. OpenAI, ElevenLabs, DeepL) ni suscripciones recurrentes.
- **Único coste admitido:** alquiler puntual de GPU en la nube (RunPod, Vast.ai, Lambda Labs, Paperspace, CoreWeave) si el hardware local es insuficiente.
- **Arquitectura desacoplada.** Cada módulo del pipeline es independiente y se comunica exclusivamente mediante archivos en disco bajo un contrato estricto. Se puede sustituir cualquier tecnología sin tocar el resto.
- **Investigación tecnológica previa.** Nunca se decide ni se fija una tecnología como definitiva sin realizar antes una fase de investigación comparativa documentada.

---

## 3. Metodología obligatoria de investigación tecnológica

**Antes de implementar o cambiar cualquier etapa del pipeline:**

1. Evaluar **como mínimo 3 alternativas** open source viables.
2. Comparar las tecnologías en una tabla usando los siguientes criterios obligatorios:
   - **Calidad:** Precisión de transcripción (WER), naturalidad de voz, similitud de speaker embedding, etc.
   - **Velocidad:** Tiempo de procesamiento por minuto de audio/vídeo en CPU y GPU.
   - **VRAM:** Requisitos mínimos y recomendados de memoria de vídeo (GB).
   - **Licencia:** Tipo de licencia libre (MIT, Apache 2.0, GPL, etc.) compatible con uso local sin costes.
   - **Comunidad:** Actividad del repositorio, frecuencia de actualización y mantenimiento.
   - **Integración:** Facilidad de uso (API Python, CLI, formatos de entrada/salida).

3. Documentar la comparación y la justificación de la alternativa seleccionada en **`/docs/decisiones-tecnicas.md`**.

> ⚠️ **No está permitido iniciar la codificación de una etapa sin haber completado y guardado previamente su informe en `/docs/decisiones-tecnicas.md`.**

---

## 4. Arquitectura del pipeline

Cada una de las etapas del pipeline se conecta mediante el intercambio de archivos intermedios en formato estandarizado.

| #  | Etapa (Requisito) | Entrada | Salida | Tecnologías Candidatas |
|----|-------------------|---------|--------|------------------------|
| 01 | **Ingesta de vídeo** (RF-01) | Archivo original (`.mp4`, `.mkv`, `.mov`, `.avi`) | `video_info.json` (metadatos) + pista de audio (`vocals_raw.wav` sin pérdida) + subtítulos embebidos extraídos si existen. | FFmpeg |
| 02 | **Separación de voces** (RF-02) | `vocals_raw.wav` | `vocals.wav` (voz limpia mono) + `accompaniment.wav` (música/efectos de fondo). | Demucs, UVR, MDX |
| 03 | **Transcripción** (RF-03) | `vocals.wav` | `transcript.json` (texto segmentado con marcas de tiempo, idioma detectado y confianza). | Whisper Large v3, Faster-Whisper, Distil-Whisper |
| 04 | **Corrección de texto** (RF-04) | `transcript.json` | `transcript_corrected.json` (corrección de nombres propios, ortografía y eliminación de muletillas/repeticiones sin alterar el significado, manteniendo trazabilidad de cambios). | LLMs locales (Llama, Qwen, Mistral) |
| 05 | **Traducción** (RF-05) | `transcript_corrected.json` | `translation.json` (traducción contextual al español manteniendo marcas de tiempo de referencia y soporte para glosario de términos). | NLLB, MarianMT, LLMs locales (Llama, Qwen, Mistral, Gemma) |
| 06 | **Adaptación del doblaje** (RF-06) | `translation.json` | `adaptation.json` (texto adaptado y reescrito cuando la duración locutada estimada difiera más de ±10% del segmento original). | Algoritmos de estimación de sílabas + LLMs de compresión de texto |
| 07 | **Clonación de voz** (RF-07) | `vocals.wav` (muestras) + `adaptation.json` | Perfil/embedding de voz clonada reutilizable (soporte para múltiples hablantes / speaker diarization). | XTTS, OpenVoice, Fish Speech, CosyVoice, Dia, Spark-TTS, E2 TTS |
| 08 | **Síntesis TTS** (RF-08) | `adaptation.json` + Perfil de voz clonada | `tts_segments/` (archivos `.wav` de audio por segmento a 44.1/48 kHz limpios de artefactos). | Motores TTS integrados con el clonador seleccionado |
| 09 | **Sincronización temporal** (RF-09) | `tts_segments/` + `adaptation.json` | `dubbed_audio_aligned.wav` (audio ajustado mediante estiramiento temporal - time stretch - sin alterar el pitch, con desfase residual < 100ms). | Time Stretch, Forced Alignment, DTW, Montreal Forced Aligner |
| 10 | **Sincronización labial** (RF-10) *(Opcional)* | Vídeo normalizado + `dubbed_audio_aligned.wav` | `video_lipsync.mp4` (vídeo con boca sincronizada al audio en español; desactivable por configuración). | Wav2Lip, MuseTalk, LatentSync, InfiniteTalk |
| 11 | **Mezcla final (Mux)** (RF-11) | Vídeo (`video_lipsync.mp4` o normalizado) + `dubbed_audio_aligned.wav` + `accompaniment.wav` | `output_video.mp4` (vídeo con audio final mezclado en el formato original de resolución y framerate). | FFmpeg |
| 12 | **Generación de subtítulos** (RF-12) | `adaptation.json` (y `transcript.json` original) | Subtítulos en español, inglés y bilingües (español + inglés) en formatos `.srt`, `.vtt` y `.ass`. | Scripts de formateo a subtítulos estándar |
| 13 | **Procesamiento por lotes** (RF-13) | Directorio de entrada con múltiples vídeos | Procesamiento desatendido en cola con reporte de resultados en `batch_report.json`. | Módulo orquestador central |

---

## 5. Contrato de carpetas y formatos intermedios

Se debe respetar de forma estricta la siguiente estructura de almacenamiento para permitir la depuración y aislamiento de módulos:

```
proyecto_doblaje/
├── input/                          # Archivos de vídeo originales (.mp4, .mkv, .mov, .avi)
├── work/
│   └── <video_id>/                 # Identificador único de vídeo (nombre del archivo o hash)
│       ├── 01_ingesta/
│       │   ├── video_info.json     # Metadatos del vídeo original (duración, resolución, fps, codecs)
│       │   ├── video_normalized.mp4# Vídeo extraído en formato de trabajo
│       │   └── subtitles_extracted/ # Subtítulos originales extraídos (si existían)
│       ├── 02_separacion/
│       │   ├── vocals.wav          # Pista de voz limpia sin música ni ruidos (WAV PCM 16-bit, mono)
│       │   └── accompaniment.wav   # Pista de música y efectos (WAV PCM 16-bit, estéreo/mono)
│       ├── 03_transcripcion/
│       │   └── transcript.json     # [{id, start, end, speaker, text, confidence}]
│       ├── 04_correccion/
│       │   ├── transcript_corrected.json # Texto limpio
│       │   └── audit_changes.json  # Registro de diferencias (antes vs después)
│       ├── 05_traduccion/
│       │   └── translation.json    # [{id, start, end, speaker, original_text, translated_text}]
│       ├── 06_adaptacion/
│       │   └── adaptation.json     # [{id, start, end, speaker, text, duration_target}]
│       ├── 07_voz_clonada/
│       │   └── voice_profiles/     # Archivos de embedding y pesos por hablante (speaker)
│       ├── 08_tts/
│       │   └── segments/
│       │       ├── 001.wav         # Audio sintetizado del segmento 1 (44.1 o 48 kHz, mono)
│       │       └── ...
│       ├── 09_sincronia/
│       │   ├── dubbed_audio_aligned.wav # Audio en español ajustado temporalmente
│       │   └── alignment_report.json    # Informe de desfase residual por segmento (debe ser < 100ms)
│       ├── 10_lipsync/             # (Opcional, se salta si está desactivado)
│       │   └── video_lipsync.mp4   # Vídeo con lip sync aplicado
│       ├── 11_mezcla/
│       │   └── output_video.mp4    # Vídeo final con audio combinado y multiplexado
│       └── 12_subtitulos/
│           ├── subtitles_es.srt    # Subtítulos en español (SRT, VTT, ASS)
│           ├── subtitles_en.srt    # Subtítulos en inglés (si corresponde)
│           └── subtitles_bil.srt   # Subtítulos bilingües combinados
├── output/                         # Carpeta donde se exporta el output final de cada vídeo
├── logs/
│   ├── <video_id>/
│   │   ├── 01_ingesta.log          # Registro en formato JSON Lines (.log)
│   │   ├── 02_separacion.log
│   │   └── ...
│   └── batch_report.json           # Resumen consolidado del procesamiento por lotes
├── docs/
│   └── decisiones-tecnicas.md      # Registro obligatorio de investigaciones tecnológicas
├── config/
│   └── config.yaml                 # Configuración del pipeline
├── tests/                          # Suite de pruebas automatizadas (pytest)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── CLAUDE.md                       # Este archivo de referencia
```

### Formatos obligatorios

- **Texto y metadatos:** JSON (UTF-8, codificado con sangría).
- **Audio intermedio:** WAV (PCM 16-bit, mono, recomendado 16kHz o 44.1kHz).
- **Audio final de síntesis:** WAV (PCM 16-bit, 44.1kHz o 48kHz).
- **Subtítulos:** SRT, VTT y ASS (deben generarse los tres formatos siempre).
- **Logs:** JSON Lines (`.log`), estructurados para fácil análisis y lectura automatizada.
- **Configuración:** YAML declarativo.

---

## 6. Requisitos no funcionales clave

- **Detección y fallback de hardware (RNF-01):**
  - Al arrancar, el sistema detecta de forma automática la VRAM disponible y el soporte CUDA/ROCm.
  - Selecciona automáticamente el tamaño de modelo óptimo (p. ej. Whisper Small vs Large) o cae a CPU (*graceful fallback*) si no hay GPU compatible.
- **Portabilidad en la nube (RNF-02):**
  - Ejecutable bajo contenedores Docker sin modificar una sola línea de código en local o en RunPod, Vast.ai, etc.
  - Se debe evitar almacenar configuraciones dentro de los contenedores; todo debe ir en variables de entorno o mediante volumen montado para `config.yaml`.
- **Tolerancia a fallos en procesamiento masivo (RNF-08 / RNF-13):**
  - Si un vídeo de la cola falla en cualquier etapa, el error se registra en los logs y en `batch_report.json`, pero el pipeline **no se detiene** e inicia automáticamente el siguiente vídeo.
  - Las etapas deben ser independientes y soportar reintentos automáticos configurables (`reintentos_por_etapa`).
  - Los archivos intermedios ya generados deben conservarse para poder reanudar el proceso desde el punto del fallo sin tener que repetir el procesamiento desde el principio.
- **Seguridad y privacidad (RNF-10):**
  - Evitar la persistencia de datos innecesarios en servidores cloud de GPU alquilada. Ofrecer mecanismos para limpiar la carpeta `work/` al finalizar.
- **Observabilidad (RNF-11):**
  - Trazabilidad completa con logs JSON Lines por cada etapa conteniendo timestamp, identificador del vídeo, nivel (`INFO`, `WARNING`, `ERROR`), mensaje y duración del procesamiento.

---

## 7. Convenciones de desarrollo

- **Lenguaje principal:** Python 3.10+ (salvo excepciones técnicas debidamente argumentadas en `/docs/decisiones-tecnicas.md`).
- **Nomenclatura de carpetas:** Nombres snake_case prefijados por número de etapa de dos dígitos (ej. `02_separacion`).
- **Nomenclatura de código:** Estándar PEP 8 (clases PascalCase, variables y funciones snake_case, constantes UPPER_SNAKE_CASE).
- **Control de Configuración:** Declarativo en `/config/config.yaml`. Los parámetros y umbrales deben ser modificables allí. No se admite hardcodear rutas, nombres de modelos ni parámetros de síntesis en el código fuente.
- **Validación de Configuración:** Utilizar `pydantic` o similar para verificar que la configuración cumple con los tipos requeridos al iniciar el pipeline.
- **Manejo de credenciales:** Ninguna credencial ni token personal de repositorios (ej. HuggingFace) debe escribirse en código ni en `config.yaml`. Se inyectan únicamente mediante variables de entorno a través de un archivo `.env` (excluido en `.gitignore`, documentado en `.env.example`).

### Ejemplo oficial de `/config/config.yaml` (Anexo B del ERS)

```yaml
pipeline:
  idioma_destino: "es"
  lip_sync: false
  formatos_subtitulos: 
    - "srt"
    - "vtt"
    - "ass"

  hardware:
    modo: "auto"                    # auto | gpu | cpu
    vram_minima_gb: 8

  procesamiento_lote:
    max_paralelo: 2
    reintentos_por_etapa: 2

  gpu_alquilada:
    proveedor: "runpod"             # runpod | vast.ai | lambda_labs | paperspace | coreweave
    activar_solo_si_local_insuficiente: true

stages:
  separation:
    model: "htdemucs"
  transcription:
    model: "whisper-large-v3"
    language: "auto"
  tts:
    engine: "coqui-xtts"
    speaker_sample_min_s: 6
```

---

## 8. Cómo ejecutar y probar el proyecto

### Ejecución con Docker

```bash
# Construir la imagen del pipeline
docker compose build

# Procesar un vídeo de forma individual
docker compose run pipeline python main.py --input input/mi_video.mp4 --config config/config.yaml

# Procesar en lote un directorio completo
docker compose run pipeline python main.py --batch input/
```

### Ejecución en desarrollo local

```bash
# Instalar el entorno virtual y dependencias
pip install -r requirements.txt

# Ejecutar el pipeline indicando etapas específicas
python main.py --input input/mi_video.mp4 --stages 01_ingesta,02_separacion,03_transcripcion
```

### Ejecución de Pruebas

```bash
# Ejecutar tests unitarios e integrales
pytest tests/ -v
```

## 9. Interfaz de usuario y modo de entrega

Para facilitar el uso por parte de usuarios finales sin conocimientos técnicos, el sistema expone una interfaz web local mínima y automatizada:

- **Interfaz Web Local (`http://localhost:8080`):**
  - **Carga de vídeos:** Botón simple y área de arrastrar y soltar (*drag and drop*) para subir uno o varios archivos de vídeo simultáneamente.
  - **Monitor de progreso:** Vista visual en tiempo real que indica en qué etapa del pipeline se encuentra cada vídeo (ej. *Procesando Etapa 03 de 12: Transcripción*) y el progreso general de la cola.
  - **Descargas:** Botones de descarga directa para el vídeo final doblado y los subtítulos generados en formatos estándar (`.srt`, `.vtt`, `.ass`) de forma independiente.
  - **Historial de la sesión:** Tabla o lista simple indicando el estado de cada vídeo subido en la sesión actual: *En cola*, *Procesando*, *Completado* o *Error* (con acceso a un resumen descriptivo del fallo).

- **Modo de Operación:**
  - Cero comandos para el uso diario. El usuario solo interactúa con la interfaz web desde su navegador favorito.
  - El backend se implementará usando un framework ágil como **FastAPI** o **Flask** para servir el cliente HTML/JS y gestionar el pipeline de procesamiento en background.

- **Empaquetamiento y auto-arranque:**
  - Todo el sistema (backend, modelos de IA e interfaz web) se ejecuta mediante contenedores Docker.
  - Se proveerá un archivo ejecutable simple (`doblar.bat` en Windows) que al hacerle doble clic se encargará de arrancar Docker Compose en segundo plano y abrir automáticamente el navegador web en `http://localhost:8080`, ocultando completamente la consola al usuario final.
  - La configuración asegura que el usuario no necesite teclear comandos en su día a día.

---

## 10. Qué NO hacer

- ❌ **No usar software propietario de pago.** Nada de APIs de traducción o síntesis comerciales de cobro por uso. Todo el pipeline debe operar de forma local/open source.
- ❌ **No importar módulos cruzados.** No acoplar el código de un módulo con otro (ej: no importar lógica de transcripción en traducción). Las etapas se comunican exclusivamente a través del sistema de archivos mediante los formatos especificados.
- ❌ **No hardcodear rutas absolutas.** Usar rutas relativas basadas en la ubicación del proyecto o definidas en `config.yaml`.
- ❌ **No codificar sin investigación previa.** No se permite añadir código a una etapa si no se ha guardado primero su respectiva investigación comparativa de 3 alternativas en `/docs/decisiones-tecnicas.md`.
- ❌ **No ignorar fallos en el procesamiento por lotes.** Un vídeo corrupto o un fallo en el procesamiento no debe interrumpir el proceso de los otros vídeos en la cola.

---

## 11. Estado actual del proyecto

```
Fase actual: Fase 0 — Investigación tecnológica inicial (Fase de evaluación de tecnologías candidatas)
```

### Historial de fases (Basado en el Plan de Implementación por Fases del ERS)

| Fase | Nombre | Descripción | Estado |
|------|--------|-------------|--------|
| **Fase 0** | Investigación tecnológica inicial | Evaluación y documentación de las 3 tecnologías candidatas por cada etapa del pipeline. | 🔄 En curso |
| **Fase 1** | Ingesta y preprocesado | Implementación de RF-01 (ingesta) y RF-02 (separación de voces). | ⏳ Pendiente |
| **Fase 2** | Transcripción y corrección | Implementación de RF-03 (transcripción) y RF-04 (corrección de texto mediante LLM local). | ⏳ Pendiente |
| **Fase 3** | Traducción y adaptación | Implementación de RF-05 (traducción) y RF-06 (adaptación temporal del guion). | ⏳ Pendiente |
| **Fase 4** | Clonación de voz y síntesis | Implementación de RF-07 (clonación) y RF-08 (síntesis TTS). | ⏳ Pendiente |
| **Fase 5** | Sincronización y mezcla | Implementación de RF-09 (sincronía temporal), RF-10 (lip sync opcional), RF-11 (mezcla final) y RF-12 (subtítulos). | ⏳ Pendiente |
| **Fase 6** | Procesamiento masivo y despliegue | Implementación de RF-13 (lotes) y RNF de despliegue Docker y hardware. | ⏳ Pendiente |
| **Fase 7** | Pruebas integrales y ajuste | Suite de pruebas final y calibración de calidad del doblaje. | ⏳ Pendiente |

---

*Última actualización: 2026-07-14 · Versión del documento: 2.1.0 (Alineado con requisitos de UI)*
