# Registro de Decisiones Tecnológicas (Etapas 01 a 05)

Este documento registra los análisis comparativos detallados e investigaciones de alternativas para las etapas iniciales del pipeline, conforme al contrato del proyecto en `CLAUDE.md`.

---

## Etapa 01 — Ingesta de vídeo (RF-01)

### 1. Tecnologías Evaluadas
- **Candidata 1: FFmpeg mediante subprocess (Línea de comandos directa)**
  - *Descripción:* Ejecución del binario nativo de FFmpeg a través del módulo `subprocess` de Python.
- **Candidata 2: ffmpeg-python**
  - *Descripción:* Wrapper oficial/comunitario en Python que construye grafos de comandos de FFmpeg.
- **Candidata 3: PyAV**
  - *Descripción:* Enlaces (bindings) de Python directo a las librerías de C de Libav/FFmpeg para procesamiento de vídeo frame a frame.

### 2. Matriz Comparativa

| Tecnología | Calidad (Métrica) | Velocidad | VRAM | Licencia | Comunidad | Facilidad de Integración |
|---|---|---|---|---|---|---|
| **FFmpeg subprocess** | Máxima (Sin pérdida) | Instantánea | 0 MB (CPU) | LGPL / GPL | Inmensa | Muy alta (sin dependencias C en Python) |
| **ffmpeg-python** | Máxima (Sin pérdida) | Instantánea | 0 MB (CPU) | Apache 2.0 | Baja (Poco mantenido) | Media (requiere ffprobe externo para metadatos) |
| **PyAV** | Máxima | Rápida | 0 MB (CPU) | BSD-3 | Alta | Baja (requiere compilación compleja de C) |

### 3. Decisión y Justificación
- **Tecnología Elegida:** **FFmpeg mediante subprocess** (combinado con `ffprobe` para la extracción de metadatos en JSON).
- **Motivos de la elección:** Es la alternativa más robusta y estándar. Al no introducir dependencias compiladas de C dentro de Python, garantiza portabilidad absoluta en Docker y evita conflictos de versiones de bibliotecas del sistema. Además, simplifica el mantenimiento a largo plazo.

---

## Etapa 02 — Separación de voces (RF-02)

### 1. Tecnologías Evaluadas
- **Candidata 1: audio-separator (Wrapper de UVR/MDX-Net en ONNX)**
  - *Descripción:* Librería de Python para separar pistas de audio utilizando modelos ONNX procedentes del proyecto Ultimate Vocal Remover.
- **Candidata 2: demucs (PyTorch oficial de Meta)**
  - *Descripción:* Modelo de separación de fuentes musicales basado en redes neuronales de Wave-U-Net y convoluciones temporales de Meta.
- **Candidata 3: spleeter (Deezer)**
  - *Descripción:* Modelo clásico de separación basado en Tensorflow desarrollado por Deezer.

### 2. Matriz Comparativa

| Tecnología | Calidad (Métrica SDR) | Velocidad | VRAM | Licencia | Comunidad | Facilidad de Integración |
|---|---|---|---|---|---|---|
| **audio-separator** | **Excepcional (> 12dB SDR)** | Muy rápida (ONNX Runtime) | 2 - 4 GB | MIT | Muy alta y activa | Muy sencilla (API limpia + CLI) |
| **demucs** | Excelente (~10-11dB SDR) | Moderada | ~4 GB | MIT | Activa | Media (Interfaz PyTorch rígida) |
| **spleeter** | Baja (Muchos artefactos) | Muy rápida | < 1 GB | MIT | Obsoleta (Incompatible con Py3.10+) | Muy difícil (dependencias viejas) |

### 3. Decisión y Justificación
- **Tecnología Elegida:** **audio-separator** empleando el modelo ONNX `UVR-MDX-NET-Inst_HQ_3.onnx` o `HTDemucs` de UVR.
- **Motivos de la elección:** Ofrece la mejor calidad de separación de voz disponible actualmente en el ámbito de código abierto. Al ejecutarse sobre ONNX Runtime, procesa el audio significativamente más rápido que demucs puro en PyTorch, reduciendo el consumo de VRAM en entornos restringidos.

---

## Etapa 03 — Transcripción (RF-03)

### 1. Tecnologías Evaluadas
- **Candidata 1: faster-whisper (CTranslate2)**
  - *Descripción:* Reimplantación de Whisper de OpenAI utilizando el motor CTranslate2, que optimiza la velocidad y reduce el consumo de memoria mediante cuantización.
- **Candidata 2: openai-whisper (PyTorch oficial)**
  - *Descripción:* El repositorio original y estándar en PyTorch proveído por OpenAI.
- **Candidata 3: whisper.cpp (C++)**
  - *Descripción:* Versión de Whisper portada a C/C++ plano optimizada para alto rendimiento sin dependencias pesadas de Python.

### 2. Matriz Comparativa

| Tecnología | Calidad (WER) | Velocidad | VRAM | Licencia | Comunidad | Facilidad de Integración |
|---|---|---|---|---|---|---|
| **faster-whisper** | Idéntica a OpenAI (~2-3% WER) | **Muy rápida (4x-6x)** | ~3 GB (INT8) | MIT | Extremadamente activa | Muy sencilla (API directa en Python) |
| **openai-whisper** | Referencia (~2-3% WER) | Lenta (1x) | ~6-10 GB (FP16) | MIT | Activa | Sencilla |
| **whisper.cpp** | Excelente | Ultra rápida (CPU) | Muy baja | MIT | Muy activa | Media (requiere compilar o usar bindings externos) |

### 3. Decisión y Justificación
- **Tecnología Elegida:** **faster-whisper** utilizando el modelo `large-v3` (o `large-v3-turbo` en caso de requerir mayor velocidad).
- **Motivos de la elección:** Es el estándar de producción actual. Mantiene la precisión matemática del modelo original de OpenAI pero se ejecuta hasta 6 veces más rápido y requiere menos de la mitad de VRAM gracias a la cuantización de enteros de 8 bits (INT8).

---

## Etapa 04 — Corrección de texto (RF-04)

### 1. Tecnologías Evaluadas
- **Candidata 1: Ollama (Inferencia externa integrada en Docker-Compose)**
  - *Descripción:* Servidor local de LLMs fácil de desplegar que optimiza dinámicamente la memoria.
- **Candidata 2: llama-cpp-python**
  - *Descripción:* Enlace directo de C++ a Python para ejecutar modelos GGUF localmente sin servicios externos.
- **Candidata 3: HuggingFace Transformers (PyTorch)**
  - *Descripción:* Carga nativa de modelos mediante la biblioteca Transformers de HuggingFace.

### 2. Matriz Comparativa

| Tecnología | Calidad | Velocidad | VRAM | Licencia | Comunidad | Facilidad de Integración |
|---|---|---|---|---|---|---|
| **Ollama** | Depende del modelo (Alta) | Muy rápida (opt. GGUF) | Dinámica | MIT | Masiva / En auge | Muy sencilla (API REST simple) |
| **llama-cpp-python** | Depende del modelo (Alta) | Rápida (opt. GGUF) | Controlada | MIT | Activa | Media (complejidad de compilar CUDA en Docker) |
| **Transformers** | Alta | Lenta (autotregresiva) | Muy alta | Apache 2.0 | Estándar científico | Fácil |

### 3. Decisión y Justificación
- **Tecnología Elegida:** **Ollama** utilizando el modelo `qwen2.5:7b-instruct` o `llama3:8b-instruct`.
- **Motivos de la elección:** La integración es sumamente limpia al delegar la carga de pesos a un contenedor independiente de Ollama en Docker Compose. Esto previene problemas al compilar bindings CUDA complejos en la imagen de desarrollo de Python y facilita cambiar de modelo simplemente cambiando una variable o configuración en `config.yaml`.

---

## Etapa 05 — Traducción (RF-05)

### 1. Tecnologías Evaluadas
- **Candidata 1: LLM local (Qwen-2.5-7B/Llama-3 vía Ollama)**
  - *Descripción:* Uso de un modelo de lenguaje de propósito general con prompts especializados para traducción contextual y adaptación del tono.
- **Candidata 2: NLLB-200 (Meta)**
  - *Descripción:* Modelo clásico de traducción automática neuronal multilingüe de Meta para más de 200 idiomas.
- **Candidata 3: MarianMT (Helsinki-NLP)**
  - *Descripción:* Colección de modelos bilingües ligeros optimizados para pares de idiomas específicos en HuggingFace.

### 2. Matriz Comparativa

| Tecnología | Calidad | Velocidad | VRAM | Licencia | Comunidad | Facilidad de Integración |
|---|---|---|---|---|---|---|
| **LLM Local (Ollama)** | **Excepcional (fluidez y modismos)** | Moderada | 0 MB extra (reutiliza Ollama) | Apache 2.0 / Llama | Muy activa | Sencilla (reutiliza el cliente Ollama) |
| **NLLB-200** | Buena (Traducción literal) | Rápida | ~2 GB | MIT | Media | Media (requiere cargar transformers en Python) |
| **MarianMT** | Aceptable | Ultra rápida | < 500 MB | Apache 2.0 | Baja | Sencilla |

### 3. Decisión y Justificación
- **Tecnología Elegida:** **LLM Local (Qwen-2.5-7B-Instruct)** utilizando la instancia existente de Ollama.
- **Motivos de la elección:** Los LLMs modernos superan ampliamente a los sistemas de traducción tradicionales en la comprensión contextual, preservación del estilo y control fino del vocabulario. Dado que ya desplegaremos Ollama para la etapa de corrección de texto, el coste de VRAM es de cero adicionales al reutilizar el mismo contenedor para la traducción, permitiendo además pasar un glosario dinámico de términos técnicos de forma sencilla en el prompt del sistema.
