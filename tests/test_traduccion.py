"""
Tests unitarios para la etapa 04 — Corrección y Traducción (LLMTranslator).

Todos los tests mockean la llamada a Ollama para no requerir
una instancia real del servidor LLM durante la CI.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import importlib

translator_module = importlib.import_module("src.04_traduccion.llm_translator")
LLMTranslator = translator_module.LLMTranslator


@pytest.fixture
def mock_config():
    return {
        "pipeline": {
            "work_dir": "./work_test",
        },
        "stages": {
            "translation": {
                "model": "qwen2.5:3b",
                "ollama_url": "http://localhost:11434",
                "temperature": 0.2,
                "max_retries": 3,
            }
        },
    }


@pytest.fixture
def translator(mock_config):
    return LLMTranslator(mock_config)


# ------------------------------------------------------------------
# Datos de prueba reutilizables
# ------------------------------------------------------------------

SAMPLE_TRANSCRIPT = {
    "video_id": "video_abc",
    "language": "en",
    "language_prob": 0.98,
    "duration": 10.0,
    "segments_count": 2,
    "segments": [
        {"id": 1, "start": 0.0, "end": 2.5, "text": "Hello world", "avg_logprob": -0.2, "no_speech_prob": 0.01, "words": []},
        {"id": 2, "start": 3.0, "end": 6.0, "text": "This is a test", "avg_logprob": -0.3, "no_speech_prob": 0.01, "words": []},
    ],
}

SAMPLE_LLM_RESPONSE = {
    "segments": [
        {"id": 1, "start": 0.0, "end": 2.5, "text_original": "Hello world", "text_es": "Hola mundo"},
        {"id": 2, "start": 3.0, "end": 6.0, "text_original": "This is a test", "text_es": "Esto es una prueba"},
    ]
}


# ------------------------------------------------------------------
# Test: construcción del cliente Ollama
# ------------------------------------------------------------------

def test_build_client_uses_correct_base_url(translator):
    """El cliente OpenAI debe apuntar a la URL de Ollama correcta."""
    mock_openai = MagicMock()
    with patch.dict("sys.modules", {"openai": mock_openai}):
        mock_openai.OpenAI.return_value = MagicMock()
        client = translator._build_client()
        mock_openai.OpenAI.assert_called_once_with(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )


# ------------------------------------------------------------------
# Test: traducción de un chunk (llamada al LLM)
# ------------------------------------------------------------------

def test_translate_chunk_success(translator):
    """_translate_chunk debe devolver los segmentos correctamente traducidos."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(SAMPLE_LLM_RESPONSE)
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    result = translator._translate_chunk(mock_client, SAMPLE_TRANSCRIPT["segments"])

    assert len(result) == 2
    assert result[0]["text_es"] == "Hola mundo"
    assert result[1]["text_es"] == "Esto es una prueba"
    # Los timestamps originales deben preservarse
    assert result[0]["start"] == 0.0
    assert result[1]["end"] == 6.0


def test_translate_chunk_invalid_json_falls_back(translator):
    """Si el LLM devuelve JSON inválido, se usa el texto original como fallback."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "esto no es JSON válido {{{"
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    # max_retries=1 para que el test no tarde demasiado
    translator.max_retries = 1
    result = translator._translate_chunk(mock_client, SAMPLE_TRANSCRIPT["segments"])

    assert len(result) == 2
    # Fallback: text_es == text original
    assert result[0]["text_es"] == "Hello world"
    assert result[1]["text_es"] == "This is a test"


# ------------------------------------------------------------------
# Test: flujo completo de run()
# ------------------------------------------------------------------

@patch.object(translator_module.Path, "exists")
@patch.object(translator_module.Path, "mkdir")
def test_run_translation_success(mock_mkdir, mock_exists, translator, tmp_path):
    """run() debe leer el transcript_raw.json y generar transcript_translated.json."""
    mock_exists.return_value = True

    mock_openai_module = MagicMock()
    mock_client_instance = MagicMock()
    mock_openai_module.OpenAI.return_value = mock_client_instance

    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(SAMPLE_LLM_RESPONSE)
    mock_client_instance.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    raw_json = json.dumps(SAMPLE_TRANSCRIPT)

    with patch("builtins.open", mock_open(read_data=raw_json)):
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            result = translator.run("video_abc", "/fake/transcript_raw.json")

    assert result["segments_count"] == 2
    assert result["target_language"] == "es"
    assert "transcript_translated" in result
    assert "04_traduccion" in result["transcript_translated"]


def test_run_raises_if_raw_not_found(translator):
    """run() debe lanzar FileNotFoundError si el transcript raw no existe."""
    with pytest.raises(FileNotFoundError):
        translator.run("video_abc", "/ruta/inexistente/transcript_raw.json")
