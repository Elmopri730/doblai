"""
Tests unitarios para la etapa 03 — Transcripción (Transcriber).

Todos los tests usan mocks para evitar cargar el modelo Whisper real
(que pesa varios GBs y requiere GPU).
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import importlib

transcriber_module = importlib.import_module("src.03_transcripcion.transcriber")
Transcriber = transcriber_module.Transcriber


@pytest.fixture
def mock_config():
    return {
        "pipeline": {
            "work_dir": "./work_test",
            "hardware": {"modo": "cpu"},
        },
        "stages": {
            "transcription": {
                "model": "whisper-large-v3",
                "language": "auto",
            }
        },
    }


@pytest.fixture
def transcriber(mock_config):
    return Transcriber(mock_config)


# ------------------------------------------------------------------
# Test: resolución de nombre de modelo
# ------------------------------------------------------------------

def test_model_name_mapping(mock_config):
    """Los aliases de config.yaml se resuelven correctamente."""
    t = Transcriber(mock_config)
    assert t.model_name == "large-v3"


def test_model_name_medium(mock_config):
    mock_config["stages"]["transcription"]["model"] = "whisper-medium"
    t = Transcriber(mock_config)
    assert t.model_name == "medium"


# ------------------------------------------------------------------
# Test: detección de dispositivo/cómputo
# ------------------------------------------------------------------

def test_detect_compute_cpu_forced(mock_config):
    """Cuando el modo es cpu, debe devolver (int8, cpu)."""
    t = Transcriber(mock_config)
    compute, device = t._detect_compute()
    assert device == "cpu"
    assert compute == "int8"


def test_detect_compute_gpu_forced_no_cuda_raises(mock_config):
    """Modo gpu forzado sin CUDA disponible debe lanzar RuntimeError."""
    mock_config["pipeline"]["hardware"]["modo"] = "gpu"
    with patch.dict("sys.modules", {"torch": MagicMock(cuda=MagicMock(is_available=lambda: False))}):
        t = Transcriber.__new__(Transcriber)
        t.config = mock_config
        t.hardware_mode = "gpu"
        with pytest.raises(RuntimeError, match="GPU"):
            t._detect_compute()


# ------------------------------------------------------------------
# Test: transcripción completa (integración con mocks)
# ------------------------------------------------------------------

@patch.object(transcriber_module.subprocess, "run")
@patch.object(transcriber_module.Path, "exists")
@patch.object(transcriber_module.Path, "mkdir")
def test_run_transcription_success(mock_mkdir, mock_exists, mock_subprocess, transcriber, tmp_path):
    """run() debe generar transcript_raw.json con los campos correctos."""
    mock_exists.return_value = True
    mock_subprocess.return_value = MagicMock(returncode=0)

    # Montar mock de faster_whisper
    mock_word = MagicMock(word="Hello", start=0.0, end=0.5, probability=0.99)
    mock_seg = MagicMock(
        id=1, start=0.0, end=1.5,
        text=" Hello world",
        avg_logprob=-0.3,
        no_speech_prob=0.01,
        words=[mock_word],
    )
    mock_info = MagicMock(
        language="en",
        language_probability=0.98,
        duration=1.5,
    )

    mock_whisper_model = MagicMock()
    mock_whisper_model.transcribe.return_value = ([mock_seg], mock_info)

    mock_fw_module = MagicMock()
    mock_fw_module.WhisperModel.return_value = mock_whisper_model

    written_data = {}

    def fake_open(path, mode="r", **kwargs):
        mo = mock_open()()
        if "w" in mode:
            # Capturar lo que se escribe
            calls = []
            mo.write = lambda d: calls.append(d)
            mo.__enter__ = lambda s: s
            mo.__exit__ = MagicMock(return_value=False)
            return mo
        return mock_open(read_data="{}")()

    with patch.dict("sys.modules", {"faster_whisper": mock_fw_module}):
        with patch("builtins.open", mock_open()) as m_open:
            result = transcriber.run("video_abc", "/fake/vocals.wav")

    assert result["language"] == "en"
    assert result["segments_count"] == 1
    assert "transcript_raw" in result
    assert "03_transcripcion" in result["transcript_raw"]
