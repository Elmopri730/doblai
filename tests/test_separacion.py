import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import importlib

sep_module = importlib.import_module("src.02_separacion.audio_separator")
AudioSeparatorWrapper = sep_module.AudioSeparatorWrapper

@pytest.fixture
def mock_config():
    return {
        "pipeline": {
            "work_dir": "./work_test",
            "hardware": {
                "modo": "auto",
                "vram_minima_gb": 8
            }
        },
        "stages": {
            "separation": {
                "model": "UVR-MDX-NET-Inst_HQ_3.onnx"
            }
        }
    }

@pytest.fixture
def audio_separator(mock_config):
    return AudioSeparatorWrapper(mock_config)

@patch.object(sep_module.onnxruntime, "get_available_providers")
def test_detect_onnx_provider_auto_gpu(mock_providers, audio_separator):
    mock_providers.return_value = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    
    provider = audio_separator.detect_onnx_provider()
    assert provider == "CUDAExecutionProvider"

@patch.object(sep_module.onnxruntime, "get_available_providers")
def test_detect_onnx_provider_auto_cpu(mock_providers, audio_separator):
    mock_providers.return_value = ["CPUExecutionProvider"]
    
    provider = audio_separator.detect_onnx_provider()
    assert provider == "CPUExecutionProvider"

@patch.object(sep_module.onnxruntime, "get_available_providers")
def test_detect_onnx_provider_forced_gpu_failure(mock_providers, mock_config):
    mock_providers.return_value = ["CPUExecutionProvider"]
    mock_config["pipeline"]["hardware"]["modo"] = "gpu"
    
    wrapper = AudioSeparatorWrapper(mock_config)
    with pytest.raises(RuntimeError):
        wrapper.detect_onnx_provider()

@patch.object(sep_module, "Separator")
@patch.object(sep_module.onnxruntime, "get_available_providers")
@patch.object(sep_module.shutil, "move")
@patch.object(sep_module.Path, "exists")
def test_run_separation_success(mock_exists, mock_move, mock_providers, mock_separator_class, audio_separator):
    mock_exists.return_value = True
    mock_providers.return_value = ["CPUExecutionProvider"]
    
    # Mock de la clase Separator y sus métodos
    mock_separator_instance = MagicMock()
    mock_separator_instance.separate.return_value = [
        "/work_test/video_123/02_separacion/vocals_output.wav",
        "/work_test/video_123/02_separacion/instrumental_output.wav"
    ]
    mock_separator_class.return_value = mock_separator_instance

    with patch.object(sep_module.Path, "mkdir") as mock_mkdir:
        # Ejecutar
        res = audio_separator.run("video_123", "/work_test/video_123/01_ingesta/vocals_raw.wav")
        
        # Verificar inicialización
        mock_separator_class.assert_called_once_with(output_dir=str(Path("./work_test/video_123/02_separacion")))
        mock_separator_instance.load_model.assert_called_once_with(model_filename="UVR-MDX-NET-Inst_HQ_3.onnx")
        mock_separator_instance.separate.assert_called_once()
        
        # Verificar renombramiento correcto
        assert "vocals.wav" in res["vocals"]
        assert "accompaniment.wav" in res["accompaniment"]
        
        # Debe haber hecho 2 movimientos de archivo
        assert mock_move.call_count == 2
