import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import json
import importlib

vp_module = importlib.import_module("src.01_ingesta.video_processor")
VideoProcessor = vp_module.VideoProcessor

@pytest.fixture
def mock_config():
    return {
        "pipeline": {
            "work_dir": "./work_test",
            "output_dir": "./output_test"
        }
    }

@pytest.fixture
def video_processor(mock_config):
    return VideoProcessor(mock_config)

@patch.object(vp_module.subprocess, "run")
@patch.object(vp_module.Path, "exists")
@patch.object(vp_module.os.path, "getsize")
def test_video_id_generation(mock_getsize, mock_exists, mock_run, video_processor):
    mock_exists.return_value = True
    mock_getsize.return_value = 1000000
    
    video_path = Path("input_dir/test_video.mp4")
    video_id = video_processor.get_video_id(video_path)
    
    assert "test_video_" in video_id
    assert len(video_id) == len("test_video_") + 8

@patch.object(vp_module.subprocess, "run")
def test_get_metadata_success(mock_run, video_processor):
    # Simular salida exitosa de ffprobe
    mock_stdout = json.dumps({
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080},
            {"codec_type": "audio", "codec_name": "aac"}
        ],
        "format": {"duration": "120.5"}
    })
    
    mock_response = MagicMock()
    mock_response.stdout = mock_stdout
    mock_run.return_value = mock_response
    
    metadata = video_processor.get_metadata(Path("test_video.mp4"))
    
    assert len(metadata["streams"]) == 2
    assert metadata["format"]["duration"] == "120.5"
    mock_run.assert_called_once()

@patch.object(vp_module.subprocess, "run")
def test_get_metadata_failure(mock_run, video_processor):
    # Simular fallo de ffprobe
    from subprocess import CalledProcessError
    mock_run.side_effect = CalledProcessError(1, "ffprobe")
    
    with pytest.raises(RuntimeError):
        video_processor.get_metadata(Path("corrupt_video.mp4"))

@patch.object(vp_module.subprocess, "run")
@patch.object(vp_module.Path, "exists")
@patch.object(vp_module.os.path, "getsize")
def test_run_ingestion_h264_direct_copy(mock_getsize, mock_exists, mock_run, video_processor):
    # Import shutil in the module if it's used inline
    if not hasattr(vp_module, "shutil"):
        import shutil
        vp_module.shutil = shutil
        
    mock_exists.return_value = True
    mock_getsize.return_value = 500000
    
    # Mock de ffprobe
    mock_stdout = json.dumps({
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "width": 1280, "height": 720, "r_frame_rate": "30/1"},
            {"codec_type": "audio", "codec_name": "aac"}
        ],
        "format": {"duration": "10.0"}
    })
    
    mock_response = MagicMock()
    mock_response.stdout = mock_stdout
    mock_run.return_value = mock_response

    # Parchear creación de directorios usando patch.object
    with patch.object(vp_module.Path, "mkdir") as mock_mkdir:
        # Parchear apertura de archivos
        with patch("builtins.open", unittest.mock.mock_open()) as mock_file:
            with patch.object(vp_module, "shutil") as mock_shutil_module:
                res = video_processor.run(Path("test_video.mp4"))
                
                # Asegurar que se copia directamente porque ya es h264/mp4
                mock_shutil_module.copy2.assert_called_once()
                assert res["video_id"] is not None
                assert "01_ingesta" in res["stage_dir"]
                assert res["video_info"]["resolution"] == "1280x720"

import unittest.mock
