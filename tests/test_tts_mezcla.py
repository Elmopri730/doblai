"""
Tests unitarios para la etapa 05 — Síntesis de voz (VoiceSynthesizer)
y la etapa 06 — Mezcla final (FinalMixer).
"""

import json
import pytest
import wave
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import importlib

synth_module = importlib.import_module("src.05_tts.voice_synthesizer")
mixer_module  = importlib.import_module("src.06_mezcla.final_mixer")
VoiceSynthesizer = synth_module.VoiceSynthesizer
FinalMixer       = mixer_module.FinalMixer


# ════════════════════════════════════════════════════════════════════
#  FIXTURES COMUNES
# ════════════════════════════════════════════════════════════════════

@pytest.fixture
def base_config():
    return {
        "pipeline": {
            "work_dir":   "./work_test",
            "output_dir": "./output_test",
            "hardware":   {"modo": "cpu"},
            "formatos_subtitulos": ["srt", "vtt"],
        },
        "stages": {
            "tts": {
                "model_name":          "tts_models/multilingual/multi-dataset/xtts_v2",
                "speaker_sample_min_s": 6,
                "speaker_sample_max_s": 30,
                "target_language":     "es",
                "speed":               1.0,
                "temperature":         0.75,
            }
        },
    }

SAMPLE_MANIFEST = {
    "video_id":             "vid_abc",
    "target_language":      "es",
    "model":                "xtts_v2",
    "sample_rate":          22050,
    "segments_total":       2,
    "segments_synthesized": 2,
    "segments": [
        {"id": 1, "text_es": "Hola mundo",        "original_start": 0.0,  "original_end": 2.5},
        {"id": 2, "text_es": "Esto es una prueba", "original_start": 3.0, "original_end": 6.0},
    ],
}

SAMPLE_TRANSCRIPT = {
    "video_id":       "vid_abc",
    "source_language": "en",
    "target_language": "es",
    "original_duration": 10.0,
    "segments_count": 2,
    "segments": [
        {"id": 1, "start": 0.0, "end": 2.5, "text_original": "Hello world", "text_es": "Hola mundo"},
        {"id": 2, "start": 3.0, "end": 6.0, "text_original": "This is a test", "text_es": "Esto es una prueba"},
    ],
}


# ════════════════════════════════════════════════════════════════════
#  TESTS: VoiceSynthesizer (etapa 05)
# ════════════════════════════════════════════════════════════════════

class TestVoiceSynthesizer:

    @pytest.fixture
    def synth(self, base_config):
        return VoiceSynthesizer(base_config)

    # -- Detección de dispositivo --

    def test_detect_device_cpu_forced(self, synth):
        assert synth._detect_device() == "cpu"

    def test_detect_device_gpu_forced_no_cuda_raises(self, base_config):
        base_config["pipeline"]["hardware"]["modo"] = "gpu"
        s = VoiceSynthesizer(base_config)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            with pytest.raises(RuntimeError, match="GPU"):
                s._detect_device()

    # -- Generación de silencio --

    def test_build_silence_correct_length(self, synth):
        silence = synth._build_silence(1.0, 22050, 1, 2)
        expected = 22050 * 1 * 2  # 1 segundo de mono 16-bit
        assert len(silence) == expected

    # -- Ensamblaje de pista --

    def test_assemble_dubbed_track_creates_file(self, synth, tmp_path):
        """_assemble_dubbed_track debe crear un WAV válido."""
        # Crear WAV de segmento de prueba
        seg_wav = tmp_path / "seg_0001.wav"
        with wave.open(str(seg_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            # 0.5 segundos de silencio
            wf.writeframes(b"\x00" * (22050 * 2 // 2))

        out_wav = tmp_path / "dubbed_voice_raw.wav"
        synth._assemble_dubbed_track(
            [(0.0, 0.5, seg_wav)],
            total_duration=2.0,
            output_path=out_wav,
        )

        assert out_wav.exists()
        with wave.open(str(out_wav), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2

    # -- run() con mocks completos --

    @patch.object(synth_module.subprocess, "run")
    @patch.object(synth_module.Path, "exists")
    @patch.object(synth_module.Path, "mkdir")
    def test_run_full_pipeline(self, mock_mkdir, mock_exists, mock_sub, synth, tmp_path):
        mock_exists.return_value = True

        # Mock TTS
        mock_tts = MagicMock()
        mock_tts_module = MagicMock()
        mock_tts_module.TTS.return_value = mock_tts

        # Mock ffprobe: devuelve duración
        mock_sub.return_value = MagicMock(stdout="2.5\n", returncode=0)

        with patch.dict("sys.modules", {"TTS": mock_tts_module, "TTS.api": mock_tts_module}):
            with patch("builtins.open", mock_open(read_data=json.dumps(SAMPLE_TRANSCRIPT))):
                with patch.object(synth_module.VoiceSynthesizer, "_assemble_dubbed_track"):
                    result = synth.run(
                        "vid_abc",
                        "/fake/vocals.wav",
                        "/fake/transcript_translated.json",
                        "en",
                    )

        assert "dubbed_voice_raw" in result
        assert result["segments_synthesized"] == 2


# ════════════════════════════════════════════════════════════════════
#  TESTS: FinalMixer (etapa 06)
# ════════════════════════════════════════════════════════════════════

class TestFinalMixer:

    @pytest.fixture
    def mixer(self, base_config):
        return FinalMixer(base_config)

    # -- Conversores de timestamps --

    def test_srt_timestamp(self):
        _secs_to_srt = getattr(mixer_module, "_secs_to_srt")
        assert _secs_to_srt(0.0)   == "00:00:00,000"
        assert _secs_to_srt(3661.5) == "01:01:01,500"

    def test_vtt_timestamp(self):
        _secs_to_vtt = getattr(mixer_module, "_secs_to_vtt")
        assert _secs_to_vtt(90.25) == "00:01:30.250"

    # -- Generación de subtítulos --

    def test_generate_srt(self, mixer, tmp_path):
        out = tmp_path / "subs.srt"
        mixer._generate_srt(SAMPLE_MANIFEST["segments"], out)
        content = out.read_text(encoding="utf-8")
        assert "Hola mundo" in content
        assert "00:00:00,000 --> 00:00:02,500" in content

    def test_generate_vtt(self, mixer, tmp_path):
        out = tmp_path / "subs.vtt"
        mixer._generate_vtt(SAMPLE_MANIFEST["segments"], out)
        content = out.read_text(encoding="utf-8")
        assert "WEBVTT" in content
        assert "Esto es una prueba" in content

    def test_generate_ass(self, mixer, tmp_path):
        out = tmp_path / "subs.ass"
        mixer._generate_ass(SAMPLE_MANIFEST["segments"], out)
        content = out.read_text(encoding="utf-8")
        assert "[Script Info]" in content
        assert "Hola mundo" in content

    # -- run() con mocks de ffmpeg --

    @patch.object(mixer_module.subprocess, "run")
    @patch.object(mixer_module.Path, "exists")
    @patch.object(mixer_module.Path, "mkdir")
    def test_run_calls_ffmpeg(self, mock_mkdir, mock_exists, mock_sub, mixer):
        mock_exists.return_value = True
        mock_sub.return_value = MagicMock(returncode=0)

        with patch("builtins.open", mock_open(read_data=json.dumps(SAMPLE_MANIFEST))):
            with patch.object(mixer_module.Path, "glob", return_value=[]):
                with patch.object(mixer_module.FinalMixer, "_export_subtitles", return_value=[]):
                    result = mixer.run(
                        "vid_abc",
                        "/fake/video_normalized.mp4",
                        "/fake/accompaniment.wav",
                        "/fake/dubbed_voice_raw.wav",
                        "/fake/tts_manifest.json",
                        "mi_video.mp4",
                    )

        # FFmpeg debe haber sido llamado al menos 2 veces (mix + mux)
        assert mock_sub.call_count >= 2
        assert "output_video" in result
