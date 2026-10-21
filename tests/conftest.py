import sys
from unittest.mock import MagicMock

# Simular los módulos pesados de IA para evitar errores de importación en los tests unitarios
dummy_audio_sep = MagicMock()
sys.modules['audio_separator'] = dummy_audio_sep
sys.modules['audio_separator.separator'] = dummy_audio_sep

# Mock de la clase Separator dentro del módulo simulado para que no falle el import
class DummySeparator:
    def __init__(self, *args, **kwargs):
        pass
    def load_model(self, *args, **kwargs):
        pass
    def separate(self, *args, **kwargs):
        return ["/mock/vocals.wav", "/mock/instrumental.wav"]

dummy_audio_sep.Separator = DummySeparator

dummy_faster_whisper = MagicMock()
sys.modules['faster_whisper'] = dummy_faster_whisper

# Mock onnxruntime
dummy_onnxruntime = MagicMock()
sys.modules['onnxruntime'] = dummy_onnxruntime

