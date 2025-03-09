from __future__ import annotations

from io import BytesIO
from pydub import AudioSegment
import wave

from piper.voice import PiperVoice
from worker import ModelWorker

from settings import (
    PiperModelSettings
)

class PiperWorker(ModelWorker):
    
    def __init__(self, model: PiperModelSettings):
        self._current_model_settings: PiperModelSettings = model
        self._current_model = self._load_model(
            self._current_model_settings
        )
        self._audio_format_map:dict[str,str] = {
            "mp3": "mp3",
            "aac": "adts",
            "pcm": "wav",
            "aac": "adts",
            "wav": "wav",
            "flac": "flac"
        }

    def synthesize(self, text: str, file_path: str, format: str) -> None:
        with BytesIO() as wav_io:
            with wave.open(wav_io, "wb") as wav_file:
                wav_file.setframerate(self._current_model.config.sample_rate)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setnchannels(1)  # mono
                for audio_bytes in self._current_model.synthesize_stream_raw(
                    text
                ):
                    wav_file.writeframes(audio_bytes)
            audio = AudioSegment(wav_io.getvalue(), sample_width=2, frame_rate=self._current_model.config.sample_rate, channels=1)
            audio.export(file_path, format=self._audio_format_map[format])

    def free(self):
        if self._current_model:
            del self._current_model

    @staticmethod
    def _load_model(settings: PiperModelSettings) -> PiperVoice:
        
        _model = PiperVoice.load(
            model_path=settings.model_path,
            config_path=settings.model_config_path,
            use_cuda=settings.use_cuda
        )
        
        return _model

