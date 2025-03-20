from io import BytesIO
from pydub import AudioSegment
from typing import Generator

from piper.voice import PiperVoice
from worker import ModelWorker, TextToSpeechRequest

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
            "wav": "wav",
            "flac": "flac"
        }

    def text_to_speech(self, request: TextToSpeechRequest) -> Generator[bytes, None, None]:
        for audio_bytes in self._current_model.synthesize_stream_raw(request["text"]):
            audio = AudioSegment(audio_bytes, sample_width=2, frame_rate=self._current_model.config.sample_rate, channels=1)
            output_format = self._audio_format_map[request["format"]]
            with BytesIO() as output:
                audio.export(output, format=output_format)
                yield output.getvalue()

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

