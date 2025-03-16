from typing import List, Optional, Union, Any
from worker import ModelWorker, SpeechToTextRequest

from pywhispercpp.model import Model as Whisper

from settings import (
    WhisperModelSettings
)

from api_types import (
    TranscriptionSegment
)

class WhisperWorker(ModelWorker):
    
    def __init__(self, model: WhisperModelSettings):
        self._current_model_settings: WhisperModelSettings = model
        self._current_model = self._load_model(
            self._current_model_settings
        )

    def speech_to_text(
        self,
        request: SpeechToTextRequest
    ) -> List[TranscriptionSegment]:
        """ Transcribes. For translate as well."""
        segments = self._current_model.transcribe(media=request["input_file"], translate=request["translate"])
        return list(map(lambda seg: TranscriptionSegment(start=seg.t0,end=seg.t1,text=seg.text), segments))
    
    def free(self):
        if self._current_model:
            del self._current_model

    @staticmethod
    def _load_model(settings: WhisperModelSettings) -> Whisper:
        _model = Whisper(
            model=settings.model_path, 
            n_threads=settings.n_threads
        )
        return _model

