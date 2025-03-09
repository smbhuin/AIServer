from __future__ import annotations

from typing import List, Optional, Union, Any
from worker import ModelWorker

from pywhispercpp.model import Model as Whisper

from settings import (
    WhisperModelSettings
)

class WhisperWorker(ModelWorker):
    
    def __init__(self, model: WhisperModelSettings):
        self._current_model_settings: WhisperModelSettings = model
        self._current_model = self._load_model(
            self._current_model_settings
        )

    def transcribe(
        self,
        file_path: str,
        translate: bool,
        **kwargs
    ) -> List[str]:
        """ Transcribes. For translate as well."""
        segments = self._current_model.transcribe(media=file_path, translate=translate, **kwargs)
        return list(map(lambda x: x.text, segments))
    
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

