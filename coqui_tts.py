import torch
from TTS.api import TTS

from io import BytesIO
from pydub import AudioSegment
from typing import Generator

from worker import ModelWorker, TextToSpeechRequest

from settings import (
    CoquiModelSettings
)

class CoquiWorker(ModelWorker):
    
    def __init__(self, model: CoquiModelSettings):
        self._current_model_settings: CoquiModelSettings = model
        self._current_model = self._load_model(
            self._current_model_settings
        )
        self._audio_format_map:dict[str,str] = {
            "mp3": "mp3",
            "aac": "adts",
            "pcm": "wav",
            "wav": "wav",
            "flac": "flac",
            "opus": "opus"
        }

    def text_to_speech(self, request: TextToSpeechRequest) -> Generator[bytes, None, None]:
        wavs = self._current_model.tts(
            text=request["text"],
            language=request["language"] if request["language"] else self._current_model.languages[0],
            speaker=request["speaker"] if request["speaker"] else self._current_model.speakers[0]
        )
        audio_bytes = BytesIO()
        self._current_model.synthesizer.save_wav(wavs, audio_bytes)
        sample_rate = self._current_model.synthesizer.output_sample_rate
        audio = AudioSegment(audio_bytes.getvalue(), sample_width=2, frame_rate=sample_rate, channels=1)
        output_format = self._audio_format_map[request["format"]]
        with BytesIO() as output:
            audio.export(output, format=output_format)
            yield output.getvalue()
            
    def free(self):
        if self._current_model:
            del self._current_model

    @staticmethod
    def _load_model(settings: CoquiModelSettings) -> TTS:

        device = "cuda" if settings.use_cuda and torch.cuda.is_available() else "cpu"

        _model = TTS(
            model_name=settings.model_id,
            model_path=settings.model_path,
            config_path=settings.model_config_path,
            vocoder_name=settings.vocoder_id,
            vocoder_path=settings.vocoder_path,
            vocoder_config_path=settings.vocoder_config_path,
            encoder_path=settings.encoder_path,
            encoder_config_path=settings.encoder_config_path,
            speakers_file_path=settings.speakers_file_path,
            language_ids_file_path=settings.language_ids_file_path
        ).to(device)
        
        return _model

