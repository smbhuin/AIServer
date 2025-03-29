from io import BytesIO
from typing import Generator, Iterator, Any, Union

from vllm import LLM, SamplingParams

from worker import ModelWorker, CreateCompletionRequest

from settings import (
    VLLMModelSettings
)

class VLLMWorker(ModelWorker):
    
    def __init__(self, model: VLLMModelSettings):
        self._current_model_settings: VLLMModelSettings = model
        self._current_model = self._load_model(
            self._current_model_settings
        )

    def completion(
        self,
        request: CreateCompletionRequest
    ) -> Union[Any, Iterator[Any]]:
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        outputs = self._current_model.generate(request.prompt, sampling_params)
        return outputs[0].outputs
    
    def free(self):
        if self._current_model:
            del self._current_model

    @staticmethod
    def _load_model(settings: VLLMModelSettings) -> LLM:
        
        _model = LLM(model="facebook/opt-125m")
        
        return _model

