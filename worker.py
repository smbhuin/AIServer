
from typing import Any, List, Iterator

from api_types import (
    TokenizeInputRequest,
    DetokenizeInputRequest,
    CreateEmbeddingRequest,
    CreateCompletionRequest,
    CreateChatCompletionRequest,
    CreateAudioTranscriptionResponse,
    CreateImageGenerationRequest,
    TranscriptionSegment
)

class ModelWorker:

    def create_completion(
        self,
        request: CreateCompletionRequest
    ) -> Any | Iterator[Any]:
        pass

    def create_chat_completion(
        self,
        request: CreateChatCompletionRequest
    ) -> Any | Iterator[Any]:
        pass

    def create_embedding(
        self,
        request: CreateEmbeddingRequest
    ) -> Any:
        pass

    def tokenize(
        self,
        request: TokenizeInputRequest
    ) -> List[int]:
        pass

    def detokenize(
        self,
        request: DetokenizeInputRequest
    ) -> str:
        pass
    
    def create_image(
        self,
        request: CreateImageGenerationRequest
    ) -> List[Any]: # List[PIL.Image]
        pass

    def speech_to_text(
        self,
        file_path: str,
        translate: bool,
        **kwargs
    ) -> List[TranscriptionSegment]:
        pass

