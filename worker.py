
from typing import Any, List, Iterator, TypedDict, Optional
from PIL.Image import Image

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

class TextToImageRequest(TypedDict):
    prompt: str
    width: Optional[int]
    height: Optional[int]
    batch_count: Optional[int]

class ImageToImageRequest(TypedDict):
    image: bytes
    mask_image: Optional[bytes]
    prompt: str
    width: Optional[int]
    height: Optional[int]
    batch_count: Optional[int]

class ImageResponse(TypedDict):
    images: List[Image]

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
    
    def txt_to_img(
        self,
        request: TextToImageRequest
    ) -> ImageResponse:
        pass

    def img_to_img(
        self,
        request: ImageToImageRequest
    ) -> ImageResponse:
        pass

    def speech_to_text(
        self,
        file_path: str,
        translate: bool,
        **kwargs
    ) -> List[TranscriptionSegment]:
        pass

