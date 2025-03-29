
from typing import Any, List, Iterator, TypedDict, Optional, Literal, Union, Generator

from api_types import (
    TokenizeInputRequest,
    DetokenizeInputRequest,
    CreateEmbeddingRequest,
    CreateCompletionRequest,
    CreateChatCompletionRequest,
    TranscriptionSegment
)

class TextToImageRequest(TypedDict):
    prompt: str
    width: int
    height: int
    batch_count: int
    upscale_factor: int

class ImageToImageRequest(TypedDict):
    image: bytes
    mask_image: Optional[bytes]
    prompt: str
    width: int
    height: int
    batch_count: int

class ImageResponse(TypedDict):
    images: List[bytes]

class SpeechToTextRequest(TypedDict):
    input_file: str
    prompt: Optional[str]
    translate: bool
    language: str
    temperature: float

class TextToSpeechRequest(TypedDict):
    text: str
    speaker: Optional[str]
    language: Optional[str]
    format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]

class ModelWorker:

    def completion(
        self,
        request: CreateCompletionRequest
    ) -> Union[Any, Iterator[Any]]:
        pass

    def chat_completion(
        self,
        request: CreateChatCompletionRequest
    ) -> Union[Any, Iterator[Any]]:
        pass

    def embeddings(
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
    
    def text_to_image(
        self,
        request: TextToImageRequest
    ) -> ImageResponse:
        pass

    def image_to_image(
        self,
        request: ImageToImageRequest
    ) -> ImageResponse:
        pass

    def speech_to_text(
        self,
        request: SpeechToTextRequest
    ) -> List[TranscriptionSegment]:
        pass

    def text_to_speech(
        self,
        request: TextToSpeechRequest
    ) -> Generator[bytes, None, None]:
        pass