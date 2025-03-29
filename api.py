from __future__ import annotations

import json
import time
import contextlib
import os

from typing import List, Optional, Union, Dict, Annotated, Callable, Coroutine, Any, Literal, Generator
from typing_extensions import TypedDict

import uuid
import base64
import anyio
from anyio import Lock
from anyio.streams.memory import MemoryObjectSendStream
from functools import partial

from starlette.concurrency import run_in_threadpool, iterate_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette_context.plugins import RequestIdPlugin
from starlette_context.middleware import RawContextMiddleware
from sse_starlette.sse import EventSourceResponse

from fastapi import FastAPI, Depends, APIRouter, Request, Response, HTTPException, status, Body, UploadFile, Form, File
from fastapi.security import HTTPBearer
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.exceptions import ValidationException, RequestValidationError, ResponseValidationError

from fastapi.encoders import jsonable_encoder
from fastapi.utils import is_body_allowed_for_status_code

from settings import (
    ModelBackend,
    ServerSettings,
    ModelSettings,
    WhisperModelSettings,
    LlamaModelSettings,
    PiperModelSettings,
    StableDiffusionModelSettings
)

from api_types import (
    CreateCompletionRequest,
    CreateCompletionResponse,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateEmbeddingRequest,
    CreateEmbeddingResponse,
    CreateAudioTranscriptionResponse,
    CreateAudioTranscriptionVerboseResponse,
    ModelList,
    ModelData,
    TokenizeInputRequest,
    TokenizeInputResponse,
    DetokenizeInputRequest,
    DetokenizeInputResponse,
    TokenizeInputCountResponse,
    CreateSpeechRequest,
    CreateSpeechResponse,
    CreateImageGenerationRequest,
    CreateImageResponse,
    GeneratedImage
)

from worker import ModelWorker, ImageToImageRequest, TextToImageRequest, ImageResponse, SpeechToTextRequest, TextToSpeechRequest
from loader import ModelWorkerLoader
import utils

class ErrorMessage(TypedDict):
    """OpenAI style error response"""

    message: str
    type: str
    param: Optional[str]
    code: Optional[str]

class ErrorResponse(TypedDict):
    error: ErrorMessage

# Setup Bearer authentication scheme
bearer_scheme = HTTPBearer(auto_error=False)

_server_settings: Optional[ServerSettings] = None
_model_loaders: dict[str, ModelWorkerLoader] = None
_ping_message_factory: Optional[Callable[[], bytes]] = None

def set_ping_message_factory(factory: Callable[[], bytes]):
    global _ping_message_factory
    _ping_message_factory = factory

def set_server_settings(server_settings: ServerSettings):
    global _server_settings
    _server_settings = server_settings

def get_server_settings():
    yield _server_settings

def set_models_settings(models_settings: List[ModelSettings]):
    global _model_loaders
    _model_loaders = {}
    for m in models_settings:
        _model_loaders[m.model_name] = ModelWorkerLoader(m)

def get_model_loader(requested_model: str, default_backend: str) -> ModelWorkerLoader:
    if requested_model == None:
        loaders = [m for m in _model_loaders.values() if m.get_settings().backend == default_backend]
        if len(loaders) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={ # type: ErrorMessage
                    "message": "No suitable model found for the request.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_request_error"
                }
            )
        return loaders[0]
    if requested_model not in _model_loaders:
        raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={ # type: ErrorMessage
                    "message": f"The model `{requested_model}` does not exist.",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "invalid_request_error"
                }
            )
    return _model_loaders[requested_model]

def get_hosted_file_url(file_name: str) -> str:
    server_settings = next(get_server_settings())
    host = server_settings.host if server_settings.host != "0.0.0.0" else "localhost"
    files_host_url = server_settings.public_host_url if server_settings.public_host_url != None else f"http://{host}:{server_settings.port}"
    return files_host_url + f"/files/{file_name}"

async def write_to_file(file_name: str, content: Union[bytes, Generator[bytes, None, None]]):
    with open(f"files/{file_name}", "wb") as file:
        if isinstance(content, bytes):
            file.write(content)
        else :
            async for chunk in iterate_in_threadpool(content):
                file.write(chunk)

async def authenticate(
    settings: ServerSettings = Depends(get_server_settings),
    authorization: Optional[str] = Depends(bearer_scheme),
):
    # Skip API key check if it's not set in settings
    if settings.api_key is None:
        return True

    # check bearer credentials against the api_key
    if authorization and authorization.credentials == settings.api_key:
        # api key is valid
        return authorization.credentials

    # raise http error 401
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={ # type: ErrorMessage
            "message": "Incorrect API key provided.",
            "type": "invalid_request_error",
            "param": None,
            "code": "invalid_api_key"
        },
    )

class AIServerRoute(APIRoute):
    """Custom APIRoute that handles application errors and exceptions"""

    def get_route_handler(
        self,
    ) -> Callable[[Request], Coroutine[None, None, Response]]:
        """Defines custom route handler that calculates the request processing time"""

        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            start_sec = time.perf_counter()
            response = await original_route_handler(request)
            elapsed_time_ms = int((time.perf_counter() - start_sec) * 1000)
            response.headers["openai-processing-ms"] = f"{elapsed_time_ms}"
            return response
            
        return custom_route_handler

async def http_exception_handler(request: Request, exc: HTTPException) -> Response:
    headers = getattr(exc, "headers", None)
    if not is_body_allowed_for_status_code(exc.status_code):
        return Response(status_code=exc.status_code, headers=headers)
    error_message: ErrorMessage = None
    if isinstance(exc.detail, dict):
        error_message = exc.detail
    else:
        error_message: ErrorMessage = {
            "message": exc.detail,
            "type": "server_error",
            "param": None,
            "code": "server_error"
        }
    return JSONResponse(
        {"error": error_message}, status_code=exc.status_code, headers=headers
    )

async def app_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    error_message: ErrorMessage = None
    error_code = None
    detail = None
    if isinstance(exc, RequestValidationError):
        error_message = {
            "message": "Your request was malformed or missing some required parameters.",
            "type": "invalid_request_error",
            "param": None,
            "code": "validation_error"
        }
        detail = jsonable_encoder(exc.errors())
        error_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    else:
        error_message = {
            "message": "The server had an error while processing your request.",
            "type": "server_error",
            "param": None,
            "code": "server_error"
        }
        error_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return JSONResponse(
        content={"error":error_message, "detail": detail},
        status_code=error_code
    )

_responses = {
    400: {
        "model": ErrorResponse,
        "description": "Item not found",
        "content": {
            "application/json": {
                "schema": {
                    "$ref": "#/components/schemas/ErrorResponse",
                    "example": {
                        "error": {
                            "message": "Item not found",
                            "type": "invalid_request_error",
                            "param": "",
                            "code": "invalid_request_error"
                        }
                    }
                }
            }
        }
    },
    404: {
        "description": "Item not found",
        "content": {
            "application/json": {
                "schema": {
                    "$ref": "#/components/schemas/ErrorResponse",
                    "example": {
                        "error": {
                            "message": "Item not found",
                            "type": "invalid_request_error",
                            "param": "",
                            "code": "not_found_error"
                        }
                    }
                }
            }
        }
    },
    422: {
        "description": "Validation Error",
        "content": {
            "application/json": {
                "schema": {
                    "$ref": "#/components/schemas/ErrorResponse",
                    "example": {
                        "error": {
                            "message": "Your request was malformed or missing some required parameters.",
                            "type": "invalid_request_error",
                            "param": "",
                            "code": "validation_error"
                        }
                    }
                }
            }
        }
    },
}

async def check_connection(request: Request):
    if await request.is_disconnected():
        print(
            f"Disconnected from client (via refresh/close) before llm invoked {request.client}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Connection is terminated by client.",
                "type": "invalid_request_error",
                "param": None,
                "code": "invalid_request_error"
            }
        )

router = APIRouter(route_class=AIServerRoute) 

def create_app(server_settings: ServerSettings, models_settings: List[ModelSettings]) -> FastAPI:
    assert len(models_settings) > 0, "No models provided!"

    set_server_settings(server_settings)
    set_models_settings(models_settings)

    if server_settings.disable_ping_events:
        set_ping_message_factory(lambda: bytes())

    middleware = [Middleware(RawContextMiddleware, plugins=(RequestIdPlugin(),))]
    app = FastAPI(
        middleware=middleware,
        title="AIServer. Host your own AI models!",
        version="1.0.0",
        root_path=server_settings.root_path,
    )

    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, app_exception_handler)
    app.add_exception_handler(ResponseValidationError, app_exception_handler)
    app.add_exception_handler(Exception, app_exception_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.include_router(router, responses=_responses)

    # mount files directory 
    app.mount("/files", StaticFiles(directory="files"), name="static")

    return app

async def get_event_publisher(
    request: Request,
    inner_send_chan: MemoryObjectSendStream[Any],
    body: Union[CreateCompletionRequest, CreateChatCompletionRequest],
    loader: ModelWorkerLoader,
):
    server_settings = next(get_server_settings())
    interrupt_requests = (
        server_settings.interrupt_requests if server_settings else False
    )
    async with contextlib.asynccontextmanager(loader.get_worker)() as worker:
        async with inner_send_chan:
            try:
                func = worker.chat_completion if isinstance(body, CreateChatCompletionRequest) else worker.completion
                iterator = await run_in_threadpool(func, request=body)
                async for chunk in iterate_in_threadpool(iterator):
                    await inner_send_chan.send(dict(data=json.dumps(chunk)))
                    if await request.is_disconnected():
                        raise anyio.get_cancelled_exc_class()()
                    if interrupt_requests and loader.outer_lock.locked():
                        await inner_send_chan.send(dict(data="[DONE]"))
                        raise anyio.get_cancelled_exc_class()()
                await inner_send_chan.send(dict(data="[DONE]"))
            except anyio.get_cancelled_exc_class() as e:
                print("disconnected")
                with anyio.move_on_after(1, shield=True):
                    print(
                        f"Disconnected from client (via refresh/close) {request.client}"
                    )
                    raise e


# Health check endpoint
@router.get(
    "/health",
    summary="Health status"
)
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@router.get(
    "/v1/models",
    summary="List models",
    description="Lists the currently available models, and provides basic information about each one such as the owner and availability.",
    dependencies=[Depends(authenticate)]
)
async def get_models(
) -> ModelList:
    return ModelList(object="list", data=[
        ModelData(
            id=loader.get_settings().model_name,
            object="model",
            owned_by="aiserver",
            created=loader.get_settings().created,
            permissions=[]
        )
        for loader in _model_loaders.values()
    ])

@router.get(
    "/v1/models/{model}",
    summary="Retrieve model",
    description="Retrieves a model instance, providing basic information about the model such as the owner and permissioning.",
    dependencies=[Depends(authenticate)]
)
async def get_models(
    model: str
) -> ModelData:
    settings = _model_loaders[model].get_settings()
    return ModelData(
        id=settings.model_name,
        object="model",
        owned_by="aiserver",
        created=settings.created,
        permissions=[]
    )

@router.post(
    "/v1/completions",
    summary="Completion",
    dependencies=[Depends(authenticate)],
    response_model=Union[
        CreateCompletionResponse,
        str,
    ],
    responses={
        200: {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "schema": {
                        "anyOf": [
                            {"$ref": "#/components/schemas/CreateCompletionResponse"}
                        ],
                        "title": "Completion response, when stream=False",
                    }
                },
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "title": "Server Side Streaming response, when stream=True. "
                        + "See SSE format: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format",  # noqa: E501
                        "example": """data: {... see CreateCompletionResponse ...} \\n\\n data: ... \\n\\n ... data: [DONE]""",
                    }
                },
            },
        }
    }
)
async def create_completion(
    request: Request,
    body: CreateCompletionRequest,
) -> CreateCompletionResponse:
    
    if isinstance(body.prompt, list):
        assert len(body.prompt) <= 1
        body.prompt = body.prompt[0] if len(body.prompt) > 0 else ""

    loader = get_model_loader(body.model, ModelBackend.llama)
    
    # handle streaming request
    if body.stream:
        send_chan, recv_chan = anyio.create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(  # type: ignore
                get_event_publisher,
                request=request,
                inner_send_chan=send_chan,
                body=body,
                loader=loader,
            ),
            sep="\n",
            ping_message_factory=_ping_message_factory,
        )

    # handle regular request
    async with contextlib.asynccontextmanager(loader.get_worker)() as worker:
        await check_connection(request)
        return await run_in_threadpool(worker.completion, request=body)


@router.post(
    "/v1/embeddings",
    summary="Embedding",
    dependencies=[Depends(authenticate)]
)
async def create_embedding(
    body: CreateEmbeddingRequest,
) -> CreateEmbeddingResponse:
    loader = get_model_loader(body.model, ModelBackend.llama)
    async with contextlib.asynccontextmanager(loader.get_worker)() as worker:
        resp = await run_in_threadpool(
            worker.embeddings,
            request=body,
        )
        return CreateEmbeddingResponse(**resp)

@router.post(
    "/v1/chat/completions",
    summary="Chat",
    dependencies=[Depends(authenticate)],
    response_model=Union[CreateChatCompletionResponse, str],
    responses={
        200: {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "schema": {
                        "anyOf": [
                            {
                                "$ref": "#/components/schemas/CreateChatCompletionResponse"
                            }
                        ],
                        "title": "Completion response, when stream=False",
                    }
                },
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "title": "Server Side Streaming response, when stream=True"
                        + "See SSE format: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format",  # noqa: E501
                        "example": """data: {... see CreateChatCompletionResponse ...} \\n\\n data: ... \\n\\n ... data: [DONE]""",
                    }
                },
            },
        }
    }
)
async def create_chat_completion(
    request: Request,
    body: CreateChatCompletionRequest = Body(
        openapi_examples={
            "normal": {
                "summary": "Chat Completion",
                "value": {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "What is the capital of France?"},
                    ],
                },
            },
            "json_mode": {
                "summary": "JSON Mode",
                "value": {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Who won the world series in 2020"},
                    ],
                    "response_format": {"type": "json_object"},
                },
            },
            "tool_calling": {
                "summary": "Tool Calling",
                "value": {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Extract Jason is 30 years old."},
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "User",
                                "description": "User record",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "age": {"type": "number"},
                                    },
                                    "required": ["name", "age"],
                                },
                            },
                        }
                    ],
                    "tool_choice": {
                        "type": "function",
                        "function": {
                            "name": "User",
                        },
                    },
                },
            },
            "logprobs": {
                "summary": "Logprobs",
                "value": {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "What is the capital of France?"},
                    ],
                    "logprobs": True,
                    "top_logprobs": 10,
                },
            },
        }
    ),
) -> CreateChatCompletionResponse:
    # This is a workaround for an issue in FastAPI dependencies
    # where the dependency is cleaned up before a StreamingResponse
    # is complete.
    # https://github.com/tiangolo/fastapi/issues/11143

    loader = get_model_loader(body.model, ModelBackend.llama)

    # handle streaming request
    if body.stream:
        send_chan, recv_chan = anyio.create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(  # type: ignore
                get_event_publisher,
                request=request,
                inner_send_chan=send_chan,
                body=body,
                loader=loader
            ),
            sep="\n",
            ping_message_factory=_ping_message_factory,
        )

    # handle regular request
    async with contextlib.asynccontextmanager(loader.get_worker)() as worker:
        await check_connection(request)
        return await run_in_threadpool(worker.chat_completion, request=body)


@router.post(
    "/extras/tokenize",
    summary="Tokenize",
    dependencies=[Depends(authenticate)]
)
async def tokenize(
    body: TokenizeInputRequest
) -> TokenizeInputResponse:
    loader = get_model_loader(body.model, ModelBackend.llama)
    async with contextlib.asynccontextmanager(loader.get_worker)() as worker:
        tokens = worker.tokenize(body)
        return TokenizeInputResponse(tokens=tokens)

@router.post(
    "/extras/tokenize/count",
    summary="Tokenize Count",
    dependencies=[Depends(authenticate)]
)
async def count_tokens(
    body: TokenizeInputRequest
) -> TokenizeInputCountResponse:
    loader = get_model_loader(body.model, ModelBackend.llama)
    async with contextlib.asynccontextmanager(loader.get_worker)() as worker:
        tokens = worker.tokenize(body)
        return TokenizeInputCountResponse(count=len(tokens))

@router.post(
    "/extras/detokenize",
    summary="Detokenize",
    dependencies=[Depends(authenticate)]
)
async def detokenize(
    body: DetokenizeInputRequest
) -> DetokenizeInputResponse:
    loader = get_model_loader(body.model, ModelBackend.llama)
    async with contextlib.asynccontextmanager(loader.get_worker)() as worker:
        text = worker.detokenize(body.tokens)
        return DetokenizeInputResponse(text=text)

@router.post(
    "/v1/audio/transcriptions",
    summary="Transcription",
    dependencies=[Depends(authenticate)]
)
async def transcription(
    request: Request,
    file: UploadFile,
    model: Annotated[str, Form()] = None,
    prompt: Annotated[str, Form()] = None,
    response_format: Annotated[Literal["json", "text", "srt", "verbose_json", "vtt"], Form()] = "json",
    temperature: Annotated[float, Form()] = 0.0,
    language: Annotated[str, Form()] = 'en'
):
    
    file_path = f"files/{uuid.uuid4()}_{file.filename}"
    kwargs: SpeechToTextRequest = {
        "input_file": file_path,
        "translate":False,
        "language": language,
        "temperature": temperature
    }
    if prompt is not None:
        kwargs["prompt"] = prompt
    loader = get_model_loader(model, ModelBackend.whisper)
    async with contextlib.asynccontextmanager(loader.get_worker)() as worker:
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        await check_connection(request)
        segments = await run_in_threadpool(worker.speech_to_text, request=kwargs)
        os.remove(file_path)
        if response_format == "text":
            return utils.text_from(segments)
        elif response_format == "srt":
            return utils.srt_from(segments)
        elif response_format == "vtt":
            return utils.vtt_from(segments)
        elif response_format == "verbose_json":
            return CreateAudioTranscriptionVerboseResponse(text=utils.text_from(segments), segments=segments)
        else:
            return CreateAudioTranscriptionResponse(text=utils.text_from(segments))
            
@router.post(
    "/v1/audio/translations",
    summary="Translation",
    dependencies=[Depends(authenticate)]
)
async def translation(
    request: Request,
    file: UploadFile,
    model: Annotated[str, Form()] = None,
    prompt: Annotated[str, Form()] = None,
    response_format: Annotated[Literal["json", "text", "srt", "verbose_json", "vtt"], Form()] = "json",
    temperature: Annotated[float, Form()] = 0.0
):
    file_path = f"files/{uuid.uuid4()}_{file.filename}"
    kwargs: SpeechToTextRequest = {
        "input_file": file_path,
        "translate":True,
        "temperature": temperature
    }
    if prompt is not None:
        kwargs["prompt"] = prompt
    loader = get_model_loader(model, ModelBackend.whisper)
    async with contextlib.asynccontextmanager(loader.get_worker)() as worker:
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        await check_connection(request)
        segments = await run_in_threadpool(worker.speech_to_text, request=kwargs)
        os.remove(file_path)
        if response_format == "text":
            return utils.text_from(segments)
        elif response_format == "srt":
            return utils.srt_from(segments)
        elif response_format == "vtt":
            return utils.vtt_from(segments)
        elif response_format == "verbose_json":
            return CreateAudioTranscriptionVerboseResponse(text=utils.text_from(segments), segments=segments)
        else:
            return CreateAudioTranscriptionResponse(text=utils.text_from(segments))

@router.post(
    "/v1/audio/speech",
    summary="Speech",
    dependencies=[Depends(authenticate)],
    responses={
        200: {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "schema": {
                        "anyOf": [
                            {
                                "$ref": "#/components/schemas/CreateSpeechResponse"
                            }
                        ],
                        "title": "Completion response, when stream=False",
                    }
                },
                "audio/mpeg": {},
                "audio/opus": {},
                "audio/aac": {},
                "audio/flac": {},
                "audio/wav": {},
                "audio/pcm": {}
            },
        }
    }
)
async def create_speech(
    request: Request,
    body: CreateSpeechRequest
) -> CreateSpeechResponse:
    kwargs: TextToSpeechRequest = {
        "text": body.input,
        "format": body.response_format,
        "language": body.language,
        "speaker": body.voice
    }
    media_types = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm"
    }
    loader = get_model_loader(body.model, ModelBackend.piper)
    async with contextlib.asynccontextmanager(loader.get_worker)() as worker:
        await check_connection(request)
        audio_generator = await run_in_threadpool(worker.text_to_speech, request=kwargs)
        if body.response_type == "url":
            file_name = f"{uuid.uuid4()}.{body.response_format}"
            await write_to_file(file_name, audio_generator)
            return CreateSpeechResponse(url=get_hosted_file_url(file_name))
        return StreamingResponse(audio_generator, media_type=media_types[body.response_format])
        

async def generated_images(images: List[bytes], response_format: str) -> List[GeneratedImage]:
    data: List[GeneratedImage] = []
    if response_format == "url":
        for img_bytes in images:
            file_name = f"{uuid.uuid4()}.png"
            await write_to_file(file_name, img_bytes)
            data.append(
                GeneratedImage(
                    url= get_hosted_file_url(file_name) 
                )
            )
    else:
        for img_bytes in images:
            data.append(
                GeneratedImage(
                    b64_json=base64.b64encode(img_bytes).decode()
                )
            )
    return data

@router.post(
    "/v1/images/generations",
    summary="Create image",
    description="Creates an image given a prompt.",
    dependencies=[Depends(authenticate)],
    response_model=CreateImageResponse,
    response_model_exclude_none=True
)
async def create_image(
    request: Request,
    body: CreateImageGenerationRequest
) -> CreateImageResponse:
    created_at = int(time.time())
    loader = get_model_loader(body.model, ModelBackend.stable_diffusion)
    width, height = body.size.split('x')
    kwargs: TextToImageRequest = {
        "prompt": body.prompt,
        "batch_count": body.n,
        "width": int(width),
        "height": int(height),
        "upscale_factor": body.upscale_factor
    }
    async with contextlib.asynccontextmanager(loader.get_worker)() as worker:
        await check_connection(request)
        image_resp: ImageResponse = await run_in_threadpool(worker.text_to_image, request=kwargs)
        gen_images = await generated_images(image_resp["images"], body.response_format)
        return CreateImageResponse(created=created_at, data=gen_images)
            

@router.post(
    "/v1/images/edits",
    summary="Create image edit",
    description="Creates an edited or extended image given an original image and a prompt.",
    dependencies=[Depends(authenticate)],
    response_model=CreateImageResponse,
    response_model_exclude_none=True
)
async def edit_image(
    request: Request,
    image: Annotated[bytes, File(description="The image to edit. Must be a valid PNG file, less than 4MB, and square. If mask is not provided, image must have transparency, which will be used as the mask.")],
    prompt: Annotated[str, Form(
        description="The prompt to generate image for."
    )],
    model: Annotated[Optional[str], Form(
        description="The model to use for generating image."
    )] = None,
    mask: Annotated[Optional[bytes], File(description="An additional image whose fully transparent areas (e.g. where alpha is zero) indicate where image should be edited. Must be a valid PNG file, less than 4MB, and have the same dimensions as image.")] = None,
    size: Annotated[str, Form(description="The size of the image to be generated in pixels.")] = "512x512",
    response_format: Annotated[Literal['url','b64_json'], Form(description="The response format. Valid values are 'url' or 'b64_json'.")] = "url",
    n: Annotated[int, Form(ge=1, le=10, description="The number of images to generate. 1-10")] = 1,
    upscale_factor: Annotated[int, Form(ge=1, le=3, description="The image upscaling factor. 1-3")] = 1,
    user: Annotated[Optional[str], Form()] = None,
) -> CreateImageResponse:
    created_at = int(time.time())
    loader = get_model_loader(model, ModelBackend.stable_diffusion)
    width, height = size.split('x')
    kwargs: ImageToImageRequest = {
        "image":image,
        "mask_image":mask,
        "prompt": prompt,
        "batch_count": n,
        "width": int(width),
        "height": int(height),
        "upscale_factor": upscale_factor
    }
    async with contextlib.asynccontextmanager(loader.get_worker)() as worker:
        await check_connection(request)
        image_resp: ImageResponse = await run_in_threadpool(worker.image_to_image, request=kwargs)
        gen_images = await generated_images(image_resp["images"], response_format)
        return CreateImageResponse(created=created_at, data=gen_images)
