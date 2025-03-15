from __future__ import annotations

from typing import List, Optional, Union, Dict, Any
from typing_extensions import TypedDict, Literal, NotRequired

from pydantic import BaseModel, Field

model_field = Field(
    default=None,
    description="The model to use for generating completions."
)

max_tokens_field = Field(
    default=16, ge=1, description="The maximum number of tokens to generate."
)

min_tokens_field = Field(
    default=0,
    ge=0,
    description="The minimum number of tokens to generate. It may return fewer tokens if another condition is met (e.g. max_tokens, stop).",
)

temperature_field = Field(
    default=0.8,
    description="Adjust the randomness of the generated text.\n\n"
    + "Temperature is a hyperparameter that controls the randomness of the generated text. It affects the probability distribution of the model's output tokens. A higher temperature (e.g., 1.5) makes the output more random and creative, while a lower temperature (e.g., 0.5) makes the output more focused, deterministic, and conservative. The default value is 0.8, which provides a balance between randomness and determinism. At the extreme, a temperature of 0 will always pick the most likely next token, leading to identical outputs in each run.",
)

top_p_field = Field(
    default=0.95,
    ge=0.0,
    le=1.0,
    description="Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P.\n\n"
    + "Top-p sampling, also known as nucleus sampling, is another text generation method that selects the next token from a subset of tokens that together have a cumulative probability of at least p. This method provides a balance between diversity and quality by considering both the probabilities of tokens and the number of tokens to sample from. A higher value for top_p (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text.",
)

min_p_field = Field(
    default=0.05,
    ge=0.0,
    le=1.0,
    description="Sets a minimum base probability threshold for token selection.\n\n"
    + "The Min-P sampling method was designed as an alternative to Top-P, and aims to ensure a balance of quality and variety. The parameter min_p represents the minimum probability for a token to be considered, relative to the probability of the most likely token. For example, with min_p=0.05 and the most likely token having a probability of 0.9, logits with a value less than 0.045 are filtered out.",
)

stop_field = Field(
    default=None,
    description="A list of tokens at which to stop generation. If None, no stop tokens are used.",
)

stream_field = Field(
    default=False,
    description="Whether to stream the results as they are generated. Useful for chatbots.",
)

top_k_field = Field(
    default=40,
    ge=0,
    description="Limit the next token selection to the K most probable tokens.\n\n"
    + "Top-k sampling is a text generation method that selects the next token only from the top k most likely tokens predicted by the model. It helps reduce the risk of generating low-probability or nonsensical tokens, but it may also limit the diversity of the output. A higher value for top_k (e.g., 100) will consider more tokens and lead to more diverse text, while a lower value (e.g., 10) will focus on the most probable tokens and generate more conservative text.",
)

repeat_penalty_field = Field(
    default=1.1,
    ge=0.0,
    description="A penalty applied to each token that is already generated. This helps prevent the model from repeating itself.\n\n"
    + "Repeat penalty is a hyperparameter used to penalize the repetition of token sequences during text generation. It helps prevent the model from generating repetitive or monotonous text. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient.",
)

presence_penalty_field = Field(
    default=0.0,
    ge=-2.0,
    le=2.0,
    description="Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.",
)

frequency_penalty_field = Field(
    default=0.0,
    ge=-2.0,
    le=2.0,
    description="Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.",
)

mirostat_mode_field = Field(
    default=0,
    ge=0,
    le=2,
    description="Enable Mirostat constant-perplexity algorithm of the specified version (1 or 2; 0 = disabled)",
)

mirostat_tau_field = Field(
    default=5.0,
    ge=0.0,
    le=10.0,
    description="Mirostat target entropy, i.e. the target perplexity - lower values produce focused and coherent text, larger values produce more diverse and less coherent text",
)

mirostat_eta_field = Field(
    default=0.1, ge=0.001, le=1.0, description="Mirostat learning rate"
)

grammar = Field(
    default=None,
    description="A CBNF grammar (as string) to be used for formatting the model's output.",
)


class CreateCompletionRequest(BaseModel):
    prompt: Union[str, List[str]] = Field(
        default="", description="The prompt to generate completions for."
    )
    suffix: Optional[str] = Field(
        default=None,
        description="A suffix to append to the generated text. If None, no suffix is appended. Useful for chatbots.",
    )
    max_tokens: Optional[int] = Field(
        default=16, ge=0, description="The maximum number of tokens to generate."
    )
    min_tokens: int = min_tokens_field
    temperature: float = temperature_field
    top_p: float = top_p_field
    min_p: float = min_p_field
    echo: bool = Field(
        default=False,
        description="Whether to echo the prompt in the generated text. Useful for chatbots.",
    )
    stop: Optional[Union[str, List[str]]] = stop_field
    stream: bool = stream_field
    logprobs: Optional[int] = Field(
        default=None,
        ge=0,
        description="The number of logprobs to generate. If None, no logprobs are generated.",
    )
    presence_penalty: Optional[float] = presence_penalty_field
    frequency_penalty: Optional[float] = frequency_penalty_field
    logit_bias: Optional[Dict[str, float]] = Field(None)
    seed: Optional[int] = Field(None)

    # ignored or currently unsupported
    model: Optional[str] = model_field
    n: Optional[int] = 1
    best_of: Optional[int] = 1
    user: Optional[str] = Field(default=None)

    # llama.cpp specific parameters
    top_k: int = top_k_field
    repeat_penalty: float = repeat_penalty_field
    logit_bias_type: Optional[Literal["input_ids", "tokens"]] = Field(None)
    mirostat_mode: int = mirostat_mode_field
    mirostat_tau: float = mirostat_tau_field
    mirostat_eta: float = mirostat_eta_field
    grammar: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "\n\n### Instructions:\nWhat is the capital of France?\n\n### Response:\n",
                    "stop": ["\n", "###"],
                }
            ]
        }
    }

class CreateEmbeddingRequest(BaseModel):
    model: Optional[str] = model_field
    input: Union[str, List[str]] = Field(description="The input to embed.")
    user: Optional[str] = Field(default=None)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "input": "The food was delicious and the waiter...",
                }
            ]
        }
    }

class ChatCompletionRequestMessage(BaseModel):
    role: Literal["system", "user", "assistant", "function"] = Field(
        default="user", description="The role of the message."
    )
    content: Optional[str] = Field(
        default="", description="The content of the message."
    )

# NOTE: Defining this correctly using annotations seems to break pydantic validation.
#       This is a workaround until we can figure out how to do this correctly
# JsonType = Union[None, int, str, bool, List["JsonType"], Dict[str, "JsonType"]]
JsonType = Union[None, int, str, bool, List[Any], Dict[str, Any]]

class ChatCompletionFunction(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: Dict[str, JsonType]  # TODO: make this more specific

class ChatCompletionRequestFunctionCallOption(TypedDict):
    name: str

ChatCompletionRequestFunctionCall = Union[
    Literal["none", "auto"], ChatCompletionRequestFunctionCallOption
]

class ChatCompletionNamedToolChoiceFunction(TypedDict):
    name: str

class ChatCompletionNamedToolChoice(TypedDict):
    type: Literal["function"]
    function: ChatCompletionNamedToolChoiceFunction

ChatCompletionToolChoiceOption = Union[
    Literal["none", "auto", "required"], ChatCompletionNamedToolChoice
]

ChatCompletionFunctionParameters = Dict[str, JsonType]  # TODO: make this more specific

class ChatCompletionToolFunction(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: ChatCompletionFunctionParameters

class ChatCompletionTool(TypedDict):
    type: Literal["function"]
    function: ChatCompletionToolFunction

class ChatCompletionRequestResponseFormat(TypedDict):
    type: Literal["text", "json_object"]
    schema: NotRequired[
        JsonType
    ]  # https://docs.endpoints.anyscale.com/guides/json_mode/

class CreateChatCompletionRequest(BaseModel):
    messages: List[ChatCompletionRequestMessage] = Field(
        default=[], description="A list of messages to generate completions for."
    )
    functions: Optional[List[ChatCompletionFunction]] = Field(
        default=None,
        description="A list of functions to apply to the generated completions.",
    )
    function_call: Optional[ChatCompletionRequestFunctionCall] = Field(
        default=None,
        description="A function to apply to the generated completions.",
    )
    tools: Optional[List[ChatCompletionTool]] = Field(
        default=None,
        description="A list of tools to apply to the generated completions.",
    )
    tool_choice: Optional[ChatCompletionToolChoiceOption] = Field(
        default=None,
        description="A tool to apply to the generated completions.",
    )  # TODO: verify
    max_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens to generate. Defaults to inf",
    )
    min_tokens: int = min_tokens_field
    logprobs: Optional[bool] = Field(
        default=False,
        description="Whether to output the logprobs or not. Default is True",
    )
    top_logprobs: Optional[int] = Field(
        default=None,
        ge=0,
        description="The number of logprobs to generate. If None, no logprobs are generated. logprobs need to set to True.",
    )
    temperature: float = temperature_field
    top_p: float = top_p_field
    min_p: float = min_p_field
    stop: Optional[Union[str, List[str]]] = stop_field
    stream: bool = stream_field
    presence_penalty: Optional[float] = presence_penalty_field
    frequency_penalty: Optional[float] = frequency_penalty_field
    logit_bias: Optional[Dict[str, float]] = Field(None)
    seed: Optional[int] = Field(None)
    response_format: Optional[ChatCompletionRequestResponseFormat] = Field(
        default=None,
    )

    # ignored or currently unsupported
    model: Optional[str] = model_field
    n: Optional[int] = 1
    user: Optional[str] = Field(None)

    # llama.cpp specific parameters
    top_k: int = top_k_field
    repeat_penalty: float = repeat_penalty_field
    logit_bias_type: Optional[Literal["input_ids", "tokens"]] = Field(None)
    mirostat_mode: int = mirostat_mode_field
    mirostat_tau: float = mirostat_tau_field
    mirostat_eta: float = mirostat_eta_field
    grammar: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [
                        ChatCompletionRequestMessage(
                            role="system", content="You are a helpful assistant."
                        ).model_dump(),
                        ChatCompletionRequestMessage(
                            role="user", content="What is the capital of France?"
                        ).model_dump(),
                    ]
                }
            ]
        }
    }

class ChatCompletionTopLogprobToken(TypedDict):
    token: str
    logprob: float
    bytes: Optional[List[int]]

class ChatCompletionLogprobToken(ChatCompletionTopLogprobToken):
    token: str
    logprob: float
    bytes: Optional[List[int]]
    top_logprobs: List[ChatCompletionTopLogprobToken]

class ChatCompletionLogprobs(TypedDict):
    content: Optional[List[ChatCompletionLogprobToken]]
    refusal: Optional[List[ChatCompletionLogprobToken]]

class ChatCompletionResponseChoice(TypedDict):
    index: int
    message: ChatCompletionResponseMessage
    logprobs: Optional[ChatCompletionLogprobs]
    finish_reason: Optional[str]

class CreateChatCompletionResponse(TypedDict):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: CompletionUsage

class ChatCompletionResponseFunctionCall(TypedDict):
    name: str
    arguments: str

class ChatCompletionMessageToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: ChatCompletionMessageToolCallFunction

class ChatCompletionMessageToolCallFunction(TypedDict):
    name: str
    arguments: str

ChatCompletionMessageToolCalls = List[ChatCompletionMessageToolCall]

class ChatCompletionResponseMessage(TypedDict):
    content: Optional[str]
    tool_calls: NotRequired["ChatCompletionMessageToolCalls"]
    role: Literal["assistant", "function"]  # NOTE: "function" may be incorrect here
    function_call: NotRequired[ChatCompletionResponseFunctionCall]  # DEPRECATED


class ChatCompletionMessageToolCallChunkFunction(TypedDict):
    name: Optional[str]
    arguments: str

class ChatCompletionMessageToolCallChunk(TypedDict):
    index: int
    id: NotRequired[str]
    type: Literal["function"]
    function: ChatCompletionMessageToolCallChunkFunction

class ChatCompletionStreamResponseDeltaEmpty(TypedDict):
    pass

class ChatCompletionStreamResponseDeltaFunctionCall(TypedDict):
    name: str
    arguments: str

class ChatCompletionStreamResponseDelta(TypedDict):
    content: NotRequired[Optional[str]]
    function_call: NotRequired[
        Optional[ChatCompletionStreamResponseDeltaFunctionCall]
    ]  # DEPRECATED
    tool_calls: NotRequired[Optional[List[ChatCompletionMessageToolCallChunk]]]
    role: NotRequired[Optional[Literal["system", "user", "assistant", "tool"]]]

class ChatCompletionStreamResponseChoice(TypedDict):
    index: int
    delta: Union[
        ChatCompletionStreamResponseDelta, ChatCompletionStreamResponseDeltaEmpty
    ]
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "function_call"]]
    logprobs: NotRequired[Optional[ChatCompletionLogprobs]]

class CreateChatCompletionStreamResponse(TypedDict):
    id: str
    model: str
    object: Literal["chat.completion.chunk"]
    created: int
    choices: List[ChatCompletionStreamResponseChoice]


class ModelData(TypedDict):
    id: str
    object: Literal["model"]
    owned_by: str
    permissions: List[str]

class ModelList(TypedDict):
    object: Literal["list"]
    data: List[ModelData]

class TokenizeInputRequest(BaseModel):
    model: Optional[str] = model_field
    input: str = Field(description="The input to tokenize.")

    model_config = {
        "json_schema_extra": {"examples": [{"input": "How many tokens in this query?"}]}
    }

class TokenizeInputResponse(BaseModel):
    tokens: List[int] = Field(description="A list of tokens.")

    model_config = {"json_schema_extra": {"example": {"tokens": [123, 321, 222]}}}

class TokenizeInputCountResponse(BaseModel):
    count: int = Field(description="The number of tokens in the input.")

    model_config = {"json_schema_extra": {"example": {"count": 5}}}

class DetokenizeInputRequest(BaseModel):
    model: Optional[str] = model_field
    tokens: List[int] = Field(description="A list of toekns to detokenize.")

    model_config = {"json_schema_extra": {"example": [{"tokens": [123, 321, 222]}]}}

class DetokenizeInputResponse(BaseModel):
    text: str = Field(description="The detokenized text.")

    model_config = {
        "json_schema_extra": {"example": {"text": "How many tokens in this query?"}}
    }

class EmbeddingUsage(TypedDict):
    prompt_tokens: int
    total_tokens: int

class Embedding(TypedDict):
    index: int
    object: str
    embedding: Union[List[float], List[List[float]]]

class CreateEmbeddingResponse(TypedDict):
    object: Literal["list"]
    model: str
    data: List[Embedding]
    usage: EmbeddingUsage

class CompletionLogprobs(TypedDict):
    text_offset: List[int]
    token_logprobs: List[Optional[float]]
    tokens: List[str]
    top_logprobs: List[Optional[Dict[str, float]]]

class CompletionChoice(TypedDict):
    text: str
    index: int
    logprobs: Optional[CompletionLogprobs]
    finish_reason: Optional[Literal["stop", "length"]]

class CompletionUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class CreateCompletionResponse(TypedDict):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: NotRequired[CompletionUsage]

class ChatCompletionResponseFunctionCall(TypedDict):
    name: str
    arguments: str

class ChatCompletionResponseMessage(TypedDict):
    content: Optional[str]
    tool_calls: NotRequired["ChatCompletionMessageToolCalls"]
    role: Literal["assistant", "function"]  # NOTE: "function" may be incorrect here
    function_call: NotRequired[ChatCompletionResponseFunctionCall]  # DEPRECATED

# Ref: https://platform.openai.com/docs/api-reference/images/create
class CreateImageGenerationRequest(BaseModel):
    prompt: str = Field(
        description="The prompt to generate image for."
    )
    model: Optional[str] = Field(
        default=None,
        description="The model to use for generating image."
    )
    size: Optional[str] = Field(default="512x512", description="The size of the image to be generated in pixels.") 
    quality: Optional[Literal['standard','hd']] = Field(default="standard", description="The quality of the image to be generated.")
    response_format: Optional[Literal['url','b64_json']] = Field(default="url", description="The response format. Valid values are 'url' or 'b64_json'.")
    n: Optional[int] = Field(default=1, ge=1, le=10, description="The number of images to generate. 1-10")
    style: Optional[Literal['vivid','natural']] = Field(default="vivid", description="The image style. 'vivid' or 'natural'")
    user: Optional[str] = Field(default=None)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "a black cat",
                    "size": "1024x1024"
                }
            ]
        }
    }

class GeneratedImage(BaseModel):
    url: Optional[str] = Field(
        default=None,
        description="The URL of the generated image, if `response_format` is `url` (default)."
    )
    b64_json: Optional[str] = Field(
        default=None,
        description="The base64-encoded JSON of the generated image, if `response_format` is `b64_json`."
    )
    revised_prompt: Optional[str] = Field(
        default=None,
        description="The prompt that was used to generate the image, if there was any revision to the prompt."
    )


class CreateImageResponse(BaseModel):
    created: int = Field(
        default=0,
        description="The time when image is generated."
    )
    data: List[GeneratedImage] = Field(
        default=[],
        description="The list of images generated."
    )

class TranscriptionSegment(BaseModel):
    start: int = Field(
        default=0,
        description="The start time."
    )
    end: int = Field(
        default=0,
        description="The end time."
    )
    text: str = Field(
        default="",
        description="The text."
    )

class CreateAudioTranscriptionResponse(BaseModel):
    text: str = Field(
        default="",
        description="The audio transription text."
    )

class CreateAudioTranscriptionVerboseResponse(BaseModel):
    text: str = Field(
        default="",
        description="The audio transription text."
    )
    segments: List[TranscriptionSegment] = Field(
        default=[],
        description="The list of text segments."
    )

# https://platform.openai.com/docs/api-reference/audio/createSpeech
class CreateSpeechRequest(BaseModel):
    model: str = Field(
        default=None,
        description="The model to use for generating audio from text."
    )
    input: str = Field(
        default=None,
        description="The text to generate audio for. The maximum length is 4096 characters."
    )
    voice: str = Field(
        default=None,
        description="The voice to use when generating the audio."
    )
    speed: Optional[float] = Field(
        default=1.0,
        description="The speed of the generated audio. Select a value from 0.25 to 4.0. 1.0 is the default."
    )
    response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = Field(
        default="mp3",
        description="The format to audio in. Supported formats are mp3, opus, aac, flac, wav, and pcm."
    )
    response_type: Optional[Literal["content", "url"]] = Field(
        default="content",
        description="The file url or file content."
    )
    stream: Optional[bool] = stream_field

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "input": "I can speak",
                    "model": "tts-1"
                }
            ]
        }
    }

class CreateSpeechResponse(BaseModel):
    url: str = Field(
        default="",
        description="The audio file url."
    )
