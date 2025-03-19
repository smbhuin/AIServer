import multiprocessing

from typing import Annotated, Optional, List, Literal, Tuple, Type, Union, Dict

from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings, YamlConfigSettingsSource, PydanticBaseSettingsSource, SettingsConfigDict
import time

# Disable warning for model and model_alias settings
BaseSettings.model_config["protected_namespaces"] = ()

class ModelBackend:
    llama = "llama"
    stable_diffusion = "stablediffusion"
    whisper = "whisper"
    piper = "piper"

class ServerSettings(BaseModel):
    """Server settings used to configure the FastAPI and Uvicorn server."""

    # Uvicorn Settings
    host: str = Field(default="localhost", description="Listen address")
    port: int = Field(default=8000, description="Listen port")
    ssl_keyfile: Optional[str] = Field(
        default=None, description="SSL key file for HTTPS"
    )
    ssl_certfile: Optional[str] = Field(
        default=None, description="SSL certificate file for HTTPS"
    )
    # FastAPI Settings
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication. If set all requests need to be authenticated.",
    )
    interrupt_requests: bool = Field(
        default=True,
        description="Whether to interrupt requests when a new request is received.",
    )
    disable_ping_events: bool = Field(
        default=False,
        description="Disable EventSource pings (may be needed for some clients).",
    )
    root_path: str = Field(
        default="",
        description="The root path for the server. Useful when running behind a reverse proxy.",
    )
    user_file_path: str = Field(
        default="files",
        description="The user files storage path.",
    )
    public_host_url: Optional[str] = Field(
        default=None,
        description="The public host url for the server.",
    )

class ModelSettings(BaseModel):
    model_name: str = Field(
        description="The alias of the model to use as model name in APIs.",
    )
    model_path: str = Field(
        description="The path to the model."
    )
    created: int = Field(
        default=int(time.time()),
        description="The Unix timestamp (in seconds) when the model was created."
    )
    backend: Literal['llama', 'stablediffusion', 'whisper', 'piper']

class LlamaModelSettings(ModelSettings):
    """Model settings used to load a llama model."""

    backend: Literal['llama']

    # Model Params
    n_gpu_layers: int = Field(
        default=0,
        ge=-1,
        description="The number of layers to put on the GPU. The rest will be on the CPU. Set -1 to move all to GPU."
    )
    split_mode: int = Field(
        default=None, # llama_cpp.LLAMA_SPLIT_MODE_LAYER
        description="The split mode to use."
    )
    main_gpu: int = Field(
        default=0,
        ge=0,
        description="Main GPU to use.",
    )
    tensor_split: Optional[List[float]] = Field(
        default=None,
        description="Split layers across multiple GPUs in proportion.",
    )
    vocab_only: bool = Field(
        default=False, description="Whether to only return the vocabulary."
    )
    use_mmap: bool = Field(
        default=None, # llama_cpp.llama_supports_mmap()
        description="Use mmap.",
    )
    use_mlock: bool = Field(
        default=None, # llama_cpp.llama_supports_mlock()
        description="Use mlock.",
    )
    kv_overrides: Optional[List[str]] = Field(
        default=None,
        description="List of model kv overrides in the format key=type:value where type is one of (bool, int, float). Valid true values are (true, TRUE, 1), otherwise false.",
    )
    rpc_servers: Optional[str] = Field(
        default=None,
        description="comma seperated list of rpc servers for offloading",
    )
    # Context Params
    seed: int = Field(
        default=None, # llama_cpp.LLAMA_DEFAULT_SEED
        description="Random seed. -1 for random."
    )
    n_ctx: int = Field(default=2048, ge=0, description="The context size.")
    n_batch: int = Field(
        default=512, ge=1, description="The batch size to use per eval."
    )
    n_ubatch: int = Field(
        default=512, ge=1, description="The physical batch size used by llama.cpp"
    )
    n_threads: int = Field(
        default=max(multiprocessing.cpu_count() // 2, 1),
        ge=1,
        description="The number of threads to use. Use -1 for max cpu threads",
    )
    n_threads_batch: int = Field(
        default=max(multiprocessing.cpu_count(), 1),
        ge=0,
        description="The number of threads to use when batch processing. Use -1 for max cpu threads",
    )
    rope_scaling_type: int = Field(
        default=None, # llama_cpp.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED
    )
    rope_freq_base: float = Field(default=0.0, description="RoPE base frequency")
    rope_freq_scale: float = Field(
        default=0.0, description="RoPE frequency scaling factor"
    )
    yarn_ext_factor: float = Field(default=-1.0)
    yarn_attn_factor: float = Field(default=1.0)
    yarn_beta_fast: float = Field(default=32.0)
    yarn_beta_slow: float = Field(default=1.0)
    yarn_orig_ctx: int = Field(default=0)
    mul_mat_q: bool = Field(
        default=True, description="if true, use experimental mul_mat_q kernels"
    )
    logits_all: bool = Field(default=True, description="Whether to return logits.")
    embedding: bool = Field(default=False, description="Whether to use embeddings.")
    offload_kqv: bool = Field(
        default=True, description="Whether to offload kqv to the GPU."
    )
    flash_attn: bool = Field(
        default=False, description="Whether to use flash attention."
    )
    # Sampling Params
    last_n_tokens_size: int = Field(
        default=64,
        ge=0,
        description="Last n tokens to keep for repeat penalty calculation.",
    )
    # LoRA Params
    lora_base: Optional[str] = Field(
        default=None,
        description="Optional path to base model, useful if using a quantized base model and you want to apply LoRA to an f16 model.",
    )
    lora_path: Optional[str] = Field(
        default=None,
        description="Path to a LoRA file to apply to the model.",
    )
    # Backend Params
    numa: Union[bool, int] = Field(
        default=False,
        description="Enable NUMA support.",
    )
    # Chat Format Params
    chat_format: Optional[str] = Field(
        default=None,
        description="Chat format to use.",
    )
    clip_model_path: Optional[str] = Field(
        default=None,
        description="Path to a CLIP model to use for multi-modal chat completion.",
    )
    # Cache Params
    cache: bool = Field(
        default=False,
        description="Use a cache to reduce processing times for evaluated prompts.",
    )
    cache_type: Literal["ram", "disk"] = Field(
        default="ram",
        description="The type of cache to use. Only used if cache is True.",
    )
    cache_size: int = Field(
        default=2 << 30,
        description="The size of the cache in bytes. Only used if cache is True.",
    )
    # Tokenizer Options
    hf_tokenizer_config_path: Optional[str] = Field(
        default=None,
        description="The path to a HuggingFace tokenizer_config.json file.",
    )
    hf_pretrained_model_name_or_path: Optional[str] = Field(
        default=None,
        description="The model name or path to a pretrained HuggingFace tokenizer model. Same as you would pass to AutoTokenizer.from_pretrained().",
    )
    # Loading from HuggingFace Model Hub
    hf_model_repo_id: Optional[str] = Field(
        default=None,
        description="The model repo id to use for the HuggingFace tokenizer model.",
    )
    # Speculative Decoding
    draft_model: Optional[str] = Field(
        default=None,
        description="Method to use for speculative decoding. One of (prompt-lookup-decoding).",
    )
    draft_model_num_pred_tokens: int = Field(
        default=10,
        description="Number of tokens to predict using the draft model.",
    )
    # KV Cache Quantization
    type_k: Optional[int] = Field(
        default=None,
        description="Type of the key cache quantization.",
    )
    type_v: Optional[int] = Field(
        default=None,
        description="Type of the value cache quantization.",
    )
    # Misc
    verbose: bool = Field(
        default=True, description="Whether to print debug information."
    )
    
class WhisperModelSettings(ModelSettings):
    """Model settings used to load a Whisper model."""

    backend: Literal['whisper']

    # Model Params
    n_threads: int = Field(
        default=max(multiprocessing.cpu_count() // 2, 1),
        ge=1,
        description="The number of threads to use. Use -1 for max cpu threads",
    )

class PiperModelSettings(ModelSettings):
    """Model settings used to load a Piper model."""

    backend: Literal['piper']

    # Model Params
    model_config_path: Optional[str] = Field(
        default=None,
        description="The path to the model config to use for tts."
    )
    use_cuda: bool = Field(
        default=False,
        description="Use nvidia cuda to generate audio.",
    )

class StableDiffusionModelSettings(ModelSettings):
    """Model settings used to load a Stable Diffusion model."""

    backend: Literal['stablediffusion']

    # Model Params
    clip_l_path: str = Field(
        default="",
        description="The path to the clip_l."
    )
    clip_g_path: str = Field(
        default="",
        description="The path to the clip_g."
    )
    t5xxl_path: str = Field(
        default="",
        description="The path to the t5xxl."
    )
    diffusion_model_path: str = Field(
        default="",
        description="The path to the diffusion model."
    )
    vae_path: str = Field(
        default="",
        description="The path to the vae."
    )
    taesd_path: str = Field(
        default="",
        description="The path to the taesd."
    )
    control_net_path: str = Field(
        default="",
        description="The path to the controlnet."
    )
    upscaler_path: str = Field(
        default="",
        description="The path to the upscaler."
    )
    lora_model_dir: str = Field(
        default="",
        description="The path to the lora."
    )
    embed_dir: str = Field(
        default="",
        description="The path to the embeddings."
    )
    stacked_id_embed_dir: str = Field(
        default="",
        description="The path to the PHOTOMAKER stacked id embeddings."
    )
    vae_decode_only: bool = Field(
        default=False,
        description="Process vae in decode only mode.",
    )
    vae_tiling: bool = Field(
        default=False,
        description="Process vae in tiles to reduce memory usage.",
    )
    n_threads: int = Field(
        default=-1,
        description="Number of threads to use for generation (default: half the number of CPUs)",
    )
    wtype: str = Field(
        default="default",
        description="The weight type (default: automatically determines the weight type of the model file)."
    )
    rng_type: str = Field(
        default="cuda",
        description="Random number generator."
    )
    schedule: str = Field(
        default="default",
        description="Denoiser sigma schedule."
    )
    keep_clip_on_cpu: bool = Field(
        default=False,
        description="Keep clip in CPU (for low vram).",
    )
    keep_control_net_cpu: bool = Field(
        default=False,
        description="Keep controlnet in CPU (for low vram).",
    )
    keep_vae_on_cpu: bool = Field(
        default=False,
        description="Keep vae in CPU (for low vram).",
    )
    diffusion_flash_attn: bool = Field(
        default=False,
        description="Use flash attention in diffusion model (can reduce memory usage significantly).",
    )
    verbose: bool = Field(
        default=True,
        description="Print verbose output to stderr.",
    )

ConfigModelSettings = Annotated[
    Union[LlamaModelSettings, WhisperModelSettings, StableDiffusionModelSettings, PiperModelSettings],
    Field(discriminator="backend")]

class ConfigFileSettings(BaseSettings):
    """Configuration file format settings."""
    model_config = SettingsConfigDict(yaml_file="aiserver.yml")
    
    server: ServerSettings = Field(default=ServerSettings(), description="Server settings")
    models: List[ConfigModelSettings] = Field(description="Model settings")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (YamlConfigSettingsSource(settings_cls),)