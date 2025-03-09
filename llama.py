from __future__ import annotations

import json

from typing import List, Optional, Union, Dict, Iterator, Any
from worker import ModelWorker

import llama_cpp
from llama_cpp import Llama
import llama_cpp.llama_speculative as llama_speculative
import llama_cpp.llama_tokenizer as llama_tokenizer

from settings import (
    LlamaModelSettings,
)
from api_types import (
    CreateCompletionRequest,
    CreateChatCompletionRequest,
    CreateEmbeddingRequest,
    TokenizeInputRequest,
    DetokenizeInputRequest
)

class LlamaWorker(ModelWorker):
    
    def __init__(self, model: LlamaModelSettings):
        self._current_model_settings: LlamaModelSettings = model
        self._current_model = self._load_model(
            self._current_model_settings
        )

    def logit_bias_tokens_to_input_ids(
        self,
        logit_bias: Dict[str, float],
    ) -> Dict[str, float]:
        to_bias: Dict[str, float] = {}
        for token, score in logit_bias.items():
            token = token.encode("utf-8")
            for input_id in self._current_model.tokenize(token, add_bos=False, special=True):
                to_bias[str(input_id)] = score
        return to_bias

    def create_completion(
        self,
        request: CreateCompletionRequest
    ) -> Any | Iterator[Any]:
        exclude = {
            "n",
            "best_of",
            "logit_bias_type",
            "user",
            "min_tokens",
        }
        kwargs = request.model_dump(exclude=exclude)
        if request.logit_bias is not None:
            kwargs["logit_bias"] = (
                self.logit_bias_tokens_to_input_ids(request.logit_bias)
                if request.logit_bias_type == "tokens"
                else request.logit_bias
            )
        if request.grammar is not None:
            kwargs["grammar"] = llama_cpp.LlamaGrammar.from_string(request.grammar)
        if request.min_tokens > 0:
            _min_tokens_logits_processor = llama_cpp.LogitsProcessorList(
                [llama_cpp.MinTokensLogitsProcessor(request.min_tokens, request.token_eos())]
            )
            if "logits_processor" not in kwargs:
                kwargs["logits_processor"] = _min_tokens_logits_processor
            else:
                kwargs["logits_processor"].extend(_min_tokens_logits_processor)
        return self._current_model.create_completion(
            **kwargs
        )

    def create_chat_completion(
        self,
        request: CreateChatCompletionRequest
    ) -> Any | Iterator[Any]:
        exclude = {
            "n",
            "logit_bias_type",
            "user",
            "min_tokens",
        }
        kwargs = request.model_dump(exclude=exclude)
        if request.logit_bias is not None:
            kwargs["logit_bias"] = (
                self.logit_bias_tokens_to_input_ids(request.logit_bias)
                if request.logit_bias_type == "tokens"
                else request.logit_bias
            )
        if request.grammar is not None:
            kwargs["grammar"] = llama_cpp.LlamaGrammar.from_string(request.grammar)
        if request.min_tokens > 0:
            _min_tokens_logits_processor = llama_cpp.LogitsProcessorList(
                [llama_cpp.MinTokensLogitsProcessor(request.min_tokens, request.token_eos())]
            )
            if "logits_processor" not in kwargs:
                kwargs["logits_processor"] = _min_tokens_logits_processor
            else:
                kwargs["logits_processor"].extend(_min_tokens_logits_processor)
        return self._current_model.create_chat_completion(
            **kwargs
        )
    
    def create_embedding(
        self,
        request: CreateEmbeddingRequest
    ) -> Any:
        return self._current_model.create_embedding(
            input=request.input,
        )
    
    def tokenize(
        self,
        request: TokenizeInputRequest
    ) -> List[int]:
        return self._current_model.tokenize(
            text=request.input.encode("utf-8"),
            special=True
        )
    
    def detokenize(
        self,
        request: DetokenizeInputRequest
    ) -> str:
        return self._current_model.detokenize(
            tokens=request.tokens,
            special=True
        ).decode("utf-8")
    
    def free(self):
        if self._current_model:
            self._current_model.close()
            del self._current_model

    @staticmethod
    def _load_model(settings: LlamaModelSettings) -> Llama:
        chat_handler = None
        if settings.chat_format == "llava-1-5":
            assert settings.clip_model_path is not None, "clip model not found"
            if settings.hf_model_repo_id is not None:
                chat_handler = (
                    llama_cpp.llama_chat_format.Llava15ChatHandler.from_pretrained(
                        repo_id=settings.hf_model_repo_id,
                        filename=settings.clip_model_path,
                        verbose=settings.verbose,
                    )
                )
            else:
                chat_handler = llama_cpp.llama_chat_format.Llava15ChatHandler(
                    clip_model_path=settings.clip_model_path, verbose=settings.verbose
                )
        elif settings.chat_format == "obsidian":
            assert settings.clip_model_path is not None, "clip model not found"
            if settings.hf_model_repo_id is not None:
                chat_handler = (
                    llama_cpp.llama_chat_format.ObsidianChatHandler.from_pretrained(
                        repo_id=settings.hf_model_repo_id,
                        filename=settings.clip_model_path,
                        verbose=settings.verbose,
                    )
                )
            else:
                chat_handler = llama_cpp.llama_chat_format.ObsidianChatHandler(
                    clip_model_path=settings.clip_model_path, verbose=settings.verbose
                )
        elif settings.chat_format == "llava-1-6":
            assert settings.clip_model_path is not None, "clip model not found"
            if settings.hf_model_repo_id is not None:
                chat_handler = (
                    llama_cpp.llama_chat_format.Llava16ChatHandler.from_pretrained(
                        repo_id=settings.hf_model_repo_id,
                        filename=settings.clip_model_path,
                        verbose=settings.verbose,
                    )
                )
            else:
                chat_handler = llama_cpp.llama_chat_format.Llava16ChatHandler(
                    clip_model_path=settings.clip_model_path, verbose=settings.verbose
                )
        elif settings.chat_format == "moondream":
            assert settings.clip_model_path is not None, "clip model not found"
            if settings.hf_model_repo_id is not None:
                chat_handler = (
                    llama_cpp.llama_chat_format.MoondreamChatHandler.from_pretrained(
                        repo_id=settings.hf_model_repo_id,
                        filename=settings.clip_model_path,
                        verbose=settings.verbose,
                    )
                )
            else:
                chat_handler = llama_cpp.llama_chat_format.MoondreamChatHandler(
                    clip_model_path=settings.clip_model_path, verbose=settings.verbose
                )
        elif settings.chat_format == "nanollava":
            assert settings.clip_model_path is not None, "clip model not found"
            if settings.hf_model_repo_id is not None:
                chat_handler = (
                    llama_cpp.llama_chat_format.NanoLlavaChatHandler.from_pretrained(
                        repo_id=settings.hf_model_repo_id,
                        filename=settings.clip_model_path,
                        verbose=settings.verbose,
                    )
                )
            else:
                chat_handler = llama_cpp.llama_chat_format.NanoLlavaChatHandler(
                    clip_model_path=settings.clip_model_path, verbose=settings.verbose
                )
        elif settings.chat_format == "llama-3-vision-alpha":
            assert settings.clip_model_path is not None, "clip model not found"
            if settings.hf_model_repo_id is not None:
                chat_handler = (
                    llama_cpp.llama_chat_format.Llama3VisionAlpha.from_pretrained(
                        repo_id=settings.hf_model_repo_id,
                        filename=settings.clip_model_path,
                        verbose=settings.verbose,
                    )
                )
            else:
                chat_handler = llama_cpp.llama_chat_format.Llama3VisionAlpha(
                    clip_model_path=settings.clip_model_path, verbose=settings.verbose
                )
        elif settings.chat_format == "minicpm-v-2.6":
            assert settings.clip_model_path is not None, "clip model not found"
            if settings.hf_model_repo_id is not None:
                chat_handler = (
                    llama_cpp.llama_chat_format.MiniCPMv26ChatHandler.from_pretrained(
                        repo_id=settings.hf_model_repo_id,
                        filename=settings.clip_model_path,
                        verbose=settings.verbose,
                    )
                )
            else:
                chat_handler = llama_cpp.llama_chat_format.MiniCPMv26ChatHandler(
                    clip_model_path=settings.clip_model_path, verbose=settings.verbose
                )
        elif settings.chat_format == "hf-autotokenizer":
            assert (
                settings.hf_pretrained_model_name_or_path is not None
            ), "hf_pretrained_model_name_or_path must be set for hf-autotokenizer"
            chat_handler = (
                llama_cpp.llama_chat_format.hf_autotokenizer_to_chat_completion_handler(
                    settings.hf_pretrained_model_name_or_path
                )
            )
        elif settings.chat_format == "hf-tokenizer-config":
            assert (
                settings.hf_tokenizer_config_path is not None
            ), "hf_tokenizer_config_path must be set for hf-tokenizer-config"
            chat_handler = llama_cpp.llama_chat_format.hf_tokenizer_config_to_chat_completion_handler(
                json.load(open(settings.hf_tokenizer_config_path))
            )

        tokenizer: Optional[llama_cpp.BaseLlamaTokenizer] = None
        if settings.hf_pretrained_model_name_or_path is not None:
            tokenizer = llama_tokenizer.LlamaHFTokenizer.from_pretrained(
                settings.hf_pretrained_model_name_or_path
            )

        draft_model = None
        if settings.draft_model is not None:
            draft_model = llama_speculative.LlamaPromptLookupDecoding(
                num_pred_tokens=settings.draft_model_num_pred_tokens
            )

        kv_overrides: Optional[Dict[str, Union[bool, int, float, str]]] = None
        if settings.kv_overrides is not None:
            assert isinstance(settings.kv_overrides, list)
            kv_overrides = {}
            for kv in settings.kv_overrides:
                key, value = kv.split("=")
                if ":" in value:
                    value_type, value = value.split(":")
                    if value_type == "bool":
                        kv_overrides[key] = value.lower() in ["true", "1"]
                    elif value_type == "int":
                        kv_overrides[key] = int(value)
                    elif value_type == "float":
                        kv_overrides[key] = float(value)
                    elif value_type == "str":
                        kv_overrides[key] = value
                    else:
                        raise ValueError(f"Unknown value type {value_type}")

        _model = llama_cpp.Llama(
            # Model Params
            model_path=settings.model_path,
            n_gpu_layers=settings.n_gpu_layers,
            split_mode=settings.split_mode if settings.split_mode != None else llama_cpp.LLAMA_SPLIT_MODE_LAYER,
            main_gpu=settings.main_gpu,
            tensor_split=settings.tensor_split,
            vocab_only=settings.vocab_only,
            use_mmap=settings.use_mmap if settings.use_mmap != None else llama_cpp.llama_supports_mmap(),
            use_mlock=settings.use_mlock if settings.use_mlock != None else llama_cpp.llama_supports_mlock(),
            kv_overrides=kv_overrides,
            rpc_servers=settings.rpc_servers,
            # Context Params
            seed=settings.seed if settings.seed != None else llama_cpp.LLAMA_DEFAULT_SEED,
            n_ctx=settings.n_ctx,
            n_batch=settings.n_batch,
            n_ubatch=settings.n_ubatch,
            n_threads=settings.n_threads,
            n_threads_batch=settings.n_threads_batch,
            rope_scaling_type=settings.rope_scaling_type if settings.rope_scaling_type != None else llama_cpp.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
            rope_freq_base=settings.rope_freq_base,
            rope_freq_scale=settings.rope_freq_scale,
            yarn_ext_factor=settings.yarn_ext_factor,
            yarn_attn_factor=settings.yarn_attn_factor,
            yarn_beta_fast=settings.yarn_beta_fast,
            yarn_beta_slow=settings.yarn_beta_slow,
            yarn_orig_ctx=settings.yarn_orig_ctx,
            mul_mat_q=settings.mul_mat_q,
            logits_all=settings.logits_all,
            embedding=settings.embedding,
            offload_kqv=settings.offload_kqv,
            flash_attn=settings.flash_attn,
            # Sampling Params
            last_n_tokens_size=settings.last_n_tokens_size,
            # LoRA Params
            lora_base=settings.lora_base,
            lora_path=settings.lora_path,
            # Backend Params
            numa=settings.numa,
            # Chat Format Params
            chat_format=settings.chat_format,
            chat_handler=chat_handler,
            # Speculative Decoding
            draft_model=draft_model,
            # KV Cache Quantization
            type_k=settings.type_k,
            type_v=settings.type_v,
            # Tokenizer
            tokenizer=tokenizer,
            # Misc
            verbose=settings.verbose,
        )
        if settings.cache:
            if settings.cache_type == "disk":
                if settings.verbose:
                    print(f"Using disk cache with size {settings.cache_size}")
                cache = llama_cpp.LlamaDiskCache(capacity_bytes=settings.cache_size)
            else:
                if settings.verbose:
                    print(f"Using ram cache with size {settings.cache_size}")
                cache = llama_cpp.LlamaRAMCache(capacity_bytes=settings.cache_size)
            _model.set_cache(cache)
        return _model

