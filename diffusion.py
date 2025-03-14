from __future__ import annotations

from typing import List, Optional, Union, Callable
from worker import ModelWorker
from PIL import Image

from stable_diffusion_cpp import StableDiffusion, SampleMethod

from settings import (
    StableDiffusionModelSettings,
)

from api_types import (
    CreateImageGenerationRequest,
    CreateImageGenerationResponse,
    GeneratedImage
)

class StableDiffusionWorker(ModelWorker):
    
    def __init__(self, model: StableDiffusionModelSettings):
        self._current_model_settings: StableDiffusionModelSettings = model
        self._current_model = self._load_model(
            self._current_model_settings
        )

    def create_image(
        self,
        request: CreateImageGenerationRequest
    ) -> List[Image.Image]:
        exclude = {
            "size",
            "model",
            "quality",
            "response_format",
            "style",
            "user",
            "n"
        }
        kwargs = request.model_dump(exclude=exclude)
        if request.n > 1:
            kwargs["batch_count"] = request.n
        if request.size is not None:
            width, height = request.size.split('x')
            kwargs["width"] = int(width)
            kwargs["height"] = int(height)
        return self._current_model.txt_to_img(**kwargs)

    def free(self):
        if self._current_model:
            del self._current_model

    @staticmethod
    def _load_model(settings: StableDiffusionModelSettings) -> StableDiffusion:
        
        _model = StableDiffusion(
            model_path=settings.model_path, 
            clip_l_path=settings.clip_l_path,
            clip_g_path=settings.clip_g_path,
            t5xxl_path=settings.t5xxl_path,
            diffusion_model_path=settings.diffusion_model_path,
            vae_path=settings.vae_path,
            taesd_path=settings.taesd_path,
            control_net_path=settings.control_net_path,
            upscaler_path=settings.upscaler_path,
            lora_model_dir=settings.lora_model_dir,
            embed_dir=settings.embed_dir,
            stacked_id_embed_dir=settings.stacked_id_embed_dir,
            vae_decode_only=settings.vae_decode_only,
            vae_tiling=settings.vae_tiling,
            n_threads=settings.n_threads,
            wtype=settings.wtype,
            rng_type=settings.rng_type,
            schedule=settings.schedule,
            keep_clip_on_cpu=settings.keep_clip_on_cpu,
            keep_control_net_cpu=settings.keep_control_net_cpu,
            keep_vae_on_cpu=settings.keep_vae_on_cpu,
            diffusion_flash_attn=settings.diffusion_flash_attn,            
            # Misc
            verbose=settings.verbose
        )
        
        return _model