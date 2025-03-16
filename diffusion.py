
from worker import ModelWorker, TextToImageRequest, ImageToImageRequest, ImageResponse
from PIL import Image
from io import BytesIO
from stable_diffusion_cpp import StableDiffusion, SampleMethod

from settings import (
    StableDiffusionModelSettings,
)

class StableDiffusionWorker(ModelWorker):
    
    def __init__(self, model: StableDiffusionModelSettings):
        self._current_model_settings: StableDiffusionModelSettings = model
        self._current_model = self._load_model(
            self._current_model_settings
        )

    def text_to_image(
        self,
        request: TextToImageRequest
    ) -> ImageResponse:
        imgs = self._current_model.txt_to_img(
            prompt=request["prompt"],
            width=request["width"],
            height=request["height"],
            batch_count=request.get("batch_count",1),
            upscale_factor=request.get("upscale_factor",1)
        )
        return {"images": imgs}

    def image_to_image(
        self,
        request: ImageToImageRequest
    ) -> ImageResponse:
        image = Image.open(BytesIO(request["image"]), formats=["PNG"])
        mask_image = request["mask_image"]
        if mask_image is not None:
            mask_image = Image.open(BytesIO(mask_image), formats=["PNG"])
        imgs = self._current_model.img_to_img(
            image=image,
            mask_image=mask_image,
            prompt=request["prompt"],
            width=request["width"],
            height=request["height"],
            batch_count=request["batch_count"]
        )
        return {"images": imgs}

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