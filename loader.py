from settings import ModelSettings, LlamaModelSettings
from typing import Optional
from anyio import Lock
from worker import ModelWorker

class ModelWorkerLoader:

    def __init__(self, model_settings: ModelSettings):
        self.model_settings = model_settings

        self._worker: Optional[ModelWorker] = None
        self.lock = Lock()
        self.outer_lock = Lock()
        self.inner_lock = Lock()

    def get_settings(self) -> ModelSettings:
        return self.model_settings
    
    async def get_worker(self):
        if isinstance(self.model_settings, LlamaModelSettings):
            # NOTE: This double lock allows the currently streaming llama model to
            # check if any other requests are pending in the same thread and cancel
            # the stream if so.
            await self.outer_lock.acquire()
            release_outer_lock = True
            try:
                await self.inner_lock.acquire()
                try:
                    self.outer_lock.release()
                    release_outer_lock = False
                    if self._worker == None:
                        self._worker = self.load_worker()
                    yield self._worker
                finally:
                    self.inner_lock.release()
            finally:
                if release_outer_lock:
                    self.outer_lock.release()
        else:
            await self.lock.acquire()
            try:
                if self._worker == None:
                    self._worker = self.load_worker()
                yield self._worker
            finally:
                self.lock.release()
        
    def load_worker(self) -> ModelWorker:
        if self.model_settings.backend == "llama":
            import llama
            return llama.LlamaWorker(model=self.model_settings)
        elif self.model_settings.backend == "stablediffusion":
            import diffusion
            return diffusion.StableDiffusionWorker(model=self.model_settings)
        elif self.model_settings.backend == "whisper":
            import whisper
            return whisper.WhisperWorker(model=self.model_settings)
        elif self.model_settings.backend == "piper":
            import piper_tts
            return piper_tts.PiperWorker(model=self.model_settings)
        else:
            raise ValueError("Model backend not found.")

