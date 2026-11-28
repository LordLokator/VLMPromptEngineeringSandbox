"""Local VideoLLaMA3 behind the provider contract. The GPU code stays in utils/vlm_wrapper.py (AGENTS rule 4)."""
import threading
import time
from typing import Any

from utils.providers import GenerateRequest, ModelInfo, Result

HF_MODEL = "DAMO-NLP-SG/VideoLLaMA3-7B"  # the only one: utils/vlm_wrapper.py loads this repo at a pinned revision
MIN_TEMPERATURE = 0.01  # generate() with sampling rejects 0


class HFProvider:
    """Loads the model when built (slow, needs a CUDA GPU). One call at a time: the GPU is shared."""

    provider = "hf"

    def __init__(self, model: str = HF_MODEL, vlm: Any = None):
        if model != HF_MODEL:
            raise ValueError(f"hf serves only {HF_MODEL}, not {model!r}")
        self.model = model
        if vlm is None:
            from utils.vlm_wrapper import VLM

            vlm = VLM()
        self._vlm = vlm
        self._gpu = threading.Lock()

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(HF_MODEL, vision=True)]

    def generate(self, request: GenerateRequest) -> Result:
        if request.video_path is None:
            raise ValueError("hf needs the clip file (request.video_path)")
        temperature = max(request.temperature if request.temperature is not None else MIN_TEMPERATURE,
                          MIN_TEMPERATURE)
        params = {
            "temperature": temperature,
            "max_new_tokens": request.max_output_tokens,
            "fps": request.fps,
            "max_frames": request.max_frames,
        }
        with self._gpu:
            start = time.perf_counter()
            text = self._vlm.forward(
                video_path=request.video_path,
                fps=request.fps,
                max_frames=request.max_frames,
                prompt=request.prompt,
                system=request.system,
                temperature=temperature,
                max_new_tokens=request.max_output_tokens,
            )
            latency_ms = (time.perf_counter() - start) * 1000
        return Result(
            text=text,
            finish_reason=None,  # the wrapper does not report it
            requested_model=self.model,
            reported_model=self.model,
            input_tokens=None,
            output_tokens=None,
            cached_tokens=None,
            reasoning_tokens=None,
            latency_ms=latency_ms,
            cost_usd=None,
            cost_source=None,
            params_sent=params,
        )
