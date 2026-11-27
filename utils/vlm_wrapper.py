"""All GPU/model code lives here (AGENTS rule 4).

Heavy imports are lazy so everything else is testable without torch.
"""
import os
from typing import Any

from loguru import logger

from prompting.prompting import Prompt

MAX_NEW_TOKENS = 1024
MODEL_PATH = "DAMO-NLP-SG/VideoLLaMA3-7B"
# Pinned because trust_remote_code=True runs code from the repo. HF DAMO-NLP-SG/VideoLLaMA3-7B,
# last modified 2025-09-02, checked 2026-11-25.
MODEL_REVISION = "d5b763e368861e7f5096e7ff1b49f92fbccf8ae6"
DEVICE = "cuda:0"
# Override e.g. on pre-Ampere GPUs or without flash-attn installed.
ATTN_IMPL = os.environ.get("VLM_ATTN_IMPL", "flash_attention_2")
# load_video (processing_videollama3.py at MODEL_REVISION) decodes with decord only when start_time and end_time are
# given and less than this many seconds apart. Otherwise it runs system ffmpeg, whose fps filter returned no frames
# for 0.5 s clips at 1 fps.
DECORD_MAX_SECONDS = 1.0


def _patch_remote_code_compat() -> None:
    """Restore a name the pinned remote code imports; call before loading anything from the repo.

    image_processing_videollama3.py (at MODEL_REVISION) imports VideoInput from transformers.image_utils
    and uses it only in a type hint. transformers 4.52 moved it to video_utils.
    """
    from transformers import image_utils, video_utils  # pyright: ignore[reportMissingImports]

    if not hasattr(image_utils, "VideoInput"):
        image_utils.VideoInput = video_utils.VideoInput


def _pinned_snapshot() -> str:
    """Download the model repo at MODEL_REVISION and return the local directory to load from.

    Passing the repo id with revision= is not enough on transformers 4.57.6 (checked 2026-11-26):
    AutoProcessor looked up processor_config.json and its other discovery files on the main branch,
    and its auto_map says which code to run; tokenizer loading also queried the repo without a
    revision. Loading from this directory made no such requests.
    """
    from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]

    return snapshot_download(MODEL_PATH, revision=MODEL_REVISION)


def _clip_seconds(video_path: str) -> float:
    """Length of a recorded clip, from its container's frame count and frame rate."""
    import cv2

    capture = cv2.VideoCapture(video_path)
    try:
        frames, fps = capture.get(cv2.CAP_PROP_FRAME_COUNT), capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()
    if frames <= 0 or fps <= 0:
        raise ValueError(f"Cannot read the length of clip {video_path}")
    seconds = frames / fps
    if seconds >= DECORD_MAX_SECONDS:
        raise ValueError(f"Clip is {seconds:.2f} s long; it must be shorter than {DECORD_MAX_SECONDS:.0f} s")
    return seconds


class VLM:
    name = "hf/VideoLLaMA3-7B"

    def __init__(self, prompt: Prompt):
        import torch  # pyright: ignore[reportMissingImports]  # GPU extra, not installed in dev
        from transformers import AutoModelForCausalLM, AutoProcessor  # pyright: ignore[reportMissingImports]

        _patch_remote_code_compat()
        model_dir = _pinned_snapshot()
        self.sys_prompt = "You are a helpful assistant."
        self.prompt_obj = prompt

        logger.info(f"Attention implementation: {ATTN_IMPL}")
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            device_map={"": DEVICE},
            dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPL
        )

        self.processor: Any = AutoProcessor.from_pretrained(
            model_dir,
            trust_remote_code=True
        )
        logger.info("Instantiated models.")

    def forward(self, video_path, fps, max_frames, temperature=0.01) -> tuple[str, float]:
        current_prompt = self.prompt_obj.get()

        conversation = self.get_conv(
            video_path=video_path,
            fps=fps,
            max_frames=max_frames,
            prompt=current_prompt
        )
        output = self._forward(conversation, temperature)
        return (output, temperature)

    def _forward(self, conversation, temperature=0.01) -> str:
        import torch  # pyright: ignore[reportMissingImports]  # GPU extra, not installed in dev

        processed_inputs = self.processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        inputs = {
            k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in processed_inputs.items()
        }

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        # VLM generate
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
        )
        decoded: list[str] = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )
        if not decoded:
            logger.warning("VLM returned no decoded output.")
            return ""
        return decoded[0].strip()

    def get_conv(self, video_path, fps, max_frames, prompt):
        return [
            {
                "role": "system",
                "content": self.sys_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": {
                            "video_path": video_path,
                            # Both bounds select the loader's decord branch, see DECORD_MAX_SECONDS.
                            "start_time": 0.0,
                            "end_time": _clip_seconds(video_path),
                            "fps": fps,
                            "max_frames": max_frames
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            },
        ]
