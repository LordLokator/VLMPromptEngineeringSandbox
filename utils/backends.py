"""Backend selection seam. No heavy imports at module level (torch is only loaded by the chosen backend)."""
import os
from typing import TYPE_CHECKING, Sequence

from loguru import logger

from prompting.prompting import Prompt

if TYPE_CHECKING:
    from main import VLMLike

BACKEND_ENV = "VLM_BACKEND"
DEFAULT_BACKEND = "hf"


class ChainBackend:
    """Try backends in order; log and continue on failure, raise the last error if all fail."""

    def __init__(self, backends: Sequence["VLMLike"]):
        if not backends:
            raise ValueError("ChainBackend needs at least one backend")
        self.backends = list(backends)
        self.name = getattr(self.backends[0], "name", type(self.backends[0]).__name__)

    def forward(self, video_path, fps, max_frames, temperature=0.01) -> tuple[str, float]:
        last_error: Exception | None = None
        for backend in self.backends:
            label = getattr(backend, "name", type(backend).__name__)
            try:
                result = backend.forward(
                    video_path=video_path, fps=fps, max_frames=max_frames, temperature=temperature
                )
            except Exception as e:  # noqa: BLE001 - any backend failure falls through to the next one
                logger.warning(f"Backend {label} failed: {e}")
                last_error = e
                continue
            self.name = label
            return result
        assert last_error is not None
        raise last_error


def make_backend(prompt: Prompt) -> "VLMLike":
    """Build the backend named by VLM_BACKEND (default: the local HF model)."""
    choice = os.environ.get(BACKEND_ENV, DEFAULT_BACKEND).strip().lower()
    if choice == "hf":
        from utils.vlm_wrapper import VLM

        return VLM(prompt)
    raise ValueError(f"Unknown {BACKEND_ENV}={choice!r}; supported: hf")
