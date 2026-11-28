"""The provider contract: what every agent (demo, local HF, later API models) takes and returns.

Importing this package must stay cheap: no torch, no provider SDK and no OpenCV. Each provider imports what it
needs when it is built or called.
"""
from dataclasses import dataclass, field
from typing import Any, Protocol

SYSTEM_PROMPT = "You are a helpful assistant."
MAX_OUTPUT_TOKENS = 1024

# The fixed list (design: providers). Only some are selectable in the app so far, see utils/providers/agents.py.
PROVIDERS = ("demo", "openai", "anthropic", "ollama", "compat", "hf")


@dataclass(frozen=True)
class ModelInfo:
    id: str
    vision: bool | None = None  # None: the provider does not say


@dataclass(frozen=True)
class GenerateRequest:
    """One question about one clip. Every agent of a round gets the same request."""

    frames: tuple[bytes, ...]  # JPEG images, in time order
    frame_times: tuple[float, ...]  # seconds, same clock as ClipSample.start_s
    system: str
    prompt: str
    temperature: float | None
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    # Local HF reads the clip file instead of the frames; its loader picks the frames by `fps` and `max_frames`.
    video_path: str | None = None
    fps: int = 10
    max_frames: int = 20


@dataclass(frozen=True)
class Result:
    text: str
    finish_reason: str | None
    requested_model: str
    reported_model: str | None
    input_tokens: int | None  # total, cached tokens included
    output_tokens: int | None
    cached_tokens: int | None
    reasoning_tokens: int | None
    latency_ms: float  # wall clock around the call
    cost_usd: float | None
    cost_source: str | None
    params_sent: dict[str, Any] = field(default_factory=dict)


class Provider(Protocol):
    provider: str  # one of PROVIDERS
    model: str

    @property
    def label(self) -> str: ...

    def list_models(self) -> list[ModelInfo]: ...

    def generate(self, request: GenerateRequest) -> Result: ...
