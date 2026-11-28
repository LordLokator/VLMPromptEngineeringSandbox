"""Demo provider: no model. It reports facts measured on the frames it received, so the app runs anywhere."""
import time

from loguru import logger

from utils.providers import GenerateRequest, ModelInfo, Result

DEMO_MODELS = ("frame-facts",)


class DemoProvider:
    provider = "demo"

    def __init__(self, model: str = DEMO_MODELS[0]):
        if model not in DEMO_MODELS:
            raise ValueError(f"Unknown demo model {model!r}; available: {', '.join(DEMO_MODELS)}")
        self.model = model

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name, vision=True) for name in DEMO_MODELS]

    def generate(self, request: GenerateRequest) -> Result:
        start = time.perf_counter()
        text = describe_frames(request.frames, request.frame_times)
        return Result(
            text=text,
            finish_reason="stop",
            requested_model=self.model,
            reported_model=self.model,
            input_tokens=None,
            output_tokens=None,
            cached_tokens=None,
            reasoning_tokens=None,
            latency_ms=(time.perf_counter() - start) * 1000,
            cost_usd=None,
            cost_source=None,
            params_sent={},
        )


def describe_frames(frames: tuple[bytes, ...], times: tuple[float, ...]) -> str:
    """Plain facts about the frames: count, size, brightness, change between frames."""
    import cv2
    import numpy as np

    grays = []
    for jpeg in frames:
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.warning("Demo provider: a frame could not be decoded.")
            continue
        grays.append(image.astype(np.float32))
    if not grays:
        return "Demo (no model): received no readable frames."

    height, width = grays[0].shape
    brightness = [float(g.mean()) for g in grays]
    parts = [
        f"Demo (no model): {len(grays)} frame{'s' if len(grays) != 1 else ''}, {width}x{height}",
    ]
    if times:
        parts[0] += f", {times[0]:.2f}-{times[-1]:.2f} s"
    parts.append(f"mean brightness {sum(brightness) / len(brightness):.0f}/255 "
                 f"(range {min(brightness):.0f}-{max(brightness):.0f})")
    changes = [float(np.abs(b - a).mean()) for a, b in zip(grays, grays[1:], strict=False) if a.shape == b.shape]
    if changes:
        parts.append(f"mean change between frames {sum(changes) / len(changes):.1f}/255")
    return "; ".join(parts) + "."
