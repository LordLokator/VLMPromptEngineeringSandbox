"""Demo provider: no model. It reports facts measured on the frames it received, so the app runs anywhere.

Two variants, so two demo slots never show the same text: `frame-facts` (size, brightness, change) and
`frame-motion` (when and where the picture changes most).
"""
import time

from loguru import logger

from utils.providers import GenerateRequest, ModelInfo, Result

DEMO_MODELS = ("frame-facts", "frame-motion")
CHANGED_PIXEL = 25  # a grey-level difference above this counts as a changed pixel


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
        describe = describe_motion if self.model == "frame-motion" else describe_frames
        text = describe(request.frames, request.frame_times)
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


def _grays(frames: tuple[bytes, ...]) -> list:
    import cv2
    import numpy as np

    grays = []
    for jpeg in frames:
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.warning("Demo provider: a frame could not be decoded.")
            continue
        grays.append(image.astype(np.float32))
    return grays


def describe_frames(frames: tuple[bytes, ...], times: tuple[float, ...]) -> str:
    """Plain facts about the frames: count, size, brightness, change between frames."""
    import numpy as np

    grays = _grays(frames)
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


_ROWS = ("top", "middle", "bottom")
_COLUMNS = ("left", "centre", "right")


def describe_motion(frames: tuple[bytes, ...], times: tuple[float, ...]) -> str:
    """When and where the picture changes most between consecutive frames."""
    import numpy as np

    grays = _grays(frames)
    if not grays:
        return "Demo (no model), motion: received no readable frames."
    span = f", {times[0]:.2f}-{times[-1]:.2f} s" if times else ""
    head = f"Demo (no model), motion: {len(grays)} frame{'s' if len(grays) != 1 else ''}{span}"
    pairs = [(i, np.abs(b - a)) for i, (a, b) in enumerate(zip(grays, grays[1:], strict=False)) if a.shape == b.shape]
    if not pairs:
        return f"{head}; one frame only, so no motion to measure."

    index, diff = max(pairs, key=lambda pair: float(pair[1].mean()))
    parts = [head]
    if len(times) == len(grays):
        parts.append(f"largest change {times[index]:.2f}-{times[index + 1]:.2f} s ({float(diff.mean()):.1f}/255)")
    else:
        parts.append(f"largest change between frames {index + 1} and {index + 2} ({float(diff.mean()):.1f}/255)")
    changed = float(np.mean([(d > CHANGED_PIXEL).mean() for _i, d in pairs])) * 100
    parts.append(f"changed area {changed:.0f}% of the picture")
    total = sum(d for _i, d in pairs)
    if float(total.max()) > 0:
        height, width = total.shape
        cells = [
            (float(total[r * height // 3:(r + 1) * height // 3, c * width // 3:(c + 1) * width // 3].sum()), r, c)
            for r in range(3) for c in range(3)
        ]
        _sum, row, column = max(cells)
        where = "centre" if (row, column) == (1, 1) else f"{_ROWS[row]} {_COLUMNS[column]}"
        parts.append(f"most change in the {where}")
    else:
        parts.append("no change at all")
    return "; ".join(parts) + "."
