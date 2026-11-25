'''Provides thread-safe Prompt and RuntimeSettings objects.'''

import math
from threading import Lock
import time

from loguru import logger

from utils.json_management import append_dict_to_json


class Prompt:
    def __init__(self):
        # Leave default as something obnoxious,
        # so errors are easily spotted.
        self._prompt = "Write a funny sentence!"
        self._lock = Lock()

    def get(self):
        with self._lock:
            return self._prompt

    def set(self, new_prompt: str):
        with self._lock:
            if new_prompt == self._prompt:
                logger.debug("Duplicate prompt detected — not recording.")
                return

            self._prompt = new_prompt
            logger.info(f"New prompt: [ {new_prompt} ]")

            data = {
                'prompt': new_prompt,
                'time': get_formatted_time()
            }

            # History is best-effort: a disk problem must not undo the prompt change.
            try:
                append_dict_to_json(file_path='./prompting/history.json', new_data=data)
            except OSError as e:
                logger.error(f"Could not write prompt history: {e}")


class RuntimeSettings:
    """Thread-safe settings the UI can change while inference runs."""

    MIN_TEMPERATURE = 0.01  # HF generate() with sampling rejects temperature=0
    MAX_TEMPERATURE = 1.0

    def __init__(self, clip_fps: int = 10, temperature: float = 0.01):
        self._lock = Lock()
        self._clip_fps = int(clip_fps)
        self._temperature = temperature
        self._resample_fps = self._clip_fps

    def get_temperature(self) -> float:
        with self._lock:
            return self._temperature

    def set_temperature(self, value: float) -> float:
        """Clamp to [0.01, 1.0]; raises ValueError for non-numeric, NaN or inf."""
        try:
            v = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"temperature must be a number, got {value!r}") from e
        if not math.isfinite(v):
            raise ValueError(f"temperature must be finite, got {value!r}")
        v = min(max(v, self.MIN_TEMPERATURE), self.MAX_TEMPERATURE)
        with self._lock:
            self._temperature = v
        return v

    def get_resample_fps(self) -> int:
        """Requested fps handed to the VLM processor (an int in 1..clip fps)."""
        with self._lock:
            return self._resample_fps

    def set_resample_fps(self, value: int) -> float:
        """Clamp to 1..clip fps; returns the effective rate the processor will use."""
        try:
            v = int(value)
        except (TypeError, ValueError, OverflowError) as e:
            raise ValueError(f"resample fps must be an integer, got {value!r}") from e
        v = min(max(v, 1), self._clip_fps)
        with self._lock:
            self._resample_fps = v
        return self.effective_resample_fps(v)

    def effective_resample_fps(self, requested: int | None = None) -> float:
        """The processor strides by floor(clip_fps / fps) frames, so the real rate is clip_fps / that stride."""
        v = self.get_resample_fps() if requested is None else requested
        return self._clip_fps / (self._clip_fps // v)


def get_formatted_time() -> str:
    _ts = time.localtime()

    month = f"{_ts.tm_mon:02d}"
    day = f"{_ts.tm_mday:02d}"
    hour = f"{_ts.tm_hour:02d}"
    minute = f"{_ts.tm_min:02d}"

    return f"{month}.{day} | {hour}:{minute}"
