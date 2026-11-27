"""Frame times and the sampling grid. No OpenCV and no threads, so both are testable with plain numbers."""
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from video_handling.source import SourceSpec

DEFAULT_FPS = 30.0
MAX_SANE_FPS = 240.0
MAX_TIMESTAMP_GAP_S = 10.0
EPS = 1e-3  # decoder timestamps are milliseconds in floating point


def sane_fps(reported: float) -> float:
    """The reported frame rate if it is plausible, else 30. Some containers report 0, NaN or 1000."""
    return reported if math.isfinite(reported) and 1.0 <= reported <= MAX_SANE_FPS else DEFAULT_FPS


class MediaTimer:
    """Seconds since the first frame of one pass, strictly increasing.

    Uses the decoder's timestamps. A frame whose timestamp stalls, goes backwards or is not a number is placed one
    nominal frame after the previous one. So is a jump of more than MAX_TIMESTAMP_GAP_S, after which later
    timestamps are counted from there.
    """

    def __init__(self, fps: float):
        self.frame_s = 1 / sane_fps(fps)
        self._origin_ms = 0.0
        self._last: float | None = None

    def stamp(self, pos_msec: float) -> float:
        known = math.isfinite(pos_msec)
        if self._last is None:
            self._origin_ms = pos_msec if known else 0.0
            self._last = 0.0
            return 0.0
        t = (pos_msec - self._origin_ms) / 1000 if known else math.nan
        if t > self._last + MAX_TIMESTAMP_GAP_S:
            self._origin_ms = pos_msec - (self._last + self.frame_s) * 1000
            t = self._last + self.frame_s
        elif not t > self._last:
            t = self._last + self.frame_s
        self._last = t
        return t


@dataclass(frozen=True)
class Segment:
    """One uninterrupted stretch of one source: a pass through a file, or a webcam run without a long gap."""

    spec: SourceSpec
    number: int
    loop: int | None  # pass through the file, from 0; None for the webcam
    base_s: float  # added to times within the segment: 0 for files, wall-clock seconds for the webcam


# (segment, start in seconds within the segment, frames, grid time of each frame)
ClipSink = Callable[[Segment, float, list[Any], list[float]], None]


class GridSampler:
    """Sample-and-hold on a fixed time grid: each grid point gets the latest frame at or before it.

    A clip is `clip_frames` consecutive grid points of one segment. A new segment discards the partial clip.
    Calls must come from one thread.
    """

    def __init__(self, fps: int, clip_frames: int, on_clip: ClipSink):
        self.fps = fps
        self.clip_frames = clip_frames
        self.on_clip = on_clip
        self._segment: Segment | None = None
        self._held: Any = None
        self._next = 0  # index of the next grid point to fill
        self._frames: list[Any] = []

    def feed(self, segment: Segment, t: float, frame: Any) -> None:
        """A frame that is on screen from `t` on."""
        if segment != self._segment:
            self._segment = segment
            self._next = math.ceil((t - EPS) * self.fps)
            self._frames = []
        else:
            self._fill(t)
        self._held = frame

    def advance(self, segment: Segment, t: float) -> None:
        """The held frame is still on screen just before `t`."""
        if segment == self._segment:
            self._fill(t)

    def _fill(self, until: float) -> None:
        while self._next / self.fps < until - EPS:
            self._frames.append(self._held)
            self._next += 1
            if len(self._frames) == self.clip_frames:
                frames, self._frames = self._frames, []
                times = [(self._next - self.clip_frames + i) / self.fps for i in range(self.clip_frames)]
                assert self._segment is not None
                self.on_clip(self._segment, times[0], frames, times)
