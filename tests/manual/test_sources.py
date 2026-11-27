"""Capture checks that need a real device or a real recording. Skipped unless VLM_MANUAL_TESTS=1.

- The webcam test opens camera 0 of this machine for a few seconds. Frames stay in memory.
- The WebM test needs VLM_MANUAL_WEBM: the path of a .webm recorded by a browser, at least 3 s long.
"""
import os
import threading
import time
from pathlib import Path

import cv2
import pytest

from video_handling import source
from video_handling.sampling import GridSampler
from video_handling.streaming import CaptureHub

pytestmark = [
    pytest.mark.manual,
    pytest.mark.skipif(os.environ.get("VLM_MANUAL_TESTS") != "1", reason="manual: set VLM_MANUAL_TESTS=1"),
]


class Clips:
    def __init__(self, hub, wanted):
        self.clips = []
        self.done = threading.Event()
        self.sampler = GridSampler(10, 5, lambda *clip: self._add(clip, wanted))
        hub.add_sink(self.sampler)

    def _add(self, clip, wanted):
        self.clips.append((time.monotonic(), *clip))
        if len(self.clips) >= wanted:
            self.done.set()


def test_webcam_delivers_clips_in_real_time():
    probe = cv2.VideoCapture(0)
    opened = probe.isOpened()
    probe.release()
    if not opened:
        pytest.skip("no webcam")
    hub = CaptureHub("webcam")
    got = Clips(hub, 6)
    hub.start()
    try:
        assert got.done.wait(20), hub.error
    finally:
        hub.stop()
    # Six half-second clips arrive about 2.5 s apart, first to last.
    assert 1.8 <= got.clips[5][0] - got.clips[0][0] <= 4.0


def test_browser_recorded_webm_plays_at_its_own_speed(monkeypatch):
    path = os.environ.get("VLM_MANUAL_WEBM", "")
    if not Path(path).is_file():
        pytest.skip("VLM_MANUAL_WEBM does not name a file")
    monkeypatch.setattr(source, "SAMPLE_PATHS", [path])
    hub = CaptureHub()
    got = Clips(hub, 5)
    hub.start()
    try:
        assert got.done.wait(30), hub.error
    finally:
        hub.stop()
    assert [clip[2] for clip in got.clips[:5]] == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])
    assert 1.6 <= got.clips[4][0] - got.clips[0][0] <= 2.8
