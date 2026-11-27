import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from utils.file_management import BASE_DIR, full_path
from video_handling import clip_handling, streaming
from video_handling.clip_handling import ClipRecorder


def test_full_path_is_cwd_independent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert Path(full_path("static_video/x.mp4")) == BASE_DIR / "static_video" / "x.mp4"


def test_capture_paths_absolute_under_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recorder = ClipRecorder()
    assert Path(recorder.out_dir) == BASE_DIR / "tmp"
    for p in [*streaming.VIDEO_PATHS, streaming.FALLBACK_IMAGE]:
        assert Path(p).is_absolute()
        assert BASE_DIR in Path(p).parents
    assert Path(streaming.FALLBACK_IMAGE).exists()


class FakeCap:
    instances: list["FakeCap"] = []

    def __init__(self, _src, opened: bool = True):
        self.opened = opened
        self.released = 0
        FakeCap.instances.append(self)

    def isOpened(self):
        return self.opened

    def read(self):
        return True, "frame"

    def get(self, _prop):
        return 1000.0

    def release(self):
        self.released += 1


def test_open_failure_raises_runtime_error(tmp_path):
    missing = str(tmp_path / "nope.mp4")
    with pytest.raises(RuntimeError):
        streaming._init_video_object(missing)


def test_static_video_close_releases_capture(monkeypatch):
    FakeCap.instances.clear()
    monkeypatch.setattr(streaming.cv2, "VideoCapture", FakeCap)
    gen = streaming.static_video()
    next(gen)
    gen.close()
    assert [c.released for c in FakeCap.instances] == [1]


def test_live_stream_close_releases_capture(monkeypatch):
    FakeCap.instances.clear()
    monkeypatch.setattr(streaming.cv2, "VideoCapture", FakeCap)
    gen = streaming.live_stream()
    next(gen)
    gen.close()
    assert [c.released for c in FakeCap.instances] == [1]


def _write_tiny_mp4(path, n_frames, fps=20):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (32, 32))
    for i in range(n_frames):
        writer.write(np.full((32, 32, 3), i * 10 % 255, dtype=np.uint8))
    writer.release()


def _instant_stream():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    while True:
        yield frame


def test_recorder_paces_once_by_timestamp(monkeypatch, tmp_path):
    seen = {"paced": []}

    def fake_source(paced: bool = True, sample_fps=None):
        seen["paced"].append(paced)
        return _instant_stream()

    clips: list[tuple[int, float]] = []
    stamp = {"t": None}

    def fake_write(frames, path, fps=60):
        clips.append((len(frames), time.monotonic()))

    monkeypatch.setattr(clip_handling, "get_stream_source", fake_source)
    monkeypatch.setattr(clip_handling, "write_video", fake_write)
    rec = ClipRecorder(buffer_seconds=0.3, fps=10)
    rec.out_dir = str(tmp_path)
    done = threading.Event()
    rec.start(lambda _p: done.set() if len(clips) >= 3 else None)
    stamp["t"] = time.monotonic()
    assert done.wait(timeout=5)
    rec.running = False
    assert seen["paced"][0] is False
    assert all(n == 3 for n, _ in clips)
    per_clip = (clips[2][1] - stamp["t"]) / 3
    assert 0.3 * 0.7 <= per_clip <= 0.3 * 1.3


def test_static_video_paced_sleeps(monkeypatch, tmp_path):
    video = tmp_path / "a.mp4"
    _write_tiny_mp4(video, 10, fps=20)
    monkeypatch.setattr(streaming, "VIDEO_PATHS", [str(video)])
    monkeypatch.setattr(streaming, "VIDEO_IDX", 0)
    gen = streaming.static_video(paced=True)
    t0 = time.monotonic()
    for _ in range(5):
        next(gen)
    gen.close()
    # 5 frames at 20 fps: the 4 gaps between yields take >= 0.2 s
    assert time.monotonic() - t0 >= 0.15

    gen = streaming.static_video(paced=False)
    t0 = time.monotonic()
    for _ in range(5):
        next(gen)
    gen.close()
    assert time.monotonic() - t0 < 0.15


def test_toggle_reopens_stream_once_at_clip_boundary(monkeypatch, tmp_path):
    opened: list[bool] = []
    closed: list[int] = []
    flag = {"value": False}

    def make_stream(mode: bool):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        try:
            while True:
                yield frame
        finally:
            closed.append(len(opened))

    def fake_source(paced: bool = True, sample_fps=None):
        opened.append(flag["value"])
        return make_stream(flag["value"])

    clips: list[int] = []
    done = threading.Event()

    def fake_write(frames, path, fps=60):
        clips.append(len(frames))
        if len(clips) == 1:
            flag["value"] = True  # flip mid-run, after the first clip is written

    def on_flush(_p):
        if len(clips) >= 3:
            done.set()

    monkeypatch.setattr(clip_handling, "get_stream_source", fake_source)
    monkeypatch.setattr(clip_handling, "write_video", fake_write)
    monkeypatch.setattr(clip_handling, "USE_LIVE_CAMERA", flag)
    rec = ClipRecorder(buffer_seconds=0.1, fps=10)
    rec.out_dir = str(tmp_path)
    rec.start(on_flush)
    assert done.wait(timeout=5)
    rec.running = False
    assert opened == [False, True]
    assert closed[0] == 1  # old stream closed before the new one was opened


def test_static_video_rotates_and_wraps_around(monkeypatch, tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    _write_tiny_mp4(a, 3)
    _write_tiny_mp4(b, 3)
    monkeypatch.setattr(streaming, "VIDEO_PATHS", [str(a), str(b)])
    monkeypatch.setattr(streaming, "VIDEO_IDX", 0)
    gen = streaming.static_video(paced=False)
    for _ in range(8):  # more than 2 clips' worth of frames
        next(gen)
    gen.close()
    assert streaming.VIDEO_IDX == 0  # a -> b -> a


class CountingCap:
    """30 fps fake source: frame i is the integer i, 90 frames (3 s) per pass."""

    def __init__(self, _src):
        self.i = 0

    def isOpened(self):
        return True

    def read(self):
        if self.i >= 90:
            return False, None
        self.i += 1
        return True, self.i - 1

    def get(self, _prop):
        return 30.0

    def release(self):
        pass


def test_sampling_follows_source_timestamps(monkeypatch):
    monkeypatch.setattr(streaming.cv2, "VideoCapture", CountingCap)
    monkeypatch.setattr(streaming, "VIDEO_IDX", 0)
    gen = streaming.static_video(paced=False, sample_fps=10)
    got = [next(gen) for _ in range(30)]
    gen.close()
    assert got == list(range(0, 90, 3))  # 30 frames spanning 3 s of source time


def _recorder(monkeypatch, tmp_path, source, fps=10, seconds=0.3):
    monkeypatch.setattr(clip_handling, "get_stream_source", source)
    monkeypatch.setattr(clip_handling, "write_video", lambda frames, path, fps=60: None)
    rec = ClipRecorder(buffer_seconds=seconds, fps=fps)
    rec.out_dir = str(tmp_path)
    return rec


def test_stop_iteration_does_not_shorten_clips(monkeypatch, tmp_path):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    calls = {"n": 0}

    def source(paced=True, sample_fps=None):
        calls["n"] += 1

        def gen():
            if calls["n"] == 1:
                yield frame  # first stream ends after one frame
                return
            while True:
                yield frame
        return gen()

    sizes = []
    rec = _recorder(monkeypatch, tmp_path, source)
    monkeypatch.setattr(clip_handling, "write_video", lambda frames, path, fps=60: sizes.append(len(frames)))
    done = threading.Event()
    rec.start(lambda _p: done.set())
    assert done.wait(5)
    rec.stop()
    assert sizes[0] == 3


def test_stop_joins_thread_and_closes_stream(monkeypatch, tmp_path):
    closed = threading.Event()

    def source(paced=True, sample_fps=None):
        def gen():
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            try:
                while True:
                    yield frame
            finally:
                closed.set()
        return gen()

    rec = _recorder(monkeypatch, tmp_path, source, seconds=5)
    rec.start()
    time.sleep(0.2)
    t0 = time.monotonic()
    rec.stop()
    assert time.monotonic() - t0 < 2
    assert rec._thread is not None and not rec._thread.is_alive()
    assert closed.is_set()


def test_open_failure_calls_on_error_once(monkeypatch, tmp_path):
    monkeypatch.setattr(streaming, "VIDEO_PATHS", [str(tmp_path / "missing.mp4")])
    monkeypatch.setattr(streaming, "VIDEO_IDX", 0)
    monkeypatch.setitem(clip_handling.USE_LIVE_CAMERA, "value", False)
    monkeypatch.setattr(clip_handling, "write_video", lambda frames, path, fps=60: None)
    errors = []
    rec = ClipRecorder(buffer_seconds=0.3, fps=10)
    rec.out_dir = str(tmp_path)
    rec.start(on_error=errors.append)
    assert rec._thread is not None
    rec._thread.join(5)
    assert errors == ["Capture source failed: Could not read or decode the data (CaptureError)"]


def test_video_feed_ends_cleanly_on_open_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(streaming, "VIDEO_PATHS", [str(tmp_path / "missing.mp4")])
    monkeypatch.setattr(streaming, "VIDEO_IDX", 0)
    monkeypatch.setitem(streaming.USE_LIVE_CAMERA, "value", False)
    assert list(streaming.video_stream_generator()) == []


def test_makedir_nested_and_idempotent(tmp_path):
    from utils.file_management import makedir

    target = str(tmp_path / "a" / "b")
    makedir(target)
    makedir(target)
    assert Path(target).is_dir()
