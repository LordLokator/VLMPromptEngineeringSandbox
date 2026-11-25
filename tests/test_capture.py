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
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 32))
    for i in range(n_frames):
        writer.write(np.full((32, 32, 3), i * 10 % 255, dtype=np.uint8))
    writer.release()


def _instant_stream():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    while True:
        yield frame


def test_recorder_paces_once_by_timestamp(monkeypatch, tmp_path):
    seen = {"paced": []}

    def fake_source(paced: bool = True):
        seen["paced"].append(paced)
        return _instant_stream()

    clips: list[tuple[int, float]] = []
    stamp = {"t": None}

    def fake_write(frames, path, fps=60):
        clips.append((len(frames), time.monotonic()))

    monkeypatch.setattr(clip_handling, "get_stream_source", fake_source)
    monkeypatch.setattr(clip_handling, "write_video", fake_write)
    monkeypatch.setattr(clip_handling, "BASE_DIR", tmp_path)
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

    def fake_source(paced: bool = True):
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
