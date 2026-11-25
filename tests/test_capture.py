from pathlib import Path

import pytest

from utils.file_management import BASE_DIR, full_path
from video_handling import streaming
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
