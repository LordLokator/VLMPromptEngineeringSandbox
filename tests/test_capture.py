from pathlib import Path

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
