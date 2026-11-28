"""Selectable sources: the media folder, keys that cannot name another file, bad files, a browser-style WebM.

All video files are generated into tmp dirs at test time. Only the capture thread's waiting uses a fake clock; files
are decoded by the real OpenCV.
"""
import itertools
import json
import os
import threading
import time
from pathlib import Path

import pytest

from tests import media_files
from utils.file_management import BASE_DIR
from video_handling import source, streaming
from video_handling.sampling import GridSampler
from video_handling.source import MediaLibrary, make_spec, selectable
from video_handling.streaming import CaptureHub


@pytest.fixture
def media(tmp_path):
    folder = tmp_path / "media"
    folder.mkdir()
    return MediaLibrary(folder)


def keys(library):
    return [(s.source_key, s.label) for s in selectable(library)]


# -- listing ---------------------------------------------------------------------------------------------------


def test_lists_samples_media_files_and_the_webcam(media):
    folder = media.folder
    for name in ("b.webm", "A.MP4", "notes.txt", ".hidden.mp4", "clip.mov", "c.mp4.part"):
        (folder / name).write_bytes(b"x")
    (folder / "sub.mp4").mkdir()  # a folder with a video name
    (folder / "sub.mp4" / "inner.mp4").write_bytes(b"x")
    assert media.scan() is True
    assert keys(media) == [
        ("sample:1", "Sample 1 (street_view_1.mp4)"), ("sample:2", "Sample 2 (street_view_2.mp4)"),
        ("media:1", "A.MP4 (media)"), ("media:2", "b.webm (media)"), ("webcam", "Webcam"),
    ]
    assert media.scan() is False


def test_numbers_stay_with_their_file_across_rescans(media):
    folder = media.folder
    (folder / "b.mp4").write_bytes(b"x")
    media.scan()
    assert make_spec("media:1", 0, media).path == str(folder / "b.mp4")
    (folder / "a.mp4").write_bytes(b"x")  # sorts first, but must not take b's number
    assert media.scan() is True
    assert media.listing() == [(2, "a.mp4"), (1, "b.mp4")]
    assert make_spec("media:1", 0, media).path == str(folder / "b.mp4")
    (folder / "b.mp4").unlink()
    media.scan()
    with pytest.raises(ValueError):
        make_spec("media:1", 0, media)  # gone: refused, not reassigned
    (folder / "b.mp4").write_bytes(b"x")
    media.scan()
    assert make_spec("media:1", 0, media).path == str(folder / "b.mp4")


def test_a_missing_folder_lists_nothing(tmp_path):
    library = MediaLibrary(tmp_path / "absent")
    assert library.scan() is False and library.listing() == []
    assert [s.source_key for s in selectable(library)] == ["sample:1", "sample:2", "webcam"]


def test_listing_is_capped(media, monkeypatch):
    monkeypatch.setattr(source, "MEDIA_MAX_FILES", 3)
    for i in range(5):
        (media.folder / f"{i}.mp4").write_bytes(b"x")
    media.scan()
    assert [name for _n, name in media.listing()] == ["0.mp4", "1.mp4", "2.mp4"]


def test_links_are_not_listed(media, tmp_path):
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"x")
    try:
        os.symlink(outside, media.folder / "link.mp4")
    except (OSError, NotImplementedError):
        pytest.skip("this account cannot create symbolic links")
    media.scan()
    assert media.listing() == []


# -- keys cannot reach another file ----------------------------------------------------------------------------


HOSTILE_KEYS = [
    "media:0", "media:-1", "media:01", "media:+1", "media: 1", "media:1 ", " media:1", "media:1\n", "media:1.0",
    "media:1/../../x", "media:../a.mp4", "media:a.mp4", "media:..", "media:", "media", "media:9999999",
    "media:\u0661",  # an Arabic-Indic digit 1: int() would take it, the key rule must not
    "media:1;media:2", "MEDIA:1", "sample:1/../../main.py", "sample:01", "../main.py", "/etc/passwd",
    "C:\\Windows\\win.ini", "file:///etc/passwd", "webcam:0", "", "a.mp4",
]


@pytest.mark.parametrize("key", HOSTILE_KEYS)
def test_keys_outside_the_issued_list_are_refused(media, key):
    (media.folder / "a.mp4").write_bytes(b"x")
    media.scan()
    with pytest.raises(ValueError):
        make_spec(key, 0, media)


def test_every_selectable_path_is_a_sample_or_directly_in_the_media_folder(media):
    for name in ("a b.mp4", "\u00fcn\u00efcode.webm", "..mp4", "-.webm", "x..webm"):
        (media.folder / name).write_bytes(b"x")
    media.scan()
    fuzz = [f"media:{n}" for n in range(0, 12)] + [f"sample:{n}" for n in range(0, 4)]
    allowed = {Path(p) for p in source.SAMPLE_PATHS}
    seen = 0
    for key in fuzz:
        try:
            spec = make_spec(key, 0, media)
        except ValueError:
            continue
        seen += 1
        path = Path(spec.path or "")
        assert path in allowed or path.parent == media.folder, key
        assert os.sep not in spec.label and "/" not in spec.label
    assert seen == 2 + 4  # "..mp4" starts with a dot: hidden, not listed


def test_the_media_folder_is_git_ignored_and_in_the_repository_root():
    assert source.MEDIA_DIR == BASE_DIR / "media"
    lines = (BASE_DIR / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "media/" in lines


# -- bad files give an error, and another source still works ----------------------------------------------------


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def wait(self, seconds):
        self.now += seconds


class Clips:
    """A hub sink: records every frame fed (media time, fake-clock time) and the clips cut from them."""

    def __init__(self, hub, clock):
        self.clock = clock
        self.fed: list[tuple[str, int, float, float]] = []
        self.clips: list[tuple[str, int, float]] = []
        self.second_pass = threading.Event()
        self.sampler = GridSampler(10, 5, lambda seg, start, *_: self.clips.append((seg.spec.source_key, seg.loop,
                                                                                    start)))
        hub.add_sink(self)

    def feed(self, segment, t, frame):
        self.fed.append((segment.spec.source_key, segment.loop, t, self.clock()))
        if segment.loop >= 1:
            self.second_pass.set()
        self.sampler.feed(segment, t, frame)

    def advance(self, segment, t):
        self.sampler.advance(segment, t)


def wait_until(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


BAD_FILES = {
    "corrupt": (lambda p: media_files.corrupt(p), "the file cannot be opened as a video"),
    "zero-frame": (lambda p: media_files.zero_frames(p), "the file cannot be opened as a video"),
    "truncated": (lambda p: media_files.truncated(p), "the file cannot be opened as a video"),
    "oversized-frames": (lambda p: media_files.write_video(p, 3, size=(1928, 1088)),
                         "frames of 1928x1088 are larger than 1920x1080"),
    "oversized-file": (lambda p: media_files.write_video(p, 10), "the file is larger than 0 MB"),
}


@pytest.mark.parametrize("kind", BAD_FILES)
def test_a_bad_file_is_a_source_error_and_another_source_still_works(media, monkeypatch, kind):
    make, reason = BAD_FILES[kind]
    make(media.folder / "bad.mp4")
    media_files.write_video(media.folder / "good.mp4", 10)
    media.scan()
    bad_key = next(f"media:{n}" for n, name in media.listing() if name == "bad.mp4")
    good_key = next(f"media:{n}" for n, name in media.listing() if name == "good.mp4")
    if kind == "oversized-file":
        size = (media.folder / "bad.mp4").stat().st_size
        monkeypatch.setattr(streaming, "MAX_FILE_BYTES", size - 1)  # the good file is the same size: switch first
    errors: list[str] = []
    clock = FakeClock()
    hub = CaptureHub(bad_key, media=media, clock=clock, wait=clock.wait, on_error=errors.append)
    sink = Clips(hub, clock)
    hub.start()
    try:
        wait_until(lambda: errors)
        assert errors == [f"Capture source failed: Could not read or decode the data (CaptureError): {reason}"]
        assert hub.error == errors[0] and "bad.mp4" not in errors[0] and str(media.folder) not in errors[0]
        assert sink.clips == []
        if kind == "oversized-file":
            monkeypatch.setattr(streaming, "MAX_FILE_BYTES", 1 << 30)
        hub.select(good_key)
        wait_until(lambda: any(key == good_key for key, *_ in sink.clips))
        assert hub.error is None and len(errors) == 1
    finally:
        hub.stop()


def test_a_file_deleted_while_selected_is_an_error(media):
    path = media_files.write_video(media.folder / "a.mp4", 10)
    media.scan()
    errors: list[str] = []
    clock = FakeClock()

    class DeletedAfterThisPass:
        """The real decoder; the file goes once it is released (an open file cannot be deleted on Windows)."""

        def __init__(self, spec):
            self.cap = streaming.open_capture(spec)

        def __getattr__(self, name):
            return getattr(self.cap, name)

        def release(self):
            self.cap.release()
            path.unlink(missing_ok=True)

    hub = CaptureHub("media:1", media=media, clock=clock, wait=clock.wait, on_error=errors.append,
                     open_source=DeletedAfterThisPass)
    sink = Clips(hub, clock)
    hub.start()
    try:
        wait_until(lambda: errors)
        assert sink.fed and {loop for _k, loop, *_ in sink.fed} == {0}
        assert errors[0].endswith(": the file is gone")
    finally:
        hub.stop()


# -- a browser recording: no frame rate, uneven timestamps ------------------------------------------------------


GAPS_MS = [17, 50, 33, 24, 41, 66, 19, 33, 45, 28]  # 35.6 ms on average: a busy tab's recording
RECORDED_MS = list(itertools.accumulate([0] + [GAPS_MS[i % len(GAPS_MS)] for i in range(59)]))  # 2.108 s


def test_a_mediarecorder_style_webm_plays_at_its_recorded_speed(media):
    import cv2

    path = media_files.mediarecorder_webm(media.folder / "recording.webm", RECORDED_MS)
    probe = cv2.VideoCapture(str(path))
    try:
        # The symptoms of a browser recording (design/capture.md): bogus frame rate and frame count.
        assert probe.get(cv2.CAP_PROP_FPS) == 1000.0
        assert probe.get(cv2.CAP_PROP_FRAME_COUNT) < 0
    finally:
        probe.release()
    media.scan()
    clock = FakeClock()
    hub = CaptureHub("media:1", media=media, clock=clock, wait=clock.wait, min_pass_s=0.5)
    sink = Clips(hub, clock)
    hub.start()
    try:
        assert sink.second_pass.wait(10)
    finally:
        hub.stop()
    first = [(t, at) for _k, loop, t, at in sink.fed if loop == 0]
    assert [t for t, _at in first] == pytest.approx([ms / 1000 for ms in RECORDED_MS], abs=1e-3)
    # Shown at its media time: on the fake clock, frame i appears RECORDED_MS[i] after the first.
    assert [at - first[0][1] for _t, at in first] == pytest.approx([ms / 1000 for ms in RECORDED_MS], abs=0.06)
    starts = [start for _k, loop, start in sink.clips if loop == 0]
    assert starts == pytest.approx([0.0, 0.5, 1.0, 1.5])  # 2.108 s plus the last frame's hold: four whole clips


# -- the protocol ---------------------------------------------------------------------------------------------


def send(ws, **frame):
    ws.send_text(json.dumps(frame))


def test_list_sources_picks_up_new_files_and_set_source_selects_them(compare_app):
    with compare_app.client.websocket_connect("/ws") as ws:
        assert [s["key"] for s in ws.receive_json()["sources"]] == ["sample:1", "sample:2", "webcam"]
        send(ws, type="list_sources")
        assert ws.receive_json()["type"] == "state"  # unchanged: to the sender only
        (compare_app.media.folder / "mine.webm").write_bytes(b"x")
        send(ws, type="list_sources")
        state = ws.receive_json()
        assert {"key": "media:1", "label": "mine.webm (media)"} in state["sources"]
        send(ws, type="set_source", source_key="media:1")
        assert ws.receive_json()["source"] == {"key": "media:1", "label": "mine.webm (media)", "epoch": 1,
                                               "error": None}
        assert compare_app.hub.spec.path == str(compare_app.media.folder / "mine.webm")
        for key in ("media:2", "media:../mine.webm", "media:" + str(compare_app.media.folder / "mine.webm")):
            send(ws, type="set_source", source_key=key)
            frame = ws.receive_json()
            assert frame["type"] == "error" and frame["request"] == "set_source"
        assert compare_app.hub.spec.source_key == "media:1"


def test_set_source_refuses_a_file_deleted_since_the_listing(compare_app):
    (compare_app.media.folder / "gone.mp4").write_bytes(b"x")
    compare_app.media.scan()
    with compare_app.client.websocket_connect("/ws") as ws:
        assert any(s["key"] == "media:1" for s in ws.receive_json()["sources"])
        (compare_app.media.folder / "gone.mp4").unlink()
        send(ws, type="set_source", source_key="media:1")
        assert [s["key"] for s in ws.receive_json()["sources"]] == ["sample:1", "sample:2", "webcam"]  # to all
        assert ws.receive_json() == {"type": "error", "text": "Unknown source: 'media:1'", "request": "set_source"}
