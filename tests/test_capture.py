"""The capture hub and the clip recorder with fake or generated sources. No test opens a camera.

Pacing is asserted with a fake clock. The only real time here is bounded waiting for a thread.
"""
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import web_engine
from utils.file_management import BASE_DIR, full_path
from video_handling import clip_handling, source, streaming
from video_handling.clip_handling import ClipRecorder
from video_handling.sampling import GridSampler, MediaTimer, Segment, sane_fps
from video_handling.source import make_spec
from video_handling.streaming import CaptureHub, mjpeg_stream

BLACK = np.zeros((8, 8, 3), dtype=np.uint8)


def test_full_path_is_cwd_independent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert Path(full_path("static_video/x.mp4")) == BASE_DIR / "static_video" / "x.mp4"


def test_capture_paths_absolute_under_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recorder = ClipRecorder(CaptureHub())
    assert Path(recorder.out_dir) == BASE_DIR / "tmp"
    for p in [*source.SAMPLE_PATHS, streaming.FALLBACK_IMAGE]:
        assert Path(p).is_absolute()
        assert BASE_DIR in Path(p).parents
        assert Path(p).exists()


def test_source_specs():
    spec = make_spec("sample:2", 4)
    assert (spec.source_key, spec.source_epoch, spec.is_webcam) == ("sample:2", 4, False)
    assert spec.label == "Sample 2 (street_view_2.mp4)"
    assert make_spec("webcam", 0).is_webcam
    for key in ("sample:0", "sample:3", "../x"):
        with pytest.raises(ValueError):
            make_spec(key, 0)
    with pytest.raises(AttributeError):
        spec.source_epoch = 5  # type: ignore[misc]


def test_select_counts_epochs_and_rejects_unknown_keys():
    hub = CaptureHub()
    assert (hub.spec.source_key, hub.spec.source_epoch) == ("sample:1", 0)
    assert hub.select("sample:1").source_epoch == 1
    with pytest.raises(ValueError):
        hub.select("nope")
    assert hub.spec.source_epoch == 1
    assert hub.toggle().source_key == "webcam"
    assert hub.toggle().source_key == "sample:1"
    assert hub.spec.source_epoch == 3


# -- frame times and the sampling grid ------------------------------------------------------------------------


def test_sane_fps():
    assert sane_fps(25.0) == 25.0
    for bogus in (0.0, -1.0, 1000.0, math.nan, math.inf):
        assert sane_fps(bogus) == 30.0


@pytest.mark.parametrize(
    ("positions", "expected"),
    [
        ([40, 73, 141, 180], [0, 0.033, 0.101, 0.14]),  # a non-zero first timestamp is the origin
        ([0, 0, 0, 0], [0, 0.1, 0.2, 0.3]),  # stalled
        ([0, 100, 90, 300], [0, 0.1, 0.2, 0.3]),  # backwards once
        ([0, math.nan, 200, 300], [0, 0.1, 0.2, 0.3]),
        ([0, 100, 3_600_000, 3_600_100], [0, 0.1, 0.2, 0.3]),  # an impossible jump, then counted from there
        ([0, 100, 4100], [0, 0.1, 4.1]),  # a real gap is kept
    ],
)
def test_media_timer(positions, expected):
    timer = MediaTimer(10.0)
    assert [timer.stamp(float(p)) for p in positions] == pytest.approx(expected)


def test_grid_sampler_holds_and_starts_late_joiners_on_the_grid():
    clips = []
    sampler = GridSampler(10, 5, lambda segment, start, frames, times: clips.append((start, frames, times)))
    segment = Segment(make_spec("sample:1", 0), 0, 0, 0.0)
    sampler.advance(segment, 9.0)  # nothing held yet
    sampler.feed(segment, 1.23, "a")
    sampler.feed(segment, 1.5, "b")
    sampler.advance(segment, 1.9)
    ((start, frames, times),) = clips
    assert start == pytest.approx(1.3)
    assert times == pytest.approx([1.3, 1.4, 1.5, 1.6, 1.7])
    assert frames == ["a", "a", "b", "b", "b"]


# -- the hub with fake captures -------------------------------------------------------------------------------


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.waits: list[float] = []

    def __call__(self):
        return self.now

    def wait(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


class ScriptCap:
    """Frame i has the timestamp positions[i] (ms) and is the value (tag, i)."""

    def __init__(self, positions, fps=30.0, tag="a", opened=True):
        self.positions = positions
        self.fps = fps
        self.tag = tag
        self.opened = opened
        self.i = -1
        self.released = 0

    def isOpened(self):
        return self.opened

    def read(self):
        if self.i + 1 >= len(self.positions):
            return False, None
        self.i += 1
        return True, (self.tag, self.i)

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else float(self.positions[self.i])

    def release(self):
        self.released += 1


class Collector:
    """A hub sink that cuts clips and records them; `until(self)` stops the hub from the capture thread."""

    def __init__(self, hub, until, clock=None, on_feed=None):
        self.hub, self.until, self.clock, self.on_feed = hub, until, clock, on_feed
        self.clips: list[tuple[Segment, float, list, list]] = []
        self.fed: list[tuple[Segment, float, float]] = []
        self.done = threading.Event()
        self.sampler = GridSampler(10, 5, lambda *clip: self.clips.append(clip))
        hub.add_sink(self)

    def feed(self, segment, t, frame):
        self.fed.append((segment, t, self.clock() if self.clock else 0.0))
        self.sampler.feed(segment, t, frame)
        if self.on_feed:
            self.on_feed(self, segment, t)
        self._check()

    def advance(self, segment, t):
        self.sampler.advance(segment, t)
        self._check()

    def _check(self):
        if not self.done.is_set() and self.until(self):
            self.done.set()
            self.hub.stop()

    def run(self):
        self.hub.start()
        try:
            assert self.done.wait(10)
        finally:
            self.hub.stop()
        assert self.hub._thread is not None
        self.hub._thread.join(5)
        assert not self.hub._thread.is_alive()


def fake_hub(make_cap, **kwargs):
    clock = FakeClock()
    caps: list = []

    def open_source(spec):
        caps.append(make_cap(spec))
        return caps[-1]

    hub = CaptureHub(open_source=open_source, clock=clock, wait=clock.wait, **kwargs)
    return hub, clock, caps


def held_index(positions_s, g):
    return max(i for i, t in enumerate(positions_s) if t <= g + 1e-3)


JITTERED = [i * 1000 / 30 + (7 if i % 3 == 1 else -6 if i % 3 == 2 else 0) for i in range(60)]
VARIABLE = [*range(0, 500, 20), *range(500, 1500, 200), *range(1500, 2000, 50)]


@pytest.mark.parametrize(
    ("positions", "fps", "times_ms"),
    [
        (JITTERED, 30.0, JITTERED),
        (VARIABLE, 30.0, VARIABLE),
        ([i * 40 for i in range(50)], 1000.0, [i * 40 for i in range(50)]),  # bogus FPS, good timestamps
        ([0] * 60, 1000.0, [i * 1000 / 30 for i in range(60)]),  # bogus FPS and no timestamps
        ([0] * 50, 25.0, [i * 40 for i in range(50)]),  # no timestamps: the frame rate
    ],
    ids=["jittered", "variable", "bogus-fps", "bogus-fps-stalled", "stalled"],
)
def test_clips_follow_timestamps_and_last_as_long_as_the_source(positions, fps, times_ms):
    hub, clock, _caps = fake_hub(lambda _spec: ScriptCap(positions, fps))
    sink = Collector(hub, lambda c: bool(c.fed) and c.fed[-1][0].loop == 1, clock=clock)
    sink.run()

    times = [t / 1000 for t in times_ms]
    first_pass = [c for c in sink.clips if c[0].loop == 0]
    # 2 s of video (the last frame lasts as long as the one before it): four half-second clips.
    assert [c[1] for c in first_pass] == pytest.approx([0.0, 0.5, 1.0, 1.5])
    for _segment, _start, frames, grid in first_pass:
        assert [f[1] for f in frames] == [held_index(times, g) for g in grid]
    # The next pass starts after that long on the clock.
    (_, _, begun), (_, _, looped) = sink.fed[0], sink.fed[-1]
    assert looped - begun == pytest.approx(2.0, abs=0.05)


def test_frames_are_published_at_their_media_time():
    positions = [i * 50 for i in range(10)]
    hub, clock, _caps = fake_hub(lambda _spec: ScriptCap(positions, 20.0))
    sink = Collector(hub, lambda c: len(c.fed) >= 10, clock=clock)
    sink.run()
    assert [at - sink.fed[0][2] for _, _, at in sink.fed] == pytest.approx([p / 1000 for p in positions])
    assert clock.waits and max(clock.waits) <= streaming.TICK_S + 1e-9


def test_clips_arrive_during_a_long_gap_not_after_it():
    hub, clock, _caps = fake_hub(lambda _spec: ScriptCap([0, 3000]))
    arrived = []
    sink = Collector(hub, lambda c: len(c.fed) >= 2, clock=clock)
    sink.sampler.on_clip = lambda *clip: arrived.append(clock.now)
    sink.run()
    gaps = [b - a for a, b in zip(arrived, arrived[1:], strict=False)]
    assert len(arrived) == 6 and gaps == pytest.approx([0.5] * 5, abs=0.06)


def test_short_source_holds_its_last_frame_for_one_clip():
    hub, clock, caps = fake_hub(lambda _spec: ScriptCap([0], 25.0))
    sink = Collector(hub, lambda c: len(c.clips) >= 3, clock=clock)
    sink.run()
    assert [[f[1] for f in frames] for _, _, frames, _ in sink.clips] == [[0] * 5] * 3
    assert [segment.loop for segment, *_ in sink.clips] == [0, 1, 2]
    assert sink.fed[2][2] - sink.fed[0][2] == pytest.approx(1.0, abs=0.05)  # not a spin
    assert all(c.released == 1 for c in caps)


def test_no_clip_spans_a_loop_point():
    hub, clock, _caps = fake_hub(lambda _spec: ScriptCap([i * 100 for i in range(7)], 10.0))  # 0.7 s
    seen: dict = {}

    def on_feed(sink, segment, _t):
        seen.setdefault(segment.loop, len(sink.fed))

    sink = Collector(hub, lambda c: len(c.clips) >= 3, clock=clock, on_feed=on_feed)
    sink.run()
    # One clip per pass, from its start; the 0.2 s left over is discarded.
    assert [(c[0].loop, c[1], [f[1] for f in c[2]]) for c in sink.clips] == [
        (loop, 0.0, [0, 1, 2, 3, 4]) for loop in range(3)
    ]


def test_a_switch_mid_clip_never_mixes_sources():
    def make_cap(spec):
        return ScriptCap([i * 100 for i in range(30)], 10.0, tag=spec.source_key)

    hub, clock, caps = fake_hub(make_cap)

    def switch(sink, segment, t):
        if segment.spec.source_epoch == 0 and t > 0.75:  # one clip done, the second one partly filled
            hub.select("sample:2")

    sink = Collector(
        hub, lambda c: len([1 for s, *_ in c.clips if s.spec.source_epoch == 1]) >= 2, clock=clock, on_feed=switch
    )
    sink.run()
    assert [(s.spec.source_key, start) for s, start, *_ in sink.clips] == [
        ("sample:1", 0.0),
        ("sample:2", 0.0),
        ("sample:2", 0.5),
    ]
    for segment, _start, frames, _times in sink.clips:
        assert {f[0] for f in frames} == {segment.spec.source_key}
    assert [(c.tag, c.released) for c in caps] == [("sample:1", 1), ("sample:2", 1)]


def wait_until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


@pytest.mark.parametrize(
    ("cap", "reason"),
    [
        (ScriptCap([]), "no frame could be decoded"),
        (ScriptCap([], opened=False), "the file cannot be opened as a video"),
    ],
    ids=["no-frames", "cannot-open"],
)
def test_unusable_file_ends_in_an_error_and_does_not_spin(cap, reason):
    errors: list[str] = []
    hub, _clock, caps = fake_hub(lambda _spec: cap, on_error=errors.append)
    sink = Collector(hub, lambda _c: False)
    hub.start()
    try:
        wait_until(lambda: errors)
        time.sleep(0.2)
        assert errors == [f"Capture source failed: Could not read or decode the data (CaptureError): {reason}"]
        assert hub.error == errors[0]
        assert len(caps) == 1 and cap.released == 1
        assert sink.clips == [] and sink.fed == []
    finally:
        hub.stop()
    assert hub._thread is not None and not hub._thread.is_alive()


def test_error_then_switch_recovers():
    def make_cap(spec):
        return ScriptCap([] if spec.source_key == "sample:1" else [i * 100 for i in range(10)])

    errors: list[str] = []
    hub, _clock, _caps = fake_hub(make_cap, on_error=errors.append)
    sink = Collector(hub, lambda c: len(c.clips) >= 2)
    hub.start()
    try:
        wait_until(lambda: errors)
        # The preview shows the fallback image; the model gets nothing of it.
        item = hub.next_preview(0, timeout=2)
        assert item is not None and item[1][:2] == b"\xff\xd8"
        assert sink.clips == []
        hub.select("sample:2")
        assert sink.done.wait(5)
        assert hub.error is None and len(errors) == 1
        assert {s.spec.source_epoch for s, *_ in sink.clips} == {1}
    finally:
        hub.stop()


def test_a_failing_error_report_does_not_end_the_capture_thread():
    def broken_report(_text):
        raise RuntimeError("no loop")

    hub, _clock, _caps = fake_hub(lambda _spec: ScriptCap([]), on_error=broken_report)
    hub.start()
    try:
        wait_until(lambda: hub.error)
        time.sleep(0.1)
        assert hub._thread is not None and hub._thread.is_alive()
    finally:
        hub.stop()


# -- webcam, always faked -------------------------------------------------------------------------------------


class FakeWebcam:
    def __init__(self, reads, opened=True):
        self.reads = iter(reads)  # True: a frame, False: a failed read; exhausted: failed reads
        self.opened = opened
        self.count = 0
        self.released = 0
        self.on_read = None

    def isOpened(self):
        return self.opened

    def read(self):
        self.count += 1
        if self.on_read:
            self.on_read()
        return (True, BLACK) if next(self.reads, False) else (False, None)

    def release(self):
        self.released += 1


def run_until_error(cam):
    clock = FakeClock()
    errors: list[str] = []
    hub = CaptureHub(
        "webcam", open_source=lambda _spec: cam, clock=clock, wait=clock.wait, on_error=errors.append
    )
    hub.start()
    try:
        wait_until(lambda: errors)
    finally:
        hub.stop()
    return hub, clock, errors


def test_webcam_read_failure_waits_between_reads_and_then_errors():
    cam = FakeWebcam([True, False, True])  # one hiccup is tolerated, then the camera goes away
    _hub, clock, errors = run_until_error(cam)
    assert len(errors) == 1 and cam.released == 1
    tries = streaming.WEBCAM_TIMEOUT_S / streaming.WEBCAM_RETRY_S
    assert cam.count <= tries + 5
    assert sum(clock.waits) == pytest.approx(streaming.WEBCAM_TIMEOUT_S, abs=0.2)


def test_missing_webcam_is_an_error():
    cam = FakeWebcam([], opened=False)
    _hub, _clock, errors = run_until_error(cam)
    assert len(errors) == 1 and cam.count == 0 and cam.released == 1


def test_webcam_clips_use_the_wall_clock_and_restart_after_a_gap():
    clock = FakeClock()
    cam = FakeWebcam([True] * 200)

    def on_read():
        clock.now += 5.0 if cam.count == 9 else 0.1  # the camera paces; one long stall

    cam.on_read = on_read
    hub = CaptureHub(
        "webcam",
        open_source=lambda _spec: cam,
        clock=clock,
        wait=clock.wait,
        wall_clock=lambda: 1_000_000.0 + clock.now,
    )
    sink = Collector(hub, lambda c: len(c.clips) >= 3)
    sink.run()
    (first, first_start, *_), (second, second_start, *_), (third, third_start, *_) = sink.clips[:3]
    assert first.loop is None and first_start == 0.0 and first.base_s == pytest.approx(1_000_100.1)
    # Frames 9 onward form a new segment: nothing was held across the stall.
    assert second != first and second_start == 0.0 and second.base_s - first.base_s == pytest.approx(5.7)
    assert third == second and third_start == pytest.approx(0.5)


# -- preview --------------------------------------------------------------------------------------------------


class EndlessCap:
    def __init__(self, shape=(8, 8, 3)):
        self.frame = np.zeros(shape, dtype=np.uint8)
        self.i = 0

    def isOpened(self):
        return True

    def read(self):
        self.i += 1
        return True, self.frame

    def get(self, prop):
        return 50.0 if prop == cv2.CAP_PROP_FPS else self.i * 20.0

    def release(self):
        pass


def test_two_viewers_share_one_capture():
    opened = []

    def open_source(spec):
        opened.append(spec)
        return EndlessCap()

    hub = CaptureHub(open_source=open_source)
    hub.start()
    viewers = [mjpeg_stream(hub), mjpeg_stream(hub)]
    try:
        for _ in range(3):
            for viewer in viewers:
                part = next(viewer)
                assert part.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\xff\xd8")
        assert len(opened) == 1
    finally:
        hub.stop()
    assert [next(v, None) for v in viewers] == [None, None]  # a stopped hub ends the streams


def test_each_frame_is_encoded_once_for_all_viewers(monkeypatch):
    encoded = []
    monkeypatch.setattr(streaming, "preview_jpeg", lambda frame: encoded.append(1) or b"jpeg")
    hub = CaptureHub()
    assert hub.next_preview(0, timeout=0.01) is None  # nothing captured yet
    hub._show(hub.spec, BLACK)
    assert hub.next_preview(0) == hub.next_preview(0) == (1, b"jpeg")
    assert hub.next_preview(1, timeout=0.01) == (1, b"jpeg")  # the latest again after the timeout
    assert len(encoded) == 1


@pytest.mark.parametrize(("shape", "expected"), [((1080, 1920, 3), (720, 1280)), ((480, 640, 3), (480, 640))])
def test_preview_is_at_most_720_high(shape, expected):
    jpeg = streaming.preview_jpeg(np.zeros(shape, dtype=np.uint8))
    assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape[:2] == expected


def test_video_feed_reads_the_app_hub(monkeypatch):
    hubs = []
    monkeypatch.setattr(web_engine, "mjpeg_stream", lambda hub: hubs.append(hub) or iter(()))
    hub = object()
    request = type("R", (), {"app": type("A", (), {"state": type("S", (), {"hub": hub})()})()})()
    response = web_engine.video_feed(request)
    assert hubs == [hub]
    assert response.media_type == "multipart/x-mixed-replace; boundary=frame"


# -- generated video files: real OpenCV timestamps ------------------------------------------------------------


def _write_video_file(path, n_frames, fps=20, shape=(32, 32)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, shape)
    for i in range(n_frames):
        writer.write(np.full((shape[1], shape[0], 3), i * 12 % 255, dtype=np.uint8))
    writer.release()


def test_generated_file_gives_clips_of_the_right_duration(monkeypatch, tmp_path):
    video = tmp_path / "a.mp4"
    _write_video_file(video, 20, fps=20)  # 1 s
    monkeypatch.setattr(source, "SAMPLE_PATHS", [str(video)])
    clock = FakeClock()
    hub = CaptureHub(clock=clock, wait=clock.wait)
    sink = Collector(hub, lambda c: bool(c.fed) and c.fed[-1][0].loop == 1, clock=clock)
    sink.run()
    assert [t for s, t, _ in sink.fed if s.loop == 0] == pytest.approx([i / 20 for i in range(20)], abs=2e-3)
    assert [start for s, start, *_ in sink.clips if s.loop == 0] == pytest.approx([0.0, 0.5])
    assert sink.fed[-1][2] - sink.fed[0][2] == pytest.approx(1.0, abs=0.05)
    # Grid point k shows the frame on screen then: every second frame of a 20 fps file.
    brightness = [round(f.mean()) for f in sink.clips[0][2]]
    assert brightness == pytest.approx([i * 24 for i in range(5)], abs=6)


def test_stop_interrupts_a_long_wait():
    hub = CaptureHub(open_source=lambda _spec: EndlessCap(), clock=lambda: 0.0)  # time never passes
    sink = Collector(hub, lambda _c: False)
    hub.start()
    wait_until(lambda: sink.fed)
    t0 = time.monotonic()
    hub.stop()
    assert time.monotonic() - t0 < 2
    assert hub._thread is not None and not hub._thread.is_alive()


# -- the clip recorder ----------------------------------------------------------------------------------------


def _feed_clips(recorder, count, spec=None, loop=0):
    segment = Segment(spec or recorder.hub.spec, loop, loop, 0.0)
    recorder._sampler.feed(segment, 0.0, BLACK)
    recorder._sampler.advance(segment, count * recorder.buffer_seconds)


def _recorder(monkeypatch, tmp_path, write=lambda frames, path, fps=60: None, seconds=0.5):
    monkeypatch.setattr(clip_handling, "write_video", write)
    recorder = ClipRecorder(CaptureHub(), buffer_seconds=seconds, fps=10)
    recorder.out_dir = str(tmp_path)
    return recorder


def test_recorder_cuts_clips_from_the_hub(monkeypatch, tmp_path):
    sizes = []
    monkeypatch.setattr(clip_handling, "write_video", lambda frames, path, fps=60: sizes.append((len(frames), fps)))
    hub = CaptureHub(open_source=lambda _spec: EndlessCap())
    recorder = ClipRecorder(hub, buffer_seconds=0.5, fps=10)
    recorder.out_dir = str(tmp_path)
    flushed: list = []
    done = threading.Event()

    def on_flush(clip):
        flushed.append(clip)
        if len(flushed) >= 2:
            done.set()

    recorder.start(on_flush)
    hub.start()
    try:
        assert done.wait(10)
    finally:
        recorder.stop()
        hub.stop()
    assert set(sizes) == {(5, 10)}
    assert recorder._thread is not None and not recorder._thread.is_alive()
    assert hub._sinks == ()


def test_a_stalled_writer_never_blocks_capture(monkeypatch, tmp_path):
    release = threading.Event()
    entered = threading.Event()

    def slow_write(frames, path, fps=60):
        entered.set()
        release.wait(5)

    recorder = _recorder(monkeypatch, tmp_path, slow_write)
    recorder.start()
    try:
        _feed_clips(recorder, 1)
        assert entered.wait(5)
        t0 = time.monotonic()
        _feed_clips(recorder, 20, loop=1)  # returns although nothing is being written
        assert time.monotonic() - t0 < 2
        assert recorder._pending.qsize() == clip_handling.MAX_PENDING_CLIPS
    finally:
        release.set()
        recorder.stop()


def test_recorder_survives_a_failed_write(monkeypatch, tmp_path):
    calls = []
    done = threading.Event()

    def write(frames, path, fps=60):
        calls.append(path)
        if len(calls) == 1:
            raise OSError("disk full")

    recorder = _recorder(monkeypatch, tmp_path, write)
    recorder.start(lambda _clip: done.set())
    _feed_clips(recorder, 2)
    assert done.wait(5)
    recorder.stop()
    assert len(calls) == 2


def test_clips_carry_jpeg_frames_and_their_times(monkeypatch, tmp_path):
    flushed: list = []
    done = threading.Event()
    recorder = _recorder(monkeypatch, tmp_path)
    recorder.start(lambda clip: flushed.append(clip) or done.set())
    segment = Segment(recorder.hub.spec, 0, None, 100.0)  # a webcam-like segment: times are base + offset
    recorder._sampler.feed(segment, 0.0, BLACK)
    recorder._sampler.advance(segment, 0.5)
    assert done.wait(5)
    recorder.stop()
    (clip,) = flushed
    assert len(clip.frames) == 5 and all(f.startswith(b"\xff\xd8") for f in clip.frames)  # JPEG start marker
    assert clip.frame_times == pytest.approx((100.0, 100.1, 100.2, 100.3, 100.4))
    assert clip.frame_times[0] == pytest.approx(clip.start_s)


def test_clips_carry_their_identity_and_times_rise_within_a_loop(monkeypatch, tmp_path):
    monkeypatch.setattr(clip_handling, "write_video", lambda frames, path, fps=60: None)
    monkeypatch.setattr(clip_handling, "encode_jpeg", lambda frame: repr(frame).encode())  # frames are tuples here
    monkeypatch.setattr(clip_handling, "MAX_PENDING_CLIPS", 1000)  # the fake clock outruns any writer
    hub, _clock, _caps = fake_hub(lambda _spec: ScriptCap([i * 100 for i in range(17)], 10.0))  # 1.7 s
    recorder = ClipRecorder(hub, buffer_seconds=0.5, fps=10)
    recorder.out_dir = str(tmp_path)
    clips: list = []
    done = threading.Event()

    def on_flush(clip):
        clips.append(clip)
        if len(clips) == 8:
            done.set()

    recorder.start(on_flush)

    def switch_in_the_third_pass(_sink, segment, _t):
        if segment.spec.source_epoch == 0 and segment.loop == 2:
            hub.select("sample:2")

    sink = Collector(
        hub, lambda c: any(s.spec.source_epoch == 1 and s.loop == 1 for s, *_ in c.clips),
        on_feed=switch_in_the_third_pass,
    )
    hub.start()
    try:
        assert sink.done.wait(10) and done.wait(10)
    finally:
        hub.stop()
        recorder.stop()

    assert [c.clip_id for c in clips] == list(range(len(clips)))
    runs: dict = {}
    for clip in clips:
        runs.setdefault((clip.source_epoch, clip.source_key, clip.loop), []).append(clip)
    assert list(runs)[:3] == [(0, "sample:1", 0), (0, "sample:1", 1), (1, "sample:2", 0)]
    for (_epoch, key, _loop), run_clips in runs.items():
        assert [c.start_s for c in run_clips] == pytest.approx([0.0, 0.5, 1.0][: len(run_clips)])
        assert [c.end_s for c in run_clips] == pytest.approx([c.start_s + 0.5 for c in run_clips])
        assert {c.label for c in run_clips} == {make_spec(key, 0).label}
    assert all(Path(c.video_path).parent == tmp_path for c in clips)


def test_webcam_clip_times_are_wall_clock(monkeypatch, tmp_path):
    recorder = _recorder(monkeypatch, tmp_path)
    clips: list = []
    done = threading.Event()
    recorder.start(lambda clip: (clips.append(clip), done.set()))
    segment = Segment(make_spec("webcam", 3), 0, None, 1_700_000_000.0)
    recorder._sampler.feed(segment, 0.0, BLACK)
    recorder._sampler.advance(segment, 0.5)
    assert done.wait(5)
    recorder.stop()
    (clip,) = clips
    assert (clip.source_key, clip.source_epoch, clip.label, clip.loop) == ("webcam", 3, "Webcam", None)
    assert (clip.start_s, clip.end_s) == (1_700_000_000.0, 1_700_000_000.5)


def test_makedir_nested_and_idempotent(tmp_path):
    from utils.file_management import makedir

    target = str(tmp_path / "a" / "b")
    makedir(target)
    makedir(target)
    assert Path(target).is_dir()
