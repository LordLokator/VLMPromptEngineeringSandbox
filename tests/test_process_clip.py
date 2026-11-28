import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from main import make_clip_handler, process_clip
from video_handling.clip_handling import ClipSample
from video_handling.streaming import CaptureHub


def sample(path="c.mp4", epoch=0):
    return ClipSample(0, "sample:1", epoch, "Sample 1", 0.0, 0.5, 0, str(path))


def run(vlm, temp=0.5, clip="c.mp4", fps=10, epoch=lambda: 0, prompt="what happens?"):
    sent = []
    process_clip(
        sample(clip), vlm, sent.append, get_prompt=lambda: prompt,
        get_temperature=lambda: temp, get_fps=lambda: fps, get_source_epoch=epoch, max_frames=20,
    )
    return sent


def test_ok_message(make_agent):
    (msg,) = run(make_agent(replies=["a car"]))
    assert msg["type"] == "vlm"
    assert msg["text"] == "a car"
    assert msg["delay"] >= 0


@pytest.mark.parametrize("fail", [False, True], ids=["answer", "error"])
def test_a_slow_result_after_a_switch_is_dropped(tmp_path, make_agent, fail):
    hub = CaptureHub()  # never started
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"x")
    # The source changes while the agent is "thinking" (its delay).
    failure = {1: RuntimeError("boom")} if fail else None
    agent = make_agent(delay=1, sleep=lambda _s: hub.select("sample:2"), raise_on=failure)
    assert run(agent, clip=clip, epoch=lambda: hub.spec.source_epoch) == []
    assert not clip.exists()


def test_a_clip_from_before_a_switch_and_back_is_still_stale(make_agent):
    hub = CaptureHub()
    hub.select("sample:2")
    hub.select("sample:1")  # the same source again, but a later epoch
    assert run(make_agent(), epoch=lambda: hub.spec.source_epoch) == []


def test_error_message(make_agent):
    (msg,) = run(make_agent(raise_on={1: RuntimeError("boom")}))
    assert msg == {"type": "error", "text": "Failed to process: Internal error (RuntimeError)"}


def test_temperature_getter_is_used(make_agent):
    vlm = make_agent()
    run(vlm, temp=0.7)
    assert [r.temperature for r in vlm.requests] == [0.7]


def test_request_carries_the_round_settings_and_the_clip(make_agent):
    vlm = make_agent()
    frames, times = (b"jpeg-1", b"jpeg-2"), (1.0, 1.1)
    clip = ClipSample(0, "sample:1", 0, "Sample 1", 1.0, 1.2, 0, "c.mp4", frames, times)
    process_clip(
        clip, vlm, lambda _m: None, get_prompt=lambda: "count the cars",
        get_temperature=lambda: 0.3, get_fps=lambda: 3, get_source_epoch=lambda: 0, max_frames=20,
    )
    (request,) = vlm.requests
    assert (request.prompt, request.temperature, request.fps, request.max_frames) == ("count the cars", 0.3, 3, 20)
    assert (request.frames, request.frame_times, request.video_path) == (frames[1:], times[1:], "c.mp4")  # stride 2
    assert request.system


def test_two_agents_get_byte_identical_frames(make_agent):
    import cv2
    import numpy as np

    from main import build_request
    from video_handling.write_video import encode_jpeg

    rng = np.random.default_rng(0)
    jpegs = tuple(encode_jpeg(rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)) for _ in range(5))
    clip = ClipSample(0, "sample:1", 0, "Sample 1", 0.0, 0.5, 0, "c.mp4", jpegs, (0.0, 0.1, 0.2, 0.3, 0.4))
    agents = (make_agent(model="a"), make_agent(model="b"))
    for agent in agents:
        agent.generate(build_request(clip, prompt="p", temperature=0.2, fps=3, max_frames=20))
    a, b = (agent.requests[0] for agent in agents)
    assert a.frames == b.frames and a.frame_times == b.frame_times == (0.1, 0.4)
    assert [cv2.imdecode(np.frombuffer(f, np.uint8), cv2.IMREAD_COLOR).shape[:2] for f in a.frames] == [(432, 768)] * 2


def test_clip_deleted_after_ok_and_error(tmp_path, make_agent):
    for fail in (False, True):
        clip = tmp_path / f"c{fail}.mp4"
        clip.write_bytes(b"x")
        run(make_agent(raise_on={1: RuntimeError("boom")} if fail else None), clip=clip)
        assert not clip.exists()


def test_busy_clip_is_dropped_and_deleted(tmp_path):
    gate = threading.Event()
    finished = threading.Semaphore(0)
    calls = []

    def process(path):
        calls.append(path)
        gate.wait(5)
        finished.release()

    executor = ThreadPoolExecutor(max_workers=1)
    handler = make_clip_handler(process, executor, threading.Lock())
    a, b, c = (tmp_path / n for n in "abc")
    for f in (a, b, c):
        f.write_bytes(b"x")

    handler(sample(a))
    handler(sample(b))  # first still running -> dropped and deleted
    assert not b.exists()
    gate.set()
    assert finished.acquire(timeout=5)
    executor.shutdown(wait=True)  # worker released the busy lock in its finally

    executor = ThreadPoolExecutor(max_workers=1)
    handler = make_clip_handler(process, executor, threading.Lock())
    handler(sample(c))
    assert finished.acquire(timeout=5)
    executor.shutdown(wait=True)
    assert [clip.video_path for clip in calls] == [str(a), str(c)]


def test_drop_notice_is_rate_limited(tmp_path):
    busy = threading.Lock()
    busy.acquire()  # inference "in flight": every clip is dropped
    now = [0.0]
    notices = []
    handler = make_clip_handler(
        lambda p: None, ThreadPoolExecutor(max_workers=1), busy, notify=notices.append, clock=lambda: now[0]
    )
    for t in (0.0, 1.0, 4.9, 5.0):
        now[0] = t
        f = tmp_path / f"c{t}.mp4"
        f.write_bytes(b"x")
        handler(sample(f))
        assert not f.exists()
    assert len(notices) == 2
    assert notices[0]["type"] == "system" and notices[0]["muted"] is True


def test_shutdown_order_and_speed():
    import time

    from main import shutdown

    order = []

    class Rec:
        def stop(self):
            order.append("recorder")

    class Hub:
        def stop(self):
            order.append("hub")

    class Exe:
        def shutdown(self, wait, cancel_futures):
            order.append("executor")

    class Srv:
        should_exit = False

    srv = Srv()
    t0 = time.monotonic()
    shutdown(Hub(), Rec(), Exe(), srv, threading.Lock())
    assert order == ["recorder", "hub", "executor"]
    assert srv.should_exit is True
    assert time.monotonic() - t0 < 2


def test_shutdown_gives_up_on_stuck_inference():
    from main import shutdown

    busy = threading.Lock()
    busy.acquire()

    class Noop:
        def stop(self):
            pass

        def shutdown(self, wait, cancel_futures):
            pass

    class Srv:
        should_exit = False

    srv = Srv()
    shutdown(Noop(), Noop(), Noop(), srv, busy, timeout=0.1)
    assert srv.should_exit is True


def test_signal_during_backend_load_exits_cleanly(monkeypatch, make_agent):
    """Handlers are installed before the slow load; a signal then skips the recorder and still shuts down.

    The capture is already running at that point: it does not wait for a backend.
    """
    import signal
    import main as app_main

    handlers = {}
    monkeypatch.setattr(app_main.signal, "signal", lambda sig, fn: handlers.__setitem__(sig, fn))
    events = []

    class Server:
        should_exit = False

    class Hub:
        def __init__(self, **_):
            pass

        def start(self):
            events.append("capture start")

        def stop(self):
            events.append("capture stop")

    class Recorder:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self, **_):
            events.append("start")

        def stop(self):
            events.append("stop")

    def slow_load(_spec):
        events.append("load")
        handlers[signal.SIGINT](signal.SIGINT, None)  # Ctrl+C while the model "loads"
        return make_agent()

    server = Server()
    monkeypatch.setattr(app_main, "configure_logging", lambda: None)  # would replace global hooks
    monkeypatch.setattr(app_main, "load_env_file", lambda: False)  # never a developer's .env in tests
    monkeypatch.setattr(app_main, "route_stdlib_logging", lambda: None)
    monkeypatch.setattr(app_main, "start_server_threaded", lambda *_: server)
    monkeypatch.setattr(app_main, "CaptureHub", Hub)
    monkeypatch.setattr(app_main, "ClipRecorder", Recorder)
    monkeypatch.setattr(app_main, "build_agent", slow_load)
    monkeypatch.delenv("VLM_AGENT_A", raising=False)
    monkeypatch.delenv("VLM_BACKEND", raising=False)

    app_main.main()

    assert events == ["capture start", "load", "stop", "capture stop"]
    assert server.should_exit is True
