import threading
from concurrent.futures import ThreadPoolExecutor

from main import make_clip_handler, process_clip


class FakeVLM:
    def __init__(self, fail=False):
        self.fail = fail
        self.temps = []

    def forward(self, video_path, fps, max_frames, temperature):
        self.temps.append(temperature)
        if self.fail:
            raise RuntimeError("boom")
        return "a car", temperature


def run(vlm, temp=0.5, clip="c.mp4", fps=10):
    sent = []
    process_clip(str(clip), vlm, sent.append, get_temperature=lambda: temp, get_fps=lambda: fps, max_frames=20)
    return sent


def test_ok_message():
    (msg,) = run(FakeVLM())
    assert msg["type"] == "vlm"
    assert msg["text"] == "a car"
    assert msg["delay"] >= 0


def test_error_message():
    (msg,) = run(FakeVLM(fail=True))
    assert msg == {"type": "error", "text": "Failed to process: Internal error (RuntimeError)"}


def test_temperature_getter_is_used():
    vlm = FakeVLM()
    run(vlm, temp=0.7)
    assert vlm.temps == [0.7]


def test_resample_fps_is_passed_to_forward():
    seen = []

    class V(FakeVLM):
        def forward(self, video_path, fps, max_frames, temperature):
            seen.append(fps)
            return super().forward(video_path, fps, max_frames, temperature)

    run(V(), fps=3)
    assert seen == [3]


def test_clip_deleted_after_ok_and_error(tmp_path):
    for fail in (False, True):
        clip = tmp_path / f"c{fail}.mp4"
        clip.write_bytes(b"x")
        run(FakeVLM(fail=fail), clip=clip)
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

    handler(str(a))
    handler(str(b))  # first still running -> dropped and deleted
    assert not b.exists()
    gate.set()
    assert finished.acquire(timeout=5)
    executor.shutdown(wait=True)  # worker released the busy lock in its finally

    executor = ThreadPoolExecutor(max_workers=1)
    handler = make_clip_handler(process, executor, threading.Lock())
    handler(str(c))
    assert finished.acquire(timeout=5)
    executor.shutdown(wait=True)
    assert calls == [str(a), str(c)]


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
        handler(str(f))
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


def test_signal_during_backend_load_exits_cleanly(monkeypatch):
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

    def slow_load(_prompt):
        events.append("load")
        handlers[signal.SIGINT](signal.SIGINT, None)  # Ctrl+C while the model "loads"
        return object()

    server = Server()
    monkeypatch.setattr(app_main, "configure_logging", lambda: None)  # would replace global hooks
    monkeypatch.setattr(app_main, "route_stdlib_logging", lambda: None)
    monkeypatch.setattr(app_main, "start_server_threaded", lambda *_: server)
    monkeypatch.setattr(app_main, "CaptureHub", Hub)
    monkeypatch.setattr(app_main, "ClipRecorder", Recorder)
    monkeypatch.setattr(app_main, "make_backend", slow_load)

    app_main.main()

    assert events == ["capture start", "load", "stop", "capture stop"]
    assert server.should_exit is True
