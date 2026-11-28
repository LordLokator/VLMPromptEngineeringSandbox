"""main.py: the round request, shutdown order and a signal during startup."""
import time

from main import build_request, shutdown
from video_handling.clip_handling import ClipSample


def test_request_carries_the_round_settings_and_the_clip():
    frames, times = (b"jpeg-1", b"jpeg-2"), (1.0, 1.1)
    clip = ClipSample(0, "sample:1", 0, "Sample 1", 1.0, 1.2, 0, "c.mp4", frames, times)
    request = build_request(clip, prompt="count the cars", temperature=0.3, fps=3, max_frames=20)
    assert (request.prompt, request.temperature, request.fps, request.max_frames) == ("count the cars", 0.3, 3, 20)
    assert (request.frames, request.frame_times, request.video_path) == (frames[1:], times[1:], "c.mp4")  # stride 2
    assert request.system


def test_two_agents_get_byte_identical_frames(make_agent):
    import cv2
    import numpy as np

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


class Stoppable:
    def __init__(self, order, name, result=True):
        self.order, self.name, self.result = order, name, result

    def stop(self, *_args):
        self.order.append(self.name)
        return self.result


class Server:
    should_exit = False


def test_shutdown_order_and_speed():
    order = []
    server = Server()
    t0 = time.monotonic()
    shutdown(Stoppable(order, "hub"), Stoppable(order, "recorder"), Stoppable(order, "session"), server)
    assert order == ["recorder", "session", "hub"]
    assert server.should_exit is True
    assert time.monotonic() - t0 < 2


def test_shutdown_gives_up_on_a_stuck_round():
    server = Server()
    shutdown(Stoppable([], "hub"), Stoppable([], "recorder"), Stoppable([], "session", result=False), server,
             timeout=0.1)
    assert server.should_exit is True


def test_signal_during_agent_load_exits_cleanly(monkeypatch):
    """Handlers are installed before the slow load; a signal then skips the recorder and still shuts down.

    The capture is already running at that point: it does not wait for an agent.
    """
    import signal

    import main as app_main

    handlers = {}
    monkeypatch.setattr(app_main.signal, "signal", lambda sig, fn: handlers.__setitem__(sig, fn))
    events = []

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

    class Registry:
        def __init__(self, **_):
            pass

        def check_spec(self, _spec):
            pass

        def configure_from_env(self):
            events.append("load")
            handlers[signal.SIGINT](signal.SIGINT, None)  # Ctrl+C while the model "loads"
            return []

    class Session:
        def __init__(self, **_):
            pass

        def broadcast_state(self):
            pass

        def stop(self, _timeout):
            events.append("session stop")
            return True

    server = Server()
    monkeypatch.setattr(app_main, "configure_logging", lambda: None)  # would replace global hooks
    monkeypatch.setattr(app_main, "load_env_file", lambda: False)  # never a developer's .env in tests
    monkeypatch.setattr(app_main, "route_stdlib_logging", lambda: None)
    monkeypatch.setattr(app_main, "start_server_threaded", lambda *_: server)
    monkeypatch.setattr(app_main, "CaptureHub", Hub)
    monkeypatch.setattr(app_main, "ClipRecorder", Recorder)
    monkeypatch.setattr(app_main, "AgentRegistry", Registry)
    monkeypatch.setattr(app_main, "CompareSession", Session)
    for name in ("VLM_AGENT_A", "VLM_AGENT_B", "VLM_BACKEND"):
        monkeypatch.delenv(name, raising=False)

    app_main.main()

    assert events == ["capture start", "load", "stop", "session stop", "capture stop"]
    assert server.should_exit is True
