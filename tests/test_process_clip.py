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
    assert msg["status"] == "OK"
    assert msg["output"] == "a car"
    assert msg["delay"] >= 0


def test_error_message():
    (msg,) = run(FakeVLM(fail=True))
    assert msg["status"] == "ERROR"
    assert "boom" in msg["msg"]


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
