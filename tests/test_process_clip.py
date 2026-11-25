from main import process_clip


class FakeVLM:
    def __init__(self, fail=False):
        self.fail = fail
        self.temps = []

    def forward(self, video_path, fps, max_frames, temperature):
        self.temps.append(temperature)
        if self.fail:
            raise RuntimeError("boom")
        return "a car", temperature


def run(vlm, temp=0.5):
    sent = []
    process_clip("c.mp4", vlm, sent.append, get_temperature=lambda: temp, fps=10, max_frames=20)
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
