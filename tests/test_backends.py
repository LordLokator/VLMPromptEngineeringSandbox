import subprocess
import sys

import pytest

from utils import backends
from utils.backends import ChainBackend, make_backend


class Fake:
    def __init__(self, name, fail=False):
        self.name = name
        self.fail = fail

    def forward(self, video_path, fps, max_frames, temperature):
        if self.fail:
            raise RuntimeError(f"{self.name} down")
        return f"from {self.name}", temperature


CALL = dict(video_path="c.mp4", fps=10, max_frames=20, temperature=0.5)


def test_import_is_lightweight():
    code = "import utils.backends, sys; sys.exit('torch' in sys.modules or 'litellm' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_default_builds_hf(monkeypatch):
    monkeypatch.delenv("VLM_BACKEND", raising=False)
    from utils import vlm_wrapper

    built = []
    monkeypatch.setattr(vlm_wrapper, "VLM", lambda prompt: built.append(prompt) or "hf-instance")
    assert make_backend("p") == "hf-instance"
    assert built == ["p"]


def test_unknown_backend_raises(monkeypatch):
    monkeypatch.setenv("VLM_BACKEND", "nope")
    with pytest.raises(ValueError, match="nope"):
        make_backend("p")


def test_chain_falls_through_and_reports_name():
    chain = ChainBackend([Fake("a", fail=True), Fake("b")])
    assert chain.forward(**CALL) == ("from b", 0.5)
    assert chain.name == "b"


def test_chain_raises_last_error_when_all_fail():
    chain = ChainBackend([Fake("a", fail=True), Fake("b", fail=True)])
    with pytest.raises(RuntimeError, match="b down"):
        chain.forward(**CALL)


def test_chain_requires_backends():
    with pytest.raises(ValueError):
        ChainBackend([])


def test_module_constants():
    assert backends.DEFAULT_BACKEND == "hf"
