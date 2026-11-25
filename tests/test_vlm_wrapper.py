import importlib
import subprocess
import sys
import types

import pytest

from utils import vlm_wrapper
from utils.vlm_wrapper import VLM


def test_import_does_not_pull_in_torch():
    code = "import utils.vlm_wrapper, sys; sys.exit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


class FakeProcessor:
    def __init__(self, decoded=("ok",), error=None):
        self.decoded = list(decoded)
        self.error = error

    def __call__(self, **_kwargs):
        if self.error:
            raise self.error
        return {"input_ids": [1, 2, 3]}

    def batch_decode(self, _ids, skip_special_tokens=True):
        return self.decoded


class FakeModel:
    def generate(self, **_kwargs):
        return [[1]]


class FakePrompt:
    def get(self):
        return "what is happening?"


def make_vlm(processor):
    vlm = object.__new__(VLM)  # skip __init__: it loads the real model
    vlm.sys_prompt = "sys"
    vlm.prompt_obj = FakePrompt()
    vlm.model = FakeModel()
    vlm.processor = processor
    return vlm


def test_get_conv_structure():
    conv = make_vlm(None).get_conv("clip.mp4", 10, 20, "p")
    assert conv[0] == {"role": "system", "content": "sys"}
    video, text = conv[1]["content"]
    assert video["video"] == {"video_path": "clip.mp4", "fps": 10, "max_frames": 20}
    assert text == {"type": "text", "text": "p"}


def test_forward_returns_str(fake_torch):
    vlm = make_vlm(FakeProcessor(decoded=["  hello "]))
    output, _ = vlm.forward("clip.mp4", 10, 20, temperature=0.1)
    assert output == "hello"


def test_preprocessing_error_is_surfaced(fake_torch):
    vlm = make_vlm(FakeProcessor(error=ValueError("bad clip")))
    with pytest.raises(ValueError, match="bad clip"):
        vlm.forward("clip.mp4", 10, 20)


def test_forward_empty_decode_returns_empty_str(fake_torch):
    vlm = make_vlm(FakeProcessor(decoded=[]))
    output, _ = vlm.forward("clip.mp4", 10, 20)
    assert output == ""


@pytest.fixture
def from_pretrained_calls(fake_torch, monkeypatch):
    calls: dict[str, dict] = {}

    def recorder(name):
        def from_pretrained(path, **kwargs):
            calls[name] = {"path": path, **kwargs}
            return object()
        return from_pretrained

    stub = types.SimpleNamespace(
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=recorder("model")),
        AutoProcessor=types.SimpleNamespace(from_pretrained=recorder("processor")),
    )
    monkeypatch.setitem(sys.modules, "transformers", stub)
    return calls


def test_model_revision_is_pinned(from_pretrained_calls):
    VLM(FakePrompt())
    assert from_pretrained_calls["model"]["revision"] == vlm_wrapper.MODEL_REVISION
    assert from_pretrained_calls["processor"]["revision"] == vlm_wrapper.MODEL_REVISION
    assert len(vlm_wrapper.MODEL_REVISION) == 40


def test_attn_impl_default_reaches_model(from_pretrained_calls):
    VLM(FakePrompt())
    assert from_pretrained_calls["model"]["attn_implementation"] == "flash_attention_2"


def test_attn_impl_env_override(from_pretrained_calls, monkeypatch):
    monkeypatch.setenv("VLM_ATTN_IMPL", "eager")
    try:
        importlib.reload(vlm_wrapper)
        vlm_wrapper.VLM(FakePrompt())
        assert from_pretrained_calls["model"]["attn_implementation"] == "eager"
    finally:
        monkeypatch.delenv("VLM_ATTN_IMPL")
        importlib.reload(vlm_wrapper)
