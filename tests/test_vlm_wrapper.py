import importlib
import subprocess
import sys
import types
from typing import cast

import numpy as np
import pytest

import main
from utils import vlm_wrapper
from video_handling.write_video import write_video
from prompting.prompting import Prompt
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
    vlm.prompt_obj = cast(Prompt, FakePrompt())
    vlm.model = FakeModel()
    vlm.processor = processor
    return vlm


@pytest.fixture(autouse=True)
def clip_seconds(request, monkeypatch):
    if "real_clip" not in request.keywords:
        monkeypatch.setattr(vlm_wrapper, "_clip_seconds", lambda _path: 0.5)


def write_clip(path, frames):
    write_video([np.zeros((64, 64, 3), dtype=np.uint8)] * frames, str(path), fps=10)
    return str(path)


@pytest.mark.real_clip
def test_clip_seconds_reads_the_recorded_length(tmp_path):
    assert vlm_wrapper._clip_seconds(write_clip(tmp_path / "c.mp4", 5)) == pytest.approx(0.5)


@pytest.mark.real_clip
def test_clip_too_long_for_decord_branch_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="shorter than"):
        vlm_wrapper._clip_seconds(write_clip(tmp_path / "c.mp4", 10))


@pytest.mark.real_clip
def test_unreadable_clip_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="Cannot read"):
        vlm_wrapper._clip_seconds(str(tmp_path / "missing.mp4"))


def test_app_clip_length_selects_the_decord_branch():
    assert main.SEC < vlm_wrapper.DECORD_MAX_SECONDS


def test_get_conv_structure():
    conv = make_vlm(None).get_conv("clip.mp4", 10, 20, "p")
    assert conv[0] == {"role": "system", "content": "sys"}
    video, text = conv[1]["content"]
    assert video["video"] == {
        "video_path": "clip.mp4", "start_time": 0.0, "end_time": 0.5, "fps": 10, "max_frames": 20,
    }
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


SNAPSHOT_DIR = "/cache/snapshots/pinned"


@pytest.fixture
def from_pretrained_calls(fake_torch, monkeypatch):
    calls: dict[str, dict] = {}

    def recorder(name):
        def from_pretrained(path, **kwargs):
            calls[name] = {"path": path, **kwargs}
            return object()
        return from_pretrained

    def snapshot_download(repo_id, **kwargs):
        calls["snapshot"] = {"repo_id": repo_id, **kwargs}
        return SNAPSHOT_DIR

    stub = types.SimpleNamespace(
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=recorder("model")),
        AutoProcessor=types.SimpleNamespace(from_pretrained=recorder("processor")),
        image_utils=types.SimpleNamespace(),
        video_utils=types.SimpleNamespace(VideoInput=object()),
    )
    monkeypatch.setitem(sys.modules, "transformers", stub)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    return calls


def test_model_revision_is_pinned(from_pretrained_calls):
    VLM(FakePrompt())
    assert from_pretrained_calls["snapshot"] == {
        "repo_id": vlm_wrapper.MODEL_PATH,
        "revision": vlm_wrapper.MODEL_REVISION,
    }
    assert len(vlm_wrapper.MODEL_REVISION) == 40


def test_loaders_only_see_the_pinned_snapshot_dir(from_pretrained_calls):
    # With a repo id, transformers 4.57.6 looks some files up on the main branch even when revision= is given.
    VLM(FakePrompt())
    for name in ("model", "processor"):
        assert from_pretrained_calls[name]["path"] == SNAPSHOT_DIR, name
        assert from_pretrained_calls[name]["trust_remote_code"] is True, name


def test_remote_code_gets_the_moved_video_input_name(from_pretrained_calls):
    VLM(FakePrompt())
    transformers = sys.modules["transformers"]
    assert transformers.image_utils.VideoInput is transformers.video_utils.VideoInput


def test_compat_alias_keeps_an_existing_name(from_pretrained_calls):
    original = object()
    sys.modules["transformers"].image_utils.VideoInput = original
    VLM(FakePrompt())
    assert sys.modules["transformers"].image_utils.VideoInput is original


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
