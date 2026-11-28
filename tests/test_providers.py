"""The provider contract, the demo and HF providers, and agent selection from the environment."""
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from utils.providers import GenerateRequest
from utils.providers.agents import AgentConfigError, AgentSpec, agent_spec_from_env, build_agent, parse_agent
from utils.providers.demo import DemoProvider
from utils.providers.hf import HF_MODEL, HFProvider

AGENT_ENV = ("VLM_AGENT_A", "VLM_BACKEND", "VLM_ENABLE_HF")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in AGENT_ENV:
        monkeypatch.delenv(name, raising=False)


def jpeg(value, size=(48, 64)):
    ok, buffer = cv2.imencode(".jpg", np.full((*size, 3), value, dtype=np.uint8))
    assert ok
    return buffer.tobytes()


def request(**overrides):
    args = dict(frames=(), frame_times=(), system="sys", prompt="what happens?", temperature=0.4)
    return GenerateRequest(**{**args, **overrides})


def test_importing_the_package_loads_no_model_or_sdk():
    code = (
        "import sys, utils.providers, utils.providers.agents, utils.providers.demo, utils.providers.hf; "
        "sys.exit(sorted({'torch', 'transformers', 'openai', 'anthropic', 'cv2'} & set(sys.modules)) or None)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_demo_reports_facts_about_the_frames_it_got():
    result = DemoProvider().generate(request(frames=(jpeg(0), jpeg(255)), frame_times=(2.0, 2.1)))
    assert result.text.startswith("Demo (no model): 2 frames, 64x48, 2.00-2.10 s")
    assert "mean brightness 128/255 (range 0-255)" in result.text
    assert "mean change between frames 255.0/255" in result.text
    assert (result.input_tokens, result.output_tokens, result.cost_usd, result.cost_source) == (None,) * 4
    assert result.requested_model == result.reported_model == "frame-facts"
    assert result.latency_ms >= 0


def test_demo_without_readable_frames_says_so():
    assert "no readable frames" in DemoProvider().generate(request(frames=(b"not a jpeg",))).text
    assert "no readable frames" in DemoProvider().generate(request()).text


def test_demo_label_and_models():
    demo = DemoProvider()
    assert demo.label == "demo:frame-facts"
    assert [m.id for m in demo.list_models()] == ["frame-facts"]
    with pytest.raises(ValueError):
        DemoProvider("gpt-9")


class FakeVLM:
    def __init__(self):
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return "a person walks"


def test_hf_passes_the_prompt_from_the_request():
    vlm = FakeVLM()
    result = HFProvider(vlm=vlm).generate(request(video_path="clip.mp4", fps=3, max_frames=20, max_output_tokens=64))
    assert vlm.calls == [{
        "video_path": "clip.mp4", "fps": 3, "max_frames": 20, "prompt": "what happens?", "system": "sys",
        "temperature": 0.4, "max_new_tokens": 64,
    }]
    assert result.text == "a person walks"
    assert result.params_sent == {"temperature": 0.4, "max_new_tokens": 64, "fps": 3, "max_frames": 20}
    assert (result.input_tokens, result.output_tokens, result.cost_usd) == (None, None, None)


def test_hf_temperature_has_a_floor():
    vlm = FakeVLM()
    HFProvider(vlm=vlm).generate(request(video_path="c.mp4", temperature=None))
    HFProvider(vlm=vlm).generate(request(video_path="c.mp4", temperature=0.0))
    assert [c["temperature"] for c in vlm.calls] == [0.01, 0.01]


def test_hf_needs_the_clip_file_and_its_own_model():
    with pytest.raises(ValueError, match="video_path"):
        HFProvider(vlm=FakeVLM()).generate(request())
    with pytest.raises(ValueError, match="only"):
        HFProvider("other/model", vlm=FakeVLM())
    assert HFProvider(vlm=FakeVLM()).label == f"hf:{HF_MODEL}"


@pytest.mark.parametrize("value, spec", [
    ("demo", AgentSpec("demo", None)),
    ("  Demo:frame-facts ", AgentSpec("demo", "frame-facts")),
    ("ollama:llava:13b", AgentSpec("ollama", "llava:13b")),  # split on the first colon only
])
def test_parse_agent(value, spec):
    assert parse_agent(value) == spec


@pytest.mark.parametrize("value", ["nope", "openai:", ""])
def test_parse_agent_rejects(value):
    with pytest.raises(AgentConfigError):
        parse_agent(value)


@pytest.mark.parametrize("value", [None, "", "   "])
def test_agent_a_defaults_to_demo(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv("VLM_AGENT_A", value)
    assert agent_spec_from_env("A") == AgentSpec("demo", None)


def test_hf_needs_the_opt_in(monkeypatch):
    monkeypatch.setenv("VLM_AGENT_A", "hf")
    with pytest.raises(AgentConfigError, match="VLM_ENABLE_HF=1"):
        agent_spec_from_env("A")
    monkeypatch.setenv("VLM_ENABLE_HF", "1")
    assert agent_spec_from_env("A") == AgentSpec("hf", None)
    monkeypatch.setenv("VLM_AGENT_A", "hf:other/model")
    with pytest.raises(AgentConfigError, match="unknown model"):
        agent_spec_from_env("A")


@pytest.mark.parametrize("provider", ["openai:gpt-x", "anthropic:claude-x", "ollama:llava", "compat:m"])
def test_api_providers_are_not_in_the_app_yet(monkeypatch, provider):
    monkeypatch.setenv("VLM_AGENT_A", provider)
    with pytest.raises(AgentConfigError, match="not available in the app yet"):
        agent_spec_from_env("A")


def test_removed_backend_variable_names_the_new_ones(monkeypatch):
    monkeypatch.setenv("VLM_BACKEND", "hf")
    with pytest.raises(AgentConfigError, match="VLM_AGENT_A.*VLM_ENABLE_HF"):
        agent_spec_from_env("A")


def test_build_agent(monkeypatch):
    from utils import vlm_wrapper

    assert build_agent(AgentSpec("demo", None)).label == "demo:frame-facts"
    built = []
    monkeypatch.setattr(vlm_wrapper, "VLM", lambda: built.append(1) or FakeVLM())
    assert isinstance(build_agent(AgentSpec("hf", None)), HFProvider)
    assert built == [1]  # the model loads when the agent is built


def test_bad_agent_settings_end_the_app_before_anything_starts(monkeypatch):
    import main as app_main

    monkeypatch.setenv("VLM_AGENT_A", "openai:gpt-x")
    monkeypatch.setattr(app_main, "configure_logging", lambda: None)
    monkeypatch.setattr(app_main, "load_env_file", lambda: False)
    started = []
    monkeypatch.setattr(app_main, "start_server_threaded", lambda *_: started.append(1))
    with pytest.raises(SystemExit) as exit_info:
        app_main.main()
    assert exit_info.value.code == 2
    assert started == []


def test_the_app_imports_and_builds_the_default_agent_without_the_gpu_stack():
    env = {k: v for k, v in os.environ.items() if k not in AGENT_ENV}
    script = str(Path(__file__).with_name("import_without_gpu.py"))
    result = subprocess.run([sys.executable, script], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok: demo:frame-facts"


def test_one_cv2_check():
    from tests.one_cv2 import cv2_distributions

    def report(*names):
        return {"install": [{"metadata": {"name": n}} for n in names]}

    assert cv2_distributions(report("numpy", "opencv-python-headless")) == ["opencv-python-headless"]
    assert len(cv2_distributions(report("opencv_python", "opencv-python-headless"))) == 2
    assert cv2_distributions(report("numpy")) == []
