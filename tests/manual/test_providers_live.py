"""One real call per provider. Skipped unless VLM_MANUAL_TESTS=1, and per provider unless its settings are there.

**Costs money** for openai, anthropic and paid compat servers: one call each, with up to 5 frames of 768 px and at
most 64 output tokens (cents at list prices). Ollama on this machine is free.

Settings: the provider's usual key or URL (README, Configuration) plus the model to use:
VLM_MANUAL_OPENAI_MODEL, VLM_MANUAL_ANTHROPIC_MODEL, VLM_MANUAL_OLLAMA_MODEL, VLM_MANUAL_COMPAT_MODEL.
Keys are never printed.
"""
import os

import pytest

from utils.providers import SYSTEM_PROMPT, GenerateRequest
from utils.providers.agents import AgentSpec, build_provider
from utils.providers.check import sample_frames
from utils.providers.settings import load_env_file

pytestmark = [
    pytest.mark.manual,
    pytest.mark.skipif(os.environ.get("VLM_MANUAL_TESTS") != "1", reason="manual: set VLM_MANUAL_TESTS=1"),
]
NEEDS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "ollama": None, "compat": "VLM_COMPAT_BASE_URL"}
MAX_OUTPUT_TOKENS = 64


@pytest.mark.parametrize("provider", sorted(NEEDS))
def test_one_real_call(provider):
    load_env_file()
    model = os.environ.get(f"VLM_MANUAL_{provider.upper()}_MODEL", "").strip()
    if not model:
        pytest.skip(f"set VLM_MANUAL_{provider.upper()}_MODEL")
    needed = NEEDS[provider]
    if needed and not os.environ.get(needed, "").strip():
        pytest.skip(f"set {needed}")
    frames, times = sample_frames()
    request = GenerateRequest(frames=frames, frame_times=times, system=SYSTEM_PROMPT,
                              prompt="Describe what happens in these frames in one sentence.", temperature=0.2,
                              max_output_tokens=MAX_OUTPUT_TOKENS)
    agent = build_provider(AgentSpec(provider, model))
    result = agent.generate(request)
    assert result.text.strip() or result.finish_reason, "no text and no finish reason"
    assert result.latency_ms > 0
    if provider in ("openai", "anthropic"):
        assert result.input_tokens and result.output_tokens, "usage missing"
    assert "frames" in result.params_sent
