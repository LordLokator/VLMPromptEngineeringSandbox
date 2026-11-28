"""Cost estimates from the vendored price snapshot."""
from types import SimpleNamespace as NS

import pytest

from utils.providers import GenerateRequest
from utils.providers.anthropic_provider import AnthropicProvider
from utils.providers.openai_compat import OpenAICompatProvider
from utils.providers.pricing import estimate, load_prices, price_date

PRICES = load_prices()["models"]
SOURCE = f"list price {price_date()}"


@pytest.fixture(autouse=True)
def no_ollama_env(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)


def cost(provider="openai", model="gpt-4o", **kwargs):
    args = dict(uncached_input=1000, output=100)
    return estimate(provider, model, **{**args, **kwargs})


def test_snapshot_names_its_source_licence_and_date():
    data = load_prices()
    assert data["_source"].startswith("https://github.com/BerriAI/litellm/blob/")
    assert "MIT License" in data["_license"] and "Permission is hereby granted" in data["_license"]
    assert data["_date"] == "2026-11-28"
    assert PRICES["openai"] and PRICES["anthropic"]


def test_exact_match():
    entry = PRICES["openai"]["gpt-4o"]
    usd, source = cost()
    assert usd == pytest.approx((1000 * entry["input"] + 100 * entry["output"]) / 1e6)
    assert source == SOURCE


@pytest.mark.parametrize("model", ["gpt-4o-2099-01-01", "GPT-4O", "gpt-4o ", "openai/gpt-4o", "nope"])
def test_unknown_model_is_null_not_zero(model):
    assert cost(model=model) == (None, None)


def test_missing_tokens_are_null():
    assert cost(uncached_input=None) == (None, None)
    assert cost(output=None) == (None, None)


def test_cached_tokens_use_the_cache_rate():
    entry = PRICES["openai"]["gpt-4o"]
    usd, _ = cost(uncached_input=488, cache_read=512)
    assert usd == pytest.approx((488 * entry["input"] + 512 * entry["cache_read"] + 100 * entry["output"]) / 1e6)


def test_cache_tokens_without_a_rate_are_null():
    no_read = next(m for m, e in PRICES["openai"].items() if e["cache_read"] is None)
    no_write = next(m for m, e in PRICES["openai"].items() if e["cache_write"] is None)
    assert cost(model=no_read, cache_read=10) == (None, None)
    assert cost(model=no_write, cache_write=10) == (None, None)
    assert cost(model=no_read)[0] is not None  # no cache tokens: priced


def test_anthropic_cache_writes_and_reads():
    entry = PRICES["anthropic"]["claude-sonnet-4-6"]
    usd, source = estimate("anthropic", "claude-sonnet-4-6", uncached_input=100, cache_read=300, cache_write=50,
                           output=20)
    expected = 100 * entry["input"] + 300 * entry["cache_read"] + 50 * entry["cache_write"] + 20 * entry["output"]
    assert usd == pytest.approx(expected / 1e6) and source == SOURCE


def test_input_above_the_long_context_limit_is_null():
    provider, model, limit = next(
        (p, m, e["priced_up_to_input_tokens"]) for p in PRICES for m, e in PRICES[p].items()
        if e["priced_up_to_input_tokens"]
    )
    assert estimate(provider, model, uncached_input=limit, output=1)[0] is not None
    assert estimate(provider, model, uncached_input=limit - 10, cache_read=11, output=1) == (None, None)


def test_free_versus_na_for_ollama_and_compat(monkeypatch):
    assert estimate("ollama", "llava:13b", uncached_input=1, output=1) == (0.0, "local")
    assert estimate("ollama", "llava:13b", base_url="http://127.0.0.1:11434/v1", uncached_input=1, output=1) == (
        0.0, "local")
    assert estimate("ollama", "gpt-oss:120b-cloud", uncached_input=1, output=1) == (None, None)
    assert estimate("ollama", "llava:13b", base_url="http://gpu-box:11434/v1", uncached_input=1, output=1) == (
        None, None)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://10.1.2.3:11434")
    assert estimate("ollama", "llava:13b", uncached_input=1, output=1) == (None, None)
    assert estimate("compat", "gpt-4o", uncached_input=1, output=1) == (None, None)  # same id, other server: unpriced


@pytest.mark.parametrize("reported, expected", [
    (0.00042, (0.00042, "api")), (0, (0.0, "api")), (float("nan"), (None, None)), (-1, (None, None)),
    (True, (None, None)), ("0.1", (None, None)),
])
def test_a_cost_the_api_reports_wins(reported, expected):
    assert estimate("compat", "m", uncached_input=1, output=1, reported_cost=reported) == expected


REQUEST = GenerateRequest(frames=(b"x",), frame_times=(0.0,), system="s", prompt="p", temperature=None)


def test_openai_adapter_fills_cost():
    usage = NS(prompt_tokens=1000, completion_tokens=100, prompt_tokens_details=NS(cached_tokens=0))
    answer = NS(model="gpt-4o-2024-08-06", choices=[NS(message=NS(content="x", refusal=None), finish_reason="stop")],
                usage=usage)
    client = NS(chat=NS(completions=NS(create=lambda **_: answer)))
    result = OpenAICompatProvider("openai", "gpt-4o", base_url="https://api.openai.com/v1", client=client).generate(
        REQUEST)
    assert (result.cost_usd, result.cost_source) == cost()


def test_compat_adapter_prefers_the_reported_cost():
    usage = NS(prompt_tokens=10, completion_tokens=2, cost=0.0123)
    answer = NS(model="m", choices=[NS(message=NS(content="x", refusal=None), finish_reason="stop")], usage=usage)
    client = NS(chat=NS(completions=NS(create=lambda **_: answer)))
    result = OpenAICompatProvider("compat", "m", base_url="https://x/v1", client=client).generate(REQUEST)
    assert (result.cost_usd, result.cost_source) == (0.0123, "api")


def test_anthropic_adapter_fills_cost():
    usage = NS(input_tokens=100, output_tokens=20, cache_read_input_tokens=300, cache_creation_input_tokens=50)
    answer = NS(content=[NS(type="text", text="x")], stop_reason="end_turn", usage=usage, model="m")
    client = NS(messages=NS(create=lambda **_: answer))
    result = AnthropicProvider("claude-sonnet-4-6", client=client).generate(REQUEST)
    assert (result.cost_usd, result.cost_source) == estimate(
        "anthropic", "claude-sonnet-4-6", uncached_input=100, cache_read=300, cache_write=50, output=20)
    assert result.input_tokens == 450
