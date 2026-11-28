"""Agent registry: availability and its reasons, env parsing, model ids, listing, HF gating, slot switching."""
import threading

import pytest

from utils.providers import GenerateRequest, ModelInfo
from utils.providers.agents import AgentConfigError
from utils.providers.registry import PAID_BLOCKED, AgentRegistry

ENV = (
    "VLM_AGENT_A", "VLM_AGENT_B", "VLM_BACKEND", "VLM_ENABLE_HF", "VLM_ALLOW_REMOTE_PAID", "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY", "OLLAMA_BASE_URL", "VLM_COMPAT_BASE_URL", "VLM_COMPAT_API_KEY",
)
SENTINEL = "sk-" + "sentinel" * 4  # built at runtime: no secret-shaped literal in the repo


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fakes(make_agent):
    """A build function that returns fake agents and records every build."""
    built = []

    def build(spec, **kwargs):
        if spec.provider == "hf" and getattr(build, "hf_error", None):
            raise build.hf_error
        agent = make_agent(spec.provider, spec.model or "default", **getattr(build, "options", {}), **kwargs)
        built.append((spec, agent))
        return agent

    build.built = built
    return build


def request():
    return GenerateRequest(frames=(b"x",), frame_times=(0.0,), system="s", prompt="p", temperature=0.2)


def test_defaults_are_two_different_demo_agents():
    registry = AgentRegistry()
    a, b = registry.configure_from_env()
    assert (a.agent.label, b.agent.label) == ("demo:frame-facts", "demo:frame-motion")
    assert a.enabled and b.enabled and a.local and b.local
    assert not registry.any_remote()


@pytest.mark.parametrize("provider, variable", [
    ("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY"), ("compat", "VLM_COMPAT_BASE_URL"),
])
def test_missing_key_or_url_makes_a_provider_unavailable_with_its_reason(fakes, monkeypatch, provider, variable):
    registry = AgentRegistry(build=fakes)
    state = registry.availability(provider)
    assert not state.available and state.reason == f"{variable} is not set"
    with pytest.raises(AgentConfigError, match=variable):
        registry.set_agent("A", provider, "some-model")
    assert fakes.built == []
    monkeypatch.setenv(variable, "http://127.0.0.1:9" if provider == "compat" else SENTINEL)
    assert registry.availability(provider).available


def test_reasons_never_quote_a_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SENTINEL)
    registry = AgentRegistry(bind_loopback=False)
    assert SENTINEL not in repr(registry.providers())


def test_demo_and_ollama_are_available_by_default():
    registry = AgentRegistry()
    states = {s.provider: s for s in registry.providers()}
    assert states["demo"].available and states["ollama"].available
    assert not states["hf"].available and "VLM_ENABLE_HF=1" in states["hf"].reason


def test_non_loopback_bind_blocks_paid_providers(fakes, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SENTINEL)
    registry = AgentRegistry(bind_loopback=False, build=fakes)
    assert registry.availability("openai").reason == PAID_BLOCKED
    with pytest.raises(AgentConfigError, match="VLM_ALLOW_REMOTE_PAID=1"):
        registry.set_agent("A", "openai", "gpt-x")
    # Local agents stay usable: demo, and Ollama on loopback with a model that runs there.
    assert registry.set_agent("A", "ollama", "llava:13b").local
    with pytest.raises(AgentConfigError, match="VLM_ALLOW_REMOTE_PAID=1"):
        registry.set_agent("B", "ollama", "gpt-oss:120b-cloud")  # runs on Ollama's servers
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://10.0.0.5:11434")
    assert registry.availability("ollama").reason == PAID_BLOCKED
    monkeypatch.setenv("VLM_ALLOW_REMOTE_PAID", "1")
    assert registry.set_agent("B", "openai", "gpt-x").remote


def test_loopback_bind_does_not_block(fakes, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SENTINEL)
    registry = AgentRegistry(build=fakes)
    slot = registry.set_agent("A", "openai", "gpt-x")
    assert slot.remote and registry.any_remote()
    registry.set_agent("A", "openai", "gpt-x", enabled=False)
    assert not registry.any_remote()  # a disabled slot never spends


def test_env_parsing_splits_ollama_names_on_the_first_colon(fakes, monkeypatch):
    monkeypatch.setenv("VLM_AGENT_B", "ollama:llava:13b")
    _a, b = AgentRegistry(build=fakes).configure_from_env()
    assert (b.spec.provider, b.spec.model, b.local) == ("ollama", "llava:13b", True)


def test_env_errors_name_the_variable(monkeypatch):
    monkeypatch.setenv("VLM_AGENT_A", "openai:gpt-x")
    with pytest.raises(AgentConfigError, match="VLM_AGENT_A=openai:gpt-x: OPENAI_API_KEY is not set"):
        AgentRegistry().configure_from_env()
    monkeypatch.setenv("VLM_AGENT_A", "nope")
    with pytest.raises(AgentConfigError, match="Unknown provider"):
        AgentRegistry().configure_from_env()
    monkeypatch.setenv("VLM_AGENT_A", "demo")
    monkeypatch.setenv("VLM_BACKEND", "hf")
    with pytest.raises(AgentConfigError, match="was removed"):
        AgentRegistry().configure_from_env()


@pytest.mark.parametrize("slot, provider, model", [
    ("A", "ollama", "bad model!"),
    ("A", "ollama", "x" * 201),
    ("A", "ollama", "<script>"),
    ("C", "demo", None),
    ("A", "nope", "m"),
    ("A", "demo", "gpt-9"),  # demo has only its own two models
])
def test_set_agent_validates_input(fakes, slot, provider, model):
    with pytest.raises(AgentConfigError):
        AgentRegistry().set_agent(slot, provider, model)


def test_listing_is_cached_until_refresh(fakes, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SENTINEL)
    fakes.options = {"models": [ModelInfo("gpt-x", vision=None)]}
    registry = AgentRegistry(build=fakes)
    first = registry.list_models("openai")
    assert first.models == (ModelInfo("gpt-x", vision=None),) and first.error is None
    assert registry.list_models("openai") is first
    assert len(fakes.built) == 1 and fakes.built[0][0].model == "list"  # placeholder id, never called
    registry.list_models("openai", refresh=True)
    assert len(fakes.built) == 2


def test_listing_failure_gives_a_reason_and_is_retried(fakes, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SENTINEL)
    fakes.options = {"list_error": ConnectionError(f"refused for key {SENTINEL}")}
    registry = AgentRegistry(build=fakes)
    listing = registry.list_models("openai")
    assert listing.models == () and listing.error == "Could not list models: Connection failed (ConnectionError)"
    assert SENTINEL not in listing.error
    registry.list_models("openai")
    assert len(fakes.built) == 2  # failures are not cached


def test_listing_an_unavailable_provider_calls_nothing(fakes):
    listing = AgentRegistry(build=fakes).list_models("anthropic")
    assert listing.error == "ANTHROPIC_API_KEY is not set" and fakes.built == []


def test_demo_listing_is_local():
    assert [m.id for m in AgentRegistry().list_models("demo").models] == ["frame-facts", "frame-motion"]


def test_hf_is_not_loaded_without_the_opt_in(fakes):
    registry = AgentRegistry(build=fakes)
    assert registry.load_hf() is False and fakes.built == []
    with pytest.raises(AgentConfigError, match="VLM_ENABLE_HF=1"):
        registry.set_agent("A", "hf")


def test_hf_failed_load_is_unavailable_and_slots_fall_back(fakes, monkeypatch):
    monkeypatch.setenv("VLM_ENABLE_HF", "1")
    monkeypatch.setenv("VLM_AGENT_A", "hf")
    fakes.hf_error = RuntimeError(f"CUDA out of memory near {SENTINEL}")
    registry = AgentRegistry(build=fakes)
    a, _b = registry.configure_from_env()
    assert a.spec.text == "demo"  # slot A's default
    state = registry.availability("hf")
    assert not state.available
    assert state.reason == "The local model failed to load: Internal error (RuntimeError)"
    assert registry.load_hf() is False  # loaded at most once, failure included


def test_hf_loads_once_and_serves_one_slot(fakes, monkeypatch):
    monkeypatch.setenv("VLM_ENABLE_HF", "1")
    monkeypatch.setenv("VLM_AGENT_A", "hf")
    registry = AgentRegistry(build=fakes)
    a, b = registry.configure_from_env()
    assert a.spec.provider == "hf" and a.local
    with pytest.raises(AgentConfigError, match="only one slot"):
        registry.set_agent("B", "hf")
    registry.set_agent("B", "hf", enabled=False)  # a disabled slot may keep it selected
    registry.set_agent("A", "demo")
    assert registry.set_agent("B", "hf").agent is a.agent  # the same loaded model
    assert [spec.provider for spec, _agent in fakes.built].count("hf") == 1
    assert registry.list_models("hf").error is None  # listed by the loaded model, nothing built
    assert [spec.provider for spec, _agent in fakes.built].count("hf") == 1


def test_a_switch_during_a_call_affects_only_the_next_round(make_agent):
    old = make_agent("demo", "frame-facts", hang_on={1}, replies=["old answer"])
    new = make_agent("demo", "frame-motion", replies=["new answer"])
    agents = iter([old, new])
    registry = AgentRegistry(build=lambda _spec: next(agents))
    registry.set_agent("A", "demo")
    round_one = registry.snapshot()
    answers = []
    call = threading.Thread(target=lambda: answers.append(round_one[0].agent.generate(request()).text))
    call.start()
    assert old.started.wait(5)

    registry.set_agent("A", "demo", "frame-motion")  # while the call hangs
    assert round_one[0].agent is old  # the running round keeps its agent
    old.release.set()
    call.join(5)
    assert answers == ["old answer"]
    (slot,) = registry.snapshot()
    assert slot.agent is new and slot.agent.generate(request()).text == "new answer"
    assert new.calls == 1 and old.calls == 1
