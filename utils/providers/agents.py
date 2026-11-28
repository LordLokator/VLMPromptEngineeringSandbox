"""Which agent the app runs, from the environment (VLM_AGENT_A; VLM_ENABLE_HF=1 allows the local model).

An empty variable counts as unset. Only `demo` and `hf` run in the app so far: the API providers become selectable
once spend control is in place.
"""
import os
from dataclasses import dataclass

from utils.providers import PROVIDERS, Provider

ENABLE_HF_ENV = "VLM_ENABLE_HF"
REMOVED_BACKEND_ENV = "VLM_BACKEND"
APP_PROVIDERS = ("demo", "hf")


class AgentConfigError(ValueError):
    """The agent settings cannot start the app; the message says what to change."""


@dataclass(frozen=True)
class AgentSpec:
    provider: str
    model: str | None  # None: the provider's default

    @property
    def text(self) -> str:
        return self.provider if self.model is None else f"{self.provider}:{self.model}"


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def parse_agent(value: str) -> AgentSpec:
    """`provider` or `provider:model`, split on the first colon only (Ollama model names contain one)."""
    provider, sep, model = value.strip().partition(":")
    provider = provider.strip().lower()
    model = model.strip()
    if provider not in PROVIDERS:
        raise AgentConfigError(f"Unknown provider {provider!r}; one of: {', '.join(PROVIDERS)}")
    if sep and not model:
        raise AgentConfigError(f"No model after {provider!r}: use {provider} or {provider}:<model>")
    return AgentSpec(provider, model or None)


def agent_spec_from_env(slot: str = "A") -> AgentSpec:
    """Check the settings for one agent slot; raises AgentConfigError with a message for the user."""
    if _env(REMOVED_BACKEND_ENV):
        raise AgentConfigError(
            f"{REMOVED_BACKEND_ENV} was removed. Use VLM_AGENT_A=demo (the default) or VLM_AGENT_A=hf with "
            f"{ENABLE_HF_ENV}=1 for the local model."
        )
    name = f"VLM_AGENT_{slot}"
    spec = parse_agent(_env(name) or "demo")
    if spec.provider not in APP_PROVIDERS:
        raise AgentConfigError(
            f"{name}={spec.text}: {spec.provider} is not available in the app yet (only {', '.join(APP_PROVIDERS)})."
        )
    if spec.provider == "hf" and _env(ENABLE_HF_ENV) != "1":
        raise AgentConfigError(
            f"{name}=hf downloads and loads a multi-GB model on a CUDA GPU: set {ENABLE_HF_ENV}=1 to allow it."
        )
    from utils.providers.demo import DEMO_MODELS
    from utils.providers.hf import HF_MODEL

    known = {"demo": DEMO_MODELS, "hf": (HF_MODEL,)}[spec.provider]
    if spec.model is not None and spec.model not in known:
        raise AgentConfigError(f"{name}={spec.text}: unknown model; {spec.provider} has {', '.join(known)}")
    return spec


def build_agent(spec: AgentSpec) -> Provider:
    """Build the agent. For hf this loads the model synchronously (slow, needs a CUDA GPU)."""
    if spec.provider == "demo":
        from utils.providers.demo import DemoProvider

        return DemoProvider() if spec.model is None else DemoProvider(spec.model)
    if spec.provider == "hf":
        from utils.providers.hf import HFProvider

        return HFProvider() if spec.model is None else HFProvider(spec.model)
    raise AgentConfigError(f"{spec.provider} is not available in the app yet")
