"""Which agent the app runs, from the environment (VLM_AGENT_A; VLM_ENABLE_HF=1 allows the local model).

An empty variable counts as unset. Only `demo` and `hf` run in the app so far: the API providers become selectable
once spend control is in place.
"""
import os
from dataclasses import dataclass

from utils.providers import PROVIDERS, Provider
from utils.providers.settings import ProviderConfigError, check_model_id

ENABLE_HF_ENV = "VLM_ENABLE_HF"
REMOVED_BACKEND_ENV = "VLM_BACKEND"
APP_PROVIDERS = ("demo", "hf")
# Agent B's default is the second demo variant, so two demo slots never show the same text.
DEFAULT_AGENTS = {"A": "demo", "B": "demo:frame-motion"}


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
    if model:
        try:
            check_model_id(model)
        except ProviderConfigError as e:
            raise AgentConfigError(str(e)) from e
    return AgentSpec(provider, model or None)


def agent_spec_from_env(slot: str = "A", providers: tuple[str, ...] = APP_PROVIDERS) -> AgentSpec:
    """Check the settings for one agent slot; raises AgentConfigError with a message for the user.

    `providers` are the ones accepted (the app's list by default; the registry passes all of them).
    """
    if _env(REMOVED_BACKEND_ENV):
        raise AgentConfigError(
            f"{REMOVED_BACKEND_ENV} was removed. Use VLM_AGENT_A=demo (the default) or VLM_AGENT_A=hf with "
            f"{ENABLE_HF_ENV}=1 for the local model."
        )
    name = f"VLM_AGENT_{slot}"
    spec = parse_agent(_env(name) or DEFAULT_AGENTS[slot])
    if spec.provider not in providers:
        raise AgentConfigError(
            f"{name}={spec.text}: {spec.provider} is not available in the app yet (only {', '.join(providers)})."
        )
    if spec.provider == "hf" and _env(ENABLE_HF_ENV) != "1":
        raise AgentConfigError(
            f"{name}=hf downloads and loads a multi-GB model on a CUDA GPU: set {ENABLE_HF_ENV}=1 to allow it."
        )
    from utils.providers.demo import DEMO_MODELS
    from utils.providers.hf import HF_MODEL

    known = {"demo": DEMO_MODELS, "hf": (HF_MODEL,)}.get(spec.provider)
    if known is not None and spec.model is not None and spec.model not in known:
        raise AgentConfigError(f"{name}={spec.text}: unknown model; {spec.provider} has {', '.join(known)}")
    return spec


def build_agent(spec: AgentSpec) -> Provider:
    """Build an agent the app accepts. For hf this loads the model synchronously (slow, needs a CUDA GPU)."""
    if spec.provider not in APP_PROVIDERS:
        raise AgentConfigError(f"{spec.provider} is not available in the app yet")
    return build_provider(spec)


def build_provider(spec: AgentSpec) -> Provider:
    """Any provider, for the check CLI and later the app. Raises ProviderConfigError when a key or URL is missing."""
    if spec.provider in ("openai", "ollama", "compat", "anthropic"):
        if spec.model is None:
            raise ProviderConfigError(f"{spec.provider} needs a model: {spec.provider}:<model>")
        if spec.provider == "anthropic":
            from utils.providers import anthropic_provider

            return anthropic_provider.build(spec.model)
        from utils.providers import openai_compat

        return openai_compat.build(spec.provider, spec.model)
    if spec.provider == "demo":
        from utils.providers.demo import DemoProvider

        return DemoProvider() if spec.model is None else DemoProvider(spec.model)
    if spec.provider == "hf":
        from utils.providers.hf import HFProvider

        return HFProvider() if spec.model is None else HFProvider(spec.model)
    raise AgentConfigError(f"{spec.provider} is not available in the app yet")
