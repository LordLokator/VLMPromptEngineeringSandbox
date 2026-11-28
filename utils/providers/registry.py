"""The two agent slots, which providers can fill them and why not, and the model lists.

Not wired into the app yet (T87). The registry holds immutable `Slot`s: a round takes `snapshot()` once, so changing
an agent while a call is in flight affects only the next round. Every reason and error here is safe to show in the
UI: it names variables and providers, never a key, a URL or an exception text.
"""
import threading
from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger

from utils.errors import log_failure
from utils.providers import PROVIDERS, ModelInfo, Provider
from utils.providers.agents import (
    DEFAULT_AGENTS,
    ENABLE_HF_ENV,
    AgentConfigError,
    AgentSpec,
    agent_spec_from_env,
    build_provider,
    parse_agent,
)
from utils.providers.settings import (
    OLLAMA_DEFAULT_URL,
    ProviderConfigError,
    env,
    is_local,
    is_loopback_url,
)

SLOTS = ("A", "B")
ALLOW_REMOTE_PAID_ENV = "VLM_ALLOW_REMOTE_PAID"
LISTING_MODEL = "list"  # a placeholder model id: building a client for listing needs one, nothing is called with it
PAID_BLOCKED = (
    f"Paid providers are off while the server is reachable from other machines: set {ALLOW_REMOTE_PAID_ENV}=1 "
    "to allow them"
)
_REQUIRED_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "compat": "VLM_COMPAT_BASE_URL"}


@dataclass(frozen=True)
class Availability:
    provider: str
    available: bool
    reason: str | None = None  # why not, for the UI


@dataclass(frozen=True)
class Slot:
    name: str  # "A" or "B"
    spec: AgentSpec
    agent: Provider
    enabled: bool
    remote: bool  # the locality rule: remote means paid, with spend control

    @property
    def local(self) -> bool:
        return not self.remote


@dataclass(frozen=True)
class ModelListing:
    provider: str
    models: tuple[ModelInfo, ...]
    error: str | None = None  # a reason to show; free-text model entry stays possible


def _slot_remote(spec: AgentSpec, agent: Provider) -> bool:
    return not is_local(spec.provider, spec.model or agent.model, getattr(agent, "base_url", None))


class AgentRegistry:
    """Slots A and B. `build` makes a provider from a spec (tests pass fakes); it must not call the network."""

    def __init__(
        self,
        *,
        bind_loopback: bool = True,
        build: Callable[[AgentSpec], Provider] = build_provider,
    ):
        self._build = build
        self._bind_loopback = bind_loopback
        self._lock = threading.Lock()
        self._slots: dict[str, Slot] = {}
        self._listings: dict[str, ModelListing] = {}
        self._hf: Provider | None = None
        self._hf_error: str | None = None

    # -- availability ---------------------------------------------------------------------------------------------

    @property
    def remote_paid_blocked(self) -> bool:
        return not self._bind_loopback and env(ALLOW_REMOTE_PAID_ENV) != "1"

    def availability(self, provider: str) -> Availability:
        if provider not in PROVIDERS:
            return Availability(provider, False, f"Unknown provider {provider!r}")
        if provider == "demo":
            return Availability(provider, True)
        if provider == "hf":
            if env(ENABLE_HF_ENV) != "1":
                return Availability(provider, False, f"The local model needs a CUDA GPU: set {ENABLE_HF_ENV}=1")
            if self._hf_error is not None:
                return Availability(provider, False, f"The local model failed to load: {self._hf_error}")
            if self._hf is None:
                return Availability(provider, False, "The local model is not loaded")
            return Availability(provider, True)
        if provider == "ollama":
            url = env("OLLAMA_BASE_URL") or OLLAMA_DEFAULT_URL
            if self.remote_paid_blocked and not is_loopback_url(url):
                return Availability(provider, False, PAID_BLOCKED)
            return Availability(provider, True)
        variable = _REQUIRED_ENV[provider]
        if env(variable) is None:
            return Availability(provider, False, f"{variable} is not set")
        if self.remote_paid_blocked:
            return Availability(provider, False, PAID_BLOCKED)
        return Availability(provider, True)

    def providers(self) -> list[Availability]:
        return [self.availability(p) for p in PROVIDERS]

    # -- the local model --------------------------------------------------------------------------------------------

    def load_hf(self) -> bool:
        """Load the local model once, synchronously, if VLM_ENABLE_HF=1. A failure marks hf unavailable."""
        if env(ENABLE_HF_ENV) != "1" or self._hf is not None or self._hf_error is not None:
            return self._hf is not None
        try:
            agent = self._build(AgentSpec("hf", None))
        except Exception as e:  # noqa: BLE001 - any load failure (CUDA, download, import) only disables hf
            self._hf_error = log_failure("The local model failed to load", e)
            return False
        self._hf = agent
        return True

    # -- slots ------------------------------------------------------------------------------------------------------

    def set_agent(self, slot: str, provider: str, model: str | None = None, enabled: bool = True) -> Slot:
        """Fill a slot from the next round on. Raises AgentConfigError with a message for the UI."""
        if slot not in SLOTS:
            raise AgentConfigError(f"Unknown slot {slot!r}; one of: {', '.join(SLOTS)}")
        spec = parse_agent(provider if not model else f"{provider}:{model}")
        return self._fill(slot, spec, enabled)

    def configure_from_env(self) -> list[Slot]:
        """Both slots from VLM_AGENT_A / VLM_AGENT_B (defaults: the two demo variants). Raises AgentConfigError.

        A slot set to hf whose load failed falls back to its demo default, so the app still starts and the UI can
        show why.
        """
        self.load_hf()
        slots = []
        for name in SLOTS:
            spec = agent_spec_from_env(name, providers=PROVIDERS)
            if spec.provider == "hf" and self._hf_error is not None:
                spec = parse_agent(DEFAULT_AGENTS[name])
                logger.warning(f"Agent {name}: the local model is unavailable, using {spec.text} instead.")
            try:
                slots.append(self._fill(name, spec, True))
            except AgentConfigError as e:
                raise AgentConfigError(f"VLM_AGENT_{name}={spec.text}: {e}") from e
        return slots

    def check_spec(self, spec: AgentSpec) -> None:
        """What can be checked before anything starts: the provider's key or URL, the remote-bind block, and that the
        provider can be built (a bad URL or a missing model id), without calling it.

        hf is checked when it loads. Raises AgentConfigError.
        """
        if spec.provider == "hf":
            return
        available = self.availability(spec.provider)
        if not available.available:
            raise AgentConfigError(available.reason or f"{spec.provider} is not available")
        if self.remote_paid_blocked and spec.model and not is_local(spec.provider, spec.model):
            raise AgentConfigError(PAID_BLOCKED)
        try:
            self._build(spec)
        except (ProviderConfigError, ValueError) as e:
            raise AgentConfigError(str(e)) from e

    def _fill(self, slot: str, spec: AgentSpec, enabled: bool) -> Slot:
        available = self.availability(spec.provider)
        if not available.available:
            raise AgentConfigError(available.reason or f"{spec.provider} is not available")
        with self._lock:
            if spec.provider == "hf":
                other = [s for name, s in self._slots.items() if name != slot and s.enabled and s.spec.provider == "hf"]
                if enabled and other:
                    raise AgentConfigError("The local model can serve only one slot at a time")
                agent = self._hf
                assert agent is not None  # availability checked it
            else:
                try:
                    agent = self._build(spec)
                except (ProviderConfigError, ValueError) as e:
                    raise AgentConfigError(str(e)) from e
            remote = _slot_remote(spec, agent)
            if remote and self.remote_paid_blocked:
                raise AgentConfigError(PAID_BLOCKED)
            new = Slot(slot, spec, agent, enabled, remote)
            self._slots[slot] = new
        logger.info(f"Agent {slot}: {agent.label}{'' if enabled else ' (disabled)'}{' [remote]' if remote else ''}")
        return new

    def snapshot(self) -> tuple[Slot, ...]:
        """The current slots, in slot order. Immutable: a round keeps the agents it started with."""
        with self._lock:
            return tuple(self._slots[name] for name in SLOTS if name in self._slots)

    def any_remote(self) -> bool:
        return any(s.enabled and s.remote for s in self.snapshot())

    # -- model lists ------------------------------------------------------------------------------------------------

    def list_models(self, provider: str, refresh: bool = False) -> ModelListing:
        """The provider's vision-capable models (unknown ones kept). Successful lists are cached per run.

        May call the provider (slow): never call it while holding a lock other code waits on.
        """
        available = self.availability(provider)
        if not available.available:
            return ModelListing(provider, (), available.reason)
        if provider == "hf":
            return ModelListing(provider, tuple(self._hf.list_models()) if self._hf is not None else ())
        with self._lock:
            cached = None if refresh else self._listings.get(provider)
        if cached is not None:
            return cached
        try:
            model = None if provider == "demo" else LISTING_MODEL
            models = tuple(self._build(AgentSpec(provider, model)).list_models())
        except Exception as e:  # noqa: BLE001 - SDK, HTTP and parse errors all mean "no list"; free text still works
            message = log_failure(f"Listing the {provider} models failed", e, level="WARNING")
            return ModelListing(provider, (), f"Could not list models: {message}")
        listing = ModelListing(provider, models)
        with self._lock:
            self._listings[provider] = listing
        return listing
