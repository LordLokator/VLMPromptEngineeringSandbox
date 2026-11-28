"""Anthropic's native Messages API through the official `anthropic` SDK.

Facts from the installed SDK (anthropic 2.12.0, read 2026-11-28):
- It sends API version 2026-10-21, whose Messages API has no temperature, top_p or top_k: none is ever sent.
- Without an explicit credential the client reads ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, profiles and federation
  settings; it also reads ANTHROPIC_BASE_URL, ANTHROPIC_CUSTOM_HEADERS and ANTHROPIC_WEBHOOK_SIGNING_KEY. Here the key
  and URL are explicit and the custom headers are cleared.
- Like the openai SDK it uses httpx2, follows redirects, retries twice and waits up to 600 s by default.
- It emits its own OpenTelemetry spans through the global tracer provider, and ANTHROPIC_OPEN_TELEMETRY* variables
  can make them carry request and response content. Here they are off (the app's own span covers the call); an
  explicit `open_telemetry` option also stops the SDK from reading those variables.
- Usage: input_tokens excludes cache reads and cache writes (cache_read_input_tokens, cache_creation_input_tokens);
  output_tokens_details.thinking_tokens may be present. stop_reason: end_turn, max_tokens, stop_sequence, tool_use,
  pause_turn, refusal, model_context_window_exceeded.
"""
import base64
import time
from typing import Any

from utils.providers import GenerateRequest, ModelInfo, Result
from utils.providers.pricing import estimate
from utils.providers.settings import ANTHROPIC_URL, TIMEOUT_S, ProviderConfigError, check_model_id, env


def make_client(api_key: str, base_url: str, timeout: float) -> Any:
    import anthropic

    client = anthropic.Anthropic(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(follow_redirects=False, timeout=timeout),
        open_telemetry={"traces": {"enabled": False}, "propagation": False},
    )
    client._custom_headers = {}  # from ANTHROPIC_CUSTOM_HEADERS; this app configures only the key and the URL
    client.webhook_key = None
    return client


def build_messages(request: GenerateRequest) -> list[dict[str, Any]]:
    """One user turn: the frames in time order, then the prompt. The system text goes in `system`."""
    content: list[dict[str, Any]] = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(frame).decode("ascii")},
        }
        for frame in request.frames
    ]
    content.append({"type": "text", "text": request.prompt})
    return [{"role": "user", "content": content}]


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class AnthropicProvider:
    provider = "anthropic"

    def __init__(self, model: str, *, client: Any):
        self.model = check_model_id(model)
        self._client = client

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    def list_models(self) -> list[ModelInfo]:
        models = []
        for item in self._client.models.list():
            supported = getattr(getattr(getattr(item, "capabilities", None), "image_input", None), "supported", None)
            vision = supported if isinstance(supported, bool) else None
            if vision is False:
                continue
            models.append(ModelInfo(str(item.id), vision=vision))
        return sorted(models, key=lambda m: m.id)

    def generate(self, request: GenerateRequest) -> Result:
        params: dict[str, Any] = {
            "model": self.model,
            "system": request.system,
            "messages": build_messages(request),
            "max_tokens": request.max_output_tokens,
        }
        start = time.perf_counter()
        response = self._client.messages.create(**params)
        latency_ms = (time.perf_counter() - start) * 1000

        blocks = getattr(response, "content", None) or []
        text = "".join(str(b.text) for b in blocks if getattr(b, "type", None) == "text" and getattr(b, "text", None))
        usage = getattr(response, "usage", None)
        uncached = _int(getattr(usage, "input_tokens", None))
        cache_read = _int(getattr(usage, "cache_read_input_tokens", None))
        cache_write = _int(getattr(usage, "cache_creation_input_tokens", None))
        input_tokens = None if uncached is None else uncached + (cache_read or 0) + (cache_write or 0)
        details = getattr(usage, "output_tokens_details", None)
        output_tokens = _int(getattr(usage, "output_tokens", None))
        cost, cost_source = estimate(self.provider, self.model, uncached_input=uncached, cache_read=cache_read or 0,
                                     cache_write=cache_write or 0, output=output_tokens)
        return Result(
            text=text,
            finish_reason=getattr(response, "stop_reason", None),
            requested_model=self.model,
            reported_model=getattr(response, "model", None),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cache_read,
            reasoning_tokens=_int(getattr(details, "thinking_tokens", None)),
            latency_ms=latency_ms,
            cost_usd=cost,
            cost_source=cost_source,
            params_sent={"max_tokens": request.max_output_tokens, "frames": len(request.frames)},
        )


def build(model: str, client: Any = None) -> AnthropicProvider:
    key = env("ANTHROPIC_API_KEY")
    if key is None:
        raise ProviderConfigError("anthropic needs ANTHROPIC_API_KEY")
    check_model_id(model)
    # The Anthropic key only ever goes to Anthropic's own URL.
    return AnthropicProvider(model, client=client if client is not None else make_client(key, ANTHROPIC_URL, TIMEOUT_S))
