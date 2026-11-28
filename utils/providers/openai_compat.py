"""OpenAI, Ollama (its `/v1` API) and any OpenAI-compatible URL, through the official `openai` SDK.

Facts from the installed SDK (openai 3.57.0, read 2026-11-28):
- The client reads OPENAI_API_KEY, OPENAI_ADMIN_KEY, OPENAI_ORG_ID, OPENAI_PROJECT_ID, OPENAI_WEBHOOK_SECRET,
  OPENAI_BASE_URL and OPENAI_CUSTOM_HEADERS by itself. `make_client` passes the key and URL explicitly and clears the
  rest, so nothing meant for OpenAI reaches another server.
- Its HTTP client (httpx2) follows redirects, retries twice and waits up to 600 s by default: here redirects are off,
  retries 0, and the timeout is ours.
- Chat Completions usage: prompt_tokens (cached included), completion_tokens, prompt_tokens_details.cached_tokens,
  completion_tokens_details.reasoning_tokens. Compatible servers may leave any of them out.
- Errors: BadRequestError (400) carries `param`; status errors carry `status_code`; APITimeoutError subclasses
  APIConnectionError. utils/errors.py classifies them by class name and status.
"""
import base64
import threading
import time
from typing import Any

from utils.providers import GenerateRequest, ModelInfo, Result
from utils.providers.settings import (
    OLLAMA_DEFAULT_URL,
    OLLAMA_TIMEOUT_S,
    OPENAI_URL,
    TIMEOUT_S,
    ProviderConfigError,
    check_model_id,
    check_url,
    env,
)

# Which field caps the answer length: OpenAI's newer models accept only max_completion_tokens; Ollama and most
# compatible servers know max_tokens.
TOKEN_LIMIT_PARAM = {"openai": "max_completion_tokens", "ollama": "max_tokens", "compat": "max_tokens"}
OLLAMA_DUMMY_KEY = "ollama"  # Ollama ignores the key, the SDK requires one
COMPAT_KEYLESS = "EMPTY"  # sent when VLM_COMPAT_API_KEY is unset; never the OpenAI key
# Model id prefixes on OpenAI's /v1/models that cannot take images; they are hidden from the picker.
OPENAI_NON_VISION_PREFIXES = (
    "text-embedding", "whisper", "tts", "dall-e", "davinci", "babbage", "omni-moderation", "text-moderation",
    "gpt-3.5", "gpt-image", "sora", "codex-mini",
)


def make_client(api_key: str, base_url: str, timeout: float) -> Any:
    import openai

    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=0,
        http_client=openai.DefaultHttpxClient(follow_redirects=False, timeout=timeout),
    )
    # Filled from OPENAI_* variables by the SDK itself; this app configures only the key and the URL.
    client.organization = None
    client.project = None
    client.admin_api_key = None
    client.webhook_secret = None
    client._custom_headers = {}  # from OPENAI_CUSTOM_HEADERS
    client._ambient_authorizations = frozenset()
    return client


def jpeg_data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def build_messages(request: GenerateRequest) -> list[dict[str, Any]]:
    """System text, then the frames in time order, then the prompt."""
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": jpeg_data_url(frame)}} for frame in request.frames
    ]
    content.append({"type": "text", "text": request.prompt})
    return [{"role": "system", "content": request.system}, {"role": "user", "content": content}]


def is_temperature_rejection(error: BaseException) -> bool:
    """A 400 that names the temperature parameter (some reasoning models accept only the default)."""
    if getattr(error, "status_code", None) != 400:
        return False
    if getattr(error, "param", None) == "temperature":
        return True
    return "temperature" in str(getattr(error, "message", "")).lower()


def _field(obj: Any, *names: str) -> Any:
    for name in names:
        obj = getattr(obj, name, None)
        if obj is None:
            return None
    return obj


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class OpenAICompatProvider:
    def __init__(self, provider: str, model: str, *, base_url: str, client: Any):
        if provider not in TOKEN_LIMIT_PARAM:
            raise ValueError(f"not an OpenAI-compatible provider: {provider}")
        self.provider = provider
        self.model = check_model_id(model)
        self.base_url = base_url
        self._client = client
        self._lock = threading.Lock()
        self._no_temperature: set[str] = set()  # models that rejected temperature in this run

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    def sends_temperature(self) -> bool:
        with self._lock:
            return self.model not in self._no_temperature

    def list_models(self) -> list[ModelInfo]:
        if self.provider == "ollama":
            from utils.providers.ollama import list_ollama_models

            return list_ollama_models(self.base_url.removesuffix("/v1"))
        models = []
        for item in self._client.models.list().data:
            model_id = str(item.id)
            if self.provider == "openai" and model_id.startswith(OPENAI_NON_VISION_PREFIXES):
                continue
            models.append(ModelInfo(model_id, vision=None))
        return sorted(models, key=lambda m: m.id)

    def generate(self, request: GenerateRequest) -> Result:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": build_messages(request),
            TOKEN_LIMIT_PARAM[self.provider]: request.max_output_tokens,
        }
        if request.temperature is not None and self.sends_temperature():
            params["temperature"] = request.temperature
        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**params)
        except Exception as e:
            if "temperature" not in params or not is_temperature_rejection(e):
                raise
            with self._lock:
                self._no_temperature.add(self.model)
            del params["temperature"]
            response = self._client.chat.completions.create(**params)
        latency_ms = (time.perf_counter() - start) * 1000
        return self._result(response, params, latency_ms)

    def _result(self, response: Any, params: dict[str, Any], latency_ms: float) -> Result:
        choices = getattr(response, "choices", None) or []
        message = _field(choices[0], "message") if choices else None
        text = _field(message, "content") or ""
        finish_reason = _field(choices[0], "finish_reason") if choices else None
        refusal = _field(message, "refusal")
        if not text and refusal:
            text, finish_reason = str(refusal), "refusal"
        usage = getattr(response, "usage", None)
        sent = {key: value for key, value in params.items() if key not in ("model", "messages")}
        sent["frames"] = sum(1 for part in params["messages"][1]["content"] if part["type"] == "image_url")
        return Result(
            text=str(text),
            finish_reason=finish_reason,
            requested_model=self.model,
            reported_model=_field(response, "model"),
            input_tokens=_int_or_none(_field(usage, "prompt_tokens")),
            output_tokens=_int_or_none(_field(usage, "completion_tokens")),
            cached_tokens=_int_or_none(_field(usage, "prompt_tokens_details", "cached_tokens")),
            reasoning_tokens=_int_or_none(_field(usage, "completion_tokens_details", "reasoning_tokens")),
            latency_ms=latency_ms,
            cost_usd=None,
            cost_source=None,
            params_sent=sent,
        )


def build(provider: str, model: str, client: Any = None) -> OpenAICompatProvider:
    """The provider with its key and URL from the environment. `client` replaces the SDK client in tests."""
    if provider == "openai":
        key = env("OPENAI_API_KEY")
        if key is None:
            raise ProviderConfigError("openai needs OPENAI_API_KEY")
        base_url, timeout = OPENAI_URL, TIMEOUT_S  # the OpenAI key only ever goes to OpenAI's own URL
    elif provider == "ollama":
        key = OLLAMA_DUMMY_KEY
        base_url = check_url("OLLAMA_BASE_URL", env("OLLAMA_BASE_URL") or OLLAMA_DEFAULT_URL) + "/v1"
        timeout = OLLAMA_TIMEOUT_S
    elif provider == "compat":
        url = env("VLM_COMPAT_BASE_URL")
        if url is None:
            raise ProviderConfigError("compat needs VLM_COMPAT_BASE_URL")
        base_url, timeout = check_url("VLM_COMPAT_BASE_URL", url), TIMEOUT_S
        key = env("VLM_COMPAT_API_KEY") or COMPAT_KEYLESS
    else:
        raise ValueError(f"not an OpenAI-compatible provider: {provider}")
    check_model_id(model)
    return OpenAICompatProvider(
        provider, model, base_url=base_url, client=client if client is not None else make_client(key, base_url, timeout)
    )
