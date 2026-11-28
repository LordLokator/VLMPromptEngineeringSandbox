"""OpenAI-compatible adapter (openai, ollama, compat): fake-client tests and one loopback wire check."""
import base64
import io
import logging
import sys
import threading
from types import SimpleNamespace as NS

import pytest

from tests.stub_server import CHAT_REPLY, stub_server
from utils.errors import classify_error
from utils.logging_setup import InterceptHandler
from utils.providers import GenerateRequest
from utils.providers import openai_compat
from utils.providers.openai_compat import OpenAICompatProvider, build, build_messages, is_temperature_rejection
from utils.providers.settings import ProviderConfigError, is_local, is_ollama_cloud_model

# Built at runtime and plain words, so secret scanners do not take them for real keys.
OPENAI_KEY = "-".join(("fake", "openai", "key", "for", "tests"))
COMPAT_KEY = "-".join(("fake", "compat", "key", "for", "tests"))
PROVIDER_ENV = (
    "OPENAI_API_KEY", "OPENAI_ADMIN_KEY", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "OPENAI_WEBHOOK_SECRET",
    "OPENAI_BASE_URL", "OPENAI_CUSTOM_HEADERS", "OLLAMA_BASE_URL", "VLM_COMPAT_BASE_URL", "VLM_COMPAT_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)


def request(**overrides):
    args = dict(frames=(b"jpeg-a", b"jpeg-b"), frame_times=(0.0, 0.4), system="sys", prompt="what happens?",
                temperature=0.3, max_output_tokens=64)
    return GenerateRequest(**{**args, **overrides})


class TemperatureError(Exception):
    status_code = 400
    param = "temperature"
    message = "Unsupported value: 'temperature' does not support 0.3 with this model."


class OtherBadRequest(Exception):
    status_code = 400
    param = "messages"
    message = "bad image"


class FakeClient:
    """Stands in for openai.OpenAI: records create() calls and answers from a script."""

    def __init__(self, *answers, models=()):
        self.calls = []
        self.answers = list(answers) or [reply()]
        self.chat = NS(completions=NS(create=self._create))
        self.models = NS(list=lambda: NS(data=[NS(id=m) for m in models]))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


def reply(text="a street", usage=None, finish_reason="stop", model="gpt-x-2026-01-01", refusal=None):
    message = NS(content=text, refusal=refusal)
    return NS(model=model, choices=[NS(message=message, finish_reason=finish_reason)], usage=usage)


def provider(name="openai", client=None, model="gpt-x"):
    return OpenAICompatProvider(name, model, base_url="http://unused", client=client or FakeClient())


def test_message_shape_frames_in_order_then_prompt():
    system, user = build_messages(request())
    assert system == {"role": "system", "content": "sys"}
    images, text = user["content"][:2], user["content"][2]
    assert [part["image_url"]["url"] for part in images] == [
        "data:image/jpeg;base64," + base64.b64encode(frame).decode() for frame in (b"jpeg-a", b"jpeg-b")
    ]
    assert text == {"type": "text", "text": "what happens?"}


@pytest.mark.parametrize("name, limit", [
    ("openai", "max_completion_tokens"), ("ollama", "max_tokens"), ("compat", "max_tokens"),
])
def test_parameter_table(name, limit):
    client = FakeClient()
    result = provider(name, client).generate(request())
    (call,) = client.calls
    assert call["model"] == "gpt-x" and call[limit] == 64 and call["temperature"] == 0.3
    other = ({"max_tokens", "max_completion_tokens"} - {limit}).pop()
    assert other not in call
    assert result.params_sent == {limit: 64, "temperature": 0.3, "frames": 2}


def test_no_temperature_when_the_request_has_none():
    client = FakeClient()
    provider(client=client).generate(request(temperature=None))
    assert "temperature" not in client.calls[0]


def test_temperature_rejected_once_then_never_sent_again_for_that_model():
    client = FakeClient(TemperatureError(), reply(), reply())
    agent = provider(client=client)
    first = agent.generate(request())
    second = agent.generate(request())
    assert ["temperature" in call for call in client.calls] == [True, False, False]
    assert "temperature" not in first.params_sent and "temperature" not in second.params_sent


def test_other_errors_are_not_retried():
    client = FakeClient(OtherBadRequest())
    with pytest.raises(OtherBadRequest):
        provider(client=client).generate(request())
    assert len(client.calls) == 1


def test_a_second_temperature_error_is_raised():
    client = FakeClient(TemperatureError(), TemperatureError())
    with pytest.raises(TemperatureError):
        provider(client=client).generate(request())
    assert len(client.calls) == 2


def test_temperature_rejection_detection():
    assert is_temperature_rejection(TemperatureError())
    assert not is_temperature_rejection(OtherBadRequest())
    assert is_temperature_rejection(NS(status_code=400, param=None, message="temperature must be 1"))
    assert not is_temperature_rejection(NS(status_code=500, param="temperature", message=""))


def test_usage_mapping():
    usage = NS(prompt_tokens=900, completion_tokens=40, prompt_tokens_details=NS(cached_tokens=512),
               completion_tokens_details=NS(reasoning_tokens=16))
    result = provider(client=FakeClient(reply(usage=usage))).generate(request())
    assert (result.input_tokens, result.output_tokens, result.cached_tokens, result.reasoning_tokens) == (
        900, 40, 512, 16)
    assert (result.text, result.finish_reason, result.requested_model, result.reported_model) == (
        "a street", "stop", "gpt-x", "gpt-x-2026-01-01")
    assert result.cost_usd is None and result.latency_ms >= 0


@pytest.mark.parametrize("usage, tokens", [
    (None, (None, None)),
    (NS(), (None, None)),
    (NS(prompt_tokens=10, completion_tokens=None, prompt_tokens_details=None), (10, None)),
    (NS(prompt_tokens=10, completion_tokens=2, prompt_tokens_details=NS(cached_tokens=None),
        completion_tokens_details=NS()), (10, 2)),
])
def test_missing_usage_fields_are_none(usage, tokens):
    result = provider(client=FakeClient(reply(usage=usage))).generate(request())
    assert (result.input_tokens, result.output_tokens) == tokens
    assert result.cached_tokens is None and result.reasoning_tokens is None


def test_empty_answers_and_refusals_are_surfaced():
    result = provider(client=FakeClient(reply(text=None, finish_reason="length"))).generate(request())
    assert (result.text, result.finish_reason) == ("", "length")
    result = provider(client=FakeClient(reply(text=None, refusal="I can't help with that."))).generate(request())
    assert (result.text, result.finish_reason) == ("I can't help with that.", "refusal")
    result = provider(client=FakeClient(NS(model=None, choices=[], usage=None))).generate(request())
    assert (result.text, result.finish_reason) == ("", None)


def _status_error(cls, status, **kwargs):
    import httpx2

    response = httpx2.Response(status, request=httpx2.Request("POST", "http://127.0.0.1/v1/chat/completions"))
    return cls("boom", response=response, body=None, **kwargs)


def test_sdk_error_classes_map_to_kinds():
    import httpx2
    import openai

    request_ = httpx2.Request("POST", "http://127.0.0.1/v1/chat/completions")
    cases = [
        (_status_error(openai.AuthenticationError, 401), "auth"),
        (_status_error(openai.PermissionDeniedError, 403), "auth"),
        (_status_error(openai.RateLimitError, 429), "rate_limit"),
        (_status_error(openai.BadRequestError, 400), "bad_request"),
        (_status_error(openai.InternalServerError, 500), "internal"),
        (openai.APITimeoutError(request=request_), "timeout"),
        (openai.APIConnectionError(request=request_), "connection"),
    ]
    for error, kind in cases:
        assert classify_error(error)[0] == kind, type(error).__name__


def test_openai_needs_its_key_and_only_uses_its_own_url(monkeypatch):
    with pytest.raises(ProviderConfigError, match="OPENAI_API_KEY"):
        build("openai", "gpt-x")
    made = []
    monkeypatch.setattr(openai_compat, "make_client", lambda *args: made.append(args) or FakeClient())
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://somewhere.else/v1")
    build("openai", "gpt-x")
    assert made == [(OPENAI_KEY, "https://api.openai.com/v1", 30.0)]


def test_compat_never_falls_back_to_the_openai_key(monkeypatch):
    made = []
    monkeypatch.setattr(openai_compat, "make_client", lambda *args: made.append(args) or FakeClient())
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    with pytest.raises(ProviderConfigError, match="VLM_COMPAT_BASE_URL"):
        build("compat", "m")
    monkeypatch.setenv("VLM_COMPAT_BASE_URL", "https://llm.example/v1/")
    build("compat", "m")
    monkeypatch.setenv("VLM_COMPAT_API_KEY", COMPAT_KEY)
    build("compat", "m")
    assert made == [("EMPTY", "https://llm.example/v1", 30.0), (COMPAT_KEY, "https://llm.example/v1", 30.0)]


def test_ollama_gets_a_dummy_key_and_a_long_timeout(monkeypatch):
    made = []
    monkeypatch.setattr(openai_compat, "make_client", lambda *args: made.append(args) or FakeClient())
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    build("ollama", "llava:13b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434/")
    build("ollama", "llava:13b")
    assert made == [("ollama", "http://localhost:11434/v1", 120.0), ("ollama", "http://gpu-box:11434/v1", 120.0)]


@pytest.mark.parametrize("url", ["ftp://x", "localhost:11434", "http://"])
def test_bad_urls_are_refused(monkeypatch, url):
    monkeypatch.setenv("VLM_COMPAT_BASE_URL", url)
    with pytest.raises(ProviderConfigError):
        build("compat", "m")


@pytest.mark.parametrize("model", ["", "x" * 201, "a b", "../etc", "m\n"])
def test_bad_model_ids_are_refused(monkeypatch, model):
    monkeypatch.setenv("VLM_COMPAT_BASE_URL", "http://127.0.0.1:1")
    with pytest.raises(ProviderConfigError):
        build("compat", model)


def test_openai_model_list_hides_known_non_vision_models():
    client = FakeClient(models=("whisper-1", "gpt-x", "text-embedding-3-small", "o-new"))
    assert [m.id for m in provider("openai", client).list_models()] == ["gpt-x", "o-new"]
    assert [m.id for m in provider("compat", client).list_models()] == sorted(
        ("whisper-1", "gpt-x", "text-embedding-3-small", "o-new"))


def test_locality_rule(monkeypatch):
    assert is_local("demo", "frame-facts") and is_local("hf", "x")
    assert not is_local("openai", "gpt-x") and not is_local("anthropic", "c") and not is_local("compat", "m")
    assert is_local("ollama", "llava:13b")  # default URL is localhost
    assert not is_local("ollama", "gpt-oss:120b-cloud")
    assert is_local("ollama", "llava", "http://127.0.0.1:11434") and is_local("ollama", "llava", "http://[::1]:11434")
    assert not is_local("ollama", "llava", "http://gpu-box:11434")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://10.0.0.5:11434")
    assert not is_local("ollama", "llava")
    assert is_ollama_cloud_model("deepseek-v3:cloud") and is_ollama_cloud_model("x-cloud")
    assert not is_ollama_cloud_model("llava:13b") and not is_ollama_cloud_model("cloudy:7b")


# Wire checks: the real SDK against a loopback stub. They show what is sent; they are not provider runs.

def _non_null_handlers():
    loggers = [logging.getLogger()] + [
        item for item in logging.Logger.manager.loggerDict.values() if isinstance(item, logging.Logger)
    ]
    found = []
    for item in loggers:
        for handler in item.handlers:
            if isinstance(handler, logging.NullHandler):
                continue
            if item is logging.getLogger() and isinstance(handler, InterceptHandler):
                continue
            found.append((item.name, type(handler).__name__))
    return found


@pytest.mark.parametrize("name, url_env, auth", [
    ("compat", "VLM_COMPAT_BASE_URL", "Bearer EMPTY"), ("ollama", "OLLAMA_BASE_URL", "Bearer ollama"),
])
def test_wire_sends_nothing_meant_for_openai(monkeypatch, name, url_env, auth):
    from loguru import logger

    from utils.logging_setup import configure_logging

    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("OPENAI_ADMIN_KEY", OPENAI_KEY + "-admin")
    monkeypatch.setenv("OPENAI_ORG_ID", "org-from-env")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-from-env")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "X-Leak: 1\nAuthorization: Bearer " + OPENAI_KEY)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    for module, name in ((sys, "excepthook"), (threading, "excepthook"), (sys, "unraisablehook")):
        monkeypatch.setattr(module, name, getattr(module, name))
    saved = [(item, item.handlers[:], item.propagate, item.level) for item in
             [logging.getLogger()] + [i for i in logging.Logger.manager.loggerDict.values()
                                      if isinstance(i, logging.Logger)]]
    sink = io.StringIO()
    configure_logging(sink)
    try:
        with stub_server() as (base, recorder):
            recorder.route("POST", "/v1/chat/completions", CHAT_REPLY)
            monkeypatch.setenv(url_env, base + ("/v1" if name == "compat" else ""))
            result = build(name, "stub-model").generate(request())
            handlers_after = _non_null_handlers()
    finally:
        logging.captureWarnings(False)
        for item, handlers, propagate, level in saved:
            item.handlers, item.propagate = handlers, propagate
            item.setLevel(level)
        logger.remove()
        logger.add(sys.stderr, diagnose=False)

    (sent,) = recorder.requests
    headers = sent["headers"]
    assert headers["authorization"] == auth
    assert OPENAI_KEY not in repr(sent)
    assert not {"openai-organization", "openai-project", "x-leak"} & set(headers)
    assert sent["body"]["max_tokens"] == 64 and sent["body"]["temperature"] == 0.3
    assert "max_completion_tokens" not in sent["body"]
    assert result.text == "a street" and (result.input_tokens, result.output_tokens) == (120, 7)
    assert result.reported_model == "stub-model-2026"
    assert handlers_after == [], "an SDK added a log handler that bypasses redaction"


def test_wire_redirects_are_not_followed(monkeypatch):
    with stub_server() as (base, recorder):
        recorder.route("POST", "/v1/chat/completions", {}, status=307,
                       headers={"Location": base + "/elsewhere/chat/completions"})
        recorder.route("POST", "/elsewhere/chat/completions", CHAT_REPLY)
        monkeypatch.setenv("VLM_COMPAT_BASE_URL", base + "/v1")
        monkeypatch.setenv("VLM_COMPAT_API_KEY", COMPAT_KEY)
        with pytest.raises(Exception):  # noqa: B017 - any error: the point is that nothing follows the redirect
            build("compat", "stub-model").generate(request())
    assert [r["path"] for r in recorder.requests] == ["/v1/chat/completions"]


def test_ollama_listing_uses_capabilities_and_no_redirects(monkeypatch):
    with stub_server() as (base, recorder):
        recorder.route("GET", "/api/tags", {"models": [{"model": "llava:13b"}, {"model": "llama3:8b"},
                                                       {"model": "mystery:1b"}]})
        recorder.route("POST", "/api/show", {"capabilities": ["completion", "vision"]})
        monkeypatch.setenv("OLLAMA_BASE_URL", base)
        agent = build("ollama", "llava:13b")

        def show(body):
            caps = {"llava:13b": ["completion", "vision"], "llama3:8b": ["completion"]}.get(body["model"])
            return caps

        # One route serves every /api/show; answer per model by rewriting it per call.
        from utils.providers import ollama

        real_call = ollama._call

        def call(url, body=None):
            if url.endswith("/api/show"):
                caps = show(body)
                if caps is None:
                    raise OSError("no such model")
                return {"capabilities": caps}
            return real_call(url, body)

        monkeypatch.setattr(ollama, "_call", call)
        models = agent.list_models()
        recorder.route("GET", "/api/tags", {}, status=302, headers={"Location": "http://example.invalid/"})
        monkeypatch.setattr(ollama, "_call", real_call)
        with pytest.raises(OSError):
            agent.list_models()
    assert [(m.id, m.vision) for m in models] == [("llava:13b", True), ("mystery:1b", None)]
