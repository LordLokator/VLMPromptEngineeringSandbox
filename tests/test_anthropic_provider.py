"""Anthropic adapter: fake-client tests and one loopback wire check."""
import base64
import logging
from types import SimpleNamespace as NS

import pytest

from tests.stub_server import MESSAGES_REPLY, stub_server
from tests.test_openai_compat import _non_null_handlers
from utils.errors import classify_error
from utils.providers import GenerateRequest, anthropic_provider
from utils.providers.anthropic_provider import AnthropicProvider, build, make_client
from utils.providers.settings import ProviderConfigError

KEY = "-".join(("fake", "anthropic", "key", "for", "tests"))
SAMPLING = {"temperature", "top_p", "top_k"}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "ANTHROPIC_CUSTOM_HEADERS"):
        monkeypatch.delenv(name, raising=False)


def request(**overrides):
    args = dict(frames=(b"jpeg-a", b"jpeg-b"), frame_times=(0.0, 0.4), system="sys", prompt="what happens?",
                temperature=0.3, max_output_tokens=64)
    return GenerateRequest(**{**args, **overrides})


TEXT = (NS(type="text", text="a street"),)


def message(blocks=TEXT, stop_reason="end_turn", usage=None, model="claude-x-1"):
    return NS(content=list(blocks), stop_reason=stop_reason, usage=usage, model=model)


class FakeClient:
    def __init__(self, answer=None, models=()):
        self.calls = []
        self.answer = answer if answer is not None else message()
        self.messages = NS(create=self._create)
        self.models = NS(list=lambda: list(models))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.answer


def run(answer=None, **overrides):
    client = FakeClient(answer)
    result = AnthropicProvider("claude-x", client=client).generate(request(**overrides))
    return client.calls[0], result


def test_request_shape_and_no_sampling_parameters():
    call, result = run()
    assert call["model"] == "claude-x" and call["system"] == "sys" and call["max_tokens"] == 64
    assert not SAMPLING & set(call)
    (turn,) = call["messages"]
    images, text = turn["content"][:2], turn["content"][2]
    assert [part["source"] for part in images] == [
        {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(f).decode()}
        for f in (b"jpeg-a", b"jpeg-b")
    ]
    assert text == {"type": "text", "text": "what happens?"}
    assert result.params_sent == {"max_tokens": 64, "frames": 2}


def test_text_blocks_are_joined_and_other_blocks_skipped():
    blocks = (NS(type="thinking", thinking="hmm"), NS(type="text", text="a "), NS(type="text", text="street"))
    _call, result = run(message(blocks))
    assert result.text == "a street"


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens", "end_turn"])
def test_refusal_truncation_and_empty_text_are_surfaced(stop_reason):
    _call, result = run(message(blocks=(), stop_reason=stop_reason))
    assert (result.text, result.finish_reason) == ("", stop_reason)


def test_input_tokens_include_cache_reads_and_writes():
    usage = NS(input_tokens=100, output_tokens=20, cache_read_input_tokens=300, cache_creation_input_tokens=50,
               output_tokens_details=NS(thinking_tokens=7))
    _call, result = run(message(usage=usage))
    assert (result.input_tokens, result.output_tokens, result.cached_tokens, result.reasoning_tokens) == (
        450, 20, 300, 7)
    assert (result.requested_model, result.reported_model) == ("claude-x", "claude-x-1")


@pytest.mark.parametrize("usage, expected", [
    (None, (None, None, None)),
    (NS(input_tokens=10, output_tokens=2), (10, 2, None)),
    (NS(input_tokens=10, output_tokens=2, cache_read_input_tokens=None, cache_creation_input_tokens=None),
     (10, 2, None)),
])
def test_missing_usage_fields(usage, expected):
    _call, result = run(message(usage=usage))
    assert (result.input_tokens, result.output_tokens, result.cached_tokens) == expected


def test_model_list_uses_image_input_capability():
    def model(id_, supported):
        caps = None if supported is None else NS(image_input=NS(supported=supported))
        return NS(id=id_, capabilities=caps)

    agent = AnthropicProvider("x", client=FakeClient(models=[model("b", True), model("a", None), model("c", False)]))
    assert [(m.id, m.vision) for m in agent.list_models()] == [("a", None), ("b", True)]


def _status_error(cls, status):
    import httpx2

    response = httpx2.Response(status, request=httpx2.Request("POST", "http://127.0.0.1/v1/messages"))
    return cls("boom", response=response, body=None)


def test_sdk_error_classes_map_to_kinds():
    import anthropic
    import httpx2

    request_ = httpx2.Request("POST", "http://127.0.0.1/v1/messages")
    cases = [
        (_status_error(anthropic.AuthenticationError, 401), "auth"),
        (_status_error(anthropic.RateLimitError, 429), "rate_limit"),
        (_status_error(anthropic.BadRequestError, 400), "bad_request"),
        (_status_error(anthropic.InternalServerError, 529), "internal"),
        (anthropic.APITimeoutError(request=request_), "timeout"),
        (anthropic.APIConnectionError(request=request_), "connection"),
    ]
    for error, kind in cases:
        assert classify_error(error)[0] == kind, type(error).__name__


def test_needs_its_key_and_only_uses_its_own_url(monkeypatch):
    with pytest.raises(ProviderConfigError, match="ANTHROPIC_API_KEY"):
        build("claude-x")
    made = []
    monkeypatch.setattr(anthropic_provider, "make_client", lambda *args: made.append(args) or FakeClient())
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://somewhere.else")
    build("claude-x")
    assert made == [(KEY, "https://api.anthropic.com", 30.0)]


def test_wire_check(monkeypatch):
    """The real SDK against a loopback stub: what it sends. Not a provider run."""
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Leak: 1")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "other-token-from-env")
    monkeypatch.setenv("ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE", "content")
    handlers_before = _non_null_handlers()
    with stub_server() as (base, recorder):
        recorder.route("POST", "/v1/messages", MESSAGES_REPLY)
        agent = AnthropicProvider("claude-x", client=make_client(KEY, base, 30.0))
        result = agent.generate(request())
        recorder.route("POST", "/v1/messages", {}, status=307, headers={"Location": base + "/elsewhere"})
        with pytest.raises(Exception):  # noqa: B017 - any error: the point is that nothing follows the redirect
            agent.generate(request())
    first, second = recorder.requests
    assert first["path"] == second["path"] == "/v1/messages"
    headers = first["headers"]
    assert KEY in (headers.get("authorization", "") + headers.get("x-api-key", ""))
    assert "other-token-from-env" not in repr(first)
    assert "x-leak" not in headers and "traceparent" not in headers
    assert not SAMPLING & set(first["body"]) and first["body"]["max_tokens"] == 64
    assert (result.text, result.input_tokens, result.cached_tokens, result.output_tokens) == ("a street", 120, 20, 5)
    assert _non_null_handlers() == handlers_before
    assert all(not isinstance(h, logging.StreamHandler) or isinstance(h, logging.NullHandler)
               for h in logging.getLogger("anthropic").handlers)
