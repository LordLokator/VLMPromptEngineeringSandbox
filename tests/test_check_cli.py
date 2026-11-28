"""The provider check CLI, with the demo provider and a loopback stub (no real provider)."""
import json

import pytest

from tests.stub_server import CHAT_REPLY, stub_server
from utils.providers import check

KEY = "-".join(("fake", "compat", "key", "for", "cli", "tests"))


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(check, "configure_logging", lambda: None)  # would replace global hooks
    monkeypatch.setattr(check, "route_stdlib_logging", lambda: None)
    monkeypatch.setattr(check, "load_env_file", lambda: False)
    for name in ("OPENAI_API_KEY", "VLM_COMPAT_BASE_URL", "VLM_COMPAT_API_KEY", "OPENAI_CUSTOM_HEADERS"):
        monkeypatch.delenv(name, raising=False)


def test_sample_frames_are_one_clip_on_the_10_fps_grid():
    frames, times = check.sample_frames()
    assert len(frames) == 5 and all(f.startswith(b"\xff\xd8") for f in frames)
    assert times == pytest.approx((0.0, 0.1, 0.2, 0.3, 0.4))


def test_demo_call_prints_the_result(capsys):
    assert check.main(["demo", "frame-facts", "--fps", "3"]) == 0
    out = capsys.readouterr().out
    result = json.loads(out)
    assert result["text"].startswith("Demo (no model): 2 frames, 768x432")
    assert result["input_tokens"] is None and result["cost_usd"] is None


def test_list(capsys):
    assert check.main(["demo", "--list"]) == 0
    assert capsys.readouterr().out == "frame-facts\tvision\nframe-motion\tvision\n"


def test_missing_key_is_a_config_error(capsys):
    assert check.main(["openai", "gpt-x"]) == 2


def test_a_model_is_required_without_list():
    with pytest.raises(SystemExit):
        check.main(["openai"])


def test_hf_is_not_offered():
    with pytest.raises(SystemExit):
        check.main(["hf", "x"])


def test_compat_call_through_the_stub_never_prints_the_key(monkeypatch, capsys):
    with stub_server() as (base, recorder):
        recorder.route("POST", "/v1/chat/completions", CHAT_REPLY)
        monkeypatch.setenv("VLM_COMPAT_BASE_URL", base + "/v1")
        monkeypatch.setenv("VLM_COMPAT_API_KEY", KEY)
        assert check.main(["compat", "stub-model", "--max-output-tokens", "32"]) == 0
    out = capsys.readouterr().out
    assert json.loads(out)["text"] == "a street"
    assert KEY not in out
    (sent,) = recorder.requests
    assert sent["headers"]["authorization"] == f"Bearer {KEY}" and sent["body"]["max_tokens"] == 32
    assert len(sent["body"]["messages"][1]["content"]) == 6  # 5 frames and the prompt


def test_a_failed_call_prints_the_safe_message_only(monkeypatch, capsys):
    with stub_server() as (base, recorder):
        recorder.route("POST", "/v1/chat/completions", {"error": {"message": f"bad key {KEY}"}}, status=401)
        monkeypatch.setenv("VLM_COMPAT_BASE_URL", base + "/v1")
        monkeypatch.setenv("VLM_COMPAT_API_KEY", KEY)
        assert check.main(["compat", "stub-model"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("FAILED: Authentication failed (AuthenticationError, HTTP 401)")
    assert KEY not in out
