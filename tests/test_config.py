"""Fail fast: a bad setting stops the app before anything starts, with a message that never quotes a value."""
import pytest

from utils import config
from utils.providers.settings import ProviderConfigError, load_env_file

CHECKED = (
    "VLM_ENABLE_HF", "VLM_ALLOW_REMOTE_PAID", "OTEL_ENABLED", "VLM_PORT", "VLM_HOST", "OLLAMA_BASE_URL",
    "VLM_COMPAT_BASE_URL", "VLM_ATTN_IMPL", "OTEL_TRACES_EXPORTER", "OTEL_METRICS_EXPORTER",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in CHECKED:
        monkeypatch.delenv(name, raising=False)


def test_the_defaults_pass():
    assert config.problems() == []


@pytest.mark.parametrize(("name", "value"), [
    ("VLM_ENABLE_HF", "true"),
    ("VLM_ALLOW_REMOTE_PAID", "yes"),
    ("OTEL_ENABLED", "on"),
    ("VLM_PORT", "80a"),
    ("VLM_PORT", "0"),
    ("VLM_PORT", "70000"),
    ("VLM_PORT", "²"),
    ("VLM_HOST", "http://0.0.0.0"),
    ("VLM_HOST", "localhost:8000"),
    ("OLLAMA_BASE_URL", "localhost:11434"),
    ("VLM_COMPAT_BASE_URL", "ftp://example.com"),
    ("VLM_ATTN_IMPL", "flash"),
    ("VLM_AGNET_A", "demo"),
])
def test_a_bad_value_is_a_problem(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    found = config.problems()
    assert len(found) == 1
    assert name in found[0]


@pytest.mark.parametrize(("name", "value"), [
    ("VLM_ENABLE_HF", "0"), ("VLM_ALLOW_REMOTE_PAID", " 1 "), ("VLM_PORT", "8080"), ("VLM_PORT", ""),
    ("VLM_HOST", "192.168.1.20"), ("VLM_HOST", "::1"), ("VLM_HOST", "mybox.local"),
    ("OLLAMA_BASE_URL", "http://192.168.1.5:11434"), ("VLM_ATTN_IMPL", "sdpa"), ("VLM_MANUAL_OPENAI_MODEL", "x"),
])
def test_a_good_value_passes(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    assert config.problems() == []


def test_otel_exporters_are_checked_only_when_enabled(monkeypatch):
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "jaeger")
    assert config.problems() == []
    monkeypatch.setenv("OTEL_ENABLED", "1")
    monkeypatch.setattr(config, "_has_module", lambda _name: True)
    assert config.problems() == ["OTEL_TRACES_EXPORTER must be one of: console, otlp"]


def test_otel_enabled_without_the_package_is_a_problem(monkeypatch):
    monkeypatch.setenv("OTEL_ENABLED", "1")
    monkeypatch.setattr(config, "_has_module", lambda _name: False)
    assert "pip install '.[otel]'" in config.problems()[0]


def test_every_problem_is_reported_at_once(monkeypatch):
    monkeypatch.setenv("VLM_PORT", "x")
    monkeypatch.setenv("VLM_ENABLE_HF", "true")
    assert len(config.problems()) == 2


def test_parse_port():
    assert config.parse_port("") == config.DEFAULT_PORT
    assert config.parse_port(" 9000 ") == 9000
    with pytest.raises(config.ConfigError):
        config.parse_port("-1")


def test_an_unparsable_env_line_stops_the_load_without_quoting_it(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('GOOD_NAME_FOR_TEST=1\nsk-ant-secret-value\nX="secret, unterminated\n', encoding="utf-8")
    monkeypatch.delenv("GOOD_NAME_FOR_TEST", raising=False)
    with pytest.raises(ProviderConfigError) as info:
        load_env_file(env_file)
    assert "line 2, 3 " in str(info.value)
    assert "secret" not in str(info.value)
    import os

    assert "GOOD_NAME_FOR_TEST" not in os.environ  # nothing is loaded from a broken file


def test_a_valid_env_file_loads(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nGOOD_NAME_FOR_TEST=1\n\n", encoding="utf-8")
    monkeypatch.delenv("GOOD_NAME_FOR_TEST", raising=False)
    assert load_env_file(env_file) is True
    import os

    assert os.environ["GOOD_NAME_FOR_TEST"] == "1"
    monkeypatch.delenv("GOOD_NAME_FOR_TEST")


def test_a_bad_setting_ends_the_app_before_anything_starts(monkeypatch):
    import main as app_main

    monkeypatch.setenv("VLM_ENABLE_HF", "true")
    monkeypatch.setattr(app_main, "configure_logging", lambda: None)
    monkeypatch.setattr(app_main, "load_env_file", lambda: False)
    started = []
    monkeypatch.setattr(app_main, "start_server_threaded", lambda *_: started.append(1))
    with pytest.raises(SystemExit) as exit_info:
        app_main.main()
    assert exit_info.value.code == 2
    assert started == []


def test_an_agent_that_cannot_be_built_ends_the_app_before_anything_starts(monkeypatch):
    """The pre-check builds the provider (no call): `compat` without a model id fails there, before the server."""
    import main as app_main

    monkeypatch.setenv("VLM_AGENT_A", "compat")
    monkeypatch.setenv("VLM_COMPAT_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setattr(app_main, "configure_logging", lambda: None)
    monkeypatch.setattr(app_main, "load_env_file", lambda: False)
    started = []
    monkeypatch.setattr(app_main, "start_server_threaded", lambda *_: started.append(1))
    with pytest.raises(SystemExit) as exit_info:
        app_main.main()
    assert exit_info.value.code == 2
    assert started == []


def test_an_env_file_that_is_not_utf8_stops_the_load(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_bytes(b"NAME=\xff\xfe\n")
    with pytest.raises(ProviderConfigError, match="not UTF-8"):
        load_env_file(env_file)
