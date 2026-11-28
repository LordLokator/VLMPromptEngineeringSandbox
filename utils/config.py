"""Fail fast: every setting is checked before the app starts, and a bad one stops it with a message.

A value the app would otherwise ignore or misread (a typo in a variable name, `VLM_ENABLE_HF=true`, an unknown OTel
exporter) is an error, not a silent default. Messages name the variable and the
accepted values; they never quote a value from `.env`, which may hold a key.
"""
import importlib.util
import ipaddress
import os
import re

from utils.providers.settings import ProviderConfigError, check_url

FLAGS = ("VLM_ENABLE_HF", "VLM_ALLOW_REMOTE_PAID", "OTEL_ENABLED")  # "1" on, "0" or unset off
KNOWN_VLM = frozenset({
    "VLM_AGENT_A", "VLM_AGENT_B", "VLM_ENABLE_HF", "VLM_ATTN_IMPL", "VLM_BACKEND", "VLM_COMPAT_BASE_URL",
    "VLM_COMPAT_API_KEY", "VLM_HOST", "VLM_PORT", "VLM_ALLOWED_HOSTS", "VLM_ALLOW_REMOTE_PAID",
})
TEST_PREFIX = "VLM_MANUAL_"  # settings of the opt-in tests in tests/manual
ATTN_IMPLS = ("flash_attention_2", "sdpa", "eager")  # the attn_implementation values transformers accepts
EXPORTERS = ("console", "otlp")  # what utils/telemetry.py builds
URLS = ("OLLAMA_BASE_URL", "VLM_COMPAT_BASE_URL")
DEFAULT_PORT = 8000
_HOST_NAME = re.compile(r"[A-Za-z0-9.\-]+")  # no scheme, port or path


class ConfigError(ValueError):
    """One or more settings are invalid; the message lists each, one per line."""


def _get(name: str) -> str:
    return os.environ.get(name, "").strip()


def _valid_host(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return bool(_HOST_NAME.fullmatch(host))
    return True


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:  # a parent package is missing
        return False


def parse_port(value: str) -> int:
    """VLM_PORT: empty means the default. Raises ConfigError."""
    value = value.strip()
    if not value:
        return DEFAULT_PORT
    if not (value.isascii() and value.isdigit()) or not 1 <= int(value) <= 65535:
        raise ConfigError("VLM_PORT must be a whole number from 1 to 65535")
    return int(value)


def problems() -> list[str]:
    """Every invalid setting in the environment, one message each. Empty when the app may start.

    The agent settings (VLM_AGENT_A / _B, keys) are checked by main.py through the agent registry.
    """
    found = [
        f"{name} is not a setting of this app (typo?); see README, Configuration"
        for name in sorted(os.environ)
        if name.upper().startswith("VLM_") and name.upper() not in KNOWN_VLM
        and not name.upper().startswith(TEST_PREFIX)
    ]
    found += [f"{name} must be 1 (on) or 0 (off)" for name in FLAGS if _get(name) not in ("", "0", "1")]
    try:
        parse_port(_get("VLM_PORT"))
    except ConfigError as e:
        found.append(str(e))
    host = _get("VLM_HOST")
    if host and not _valid_host(host):
        found.append("VLM_HOST must be a host name or an IP address, without scheme, port or path")
    for name in URLS:
        if _get(name):
            try:
                check_url(name, _get(name))
            except ProviderConfigError as e:
                found.append(str(e))
    attn = _get("VLM_ATTN_IMPL")
    if attn and attn not in ATTN_IMPLS:
        found.append(f"VLM_ATTN_IMPL must be one of: {', '.join(ATTN_IMPLS)}")
    if _get("OTEL_ENABLED") == "1":
        if not _has_module("opentelemetry.sdk"):
            found.append("OTEL_ENABLED=1 but OpenTelemetry is missing: pip install '.[otel]'")
        for name in ("OTEL_TRACES_EXPORTER", "OTEL_METRICS_EXPORTER"):
            if _get(name) and _get(name).lower() not in EXPORTERS:
                found.append(f"{name} must be one of: {', '.join(EXPORTERS)}")
    return found
