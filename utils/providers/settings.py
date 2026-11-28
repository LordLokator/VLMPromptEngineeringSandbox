"""Provider settings from the environment, the `.env` file, and the one locality rule.

Keys and URLs come only from environment variables (or a git-ignored `.env`, which never overrides a real variable).
An empty variable counts as unset. The browser can never set a key or a URL.
"""
import ipaddress
import os
import re
from urllib.parse import urlsplit

from utils.file_management import BASE_DIR

OPENAI_URL = "https://api.openai.com/v1"
ANTHROPIC_URL = "https://api.anthropic.com"
OLLAMA_DEFAULT_URL = "http://localhost:11434"

TIMEOUT_S = 30.0
OLLAMA_TIMEOUT_S = 120.0  # a cold model load on the Ollama host can take a while

MAX_MODEL_ID_CHARS = 200
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+\-]*")


class ProviderConfigError(ValueError):
    """A provider cannot be built from the current settings; the message says what to set."""


def env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def load_env_file() -> bool:
    """Load `<repo>/.env` if it exists; real environment variables win, even empty ones."""
    from dotenv import load_dotenv

    return load_dotenv(BASE_DIR / ".env", override=False)


def check_model_id(model: str) -> str:
    """A model id typed by a person: bounded length and a character allowlist."""
    if len(model) > MAX_MODEL_ID_CHARS or not _MODEL_ID.fullmatch(model):
        raise ProviderConfigError(
            f"Invalid model id {model[:40]!r}: use at most {MAX_MODEL_ID_CHARS} letters, digits and ._:/@+-"
        )
    return model


def check_url(name: str, url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ProviderConfigError(f"{name} must be an http(s) URL with a host")
    return url.rstrip("/")


def is_loopback_url(url: str) -> bool:
    host = urlsplit(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def is_ollama_cloud_model(model: str) -> bool:
    """Ollama runs models tagged `cloud` (for example `gpt-oss:120b-cloud`) on its servers, not on the host."""
    _name, _sep, tag = model.rpartition(":")
    return "cloud" in tag.split("-") if _sep else model.endswith("-cloud")


def is_local(provider: str, model: str, base_url: str | None = None) -> bool:
    """The one locality rule. Local: demo, hf, or Ollama on a loopback URL with a model that is not a cloud model.

    Everything else is remote and treated as paid. Spend control, pricing, the remote-bind block, the privacy badge
    and the webcam confirmation all use this function.
    """
    if provider in ("demo", "hf"):
        return True
    if provider == "ollama":
        url = base_url or env("OLLAMA_BASE_URL") or OLLAMA_DEFAULT_URL
        return is_loopback_url(url) and not is_ollama_cloud_model(model)
    return False
