"""Ollama's native model listing (`/api/tags`, `/api/show`), with the standard library and no redirects."""
import json
import urllib.request
from typing import Any

from utils.providers import ModelInfo

LIST_TIMEOUT_S = 10.0
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # urllib then raises HTTPError for the 3xx instead of following it


_OPENER = urllib.request.build_opener(_NoRedirect())


def _call(url: str, body: dict[str, Any] | None = None) -> Any:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with _OPENER.open(request, timeout=LIST_TIMEOUT_S) as response:
        return json.loads(response.read(MAX_RESPONSE_BYTES))


def list_ollama_models(base_url: str) -> list[ModelInfo]:
    """Installed models; `vision` from `/api/show` capabilities, None when the server does not say."""
    tags = _call(f"{base_url}/api/tags")
    models = []
    for item in tags.get("models", []):
        name = item.get("model") or item.get("name")
        if not isinstance(name, str):
            continue
        try:
            capabilities = _call(f"{base_url}/api/show", {"model": name}).get("capabilities")
        except (OSError, ValueError):
            capabilities = None
        vision = "vision" in capabilities if isinstance(capabilities, list) else None
        if vision is False:
            continue  # known not to take images
        models.append(ModelInfo(name, vision=vision))
    return sorted(models, key=lambda m: m.id)
