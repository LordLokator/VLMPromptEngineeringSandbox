"""What may be said about a failure.

Exception messages can quote keys, file paths, prompt text or response bodies, so for backend and capture
errors they are neither sent to the UI nor logged. A fixed kind, the exception class and the HTTP status are.
"""
import json
import traceback

from loguru import logger

ERROR_KINDS: dict[str, str] = {
    "auth": "Authentication failed",
    "rate_limit": "Rate limit reached",
    "timeout": "Request timed out",
    "connection": "Connection failed",
    "bad_request": "Request rejected",
    "decode": "Could not read or decode the data",
    "internal": "Internal error",
}


class DecodeError(Exception):
    """Mixin for our own media errors; classified as `decode`."""


def _http_status(e: BaseException) -> int | None:
    for holder in (e, getattr(e, "response", None)):
        status = getattr(holder, "status_code", None)
        if isinstance(status, int) and 100 <= status <= 599:
            return status
    return None


def _kind(e: BaseException, status: int | None) -> str:
    if status is not None:
        if status in (401, 403):
            return "auth"
        if status == 429:
            return "rate_limit"
        if status in (408, 504):
            return "timeout"
        if 400 <= status < 500:
            return "bad_request"
        return "internal"
    # SDK exceptions are matched by class name, so no SDK has to be imported.
    name = type(e).__name__
    if "Authentication" in name or "PermissionDenied" in name:
        return "auth"
    if "RateLimit" in name:
        return "rate_limit"
    # Before "connection": SDK timeout classes can subclass their connection error.
    if isinstance(e, TimeoutError) or "Timeout" in name:
        return "timeout"
    if isinstance(e, ConnectionError) or "Connect" in name:
        return "connection"
    if isinstance(e, (DecodeError, UnicodeDecodeError, json.JSONDecodeError)):
        return "decode"
    return "internal"


def classify_error(e: BaseException) -> tuple[str, str]:
    """Return (kind, message). The message never contains the exception text."""
    status = _http_status(e)
    kind = _kind(e, status)
    detail = type(e).__name__ if status is None else f"{type(e).__name__}, HTTP {status}"
    return kind, f"{ERROR_KINDS[kind]} ({detail})"


def log_failure(context: str, e: BaseException, level: str = "ERROR") -> str:
    """Log the safe message plus the stack frames (code locations only); return the safe message."""
    kind, message = classify_error(e)
    frames = "".join(traceback.format_tb(e.__traceback__)).rstrip()
    text = f"{context}: [{kind}] {message}"
    logger.opt(depth=1).log(level, f"{text}\n{frames}" if frames else text)
    return message
