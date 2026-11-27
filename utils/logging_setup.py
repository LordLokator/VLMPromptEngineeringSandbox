"""One log path for everything: loguru, stdlib logging, warnings and uncaught errors.

Configured key values are replaced in the final sink text, so tracebacks are covered too. `diagnose` is off:
loguru would otherwise print local variable values in tracebacks. Output that bypasses Python logging (native
libraries, the OpenTelemetry console exporter) is not covered.
"""
import logging
import os
import sys
import threading
from types import TracebackType
from typing import TextIO

from loguru import logger

SECRET_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "VLM_COMPAT_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)
# Shorter values are placeholders such as "ollama" or "EMPTY"; replacing them would only mangle the logs.
MIN_SECRET_CHARS = 8
REDACTED = "[redacted]"
# These log requests, headers or bodies at DEBUG.
QUIET_LOGGERS = ("openai", "anthropic", "httpx", "httpcore")
# Either of these makes the SDK switch on its own debug logging.
SDK_LOG_ENV_VARS = ("OPENAI_LOG", "ANTHROPIC_LOG")


def redact(text: str) -> str:
    """Replace every configured key value. Read from the environment on each call, so load order does not matter."""
    secrets = {os.environ.get(name, "").strip() for name in SECRET_ENV_VARS}
    # Longest first, so a key that contains another is removed whole.
    for secret in sorted(secrets, key=len, reverse=True):
        if len(secret) >= MIN_SECRET_CHARS:
            text = text.replace(secret, REDACTED)
    return text


class RedactingSink:
    def __init__(self, stream: TextIO):
        self._stream = stream

    def write(self, message: str) -> None:
        try:
            self._stream.write(redact(str(message)))
            self._stream.flush()
        except (OSError, ValueError):
            pass  # closed or broken stream: there is nowhere safe left to report it


class InterceptHandler(logging.Handler):
    """Forward stdlib records to loguru, keeping the caller's location."""

    def emit(self, record: logging.LogRecord) -> None:
        # Never fall back to Handler.handleError: it prints the raw message and arguments to stderr.
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a bad format string must not reach handleError
            message = f"Unformattable log message from logger {record.name!r}"
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 0
        while frame is not None and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, message)


def route_stdlib_logging() -> None:
    """Send every stdlib logger through loguru.

    Call again after importing a library that installs its own handler (transformers does).
    """
    root = logging.getLogger()
    root.handlers = [InterceptHandler()]
    root.setLevel(logging.INFO)
    for item in list(logging.Logger.manager.loggerDict.values()):
        if isinstance(item, logging.Logger):
            item.handlers = [h for h in item.handlers if isinstance(h, logging.NullHandler)]
            item.propagate = True
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _excepthook(exc_type: type[BaseException], value: BaseException, tb: TracebackType | None) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, value, tb)
        return
    logger.opt(exception=(exc_type, value, tb)).critical("Uncaught exception")


def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
    if issubclass(args.exc_type, SystemExit):
        return
    name = args.thread.name if args.thread is not None else "unknown"
    logger.opt(exception=(args.exc_type, args.exc_value, args.exc_traceback)).error(
        f"Uncaught exception in thread {name}"
    )


def _unraisablehook(unraisable) -> None:
    # The object's repr is left out on purpose: it can hold anything.
    logger.opt(exception=(unraisable.exc_type, unraisable.exc_value, unraisable.exc_traceback)).error(
        "Unraisable exception"
    )


def configure_logging(stream: TextIO | None = None, level: str = "DEBUG") -> None:
    """Install the only log sink and the hooks. Call before anything else logs."""
    target = stream if stream is not None else sys.stderr
    for name in SDK_LOG_ENV_VARS:
        os.environ.pop(name, None)
    logger.remove()
    # catch=False: loguru's fallback would print the failed record unredacted.
    logger.add(
        RedactingSink(target),
        level=level,
        diagnose=False,
        catch=False,
        colorize=bool(getattr(target, "isatty", lambda: False)()),
    )
    route_stdlib_logging()
    logging.captureWarnings(True)
    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook
    sys.unraisablehook = _unraisablehook
