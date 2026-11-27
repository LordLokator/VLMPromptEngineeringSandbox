"""A sentinel secret must reach no log sink and no WebSocket frame."""
import io
import logging
import os
import sys
import threading
import warnings

import numpy as np
import pytest
from fastapi.testclient import TestClient
from loguru import logger

import web_engine
from main import process_clip
from utils import logging_setup
from utils.backends import ChainBackend
from utils.logging_setup import REDACTED, configure_logging, redact
from video_handling import clip_handling, streaming
from video_handling.clip_handling import ClipRecorder

KEY = "sk-sentinel-0123456789abcdef"  # configured as a key: redaction must remove it
LOCAL_ONLY = "local-only-sentinel-fedcba"  # not configured: only diagnose=False keeps it out


@pytest.fixture
def sink(monkeypatch):
    """Install the real logging setup on a buffer, then restore the global state it touches."""
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    for module, name in ((sys, "excepthook"), (threading, "excepthook"), (sys, "unraisablehook")):
        monkeypatch.setattr(module, name, getattr(module, name))
    loggers = [logging.getLogger()] + [
        item for item in logging.Logger.manager.loggerDict.values() if isinstance(item, logging.Logger)
    ]
    saved = [(item, item.handlers[:], item.propagate, item.level) for item in loggers]
    stream = io.StringIO()
    configure_logging(stream)
    yield stream
    logging.captureWarnings(False)
    for item, handlers, propagate, level in saved:
        item.handlers, item.propagate = handlers, propagate
        item.setLevel(level)
    logger.remove()
    logger.add(sys.stderr, diagnose=False)


def fail():
    hidden = LOCAL_ONLY
    assert hidden
    raise RuntimeError(f"request failed for key {KEY}")


def assert_clean(text):
    assert KEY not in text
    assert LOCAL_ONLY not in text


class LeakyBackend:
    name = "leaky"

    def forward(self, video_path, fps, max_frames, temperature):
        fail()


def test_loguru_traceback_is_redacted_without_locals(sink):
    try:
        fail()
    except RuntimeError:
        logger.exception("loguru path")
    out = sink.getvalue()
    assert "loguru path" in out and "Traceback" in out and REDACTED in out
    assert_clean(out)


def test_stdlib_logging_and_warnings_are_redacted(sink):
    names = ("uvicorn.error", "uvicorn.access", "some.library")
    for name in names:
        try:
            fail()
        except RuntimeError:
            logging.getLogger(name).exception("stdlib %s %s", name, KEY)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        logging.captureWarnings(False)
        logging.captureWarnings(True)  # catch_warnings and pytest replace showwarning
        warnings.warn(f"careful {KEY}", stacklevel=1)
    out = sink.getvalue()
    for name in names:
        assert f"stdlib {name} {REDACTED}" in out
    assert f"careful {REDACTED}" in out
    assert_clean(out)


def test_bad_stdlib_format_string_does_not_print_arguments(sink, capsys):
    logging_setup.route_stdlib_logging()  # drops pytest's capture handler, which re-raises format errors
    logging.getLogger("some.library").error("%d", KEY)
    captured = capsys.readouterr()
    out = sink.getvalue() + captured.out + captured.err
    assert "Unformattable log message" in out
    assert_clean(out)


def test_library_handlers_added_later_are_rerouted(sink):
    own_stream = io.StringIO()
    library = logging.getLogger("late.library")
    library.addHandler(logging.StreamHandler(own_stream))
    library.propagate = False
    logging_setup.route_stdlib_logging()
    library.error("late %s", KEY)
    assert own_stream.getvalue() == ""
    assert f"late {REDACTED}" in sink.getvalue()


def test_sdk_debug_logging_is_off(monkeypatch, sink):
    monkeypatch.setenv("OPENAI_LOG", "debug")
    monkeypatch.setenv("ANTHROPIC_LOG", "debug")
    configure_logging(sink)
    assert "OPENAI_LOG" not in os.environ and "ANTHROPIC_LOG" not in os.environ
    for name in logging_setup.QUIET_LOGGERS:
        assert not logging.getLogger(name).isEnabledFor(logging.INFO)


def test_uncaught_errors_are_redacted(sink, capsys):
    thread = threading.Thread(target=fail, name="doomed")
    thread.start()
    thread.join(5)
    try:
        fail()
    except RuntimeError:
        sys.excepthook(*sys.exc_info())
    out = sink.getvalue()
    assert "Uncaught exception in thread doomed" in out and out.count("RuntimeError") >= 2
    assert_clean(out)
    captured = capsys.readouterr()
    assert_clean(captured.out + captured.err)


def test_uvicorn_config_adds_no_handlers_of_its_own(sink):
    web_engine.make_server()
    for name in ("uvicorn", "uvicorn.access"):
        assert logging.getLogger(name).handlers == []
        assert logging.getLogger(name).propagate


def test_backend_failure_reaches_neither_log_nor_websocket(sink, monkeypatch, tmp_path):
    monkeypatch.setattr(web_engine.app.state, "prompt", None, raising=False)
    monkeypatch.setattr(web_engine.app.state, "settings", None, raising=False)
    web_engine.active_connections.clear()
    with TestClient(web_engine.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_text('{"role": "x"}')
            ws.receive_json()  # the connection is registered
            process_clip(
                str(tmp_path / "c.mp4"),
                ChainBackend([LeakyBackend()]),
                web_engine.broadcast_sync,
                get_temperature=lambda: 0.1,
                get_fps=lambda: 10,
                max_frames=20,
            )
            frame = ws.receive_text()
    assert "Internal error (RuntimeError)" in frame
    assert_clean(frame)
    out = sink.getvalue()
    assert "Backend leaky failed" in out and "Failed to process clip" in out
    assert "in fail" in out  # stack frames stay, for debugging
    assert f"request failed for key {REDACTED}" not in out  # the rendered exception text is not logged
    assert_clean(out)


class _Frames:
    def __next__(self):
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def close(self):
        pass


def test_capture_and_clip_write_errors_carry_no_exception_text(sink, monkeypatch, tmp_path):
    monkeypatch.setattr(streaming, "VIDEO_PATHS", [str(tmp_path / "private-dir" / "missing.mp4")])
    monkeypatch.setattr(streaming, "VIDEO_IDX", 0)
    monkeypatch.setitem(clip_handling.USE_LIVE_CAMERA, "value", False)
    errors = []
    recorder = ClipRecorder(buffer_seconds=0.3, fps=10)
    recorder.out_dir = str(tmp_path)
    recorder.start(on_error=errors.append)
    assert recorder._thread is not None
    recorder._thread.join(5)
    assert len(errors) == 1 and "private-dir" not in errors[0] and "missing" not in errors[0]
    assert "private-dir" not in sink.getvalue()

    attempted = threading.Event()

    def broken_write(frames, path, fps=60):
        attempted.set()
        fail()

    monkeypatch.setattr(clip_handling, "write_video", broken_write)
    monkeypatch.setattr(recorder, "_open", _Frames)
    recorder.start()
    assert attempted.wait(5)
    recorder.stop()
    out = sink.getvalue()
    assert "Video writing or callback failed" in out and f"request failed for key {REDACTED}" not in out
    assert_clean(out)


def test_redact_handles_several_keys_and_ignores_placeholders(monkeypatch):
    for name in logging_setup.SECRET_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "abcdefgh")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "abcdefgh-longer")
    monkeypatch.setenv("VLM_COMPAT_API_KEY", "ollama")
    assert redact("abcdefgh-longer abcdefgh ollama") == f"{REDACTED} {REDACTED} ollama"
