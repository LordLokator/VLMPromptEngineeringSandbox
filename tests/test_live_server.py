"""The real uvicorn server on a loopback port: TestClient bypasses its HTTP parsing and its logging."""
import io
import logging
import socket
import sys
import threading
import time

import pytest
from loguru import logger

import web_engine
from utils.logging_setup import configure_logging


@pytest.fixture
def server(monkeypatch):
    for module, name in ((sys, "excepthook"), (threading, "excepthook"), (sys, "unraisablehook")):
        monkeypatch.setattr(module, name, getattr(module, name))
    root = logging.getLogger()
    saved = root.handlers[:], root.level
    stream = io.StringIO()
    configure_logging(stream)
    monkeypatch.setattr(web_engine, "PORT", 0)  # any free port
    monkeypatch.setattr(web_engine.app.state, "prompt", None, raising=False)
    monkeypatch.setattr(web_engine.app.state, "settings", None, raising=False)
    web_engine.active_connections.clear()
    instance = web_engine.make_server()
    thread = threading.Thread(target=instance.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not instance.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert instance.started
    port = instance.servers[0].sockets[0].getsockname()[1]
    yield port, stream
    instance.should_exit = True
    thread.join(10)
    logging.captureWarnings(False)
    root.handlers = saved[0]
    root.setLevel(saved[1])
    logger.remove()
    logger.add(sys.stderr, diagnose=False)


def status_line(port, path, host, upgrade=False):
    lines = [f"GET {path} HTTP/1.1", f"Host: {host}"]
    if upgrade:
        lines += [
            "Upgrade: websocket",
            "Connection: Upgrade",
            "Sec-WebSocket-Version: 13",
            "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==",
            f"Origin: http://{host}",
        ]
    else:
        lines.append("Connection: close")
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        with sock.makefile("rb") as reply:
            return reply.readline().decode().strip()


def test_host_allowlist_holds_on_the_real_server(server):
    port, log = server
    assert status_line(port, "/", f"localhost:{port}").split()[1] == "200"
    assert status_line(port, "/", f"[::1]:{port}").split()[1] == "200"
    assert status_line(port, "/", "evil.example").split()[1] == "400"
    assert status_line(port, "/ws", f"localhost:{port}", upgrade=True).split()[1] == "101"
    assert status_line(port, "/ws", "evil.example", upgrade=True).split()[1] == "400"
    # uvicorn's own records arrive in the one redacting sink.
    out = log.getvalue()
    assert "Uvicorn running on" in out and '"GET / HTTP/1.1" 200' in out
