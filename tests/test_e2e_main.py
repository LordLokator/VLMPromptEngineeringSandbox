"""The real `python main.py` with both demo agents on a loopback port: startup, Run, one round over /ws, clean stop.

A subprocess, because the signal handlers need the main thread. Every variable the app reads is set (empty counts as
unset), so a developer's `.env` cannot change the run. Bundled sample, no webcam, no network beyond loopback.
"""
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parent.parent
START_LIMIT_S = 30.0
ANSWER_LIMIT_S = 20.0
STOP_LIMIT_S = 10.0
# loguru's default format; continuation lines (stack locations from log_failure) are indented.
LOG_LINE = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}(?: [+-]\d\d:\d\d)? \| [A-Z]+ +\| ")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def app_env(port: int, token: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("VLM_", "OTEL_"))}
    env.update({
        "VLM_AGENT_A": "demo",
        "VLM_AGENT_B": "",
        "VLM_ENABLE_HF": "",
        "VLM_BACKEND": "",
        "VLM_HOST": "127.0.0.1",
        "VLM_PORT": str(port),
        "VLM_ALLOWED_HOSTS": "",
        "VLM_ALLOW_REMOTE_PAID": "",
        "OPENAI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "OLLAMA_BASE_URL": "",
        "VLM_COMPAT_BASE_URL": "",
        "VLM_COMPAT_API_KEY": "",
        "HUGGING_FACE_HUB_TOKEN": "",
        "HF_TOKEN": token,
        "OTEL_ENABLED": "",
        "PYTHONUNBUFFERED": "1",
    })
    return env


def wait_for_socket(url: str, proc: subprocess.Popen):
    deadline = time.monotonic() + START_LIMIT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"main.py exited early with {proc.returncode}")
        try:
            return connect(url, open_timeout=2)
        except OSError:
            time.sleep(0.2)
    pytest.fail("the server did not accept a WebSocket in time")


def test_demo_app_answers_and_stops_cleanly():
    port = free_port()
    token = "e2e-dummy-" + "t" * 12  # built here: no secret-shaped literal in the repo
    proc = subprocess.Popen(
        [sys.executable, "main.py"], cwd=ROOT, env=app_env(port, token),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    )
    try:
        with wait_for_socket(f"ws://127.0.0.1:{port}/ws", proc) as ws:
            assert json.loads(ws.recv(timeout=10))["type"] == "state"
            ws.send(json.dumps({"type": "control", "action": "run"}))  # the app starts paused
            deadline = time.monotonic() + ANSWER_LIMIT_S
            frames = []
            while not any(f["type"] == "round_end" for f in frames) and time.monotonic() < deadline:
                try:
                    frames.append(json.loads(ws.recv(timeout=max(0.1, deadline - time.monotonic()))))
                except TimeoutError:
                    break
            kinds = [f["type"] for f in frames if f["type"] in ("round", "result", "round_end")]
            assert kinds[:1] == ["round"] and kinds[-1:] == ["round_end"], kinds
            results = {f["slot"]: f for f in frames if f["type"] == "result"}
            assert results["A"]["text"].startswith("Demo (no model): ")
            assert results["B"]["text"].startswith("Demo (no model), motion: ")
            assert results["A"]["round_id"] == results["B"]["round_id"]

        if sys.platform == "win32":
            proc.terminate()  # no SIGINT for a child here; the exit code says nothing on Windows
            output, _ = proc.communicate(timeout=STOP_LIMIT_S)
        else:
            proc.send_signal(signal.SIGINT)
            output, _ = proc.communicate(timeout=STOP_LIMIT_S)
            assert proc.returncode == 0, output
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert "Traceback" not in output
    assert token not in output
    lines = [line for line in output.splitlines() if line.strip()]
    foreign = [line for line in lines if not LOG_LINE.match(line) and not line[:1].isspace()]
    assert not foreign, foreign
    assert "Agent A: demo:frame-facts" in output and "Agent B: demo:frame-motion" in output
