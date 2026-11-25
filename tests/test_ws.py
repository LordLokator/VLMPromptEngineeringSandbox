import json

import pytest
from fastapi.testclient import TestClient

import web_engine
from prompting.prompting import RuntimeSettings
from video_handling.streaming import USE_LIVE_CAMERA


class StubPrompt:
    def __init__(self):
        self.value = None

    def set(self, text):
        self.value = text


@pytest.fixture
def client(monkeypatch):
    stub = StubPrompt()
    monkeypatch.setattr(web_engine.app.state, "prompt", stub, raising=False)
    monkeypatch.setitem(USE_LIVE_CAMERA, "value", False)
    settings = RuntimeSettings()
    monkeypatch.setattr(web_engine.app.state, "settings", settings, raising=False)
    web_engine.active_connections.clear()
    client = TestClient(web_engine.app)
    client.stub = stub
    client.settings = settings
    return client


def test_toggle_stream_echo(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"role": "user", "text": "toggle_stream"}))
        assert ws.receive_json() == {"type": "system", "text": "Switched to: Live camera"}
    assert USE_LIVE_CAMERA["value"] is True


def test_prompt_change_echo(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"role": "user", "text": "hello"}))
        assert "Prompt changed to: hello" in ws.receive_json()["text"]
    assert client.stub.value == "hello"


def test_malformed_frame_does_not_close_socket(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text("{not json")
        assert ws.receive_json()["type"] == "error"
        ws.send_text(json.dumps({"role": "user", "text": "still alive"}))
        assert "still alive" in ws.receive_json()["text"]


def test_temperature_reaches_settings(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"role": "user", "text": "hi", "temperature": 0.7}))
        assert "Temperature: 0.70" in ws.receive_json()["text"]
    assert client.settings.get_temperature() == 0.7


def test_resample_fps_echoes_effective_rate(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"role": "user", "text": "hi", "resample_fps": 3}))
        assert "Resample: 3.33 fps" in ws.receive_json()["text"]
    assert client.settings.get_resample_fps() == 3


def test_oversized_text_rejected(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"role": "user", "text": "x" * 2001}))
        assert ws.receive_json()["type"] == "error"
    assert client.stub.value is None


def test_json_list_rejected(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text("[1, 2]")
        assert ws.receive_json()["type"] == "error"


def test_bad_temperature_replies_error_and_keeps_socket(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"role": "user", "text": "a", "temperature": "abc"}))
        assert ws.receive_json()["type"] == "error"
        ws.send_text(json.dumps({"role": "user", "text": "b"}))
        assert "Prompt changed to: b" in ws.receive_json()["text"]


def test_connection_removed_after_close(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text("{bad")
        ws.receive_json()
    assert web_engine.active_connections == []


def test_server_caps_raw_frame_size():
    # TestClient bypasses uvicorn, so assert the cap is wired into the real server config.
    config = web_engine.make_server().config
    assert config.ws_max_size == web_engine.WS_MAX_BYTES
    assert web_engine.WS_MAX_BYTES < 16 * 1024 * 1024
    assert web_engine.WS_MAX_BYTES > web_engine.MAX_TEXT_CHARS * 4  # UTF-8 worst case still fits
