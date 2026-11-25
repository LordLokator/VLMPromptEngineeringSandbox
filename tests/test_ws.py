import json

import pytest
from fastapi.testclient import TestClient

import web_engine
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
    monkeypatch.setattr(web_engine, "TEMPERATURE", 0.01)
    web_engine.active_connections.clear()
    client = TestClient(web_engine.app)
    client.stub = stub
    return client


def test_toggle_stream_echo(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"role": "user", "text": "toggle_stream"}))
        assert ws.receive_json() == {"role": "system", "text": "Switched to: Live camera"}
    assert USE_LIVE_CAMERA["value"] is True


def test_prompt_change_echo(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"role": "user", "text": "hello"}))
        assert "Prompt changed to: hello" in ws.receive_json()["text"]
    assert client.stub.value == "hello"


@pytest.mark.xfail(strict=True, reason="F10")
def test_malformed_frame_does_not_close_socket(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text("{not json")
        ws.send_text(json.dumps({"role": "user", "text": "still alive"}))
        assert "still alive" in ws.receive_json()["text"]
