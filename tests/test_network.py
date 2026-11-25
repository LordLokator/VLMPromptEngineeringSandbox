import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import web_engine


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(web_engine.app.state, "prompt", None, raising=False)
    monkeypatch.setattr(web_engine.app.state, "settings", None, raising=False)
    web_engine.active_connections.clear()
    return TestClient(web_engine.app)


def test_default_host_is_loopback():
    assert web_engine.HOST == "127.0.0.1"
    assert web_engine.IS_LOOPBACK


def test_mismatching_origin_rejected(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"origin": "http://evil.example"}):
            pass


def test_matching_and_missing_origin_accepted(client):
    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_text(json.dumps({"role": "other"}))
        assert ws.receive_json()["type"] == "error"
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"role": "other"}))
        assert ws.receive_json()["type"] == "error"


def test_save_preset_requires_json_content_type(client):
    res = client.post("/save_preset", content='{"label":"a","text":"b"}', headers={"content-type": "text/plain"})
    assert res.status_code == 415


def test_index_served_and_metadata_removed(client):
    res = client.get("/")
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/html")
    assert client.get("/metadata").status_code == 404
