from pathlib import Path
import json

import pytest
from fastapi.testclient import TestClient
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

import web_engine


@pytest.fixture
def client(compare_app):
    return compare_app.client


def test_default_host_is_loopback():
    assert web_engine.HOST == "127.0.0.1"
    assert web_engine.IS_LOOPBACK


def test_mismatching_origin_rejected(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"origin": "http://evil.example"}):
            pass


def test_matching_and_missing_origin_accepted(client):
    for headers in ({"origin": "http://testserver"}, {}):
        with client.websocket_connect("/ws", headers=headers) as ws:
            assert ws.receive_json()["type"] == "state"
            ws.send_text(json.dumps({"type": "other"}))
            assert ws.receive_json()["type"] == "error"


def test_save_preset_requires_json_content_type(client):
    res = client.post("/save_preset", content='{"label":"a","text":"b"}', headers={"content-type": "text/plain"})
    assert res.status_code == 415


def test_index_served_and_metadata_removed(client):
    res = client.get("/")
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/html")
    assert client.get("/metadata").status_code == 404


def test_paths_do_not_depend_on_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    c = TestClient(web_engine.app)
    assert c.get("/").status_code == 200
    assert c.get("/preset_buttons").status_code == 200
    assert c.get("/src/script.js").status_code == 200
    from prompting import prompting
    from utils.button_presets import USER_PRESETS_FILE
    assert prompting.HISTORY_FILE.is_absolute() and USER_PRESETS_FILE.is_absolute()


def test_static_mounts_are_absolute():
    mounts = {
        r.path: r.app.directory
        for r in web_engine.app.routes
        if isinstance(r, Mount) and isinstance(r.app, StaticFiles) and r.path in ("/src", "/presets")
    }
    assert set(mounts) == {"/src", "/presets"}
    assert all(Path(d).is_absolute() for d in mounts.values())
