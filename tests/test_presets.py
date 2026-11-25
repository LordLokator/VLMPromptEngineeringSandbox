import json

import pytest
from fastapi.testclient import TestClient

from utils import button_presets
from web_engine import app


@pytest.fixture
def user_file(tmp_path, monkeypatch):
    path = tmp_path / "user_presets.json"
    monkeypatch.setattr(button_presets, "USER_PRESETS_FILE", path)
    return path


def test_missing_user_file_returns_empty(user_file):
    assert button_presets.load_user_presets() == []


def test_user_preset_roundtrip(user_file):
    button_presets.save_user_preset("a", "text a")
    button_presets.save_user_preset("b", "text b")
    assert button_presets.load_user_presets() == [
        {"label": "a", "text": "text a"},
        {"label": "b", "text": "text b"},
    ]


def test_saving_user_preset_leaves_system_presets_untouched(user_file):
    before = button_presets.PRESETS_FILE.read_text(encoding="utf-8")
    button_presets.save_user_preset("x", "y")
    assert button_presets.PRESETS_FILE.read_text(encoding="utf-8") == before
    assert json.loads(before)  # system presets exist


def test_label_truncated_to_25_chars(user_file):
    res = TestClient(app).post("/save_preset", json={"label": "L" * 40, "text": "t"})
    assert res.json() == {"status": "ok", "label": "L" * 25}
    assert button_presets.load_user_presets()[0]["label"] == "L" * 25


def test_empty_input_rejected(user_file):
    res = TestClient(app).post("/save_preset", json={"label": " ", "text": "t"})
    assert res.status_code == 400 and res.json()["status"] == "error"


def test_invalid_bodies_return_400(user_file):
    c = TestClient(app)
    assert c.post("/save_preset", json=["x"]).status_code == 400
    assert c.post("/save_preset", json={"text": "t"}).status_code == 400
    assert c.post("/save_preset", json={"label": "l", "text": "x" * 2001}).status_code == 400
    assert c.post("/save_preset", json={"label": 5, "text": "t"}).status_code == 400
    assert button_presets.load_user_presets() == []


def test_concurrent_saves_stay_valid(user_file):
    import threading

    threads = [threading.Thread(target=button_presets.save_user_preset, args=(f"l{i}", "t")) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(json.loads(user_file.read_text(encoding="utf-8"))) == 8
