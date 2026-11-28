"""The /ws protocol: state on connect, the client frames, malformed input, spend control, no key in any frame."""
import json
import time

import pytest

import web_engine
from utils.session import CONFIRM_WEBCAM
from video_handling.clip_handling import ClipSample

SENTINEL = "sk-" + "sentinel" * 4  # built at runtime: no secret-shaped literal in the repo


def send(ws, **frame):
    ws.send_text(json.dumps(frame))


def eventually(check, limit=5.0):
    deadline = time.monotonic() + limit
    while not check() and time.monotonic() < deadline:
        time.sleep(0.01)
    return check()


def receive_until(ws, kind, limit=20):
    """Frames up to and including the first of type `kind`."""
    seen = []
    for _ in range(limit):
        frame = ws.receive_json()
        seen.append(frame)
        if frame["type"] == kind:
            return seen
    raise AssertionError(f"no {kind} frame in {[f['type'] for f in seen]}")


def test_state_on_connect_starts_paused(compare_app):
    with compare_app.client.websocket_connect("/ws") as ws:
        state = ws.receive_json()
    assert state["type"] == "state"
    assert state["run"]["running"] is False and state["run"]["pause_reason"] == "start"
    assert [(s["slot"], s["label"], s["remote"], s["destination"]) for s in state["slots"]] == [
        ("A", "demo:frame-facts", False, None), ("B", "demo:frame-motion", False, None),
    ]
    providers = {p["id"]: p for p in state["providers"]}
    assert providers["demo"]["available"] and providers["openai"]["reason"] == "OPENAI_API_KEY is not set"
    assert state["source"]["key"] == "sample:1" and state["source"]["error"] is None
    assert [s["key"] for s in state["sources"]] == ["sample:1", "sample:2", "webcam"]
    assert state["settings"]["prompt"] == "describe" and state["price_date"]


def test_run_pause_step_and_reset(compare_app):
    with compare_app.client.websocket_connect("/ws") as ws:
        ws.receive_json()
        send(ws, type="control", action="run")
        assert ws.receive_json()["run"]["running"] is True
        send(ws, type="control", action="step")
        assert ws.receive_json() == {"type": "error", "text": "Already running: pause first to step",
                                     "request": "control"}
        send(ws, type="control", action="pause")
        assert ws.receive_json()["run"]["pause_reason"] == "user"
        send(ws, type="control", action="step")
        assert ws.receive_json()["run"]["step_pending"] is True
        send(ws, type="control", action="reset_budget")
        assert ws.receive_json()["run"]["rounds_used"] == 0
        send(ws, type="control", action="explode")
        assert ws.receive_json()["type"] == "error"


def test_update_changes_the_next_round_settings(compare_app):
    with compare_app.client.websocket_connect("/ws") as ws:
        ws.receive_json()
        send(ws, type="update", text="count the cars", temperature=0.7, resample_fps=3)
        state = ws.receive_json()
        assert state["settings"]["prompt"] == "count the cars"
        assert (state["settings"]["temperature"], state["settings"]["resample_fps"]) == (0.7, 3)
        assert state["settings"]["effective_fps"] == 4.0
        assert ws.receive_json()["type"] == "system"
        send(ws, type="update", temperature="abc")
        assert ws.receive_json()["type"] == "error"
    assert compare_app.prompt.value == "count the cars"


def test_set_agent_and_its_errors(compare_app):
    with compare_app.client.websocket_connect("/ws") as ws:
        ws.receive_json()
        send(ws, type="set_agent", slot="B", provider="ollama", model="llava:13b")
        slots = ws.receive_json()["slots"]
        assert (slots[1]["label"], slots[1]["remote"]) == ("ollama:llava:13b", False)
        send(ws, type="set_agent", slot="A", provider="openai", model="gpt-x")
        assert ws.receive_json()["text"] == "OPENAI_API_KEY is not set"
        send(ws, type="set_agent", slot="A", provider="demo", model="bad model!")
        assert ws.receive_json()["type"] == "error"
        send(ws, type="set_agent", slot="C", provider="demo")
        assert ws.receive_json()["type"] == "error"
        send(ws, type="set_agent", slot="A", provider="demo", model="frame-facts", enabled=False)
        assert ws.receive_json()["slots"][0]["enabled"] is False


def test_list_models(compare_app):
    compare_app.options = {}
    with compare_app.client.websocket_connect("/ws") as ws:
        ws.receive_json()
        send(ws, type="list_models", provider="anthropic")
        assert ws.receive_json() == {"type": "models", "provider": "anthropic", "models": [],
                                     "error": "ANTHROPIC_API_KEY is not set"}
        send(ws, type="list_models", provider="demo", refresh="yes")
        assert ws.receive_json()["type"] == "error"


def test_set_source_and_webcam_confirm(compare_app, monkeypatch):
    with compare_app.client.websocket_connect("/ws") as ws:
        ws.receive_json()
        send(ws, type="set_source", source_key="sample:2")
        assert ws.receive_json()["source"] == {"key": "sample:2", "label": "Sample 2 (street_view_2.mp4)",
                                               "epoch": 1, "error": None}
        send(ws, type="set_source", source_key="../etc/passwd")
        assert ws.receive_json()["type"] == "error"

        monkeypatch.setenv("OPENAI_API_KEY", SENTINEL)
        send(ws, type="set_agent", slot="A", provider="openai", model="gpt-x")
        assert ws.receive_json()["slots"][0]["destination"] == "api.openai.com"
        send(ws, type="set_source", source_key="webcam")
        refusal = ws.receive_json()
        assert refusal["text"] == CONFIRM_WEBCAM and refusal["confirm"] == "webcam_remote"
        assert compare_app.hub.spec.source_key == "sample:2"
        send(ws, type="set_source", source_key="webcam", confirm=True)
        assert ws.receive_json()["source"]["key"] == "webcam"
        # The other way round: the webcam is on, a remote agent needs the same confirmation.
        send(ws, type="set_agent", slot="B", provider="openai", model="gpt-y")
        assert ws.receive_json()["confirm"] == "webcam_remote"
        send(ws, type="set_agent", slot="B", provider="openai", model="gpt-y", confirm=True)
        assert ws.receive_json()["slots"][1]["remote"] is True


@pytest.mark.parametrize("raw", ["{not json", "[1, 2]", '{"type": "toggle_stream"}', '{"role": "user", "text": "x"}',
                                 json.dumps({"type": "update", "text": "x" * 2001})])
def test_malformed_frame_gets_an_error_and_keeps_the_socket(compare_app, raw):
    with compare_app.client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(raw)
        assert ws.receive_json()["type"] == "error"
        send(ws, type="control", action="pause")
        assert ws.receive_json()["type"] == "state"
    assert compare_app.prompt.value == "describe"


def test_a_round_over_the_socket(compare_app):
    clip = ClipSample(3, "sample:1", 0, "Sample 1", 1.0, 1.5, 0, "c.mp4", (b"x",) * 5, (1.0, 1.1, 1.2, 1.3, 1.4))
    with compare_app.client.websocket_connect("/ws") as ws:
        ws.receive_json()
        send(ws, type="control", action="step")
        ws.receive_json()
        compare_app.session.on_clip(clip)
        frames = receive_until(ws, "round_end")
    kinds = [f["type"] for f in frames if f["type"] != "state"]
    assert kinds == ["round", "result", "result", "round_end"]
    round_frame = next(f for f in frames if f["type"] == "round")
    assert (round_frame["clip_id"], round_frame["start_s"], round_frame["prompt"]) == (3, 1.0, "describe")
    results = {f["slot"]: f for f in frames if f["type"] == "result"}
    assert results["A"]["text"] == "fake answer" and results["A"]["stale"] is False
    assert results["A"]["metrics"]["cost_usd"] is None
    assert eventually(lambda: compare_app.deleted == ["c.mp4"])  # after round_end, in the round's thread


def test_paused_clips_are_deleted_without_a_round(compare_app):
    clip = ClipSample(4, "sample:1", 0, "Sample 1", 0.0, 0.5, 0, "d.mp4", (b"x",) * 5, (0.0,) * 5)
    compare_app.session.on_clip(clip)
    assert compare_app.deleted == ["d.mp4"] and compare_app.session.busy_drops == 0
    assert all(agent.calls == 0 for agent in compare_app.agents.values())


def test_last_disconnect_pauses(compare_app):
    with compare_app.client.websocket_connect("/ws") as ws:
        ws.receive_json()
        send(ws, type="control", action="run")
        ws.receive_json()
    assert eventually(lambda: compare_app.session.spend.snapshot()["pause_reason"] == "disconnect")
    assert web_engine.active_connections == []


def test_no_frame_ever_carries_a_key(compare_app, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SENTINEL)
    monkeypatch.setenv("VLM_COMPAT_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("VLM_COMPAT_API_KEY", SENTINEL)
    compare_app.options = {"raise_on": {1: RuntimeError(f"rejected key {SENTINEL}")},
                           "list_error": ConnectionError(f"bad key {SENTINEL}")}
    clip = ClipSample(5, "sample:1", 0, "Sample 1", 0.0, 0.5, 0, "e.mp4", (b"x",) * 5, (0.0,) * 5)
    seen = []
    with compare_app.client.websocket_connect("/ws") as ws:
        seen.append(ws.receive_text())
        for frame in ({"type": "set_agent", "slot": "A", "provider": "openai", "model": "gpt-x"},
                      {"type": "set_agent", "slot": "B", "provider": "compat", "model": "m"},
                      {"type": "list_models", "provider": "openai"},
                      {"type": "control", "action": "step"}):
            ws.send_text(json.dumps(frame))
            seen.append(ws.receive_text())
        compare_app.session.on_clip(clip)
        while '"round_end"' not in seen[-1]:
            seen.append(ws.receive_text())
    everything = "\n".join(seen)
    assert '"status": "error"' in everything and "Internal error (RuntimeError)" in everything
    assert SENTINEL not in everything


def test_connection_removed_after_close(compare_app):
    with compare_app.client.websocket_connect("/ws") as ws:
        ws.receive_json()
    assert web_engine.active_connections == []


def test_server_caps_raw_frame_size():
    # TestClient bypasses uvicorn, so assert the cap is wired into the real server config.
    config = web_engine.make_server().config
    assert config.ws_max_size == web_engine.WS_MAX_BYTES
    assert web_engine.WS_MAX_BYTES < 16 * 1024 * 1024
    assert web_engine.WS_MAX_BYTES > web_engine.MAX_TEXT_CHARS * 4  # UTF-8 worst case still fits
