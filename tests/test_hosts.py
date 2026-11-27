import json
import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.websockets import WebSocketDisconnect

import web_engine
from utils.file_management import BASE_DIR
from utils.hosts import LOOPBACK_HOSTS, parse_allowed_hosts, require_extra_hosts

FOREIGN = ["evil.example", "evil.example:8000", "localhost.evil.example", "127.0.0.2", "[::2]:8000", "www.localhost"]
LOOPBACK = ["localhost", "localhost:8000", "LOCALHOST:8000", "127.0.0.1", "127.0.0.1:8000", "[::1]", "[::1]:8000"]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(web_engine.app.state, "prompt", None, raising=False)
    monkeypatch.setattr(web_engine.app.state, "settings", None, raising=False)
    web_engine.active_connections.clear()
    return TestClient(web_engine.app)


def ws_round_trip(client, headers):
    with client.websocket_connect("/ws", headers=headers) as ws:
        ws.send_text(json.dumps({"role": "other"}))
        assert ws.receive_json()["type"] == "error"


@pytest.mark.parametrize("host", FOREIGN + [""])
def test_foreign_host_rejected_for_http_and_static_files(client, host):
    for path in ("/", "/preset_buttons", "/src/script.js"):
        assert client.get(path, headers={"host": host}).status_code == 400
    res = client.post("/save_preset", json={"label": "a", "text": "b"}, headers={"host": host})
    assert res.status_code == 400


@pytest.mark.parametrize("host", FOREIGN)
def test_foreign_host_rejected_for_websocket_even_with_matching_origin(client, host):
    # What a rebound page sends: Origin and Host agree, so only the allowlist can stop it.
    with pytest.raises(WebSocketDisconnect):
        ws_round_trip(client, {"host": host, "origin": f"http://{host}"})
    assert web_engine.active_connections == []


@pytest.mark.parametrize("host", LOOPBACK)
def test_loopback_names_accepted_for_http_and_websocket(client, host):
    assert client.get("/", headers={"host": host}).status_code == 200
    ws_round_trip(client, {"host": host, "origin": f"http://{host}"})


def test_testserver_is_allowed_only_by_the_test_configuration():
    (middleware,) = [m for m in web_engine.app.user_middleware if m.cls is TrustedHostMiddleware]
    assert middleware.kwargs["allowed_hosts"] == [*LOOPBACK_HOSTS, "testserver"]
    assert os.environ["VLM_ALLOWED_HOSTS"] == "testserver"
    assert "testserver" not in LOOPBACK_HOSTS and parse_allowed_hosts("") == []


def run_python(code, **env):
    """A fresh interpreter, because the app reads its configuration at import."""
    clean = {k: v for k, v in os.environ.items() if not k.startswith("VLM_")}
    return subprocess.run(
        [sys.executable, "-c", code], cwd=BASE_DIR, env=clean | env, capture_output=True, text=True, timeout=60
    )


def test_default_configuration_rejects_testserver():
    code = (
        "from fastapi.testclient import TestClient; import web_engine; "
        "print(TestClient(web_engine.app).get('/').status_code, "
        "TestClient(web_engine.app, base_url='http://localhost').get('/').status_code)"
    )
    result = run_python(code)
    assert result.stdout.split()[-2:] == ["400", "200"], result.stderr


def test_non_loopback_without_allowed_hosts_stops_with_a_clear_error():
    result = run_python("import web_engine; web_engine.make_server()", VLM_HOST="0.0.0.0")  # noqa: S104 - configuration value under test
    assert result.returncode == 1
    assert "set VLM_ALLOWED_HOSTS" in result.stderr and "Traceback" not in result.stderr


def test_non_loopback_with_allowed_hosts_serves_that_name():
    code = (
        "from fastapi.testclient import TestClient; import web_engine; web_engine.make_server(); "
        "print(*(TestClient(web_engine.app, base_url=u).get('/').status_code "
        "for u in ('http://192.168.1.20:8000', 'http://localhost', 'http://evil.example')))"
    )
    result = run_python(code, VLM_HOST="0.0.0.0", VLM_ALLOWED_HOSTS="192.168.1.20")  # noqa: S104 - configuration value under test
    assert result.stdout.split()[-3:] == ["200", "200", "400"], result.stderr


def test_parse_allowed_hosts():
    assert parse_allowed_hosts(" MyBox.local, 192.168.1.20 ,,fd00::5,[FD00::6], *.lan.example ") == [
        "mybox.local",
        "192.168.1.20",
        "[fd00::5]",
        "[fd00::6]",
        "*.lan.example",
    ]


@pytest.mark.parametrize(
    "value", ["*", "*.", "foo*", "a.*.b", "myhost:8000", "http://myhost", "[::1]:8000", "[nonsense]", "a b"]
)
def test_parse_allowed_hosts_refuses_bad_entries(value):
    with pytest.raises(SystemExit, match="VLM_ALLOWED_HOSTS"):
        parse_allowed_hosts(value)


def test_require_extra_hosts():
    require_extra_hosts("0.0.0.0", ["mybox"])  # noqa: S104 - configuration value under test
    with pytest.raises(SystemExit, match="VLM_ALLOWED_HOSTS"):
        require_extra_hosts("0.0.0.0", [])  # noqa: S104 - configuration value under test
