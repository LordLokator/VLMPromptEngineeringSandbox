import re
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "src" / "script.js"


def test_no_interpolated_innerhtml():
    bad = [
        line for line in SCRIPT.read_text(encoding="utf-8").splitlines()
        if re.search(r"\.innerHTML\s*\+?=", line) and ("${" in line or "+" in line.split("=", 1)[1])
    ]
    assert bad == []


def test_message_handler_covers_all_shapes():
    src = SCRIPT.read_text(encoding="utf-8")
    assert "data.type" in src
    assert "status === \"ERROR\"" not in src, "legacy shapes are gone"
    assert "data.output" not in src
    assert src.count("let reconnectTimer") == 1 and "clearTimeout(reconnectTimer)" in src
    onerror = src[src.index("ws.onerror"):src.index("ws.onclose")]
    assert "addChatMessage" not in onerror
    assert src.count("const presetBar") == 1
    assert "addChatMessage(\"System\"" not in src


def test_muted_notice_supported():
    src = SCRIPT.read_text(encoding="utf-8")
    assert "data.muted === true" in src
    assert ".chat-message.muted" in (SCRIPT.parent / "style.css").read_text(encoding="utf-8")


def test_error_style_exists():
    css = (SCRIPT.parent / "style.css").read_text(encoding="utf-8")
    assert ".chat-message.error" in css


def test_websocket_client_is_robust():
    src = SCRIPT.read_text(encoding="utf-8")
    assert "wss" in src
    assert "ws://${" not in src
    assert "onclose" in src and "onopen" in src and "onerror" in src
    sends = [ln for ln in src.splitlines() if "ws.send(" in ln]
    assert len(sends) == 1, "all sends must go through the send() helper"


def test_save_flow_clears_after_success():
    src = SCRIPT.read_text(encoding="utf-8")
    assert "AbortController" in src
    start = src.index("saveBtn.onclick")
    handler = src[start:]
    assert "catch" in handler
    check = handler.index('result.status !== "ok"')
    clear = handler.index('msgInput.value = ""')
    assert clear > check
    assert src.count("fetch(") == 1, "all requests go through fetchWithTimeout"


def test_resample_slider_and_payload():
    html = (SCRIPT.parent / "page.html").read_text(encoding="utf-8")
    js = SCRIPT.read_text(encoding="utf-8")
    assert 'id="fpsSlider"' in html and 'min="1" max="10" step="1"' in html
    assert 'min="0.01"' in html  # temperature slider must not allow 0
    assert "resample_fps" in js


def test_page_has_no_unprotected_external_assets():
    html = (SCRIPT.parent / "page.html").read_text(encoding="utf-8")
    tags = re.findall(r"<(?:script|link)[^>]*>", html)
    bad = [t for t in tags if re.search(r"(?:src|href)=[\"']?(?:https?:)?//", t) and "integrity=" not in t]
    assert bad == []
    assert (SCRIPT.parent / "vendor" / "bootstrap.min.css").is_file()
    assert (SCRIPT.parent / "vendor" / "bootstrap.LICENSE").is_file()
