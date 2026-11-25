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
    assert "status === \"ERROR\"" in src
    assert "data.type" in src
    assert "typeof data.output === \"string\"" in src
    assert "addChatMessage(\"System\"" not in src


def test_error_style_exists():
    css = (SCRIPT.parent / "style.css").read_text(encoding="utf-8")
    assert ".chat-message.error" in css
