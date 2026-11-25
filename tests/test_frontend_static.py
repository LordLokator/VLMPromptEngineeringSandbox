import re
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "src" / "script.js"


@pytest.mark.xfail(strict=True, reason="F3")
def test_no_interpolated_innerhtml():
    bad = [
        line for line in SCRIPT.read_text(encoding="utf-8").splitlines()
        if re.search(r"\.innerHTML\s*\+?=", line) and ("${" in line or "+" in line.split("=", 1)[1])
    ]
    assert bad == []
