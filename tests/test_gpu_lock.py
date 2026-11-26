"""requirements-gpu.txt exists so scanners can read the `gpu` extra; it must not drift from pyproject.toml."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9._-]+", requirement)
    assert match, requirement
    return re.sub(r"[-_.]+", "-", match.group()).lower()


def _locked() -> dict[str, str]:
    lines = (ROOT / "requirements-gpu.txt").read_text(encoding="utf-8").splitlines()
    pins = [line.partition("==") for line in lines if re.match(r"[A-Za-z0-9]", line)]
    return {_name(name): version.strip() for name, _, version in pins}


def _project() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_lock_has_every_direct_dependency() -> None:
    project = _project()
    direct = project["dependencies"] + project["optional-dependencies"]["gpu"]
    assert {_name(r) for r in direct} <= _locked().keys()


def test_lock_matches_exact_gpu_pins() -> None:
    pins = dict(r.split("==") for r in _project()["optional-dependencies"]["gpu"] if "==" in r)
    assert {"torch", "transformers", "numpy"} <= pins.keys()
    locked = _locked()
    assert {name: locked.get(name) for name in pins} == pins
