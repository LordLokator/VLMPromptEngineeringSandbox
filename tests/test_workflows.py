"""Structure checks for GitHub workflow and Dependabot files (CI cannot run locally)."""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

GITHUB = Path(__file__).resolve().parent.parent / ".github"
WORKFLOWS = sorted((GITHUB / "workflows").glob("*.yml"))
# A full 40-char sha or a version-like tag (v7, v7.0.1); branches such as main are rejected.
PINNED = re.compile(r"^(?:[0-9a-f]{40}|v\d+(?:\.\d+){0,2})$")


def _load(path: Path) -> dict[Any, Any]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    return doc


def test_workflows_exist() -> None:
    assert WORKFLOWS


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_structure(path: Path) -> None:
    doc = _load(path)
    # PyYAML (YAML 1.1) loads the key `on` as boolean True.
    assert True in doc or "on" in doc
    jobs = doc.get("jobs")
    assert jobs
    for name, job in jobs.items():
        assert "runs-on" in job, name
        assert job.get("steps"), name
        for step in job["steps"]:
            uses = step.get("uses")
            if uses is None or uses.startswith("./"):
                continue
            action, _, ref = uses.partition("@")
            assert ref, f"{name}: {uses} has no ref"
            assert PINNED.match(ref), f"{name}: {uses} is not pinned to a tag or sha"


def test_dependabot_structure() -> None:
    doc = _load(GITHUB / "dependabot.yml")
    assert doc.get("version") == 2
    assert doc.get("updates")


def _ignored(doc: dict[Any, Any]) -> dict[str, dict[Any, Any]]:
    pip = next(u for u in doc["updates"] if u["package-ecosystem"] == "pip")
    return {e["dependency-name"]: e for e in pip.get("ignore", [])}


def test_dependabot_ignores_gpu_stack_entirely() -> None:
    # An `update-types` filter would let a minor bump (torch 2.2 -> 2.15) through the group.
    ignored = _ignored(_load(GITHUB / "dependabot.yml"))
    for name in ("torch", "transformers", "numpy"):
        assert name in ignored, name
        assert "update-types" not in ignored[name], name


def test_no_action_uses_a_branch_ref() -> None:
    for path in WORKFLOWS:
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.search(r"uses:\s*\S+@(\S+)", line)
            if m:
                assert PINNED.match(m.group(1)), f"{path.name}: {line.strip()}"


def test_security_workflow_has_osv_job() -> None:
    assert "osv-scanner" in _load(GITHUB / "workflows" / "security.yml")["jobs"]


def test_ci_matrix_covers_311_and_312() -> None:
    jobs = _load(GITHUB / "workflows" / "ci.yml")["jobs"]
    versions = [v for job in jobs.values() for v in job.get("strategy", {}).get("matrix", {}).get("python-version", [])]
    assert {"3.11", "3.12"} <= set(versions)
