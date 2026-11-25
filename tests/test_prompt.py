import json
import threading

import pytest

from prompting import prompting
from prompting.prompting import Prompt


@pytest.fixture
def prompt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(prompting, "HISTORY_FILE", tmp_path / "prompting" / "history.json")
    return Prompt()


def history(tmp_path):
    return json.loads((tmp_path / "prompting" / "history.json").read_text())


def test_set_and_get(prompt):
    prompt.set("describe the scene")
    assert prompt.get() == "describe the scene"


def test_equal_prompt_not_recorded_twice(prompt, tmp_path):
    prompt.set("same")
    prompt.set("same")
    assert [h["prompt"] for h in history(tmp_path)] == ["same"]


def test_concurrent_sets_keep_a_valid_value(prompt):
    values = [f"p{i}" for i in range(8)]
    threads = [threading.Thread(target=prompt.set, args=(v,)) for v in values]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert prompt.get() in values


def test_history_write_failure_keeps_prompt(prompt, monkeypatch):
    import os

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    prompt.set("still set")
    assert prompt.get() == "still set"
