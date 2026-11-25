import json

from utils.json_management import append_dict_to_json


def test_creates_file(tmp_path):
    f = tmp_path / "h.json"
    append_dict_to_json(str(f), {"a": 1})
    assert json.loads(f.read_text()) == [{"a": 1}]


def test_appends_and_dedupes_last_entry(tmp_path):
    f = tmp_path / "h.json"
    for item in ({"a": 1}, {"a": 1}, {"a": 2}):
        append_dict_to_json(str(f), item)
    assert json.loads(f.read_text()) == [{"a": 1}, {"a": 2}]


def test_empty_list_file_does_not_raise(tmp_path):
    f = tmp_path / "h.json"
    f.write_text("[]")
    append_dict_to_json(str(f), {"a": 1})
    assert json.loads(f.read_text()) == [{"a": 1}]


def test_corrupt_file_starts_fresh(tmp_path):
    f = tmp_path / "h.json"
    f.write_text("{not json")
    append_dict_to_json(str(f), {"a": 1})
    assert json.loads(f.read_text()) == [{"a": 1}]
