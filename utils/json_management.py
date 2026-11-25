import json
import os
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger


def write_json_atomic(path: str | Path, data: Any) -> None:
    """Write JSON to a temp file in the same directory, then os.replace it over the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def append_dict_to_json(file_path: str, new_data: dict) -> None:
    """Append new_data to a JSON list file unless it equals the last entry. A corrupt file is reset."""
    data: list = []
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            data = loaded if isinstance(loaded, list) else [loaded]
        except json.JSONDecodeError:
            logger.info("JSON file is empty or invalid. Starting fresh.")

    if data and data[-1] == new_data:
        logger.debug("Prompt remains the same, history not updated.")
        return

    data.append(new_data)
    write_json_atomic(file_path, data)
    logger.info(f"Appended data to JSON: {new_data}")
