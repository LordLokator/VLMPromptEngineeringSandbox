# button_presets.py
import json
from pathlib import Path
from loguru import logger

PRESETS_FILE = Path("src/button_presets.json")


def load_presets():
    if not PRESETS_FILE.exists():
        logger.warning("No presets file found.")
        return []
    with open(PRESETS_FILE, "r") as f:
        return json.load(f)


def save_presets(presets):
    with open(PRESETS_FILE, "w") as f:
        json.dump(presets, f, indent=2)
