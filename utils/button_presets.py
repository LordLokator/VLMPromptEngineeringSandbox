# button_presets.py
import json
from pathlib import Path
from loguru import logger

PRESETS_FILE = Path("presets/button_presets.json")
USER_PRESETS_FILE = Path("presets/user_presets.json")
USER_PRESETS_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_presets() -> list:
    if not PRESETS_FILE.exists():
        logger.warning("No presets file found.")
        return []
    with open(PRESETS_FILE, "r") as f:
        return json.load(f)


def save_presets(presets):
    with open(PRESETS_FILE, "w") as f:
        json.dump(presets, f, indent=2)


def load_user_presets() -> list:
    if not USER_PRESETS_FILE.exists():
        return []
    with open(USER_PRESETS_FILE, "r") as f:
        return json.load(f)


def save_user_preset(label, text):
    new_preset = {"label": label, "text": text}
    presets = load_user_presets()
    presets.append(new_preset)

    with open(USER_PRESETS_FILE, "w") as f:
        json.dump(presets, f, indent=2)
    logger.info(f"Saved new user preset: {label}")
