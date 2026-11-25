# button_presets.py
import json
import threading
from loguru import logger

from utils.file_management import BASE_DIR
from utils.json_management import write_json_atomic

PRESETS_FILE = BASE_DIR / "presets" / "button_presets.json"
USER_PRESETS_FILE = BASE_DIR / "presets" / "user_presets.json"
_save_lock = threading.Lock()


def load_presets() -> list:
    if not PRESETS_FILE.exists():
        logger.warning("No presets file found.")
        return []
    with open(PRESETS_FILE, "r") as f:
        return json.load(f)


def load_user_presets() -> list:
    if not USER_PRESETS_FILE.exists():
        return []
    with open(USER_PRESETS_FILE, "r") as f:
        return json.load(f)


def save_user_preset(label, text):
    new_preset = {"label": label, "text": text}
    with _save_lock:
        presets = load_user_presets()
        presets.append(new_preset)
        write_json_atomic(USER_PRESETS_FILE, presets)  # creates the directory if needed
    logger.info(f"Saved new user preset: {label}")
