'''Provides a thread-safe Prompt object.'''

from threading import Lock
import time

from loguru import logger

from utils.json_management import append_dict_to_json


class Prompt:
    def __init__(self):
        # Leave default as something obnoxious,
        # so errors are easily spotted.
        self._prompt = "Write a funny sentence!"
        self._lock = Lock()

    def get(self):
        with self._lock:
            return self._prompt

    def set(self, new_prompt: str):
        with self._lock:
            if new_prompt == self._prompt:
                logger.debug("Duplicate prompt detected — not recording.")
                return

            self._prompt = new_prompt

        logger.info(f"New prompt: [ {new_prompt} ]")

        data = {
            'prompt': new_prompt,
            'time': get_formatted_time()
        }

        append_dict_to_json(file_path='./prompting/history.json', new_data=data)


def get_formatted_time() -> str:
    _ts = time.localtime()

    month = f"{_ts.tm_mon:02d}"
    day = f"{_ts.tm_mday:02d}"
    hour = f"{_ts.tm_hour:02d}"
    minute = f"{_ts.tm_min:02d}"

    return f"{month}.{day} | {hour}:{minute}"
