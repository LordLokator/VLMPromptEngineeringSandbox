# prompting/prompting.py
from threading import Lock

from loguru import logger


class Prompt:
    def __init__(self):
        self._prompt = "Write a funny sentence!"
        self._lock = Lock()

    def get(self):
        with self._lock:
            return self._prompt

    def set(self, new_prompt: str):
        logger.info(f"New prompt: [ {new_prompt} ]")
        with self._lock:
            self._prompt = new_prompt
