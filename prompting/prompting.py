# prompting/prompting.py
from threading import Lock

from loguru import logger


class Prompt:
    _prompt = "Describe the scene."
    _lock = Lock()

    @classmethod
    def get(cls):
        with cls._lock:
            return cls._prompt

    @classmethod
    def set(cls, new_prompt: str):
        logger.info(f"New prompt: [ {new_prompt} ]")
        with cls._lock:
            cls._prompt = new_prompt
