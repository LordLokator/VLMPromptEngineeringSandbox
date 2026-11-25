from types import NoneType
import os

from pathlib import Path

from loguru import logger

# Repository root; relative paths are resolved against it, never against the cwd.
BASE_DIR = Path(__file__).resolve().parent.parent


def full_path(path: str) -> str:
    """Returns a sanitized, absolute path from absolute, relative or ~/ like paths.

    Relative paths resolve against BASE_DIR (the repository root), not the cwd.

    Args:
        path (str)

    Returns:
        str: Absolute path.
    """
    # Expand ~ to the user's home directory
    expanded_path = os.path.expanduser(path)
    # Convert to absolute path (relative ones are anchored at the repo root)
    absolute_path = os.path.abspath(os.path.join(BASE_DIR, expanded_path))
    return str(absolute_path)


def makedir(path: str) -> NoneType:
    path = full_path(path)
    if not os.path.exists(path):
        os.mkdir(path)


def delete_files(files: list[str]):
    for file in files:
        try:
            os.remove(file)
            logger.debug(f"Deleted {file}.")
        except Exception as e:
            logger.error(f"Error deleting {file}: {e}")
