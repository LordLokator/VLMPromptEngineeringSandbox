from types import NoneType
import os


def full_path(path: str) -> str:
    """Returns a sanitized, absolute path from absolute, relative or ~/ like paths.

    Args:
        path (str)

    Returns:
        str: Absolute path.
    """
    # Expand ~ to the user's home directory
    expanded_path = os.path.expanduser(path)
    # Convert to absolute path
    absolute_path = os.path.abspath(expanded_path)
    return str(absolute_path)


def makedir(path: str) -> NoneType:
    path = full_path(path)
    if not os.path.exists(path):
        os.mkdir(path)
