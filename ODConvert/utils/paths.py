from pathlib import Path


def valid_path(path: str) -> bool:
    """
    Check if the given path is valid.
    :param path: The path to check.
    :return: True if the path is valid, False otherwise.
    """
    return Path(path).exists()
