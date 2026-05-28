import os
import shutil
from typing import Iterable


def safe_remove_file(path: str) -> None:
    if not path:
        return
    try:
        if os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass


def safe_remove_dir(path: str) -> None:
    if not path:
        return
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def cleanup_paths(paths: Iterable[str]) -> None:
    for path in paths:
        if not path:
            continue
        if os.path.isdir(path):
            safe_remove_dir(path)
        else:
            safe_remove_file(path)
