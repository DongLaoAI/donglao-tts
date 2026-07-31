"""Small filesystem helpers shared by data-preparation commands."""

import os
import tempfile
from contextlib import contextmanager


@contextmanager
def atomic_text_writer(path):
    """Write UTF-8 text to a sibling temporary file and atomically replace ``path``."""
    destination = os.path.abspath(path)
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".donglao-", suffix=".tmp", dir=parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
