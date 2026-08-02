"""donglao-tts public package metadata."""

from importlib.metadata import PackageNotFoundError, version

from donglao_tts.api import DongLaoTTS

try:
    __version__ = version("donglao-tts")
except PackageNotFoundError:
    __version__ = "0.1.6"

__all__ = ["DongLaoTTS", "__version__"]
