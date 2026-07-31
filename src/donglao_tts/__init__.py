"""donglao-tts public package metadata."""

from importlib.metadata import version

from donglao_tts.api import DongLaoTTS

__version__ = version("donglao-tts")

__all__ = ["DongLaoTTS", "__version__"]
