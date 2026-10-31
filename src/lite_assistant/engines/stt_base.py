"""
Base interfaces and definitions for Speech-To-Text (STT) engines.
"""
from typing import Optional

class STTEngine:
    """
    Abstract base class for all Speech-to-Text engines.
    """
    def transcribe(self, audio_bytes: bytes, language: Optional[str] = None) -> str:
        """
        Transcribes audio bytes (e.g. WAV format) into text. language, an ISO
        639-1 code such as "en", says what's spoken; an engine that hears only
        one language ignores it.
        """
        raise NotImplementedError
        
    @classmethod
    def setup(cls):
        """
        Performs first-time setup, such as downloading default models.
        """
        pass
