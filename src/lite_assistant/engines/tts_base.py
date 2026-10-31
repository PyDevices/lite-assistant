"""
Base interfaces and definitions for Text-To-Speech (TTS) engines.
"""
from typing import Iterator, Dict, Any, List

class TTSEngine:
    """
    Abstract base class for all TTS engines.
    """
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        Returns a list of dictionaries, each containing at least 'id' and 'name'.
        """
        raise NotImplementedError
        
    def get_default_voice(self) -> str:
        """
        Returns the ID of the default voice for this engine.
        """
        raise NotImplementedError
        
    def generate_pcm(self, text: str, voice: str) -> Iterator[bytes]:
        """
        Generates and yields raw 16-bit 24kHz Mono PCM audio chunks.
        """
        raise NotImplementedError
        
    @classmethod
    def setup(cls):
        """
        Performs first-time setup, such as downloading default models or voices.
        """
        pass
