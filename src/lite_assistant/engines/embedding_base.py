"""
Base interfaces and definitions for embedding engines.
"""
from typing import Any, Dict, List, Optional, Union

class EmbeddingEngine:
    """
    Abstract base class for all embedding engines.
    """

    def get_model_id(self) -> str:
        """
        Returns the ID of the model this engine has loaded.
        """
        raise NotImplementedError

    def get_dimensions(self) -> List[int]:
        """
        Returns the vector sizes embed() can produce, smallest first; the last is the model's full size.
        """
        raise NotImplementedError

    def embed(self, item: Union[str, Dict[str, Any]], dimensions: Optional[int] = None) -> List[float]:
        """
        Embeds one input: a string, or an OpenAI content part (image_url with a
        data: URL, or input_audio). Returns a unit-length vector of the given
        size, or of the full size.
        """
        raise NotImplementedError

    @classmethod
    def setup(cls):
        """
        Performs first-time setup, such as downloading default models.
        """
        pass
