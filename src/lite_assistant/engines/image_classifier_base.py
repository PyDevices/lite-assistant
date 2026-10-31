"""
Base interface for image classification engines, which name what a photo
shows from a fixed list, such as a plant's species.
"""
from typing import List, Tuple


class ImageClassifierEngine:
    """
    Abstract base class for all image classification engines. TASK says
    what it names, such as "plant".
    """
    MODEL_ID = ""
    TASK = ""

    def get_model_id(self) -> str:
        return self.MODEL_ID

    def classify(self, image: bytes, top: int = 5) -> List[Tuple[str, float]]:
        """
        The top likeliest labels for a photo, each with how likely it is,
        from 0 to 1, likeliest first. Raises ValueError if it isn't an image.
        """
        raise NotImplementedError

    @classmethod
    def setup(cls):
        """
        Performs first-time setup, such as downloading default models.
        """
        pass
