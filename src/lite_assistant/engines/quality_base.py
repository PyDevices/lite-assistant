"""
Base interface for image quality engines, which score how good a photo is.
"""
from typing import Dict


class QualityEngine:
    """
    Abstract base class for all image quality engines.
    """
    MODEL_ID = ""

    def get_model_id(self) -> str:
        return self.MODEL_ID

    def score(self, image: bytes) -> Dict[str, float]:
        """
        Scores from 1 to 10, by what each judges: "technical" for sharpness,
        noise and exposure, and "aesthetic" for how pleasing it is. Raises
        ValueError if it isn't an image.
        """
        raise NotImplementedError

    @classmethod
    def setup(cls):
        """
        Performs first-time setup, such as downloading default models.
        """
        pass
