"""
Base interface for segmentation engines, which find where things are in a
photo from what they're called.
"""
from typing import List

import numpy as np


class SegmentEngine:
    """
    Abstract base class for all segmentation engines.
    """
    MODEL_ID = ""

    def get_model_id(self) -> str:
        return self.MODEL_ID

    def segment(self, image: bytes, prompts: List[str]) -> List[np.ndarray]:
        """
        For each prompt, such as "the cup", a map of how likely each point of
        the photo is to be it, from 0 to 1. A map spans the whole photo,
        whatever its shape: a point's row and column, as fractions of the
        map's height and width, are the same fractions of the photo's.
        Raises ValueError if it isn't an image.
        """
        raise NotImplementedError

    @classmethod
    def setup(cls):
        """
        Performs first-time setup, such as downloading default models.
        """
        pass
