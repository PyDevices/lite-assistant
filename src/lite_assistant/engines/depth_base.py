"""
Base interfaces and definitions for depth engines.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DepthMap:
    """
    How far each point in a photo is from the camera, in meters, in a map
    with the photo's aspect ratio. NaN marks points with no estimate, such as
    the sky.
    """
    distances: np.ndarray
    # The photo's horizontal field of view in degrees, if the engine estimates it
    horizontal_fov: Optional[float] = None


class DepthEngine:
    """
    Abstract base class for all depth engines.
    """

    def get_model_id(self) -> str:
        """
        Returns the ID of the model this engine has loaded.
        """
        raise NotImplementedError

    def estimate(self, image: bytes) -> DepthMap:
        """
        Estimates the distances in a photo, given as JPEG, PNG or another
        format Pillow reads. Raises ValueError if it isn't an image.
        """
        raise NotImplementedError

    @classmethod
    def setup(cls):
        """
        Performs first-time setup, such as downloading default models.
        """
        pass
