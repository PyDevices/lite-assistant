"""
Base interface for action engines, which recognize what people are doing
across a video's frames, one frame at a time.
"""
from typing import Any, List

import numpy as np


class ActionEngine:
    """
    Abstract base class for all action engines. FPS is the frame rate it
    expects; labels are the actions it knows.
    """
    MODEL_ID = ""
    FPS = 5
    labels: List[str] = []

    def get_model_id(self) -> str:
        return self.MODEL_ID

    def start(self) -> Any:
        """ The memory of a new video, before its first frame. """
        raise NotImplementedError

    def watch(self, memory: Any, frame) -> np.ndarray:
        """
        Takes the video's next frame, a PIL image, into its memory, and
        returns how likely each label is, from 0 to 1, given every frame so far.
        """
        raise NotImplementedError

    @classmethod
    def setup(cls):
        """
        Performs first-time setup, such as downloading default models.
        """
        pass
