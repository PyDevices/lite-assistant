"""
Base interface for pitch engines, which hear the note a voice or an
instrument plays.
"""
from typing import List, Tuple

import numpy as np


class PitchEngine:
    """
    Abstract base class for all pitch engines. RATE is the sample rate it
    takes, and HOP how far apart, in seconds, it hears the pitch.
    """
    MODEL_ID = ""
    RATE = 16000
    HOP = 0.05

    def get_model_id(self) -> str:
        return self.MODEL_ID

    def pitches(self, samples: np.ndarray) -> List[Tuple[float, float]]:
        """
        The pitch every HOP seconds of mono audio at RATE, as its
        frequency in Hz and how sure the engine is that a note is playing,
        from 0 to 1.
        """
        raise NotImplementedError

    @classmethod
    def setup(cls):
        """
        Performs first-time setup, such as downloading default models.
        """
        pass
