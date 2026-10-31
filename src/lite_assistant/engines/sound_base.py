"""
Base interfaces and definitions for sound engines, which name the sounds in
audio, such as a doorbell or a dog barking.
"""
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Sounds:
    """
    How sure the model is of each kind of sound, in each window of the audio:
    scores has a row per window and a column per label, from 0 to 1. Several
    kinds can score high at once, such as speech over music.
    """
    labels: List[str]
    scores: np.ndarray
    # Each window's start and end, in seconds
    windows: List[tuple]
    # The audio's length in seconds
    duration: float


class SoundEngine:
    """
    Abstract base class for all sound engines.
    """

    def get_model_id(self) -> str:
        """
        Returns the ID of the model this engine has loaded.
        """
        raise NotImplementedError

    def classify(self, audio: bytes) -> Sounds:
        """
        The sounds in audio, given as a WAV, FLAC, MP3 or another file libsndfile
        reads. Raises ValueError if it isn't audio.
        """
        raise NotImplementedError

    @classmethod
    def setup(cls):
        """
        Performs first-time setup, such as downloading default models.
        """
        pass
