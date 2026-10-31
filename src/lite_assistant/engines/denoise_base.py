"""
Base interface for noise removal engines, which take the background noise
out of speech, such as a fan or traffic heard by a cheap microphone.
"""
import numpy as np


class DenoiseEngine:
    """
    Abstract base class for all noise removal engines. RATE is the sample
    rate they work at.
    """
    MODEL_ID = ""
    RATE = 16000

    def get_model_id(self) -> str:
        return self.MODEL_ID

    def denoise(self, samples: np.ndarray) -> np.ndarray:
        """ Mono float samples at RATE, with the noise taken out, the same length. """
        raise NotImplementedError

    @classmethod
    def setup(cls):
        """
        Performs first-time setup, such as downloading default models.
        """
        pass
