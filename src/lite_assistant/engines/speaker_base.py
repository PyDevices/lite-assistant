"""
Base interface for speaker engines, which tell voices apart: an embedding
of a voice, close to the same voice's and far from others', and when each
of the few voices in a short window is speaking.
"""
import numpy as np


class SpeakerEngine:
    """
    Abstract base class for all speaker engines. RATE is the sample rate
    they work at; WINDOW is how many seconds activity takes at once, and
    FRAME how long each of its frames is.
    """
    MODEL_ID = ""
    RATE = 16000
    WINDOW = 10.0
    FRAME = 0.017
    # Two voices' embeddings closer than this, as the distance between unit vectors, are one voice
    SAME = 0.7

    def get_model_id(self) -> str:
        return self.MODEL_ID

    def embed(self, samples: np.ndarray) -> np.ndarray:
        """ A voice's embedding, a unit vector, from mono float samples at RATE of it speaking alone. """
        raise NotImplementedError

    def activity(self, window: np.ndarray) -> np.ndarray:
        """
        Which voices speak in each frame of a window of WINDOW seconds of
        samples: a boolean array with a row per frame and a column per
        voice, numbered within the window only.
        """
        raise NotImplementedError

    @classmethod
    def setup(cls):
        """
        Performs first-time setup, such as downloading default models.
        """
        pass
