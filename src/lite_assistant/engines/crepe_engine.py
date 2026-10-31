"""
CREPE pitch engine, on LiteRT.
"""
from typing import List, Tuple

import numpy as np

from . import tflite
from .pitch_base import PitchEngine

# Kim, Salamon, Li and Bello's CREPE, the full model, converted by litert-community; MIT, 45 MB
REPO = "litert-community/CREPE-pitch-LiteRT"
FILE = "crepe_full_fp16.tflite"
WINDOW = 1024
# Its 360 outputs are pitches 20 cents apart, the first this many cents above 10 Hz
CENTS_PER_BIN, FIRST_CENTS = 20, 1997.3794084376191


class CREPEEngine(PitchEngine):
    """
    CREPE: the pitch of 64 ms of sound, from about 32 Hz to 2 kHz, to a
    few cents, as the likelihood of each of 360 pitches 20 cents apart.
    About 20 ms a window on a desktop CPU, which can't take several at
    once, so it hears one every 50 ms.
    """
    MODEL_ID = "crepe-full"

    def __init__(self):
        paths = tflite.files("CREPE", REPO, [FILE])
        graphs = tflite.load_graphs("CREPE", {"net": paths[0]}) if paths else None
        self.graph = graphs["net"] if graphs else None

    def pitches(self, samples: np.ndarray) -> List[Tuple[float, float]]:
        if not self.graph:
            raise RuntimeError("CREPE is not available.")
        hop = int(self.HOP * self.RATE)
        # Each window is centered on its time, as CREPE's own code does
        padded = np.pad(samples.astype(np.float32), WINDOW // 2)
        found = []
        for start in range(0, len(samples), hop):
            window = padded[start:start + WINDOW]
            window = (window - window.mean()) / max(float(window.std()), 1e-10)
            chances = self.graph(window[None])[0][0]
            # The likeliest pitch, refined by its neighbors' likelihoods
            peak = int(chances.argmax())
            near = slice(max(0, peak - 4), min(len(chances), peak + 5))
            weights = chances[near]
            cents = CENTS_PER_BIN * float(weights @ np.arange(len(chances))[near] / weights.sum()) + FIRST_CENTS
            found.append((10 * 2 ** (cents / 1200), float(chances[peak])))
        return found

    @classmethod
    def setup(cls):
        tflite.download("CREPE", REPO, [FILE])
