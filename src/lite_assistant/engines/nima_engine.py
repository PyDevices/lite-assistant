"""
NIMA image quality engine, on LiteRT.
"""
from typing import Dict

import numpy as np

from . import tflite
from .quality_base import QualityEngine

# idealo's NIMA MobileNets, converted by litert-community; Apache-2.0, 6 MB each
REPO = "litert-community/NIMA-LiteRT"
FILES = {"technical": "nima_technical_fp16.tflite", "aesthetic": "nima_aesthetic_fp16.tflite"}
SIZE = 224


class NIMAEngine(QualityEngine):
    """
    Google's NIMA, as two MobileNets: one scores a photo's technical quality
    (sharpness, noise, exposure), the other how pleasing it is, each from 1
    to 10, in about 20 ms each on the CPU.
    """
    MODEL_ID = "nima-mobilenet"

    def __init__(self):
        paths = tflite.files("NIMA", REPO, list(FILES.values()))
        self.graphs = tflite.load_graphs("NIMA", dict(zip(FILES, paths))) if paths else None

    def score(self, image: bytes) -> Dict[str, float]:
        if not self.graphs:
            raise RuntimeError("NIMA is not available.")
        photo = tflite.open_photo(image).resize((SIZE, SIZE))
        x = (np.asarray(photo, np.float32) / 127.5 - 1)[None]
        # Each graph gives how likely each score from 1 to 10 is; the score is the mean
        return {name: round(float(np.arange(1, 11) @ graph(x)[0][0]), 2) for name, graph in self.graphs.items()}

    @classmethod
    def setup(cls):
        tflite.download("NIMA", REPO, list(FILES.values()))
