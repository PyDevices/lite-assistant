"""
HSEmotion facial expression engine, on LiteRT.
"""
from typing import Any, Dict, Tuple

import numpy as np

from . import tflite
from .face_base import FaceEngine

# Savchenko's HSEmotion EfficientNet-B0, converted by litert-community; Apache-2.0, 8 MB
REPO = "litert-community/HSEmotion-B0-LiteRT"
FILE = "hsemotion_b0_fp16.tflite"
SIZE = 224
# AffectNet's eight expressions, in the order the graph scores them
EXPRESSIONS = ["angry", "contemptuous", "disgusted", "afraid", "happy", "neutral", "sad", "surprised"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
# How much bigger than the detector's box the crop is, as the face crops it was trained on
MARGIN = 1.1


class HSEmotionEngine(FaceEngine):
    """
    HSEmotion: a face's expression, one of AffectNet's eight, in about 15 ms
    on the CPU.
    """
    MODEL_ID = "hsemotion-b0"

    def __init__(self):
        paths = tflite.files("HSEmotion", REPO, [FILE])
        self.graphs = tflite.load_graphs("HSEmotion", {"expression": paths[0]}) if paths else None

    def describe(self, photo, box: Tuple[float, float, float, float]) -> Dict[str, Any]:
        if not self.graphs:
            raise RuntimeError("HSEmotion is not available.")
        face = tflite.crop(photo, box, MARGIN, aspect=1.0)[0].resize((SIZE, SIZE))
        x = ((np.asarray(face, np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
        logits = self.graphs["expression"](x)[0][0].astype(np.float64)
        chances = np.exp(logits - logits.max())
        chances /= chances.sum()
        ranked = np.argsort(-chances)
        return {"expression": EXPRESSIONS[ranked[0]],
                "expressions": {EXPRESSIONS[i]: round(float(chances[i]), 3) for i in ranked[:3]}}

    @classmethod
    def setup(cls):
        tflite.download("HSEmotion", REPO, [FILE])
