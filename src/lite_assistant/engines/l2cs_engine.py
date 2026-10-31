"""
L2CS-Net gaze engine, on LiteRT.
"""
from typing import Any, Dict, Tuple

import numpy as np

from . import tflite
from .face_base import FaceEngine

# Abdelrahman et al.'s L2CS-Net, trained on Gaze360, converted by litert-community; MIT, 48 MB
REPO = "litert-community/L2CS-Gaze360-LiteRT"
FILE = "gaze_fp16.tflite"
SIZE = 448
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
# L2CS's demo grows the face detector's box by a fifth on each side
MARGIN = 1.4
# Each output scores 90 bins of 4 degrees, from -180
BINS = np.arange(90) * 4 - 180
# A gaze within this many degrees of straight at the camera, both ways, is at it
AT_CAMERA = 25


class L2CSEngine(FaceEngine):
    """
    L2CS-Net: where a face is looking, as degrees left or right (yaw) and up
    or down (pitch) of straight at the camera, and so whether it's looking
    at the camera, in about 0.1 s on the CPU.
    """
    MODEL_ID = "l2cs-gaze360"

    def __init__(self):
        paths = tflite.files("L2CS", REPO, [FILE])
        self.graphs = tflite.load_graphs("L2CS", {"gaze": paths[0]}) if paths else None

    def describe(self, photo, box: Tuple[float, float, float, float]) -> Dict[str, Any]:
        if not self.graphs:
            raise RuntimeError("L2CS is not available.")
        face = tflite.crop(photo, box, MARGIN)[0].resize((SIZE, SIZE))
        x = ((np.asarray(face, np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
        yaw, pitch = (float(chances[0] @ BINS) for chances in self.graphs["gaze"](x))
        return {"gaze": {"yaw": round(yaw, 1), "pitch": round(pitch, 1)},
                "looking_at_camera": abs(yaw) <= AT_CAMERA and abs(pitch) <= AT_CAMERA}

    @classmethod
    def setup(cls):
        tflite.download("L2CS", REPO, [FILE])
