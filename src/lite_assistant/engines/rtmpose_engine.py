"""
RTMPose body and hand pose engines, on LiteRT.
"""
from typing import Dict, Tuple

import numpy as np

from . import tflite
from .pose_base import PoseEngine

MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
# mmpose grows a detector's box by a quarter before cropping it
MARGIN = 1.25
# The heads score each joint's x and y in bins of half a pixel
SPLIT = 2.0

BODY = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"]
HAND = ["wrist"] + [f"{finger}_{joint}" for finger, joints in (
    ("thumb", ("cmc", "mcp", "ip", "tip")), ("index", ("mcp", "pip", "dip", "tip")),
    ("middle", ("mcp", "pip", "dip", "tip")), ("ring", ("mcp", "pip", "dip", "tip")),
    ("pinky", ("mcp", "pip", "dip", "tip"))) for joint in joints]


class _RTMPose(PoseEngine):
    """ What RTMPose's body and hand models share: a graph that takes a crop of a size, and SimCC heads. """
    TAG = REPO = FILE = ""
    WIDTH = HEIGHT = 256

    def __init__(self):
        paths = tflite.files(self.TAG, self.REPO, [self.FILE])
        self.graphs = tflite.load_graphs(self.TAG, {"pose": paths[0]}) if paths else None

    def keypoints(self, photo, box: Tuple[float, float, float, float]) -> Dict[str, Tuple[float, float, float]]:
        if not self.graphs:
            raise RuntimeError(f"{self.TAG} is not available.")
        region, (left, top, width, height) = tflite.crop(photo, box, MARGIN, aspect=self.WIDTH / self.HEIGHT)
        x = ((np.asarray(region.resize((self.WIDTH, self.HEIGHT)), np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
        heads = self.graphs["pose"](x)
        # The x head has a bin for each half pixel of the width, the y head of the height
        across, down = sorted((h[0] for h in heads), key=lambda h: h.shape[-1] != self.WIDTH * SPLIT)
        xs, ys = across.argmax(-1) / SPLIT, down.argmax(-1) / SPLIT
        # As mmpose's: the lesser of the two heads' peaks
        scores = np.minimum(across.max(-1), down.max(-1))
        return {name: (round(float((left + px / self.WIDTH * width) / photo.width), 4),
                       round(float((top + py / self.HEIGHT * height) / photo.height), 4),
                       round(float(np.clip(score, 0, 1)), 3))
                for name, px, py, score in zip(self.KEYPOINTS, xs, ys, scores)}

    @classmethod
    def setup(cls):
        tflite.download(cls.TAG, cls.REPO, [cls.FILE])


class RTMPoseBodyEngine(_RTMPose):
    """
    RTMPose-s: the 17 COCO joints of a person, from the eyes to the ankles,
    in about 30 ms on the CPU.
    """
    MODEL_ID, TASK, KEYPOINTS = "rtmpose-s", "body", BODY
    TAG, REPO, FILE = "RTMPose", "litert-community/RTMPose-s-LiteRT", "rtmpose_s_fp16.tflite"
    WIDTH, HEIGHT = 192, 256


class RTMPoseHandEngine(_RTMPose):
    """
    RTMPose hand: the 21 joints of a hand, the wrist and four on each
    finger, in about 40 ms on the CPU.
    """
    MODEL_ID, TASK, KEYPOINTS = "rtmpose-hand", "hand", HAND
    TAG, REPO, FILE = "RTMPose-Hand", "litert-community/RTMPose-Hand-LiteRT", "rtmhand_fp16.tflite"
