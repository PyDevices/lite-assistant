"""
MoViNet-A0 Stream action recognition engine, on LiteRT.
"""
import os
from typing import Any, Dict

import numpy as np

from . import tflite
from .action_base import ActionEngine

# Google's MoViNet-A0 streaming, trained on Kinetics-600, ported to PyTorch by Atze00 and converted by
# litert-community; Apache-2.0, 15 MB
REPO = "litert-community/MoViNet-A0-Stream-LiteRT"
FILE = "movinet_a0_stream.tflite"
SIZE = 172
# Kinetics-600's labels, in the order of its outputs, from TensorFlow's model garden
LABELS = os.path.join(os.path.dirname(__file__), "kinetics600_labels.txt")
# How many past frames each of the 11 temporal convolutions keeps
DEPTHS = [2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 4]
POOLS = 16


class MoViNetEngine(ActionEngine):
    """
    MoViNet-A0's streaming graph: 600 actions, such as jumping jacks or
    clapping, sharpening as more frames of one arrive. The graph takes a
    frame and the memory of those before it, and gives back what to add
    to the memory: each convolution's latest frame, and running sums for
    its averages. About 5 ms a frame on the CPU.
    """
    MODEL_ID = "movinet-a0-stream"
    FPS = 5

    def __init__(self):
        paths = tflite.files("MoViNet", REPO, [FILE])
        graphs = tflite.load_graphs("MoViNet", {"net": paths[0]}) if paths else None
        self.graph = graphs["net"] if graphs else None
        with open(LABELS, encoding="utf-8") as f:
            self.labels = f.read().splitlines()

    def start(self) -> Dict[str, Any]:
        if not self.graph:
            raise RuntimeError("MoViNet is not available.")
        shapes = [self.graph.shape(i) for i in range(1 + sum(DEPTHS) + POOLS)]
        recent, k = [], 1
        for depth in DEPTHS:
            recent.append([np.zeros(shapes[k + i], np.float32) for i in range(depth)])
            k += depth
        return {"recent": recent, "sums": [np.zeros(shapes[k + i], np.float32) for i in range(POOLS)], "frames": 0}

    def watch(self, memory: Dict[str, Any], frame) -> np.ndarray:
        # Filled to a square from the middle, as it was trained
        side = min(frame.size)
        left, top = (frame.width - side) // 2, (frame.height - side) // 2
        square = frame.convert("RGB").crop((left, top, left + side, top + side)).resize((SIZE, SIZE))
        pixels = (np.asarray(square, np.float32) / 255).transpose(2, 0, 1)[None]
        memory["frames"] += 1
        inputs = [pixels] + [past for recent in memory["recent"] for past in recent] + memory["sums"] + [
            np.full((1, 1, 1, 1), 1 / memory["frames"], np.float32), np.ones((1, 1, 1, 1), np.float32)]
        outputs = self.graph(*inputs)
        memory["recent"] = [recent[1:] + [latest] for recent, latest in zip(memory["recent"], outputs[1:12])]
        memory["sums"] = [total + fresh for total, fresh in zip(memory["sums"], outputs[12:28])]
        logits = outputs[0][0]
        chances = np.exp(logits - logits.max())
        return chances / chances.sum()

    @classmethod
    def setup(cls):
        tflite.download("MoViNet", REPO, [FILE])
