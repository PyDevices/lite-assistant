"""
Ultra-Fast-Lane-Detection engine for road lanes, on LiteRT.
"""
from typing import List, Optional

import numpy as np

from . import tflite
from .line_base import LineEngine, Point

# cfzd's Ultra-Fast-Lane-Detection, ResNet18 trained on CULane, converted by litert-community; MIT, 178 MB
REPO = "litert-community/Ultra-Fast-Lane-Detection-LiteRT"
FILE = "ufld.tflite"
WIDTH, HEIGHT = 800, 288
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
# CULane's rows, on the 288 high input, from the top: each lane gives its column on each
ROWS = np.array([121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287])
# Across, each row is split into 200 cells; the 201st means the lane isn't on that row
CELLS = 200
# A lane needs this many rows to count
MIN_ROWS = 3


class UFLDEngine(LineEngine):
    """
    Up to four lane markings of a road from a dashcam's view, by which of
    200 columns each crosses 18 rows at: the car's own lane's edges and
    the ones beyond them. Trained on Chinese highways and city streets, it
    doesn't see tape on a floor. About 60 ms on a desktop CPU.
    """
    MODEL_ID, TASK = "ufld-culane-resnet18", "lanes"

    def __init__(self):
        paths = tflite.files("UFLD", REPO, [FILE])
        graphs = tflite.load_graphs("UFLD", {"net": paths[0]}) if paths else None
        self.graph = graphs["net"] if graphs else None

    def find(self, photo) -> List[List[Point]]:
        if not self.graph:
            raise RuntimeError("UFLD is not available.")
        # It was trained on OpenCV's BGR photos: given RGB, it finds almost nothing
        x = np.asarray(photo.convert("RGB").resize((WIDTH, HEIGHT)), np.float32)[..., ::-1] / 255
        scores = self.graph(((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32))[0][0]
        cells = scores[:CELLS] - scores[:CELLS].max(axis=0)
        chances = np.exp(cells) / np.exp(cells).sum(axis=0)
        # Each lane's column on each row: the expected cell, if the likeliest isn't "no lane"
        columns = (chances * np.arange(CELLS)[:, None, None]).sum(axis=0) / (CELLS - 1)
        present = scores.argmax(axis=0) != CELLS
        lanes = []
        for lane in range(scores.shape[2]):
            rows = [r for r in range(len(ROWS)) if present[r, lane]]
            if len(rows) >= MIN_ROWS:
                lanes.append([(round(float(columns[r, lane]), 4), round(float(ROWS[r] / HEIGHT), 4))
                              for r in reversed(rows)])
        return sorted(lanes, key=lambda lane: lane[0][0])

    def path(self, lines: List[List[Point]]) -> Optional[List[Point]]:
        """ The middle of the car's own lane: between the nearest lanes left and right of the photo's middle. """
        left = [line for line in lines if line[0][0] < 0.5]
        right = [line for line in lines if line[0][0] >= 0.5]
        if not left or not right:
            return None
        left_x, right_x = dict((y, x) for x, y in left[-1]), dict((y, x) for x, y in right[0])
        middle = [(round((left_x[y] + right_x[y]) / 2, 4), y) for y in sorted(set(left_x) & set(right_x), reverse=True)]
        return middle or None

    @classmethod
    def setup(cls):
        tflite.download("UFLD", REPO, [FILE])
