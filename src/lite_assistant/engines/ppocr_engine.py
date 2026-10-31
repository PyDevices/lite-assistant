"""
PP-OCRv6 OCR engine, on LiteRT.
"""
import json
import math
from typing import List, Optional

import numpy as np

from . import tflite
from .ocr_base import OCREngine, TextLine

# PaddleOCR's PP-OCRv6 small detector and recognizers, converted by
# litert-community; Apache-2.0, 77 MB in all
REPO = "litert-community/PP-OCRv6-Small-LiteRT"
DETECTOR = "ppocrv6_small_det_640_fp32.tflite"
# The recognizers, by the widest line they take at a height of 48
READERS = {320: "ppocrv6_small_rec_320_fp32.tflite", 640: "ppocrv6_small_rec_640_fp32.tflite",
           960: "ppocrv6_small_rec_960_fp32.tflite"}
CHARACTERS = "characters.json"
SIZE = 640
HEIGHT = 48
# The detector's settings, as PaddleOCR's: a point is text above THRESHOLD,
# a box is kept if its mean is BOX_THRESHOLD, and grown by UNCLIP times its
# area over its perimeter
THRESHOLD, BOX_THRESHOLD, UNCLIP, MIN_SIDE, MAX_BOXES = 0.2, 0.45, 1.4, 3, 3000
# The detector's normalization, in RGB order; the graph takes BGR
MEAN = np.array([0.406, 0.456, 0.485], np.float32) * 255
STD = np.array([0.225, 0.224, 0.229], np.float32) * 255
# A line this much taller than wide is read turned on its side, both ways
TALL = 1.5
# Lines whose tops are closer than this, in detector pixels, are in one row
ROW = 10


def min_area_rect(points: np.ndarray) -> np.ndarray:
    """ The smallest rotated rectangle around points, as its 4 corners. """
    from scipy.spatial import ConvexHull, QhullError
    try:
        hull = points[ConvexHull(points).vertices]
    except (QhullError, ValueError):
        # A line or a point: its bounding box, which is as small
        hull = points
    best = None
    edges = np.diff(np.vstack([hull, hull[:1]]), axis=0)
    for angle in np.unique(np.arctan2(edges[:, 1], edges[:, 0]) % (np.pi / 2)):
        turn = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        turned = hull @ turn.T
        low, high = turned.min(0), turned.max(0)
        area = float(np.prod(high - low))
        if best is None or area < best[0]:
            corners = np.array([low, [high[0], low[1]], high, [low[0], high[1]]])
            best = (area, corners @ turn)
    return best[1]


def ordered(box: np.ndarray) -> np.ndarray:
    """ A rectangle's corners in order: top left, top right, bottom right, bottom left. """
    box = box[np.argsort(box[:, 0])]
    left, right = box[:2][np.argsort(box[:2, 1])], box[2:][np.argsort(box[2:, 1])]
    return np.array([left[0], right[0], right[1], left[1]])


def sides(box: np.ndarray) -> tuple:
    return np.linalg.norm(box[1] - box[0]), np.linalg.norm(box[3] - box[0])


def inside_mean(probability: np.ndarray, box: np.ndarray) -> float:
    """ The mean probability inside a rotated rectangle. """
    height, width = probability.shape
    left, top = np.clip(np.floor(box.min(0)).astype(int), 0, [width - 1, height - 1])
    right, bottom = np.clip(np.ceil(box.max(0)).astype(int), 0, [width - 1, height - 1])
    ys, xs = np.mgrid[top:bottom + 1, left:right + 1]
    points = np.stack([xs, ys], -1).astype(np.float32) - box[0]
    across, down = box[1] - box[0], box[3] - box[0]
    a = points @ across / max(float(across @ across), 1e-6)
    d = points @ down / max(float(down @ down), 1e-6)
    mask = (a >= 0) & (a <= 1) & (d >= 0) & (d <= 1)
    region = probability[top:bottom + 1, left:right + 1]
    return float(region[mask].mean()) if mask.any() else 0.0


def boxes(probability: np.ndarray) -> List[np.ndarray]:
    """ The boxes around the text in the detector's map, as PaddleOCR's DB postprocessing finds them. """
    from scipy import ndimage
    labels, count = ndimage.label(probability > THRESHOLD, structure=np.ones((3, 3)))
    found = []
    for index, where in enumerate(ndimage.find_objects(labels)[:MAX_BOXES], 1):
        ys, xs = np.nonzero(labels[where] == index)
        points = np.stack([xs + where[1].start, ys + where[0].start], -1).astype(np.float32)
        # The corners of the pixels, so a one-pixel-wide line has an area
        points = np.concatenate([points, points + [1, 0], points + [0, 1], points + 1])
        box = ordered(min_area_rect(points))
        if min(sides(box)) < MIN_SIDE or inside_mean(probability, box) < BOX_THRESHOLD:
            continue
        # Grow it, as PaddleOCR's unclip does, by its area times UNCLIP over its perimeter
        width, height = sides(box)
        grow = width * height * UNCLIP / (2 * (width + height))
        center = box.mean(0)
        along, down = (box[1] - box[0]) / max(width, 1e-6), (box[3] - box[0]) / max(height, 1e-6)
        width, height = width + 2 * grow, height + 2 * grow
        if min(width, height) < MIN_SIDE + 2:
            continue
        half = along * width / 2, down * height / 2
        found.append(ordered(np.array([center - half[0] - half[1], center + half[0] - half[1],
                                       center + half[0] + half[1], center - half[0] + half[1]])))
    # Reading order: by top, then left, with tops less than ROW apart counting as one row
    found.sort(key=lambda b: (b[0][1], b[0][0]))
    for i in range(len(found) - 1):
        for j in range(i, -1, -1):
            if abs(found[j + 1][0][1] - found[j][0][1]) < ROW and found[j + 1][0][0] < found[j][0][0]:
                found[j], found[j + 1] = found[j + 1], found[j]
            else:
                break
    return found


class PPOCREngine(OCREngine):
    """
    PaddleOCR's PP-OCRv6 small: a detector finds the lines of text in a
    photo, scaled to fit 640 pixels, and a recognizer reads each line from
    the full-size photo, in 85 languages and scripts.
    """
    MODEL_ID = "pp-ocrv6-small"

    def __init__(self):
        paths = tflite.files("PP-OCR", REPO, [DETECTOR, *READERS.values(), CHARACTERS])
        self.graphs = None
        if paths:
            with open(paths[-1], encoding="utf-8") as f:
                self.characters = json.load(f)
            self.graphs = tflite.load_graphs("PP-OCR", {"detector": paths[0], **dict(zip(READERS, paths[1:-1]))})

    def find(self, photo) -> List[np.ndarray]:
        """ The boxes around the lines of text in a photo, in its pixels. """
        scale = SIZE / max(photo.size)
        size = (max(1, round(photo.width * scale)), max(1, round(photo.height * scale)))
        page = np.zeros((SIZE, SIZE, 3), np.float32)
        page[:] = MEAN
        page[:size[1], :size[0]] = np.asarray(photo.resize(size), np.float32)
        x = ((page - MEAN) / STD)[..., ::-1].transpose(2, 0, 1)[None]
        probability = self.graphs["detector"](x)[0][0, 0]
        return [np.clip(box / scale, 0, [photo.width, photo.height]) for box in boxes(probability)]

    def crops(self, photo, box: np.ndarray) -> list:
        """ A line of text, straightened out of its box, or if it's tall, turned each way it might read. """
        from PIL import Image
        width = max(1, int(max(np.linalg.norm(box[1] - box[0]), np.linalg.norm(box[2] - box[3]))))
        height = max(1, int(max(np.linalg.norm(box[3] - box[0]), np.linalg.norm(box[2] - box[1]))))
        # Image.QUAD takes the corners top left, bottom left, bottom right, top right
        quad = tuple(float(v) for v in box[[0, 3, 2, 1]].ravel())
        line = photo.transform((width, height), Image.QUAD, quad, Image.BICUBIC)
        return [line.transpose(Image.ROTATE_90), line.transpose(Image.ROTATE_270)] if height / width >= TALL else [line]

    def recognize(self, line) -> Optional[TextLine]:
        """ The text in one line, and the mean probability of its characters. """
        width = min(max(1, math.ceil(HEIGHT * line.width / line.height)), max(READERS))
        reader = min(w for w in READERS if w >= width)
        values = np.asarray(line.resize((width, HEIGHT)), np.float32)[..., ::-1]
        x = np.zeros((HEIGHT, reader, 3), np.float32)
        x[:, :width] = (values - 127.5) / 127.5
        probabilities = self.graphs[reader](x.transpose(2, 0, 1)[None])[0][0]
        # Greedy CTC: repeated characters collapse, and 0, the blank, separates them
        ids = probabilities.argmax(-1)
        keep = (ids != 0) & np.concatenate([[True], ids[1:] != ids[:-1]])
        if not keep.any():
            return None
        text = "".join(self.characters[i] for i in ids[keep])
        return TextLine(text, float(probabilities.max(-1)[keep].mean()), (0, 0, 0, 0))

    def read(self, image: bytes) -> List[TextLine]:
        if not self.graphs:
            raise RuntimeError("PP-OCR is not available.")
        photo = tflite.open_photo(image)
        lines = []
        for box in self.find(photo):
            readings = [r for r in map(self.recognize, self.crops(photo, box)) if r]
            found = max(readings, key=lambda r: r.score, default=None)
            if found and found.text.strip():
                (left, top), (right, bottom) = box.min(0), box.max(0)
                found.text = found.text.strip()
                found.box = (round(float(left) / photo.width, 4), round(float(top) / photo.height, 4),
                             round(float(right) / photo.width, 4), round(float(bottom) / photo.height, 4))
                lines.append(found)
        return lines

    @classmethod
    def setup(cls):
        tflite.download("PP-OCR", REPO, [DETECTOR, *READERS.values(), CHARACTERS])
