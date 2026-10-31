"""
YOLOX object detection engine, on LiteRT.
"""
import os
import threading
import time
from typing import List, Optional

import numpy as np
from .detection_base import Detection, DetectionEngine, letterbox, suppress
from ..paths import cached_hf_file

# Megvii's YOLOX, converted by Google; Apache-2.0. "nano" (2 MB) or "tiny"
# (10 MB, COCO AP 32.8 against nano's 25.8)
SIZE_NAME = os.environ.get("YOLOX_MODEL", "tiny").lower()
REPO = f"litert-community/yolox-{SIZE_NAME}-litert"
FILE = f"yolox_{SIZE_NAME}.tflite"
# The model's square input, and the gray YOLOX pads it with
SIZE = 416
PADDING = (114, 114, 114)
STRIDES = (8, 16, 32)
# Boxes of one kind that overlap more than this are taken as the same object
OVERLAP = 0.45
THRESHOLD = 0.35

# COCO's 80 kinds of object, in the model's order
LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


def anchors() -> np.ndarray:
    """ Each output row's grid cell and stride, as (x, y, stride), smallest stride first. """
    found = []
    for stride in STRIDES:
        cells = SIZE // stride
        y, x = np.mgrid[:cells, :cells]
        found.append(np.stack([x.ravel(), y.ravel(), np.full(cells * cells, stride)], axis=-1))
    return np.concatenate(found).astype(np.float32)


class YOLOXEngine(DetectionEngine):
    """
    Megvii's YOLOX, which finds COCO's 80 kinds of object in a photo, run by
    LiteRT on the CPU in tens of milliseconds.
    """
    MODEL_ID = f"yolox-{SIZE_NAME}"

    def __init__(self):
        self.interpreter = None
        # One photo at a time: the interpreter's tensors are shared
        self._lock = threading.Lock()
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            print("[YOLOX] ai-edge-litert package is not installed. Skipping YOLOXEngine.")
            return
        path = cached_hf_file(REPO, FILE)
        if not path:
            print(f"[YOLOX] {FILE} isn't downloaded; run 'lite-assistant setup'.")
            return
        print(f"[YOLOX] Loading {FILE}...")
        start = time.time()
        try:
            self.interpreter = Interpreter(model_path=path, num_threads=os.cpu_count())
            self.interpreter.allocate_tensors()
        except Exception as e:
            print(f"[YOLOX] Failed to load: {e}")
            self.interpreter = None
            return
        self.anchors = anchors()
        print(f"[YOLOX] Loaded in {time.time() - start:.1f}s.")

    def get_model_id(self) -> str:
        return self.MODEL_ID

    def get_labels(self) -> List[str]:
        return list(LABELS)

    def detect(self, image: bytes, threshold: Optional[float] = None) -> List[Detection]:
        if not self.interpreter:
            raise RuntimeError("YOLOX detection engine is not available.")
        threshold = THRESHOLD if threshold is None else threshold
        # Letterboxed at the top left, as YOLOX was trained
        pixels, fit, width, height = letterbox(image, SIZE, PADDING)
        pixels = pixels[None]

        with self._lock:
            interpreter = self.interpreter
            interpreter.set_tensor(interpreter.get_input_details()[0]["index"], pixels)
            interpreter.invoke()
            raw = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0]

        # Each row: the box's center and size in grid cells, then how sure the
        # model is that it holds an object, and of each kind (already sigmoids)
        scores = raw[:, 4:5] * raw[:, 5:]
        kinds, best = scores.argmax(1), scores.max(1)
        rows = np.nonzero(best >= threshold)[0]
        if not rows.size:
            return []
        grid, stride = self.anchors[rows, :2], self.anchors[rows, 2:]
        middle = (raw[rows, :2] + grid) * stride
        size = np.exp(raw[rows, 2:4]) * stride
        boxes = np.concatenate([middle - size / 2, middle + size / 2], axis=1) / fit
        boxes = np.clip(boxes / [width, height, width, height], 0, 1)

        found = []
        for kind in np.unique(kinds[rows]):
            same = np.nonzero(kinds[rows] == kind)[0]
            for i in suppress(boxes[same], best[rows][same], OVERLAP):
                row = same[i]
                found.append(Detection(LABELS[kind], round(float(best[rows][row]), 3),
                                       tuple(round(float(v), 4) for v in boxes[row])))
        return sorted(found, key=lambda d: -d.score)

    @classmethod
    def setup(cls):
        if not cached_hf_file(REPO, FILE):
            from huggingface_hub import hf_hub_download
            print(f"[YOLOX] Downloading {FILE} from {REPO}...")
            hf_hub_download(REPO, FILE)
