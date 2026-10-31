"""
YuNet face detection engine, on LiteRT.
"""
import os
import re
import threading
import time
from typing import List, Optional

import numpy as np
from .detection_base import Detection, DetectionEngine, letterbox, suppress
from ..paths import cached_hf_file

# ShiqiYu's libfacedetection YuNet, converted by litert-community; BSD-3-Clause, 0.3 MB
REPO = "litert-community/YuNet-Face-LiteRT"
FILE = "yunet_fp16.tflite"
SIZE = 640
STRIDES = (8, 16, 32)
THRESHOLD = 0.6
OVERLAP = 0.45
# The five landmarks, in the model's order: the person's own right eye, the
# one on the photo's left when they face the camera
POINTS = ("right_eye", "left_eye", "nose", "right_mouth", "left_mouth")


class YuNetEngine(DetectionEngine):
    """
    YuNet, which finds faces, with their eyes, nose and mouth corners, in
    about 15 ms on the CPU.
    """
    MODEL_ID = "yunet"

    def __init__(self):
        self.interpreter = None
        # One photo at a time: the interpreter's tensors are shared
        self._lock = threading.Lock()
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            print("[YuNet] ai-edge-litert package is not installed. Skipping YuNetEngine.")
            return
        path = cached_hf_file(REPO, FILE)
        if not path:
            print(f"[YuNet] {FILE} isn't downloaded; run 'lite-assistant setup'.")
            return
        print(f"[YuNet] Loading {FILE}...")
        start = time.time()
        try:
            self.interpreter = Interpreter(model_path=path, num_threads=os.cpu_count())
            self.interpreter.allocate_tensors()
        except Exception as e:
            print(f"[YuNet] Failed to load: {e}")
            self.interpreter = None
            return
        # Twelve outputs, numbered in their names: scores, objectness, boxes and landmarks, each for strides 8, 16, 32
        number = lambda d: int(re.findall(r"\d+", d["name"])[-1])
        self.outputs = [d["index"] for d in sorted(self.interpreter.get_output_details(), key=number)]
        self.cells = []
        for stride in STRIDES:
            y, x = np.mgrid[:SIZE // stride, :SIZE // stride]
            self.cells.append(np.stack([x.ravel(), y.ravel()], axis=-1).astype(np.float32) * stride)
        print(f"[YuNet] Loaded in {time.time() - start:.1f}s.")

    def get_model_id(self) -> str:
        return self.MODEL_ID

    def get_labels(self) -> List[str]:
        return ["face"]

    def detect(self, image: bytes, threshold: Optional[float] = None) -> List[Detection]:
        if not self.interpreter:
            raise RuntimeError("YuNet face detection engine is not available.")
        threshold = THRESHOLD if threshold is None else threshold
        # BGR, 0 to 255, channels first
        pixels, fit, width, height = letterbox(image, SIZE, (0, 0, 0))
        pixels = np.ascontiguousarray(pixels.transpose(2, 0, 1))[None]

        with self._lock:
            interpreter = self.interpreter
            interpreter.set_tensor(interpreter.get_input_details()[0]["index"], pixels)
            interpreter.invoke()
            out = [interpreter.get_tensor(i)[0] for i in self.outputs]

        boxes, scores, points = [], [], []
        for level, stride in enumerate(STRIDES):
            score = out[level][:, 0] * out[3 + level][:, 0]
            rows = np.nonzero(score >= threshold)[0]
            cell = self.cells[level][rows]
            middle = out[6 + level][rows, :2] * stride + cell
            size = np.exp(out[6 + level][rows, 2:4]) * stride
            boxes.append(np.concatenate([middle - size / 2, middle + size / 2], axis=1))
            points.append(out[9 + level][rows].reshape(-1, 5, 2) * stride + cell[:, None])
            scores.append(score[rows])
        boxes, scores, points = np.concatenate(boxes), np.concatenate(scores), np.concatenate(points)
        # From the square's pixels to fractions of the photo
        scale = np.array([width, height]) * fit
        boxes = np.clip(boxes / np.tile(scale, 2), 0, 1)
        points = np.clip(points / scale, 0, 1)

        found = []
        for i in suppress(boxes, scores, OVERLAP):
            found.append(Detection("face", round(float(scores[i]), 3), tuple(round(float(v), 4) for v in boxes[i]),
                                   {name: (round(float(x), 4), round(float(y), 4))
                                    for name, (x, y) in zip(POINTS, points[i])}))
        return found

    @classmethod
    def setup(cls):
        if not cached_hf_file(REPO, FILE):
            from huggingface_hub import hf_hub_download
            print(f"[YuNet] Downloading {FILE} from {REPO}...")
            hf_hub_download(REPO, FILE)
