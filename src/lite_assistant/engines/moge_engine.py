"""
MoGe-2 depth engine, on LiteRT.
"""
import io
import math
import os
import threading
import time
from typing import Tuple

import numpy as np
from .depth_base import DepthEngine, DepthMap
from ..paths import cached_hf_file

# Microsoft's MoGe-2 ViT-S, converted by Google; MIT licensed
REPO = "litert-community/MoGe-2-LiteRT"
# The fp32 graph: XNNPACK declines the fp16 one, which then runs ~30x slower on the CPU
FILE = "moge.tflite"
# The model's square input
SIZE = 448
# Gray padding looks to the model like plain background, unlike black bars
PADDING = (127, 127, 127)
# The point map is sampled on this grid to recover the depth shift
FIT_GRID = 64


def view_plane(size: int) -> np.ndarray:
    """ Each pixel's direction on a square image's view plane, as MoGe normalizes it: the half diagonal is 1. """
    span = (size - 1) / size / math.sqrt(2)
    u, v = np.meshgrid(np.linspace(-span, span, size), np.linspace(-span, span, size), indexing="xy")
    return np.stack([u, v], axis=-1)


def recover_shift(points: np.ndarray, uv: np.ndarray) -> Tuple[float, float]:
    """
    MoGe's point map is right up to a shift in depth, found as the one that
    best projects the points back onto their pixels. Returns the shift and
    the focal length (relative to the half diagonal) that goes with it.
    """
    from scipy.optimize import least_squares
    xy, z = points[:, :2], points[:, 2]

    def focal(shift):
        projected = xy / (z + shift)[:, None]
        return projected, (projected * uv).sum() / np.square(projected).sum()

    def error(shift):
        projected, f = focal(shift[0])
        return (f * projected - uv).ravel()

    shift = float(least_squares(error, x0=[0.0], ftol=1e-3, method="lm").x[0])
    return shift, float(focal(shift)[1])


class MoGeEngine(DepthEngine):
    """
    Microsoft's MoGe-2, which estimates a photo's 3D shape and its metric
    scale, run by LiteRT on the CPU in about half a second.
    """
    def __init__(self):
        self.interpreter = None
        # One photo at a time: the interpreter's tensors are shared
        self._lock = threading.Lock()
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            print("[MoGe] ai-edge-litert package is not installed. Skipping MoGeEngine.")
            return
        path = cached_hf_file(REPO, FILE)
        if not path:
            print(f"[MoGe] {FILE} isn't downloaded; run 'lite-assistant setup'.")
            return
        print(f"[MoGe] Loading {FILE}...")
        start = time.time()
        try:
            self.interpreter = Interpreter(model_path=path, num_threads=os.cpu_count())
            self.interpreter.allocate_tensors()
        except Exception as e:
            print(f"[MoGe] Failed to load: {e}")
            self.interpreter = None
            return
        self.uv = view_plane(SIZE)
        print(f"[MoGe] Loaded in {time.time() - start:.1f}s.")

    def get_model_id(self) -> str:
        return "moge-2-vits"

    def estimate(self, image: bytes) -> DepthMap:
        if not self.interpreter:
            raise RuntimeError("MoGe depth engine is not available.")
        from PIL import Image, ImageOps, UnidentifiedImageError
        try:
            photo = ImageOps.exif_transpose(Image.open(io.BytesIO(image))).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(f"not an image Pillow can read: {e}")

        # Letterboxed, not stretched, so the shapes in it stay true
        width, height = photo.size
        fit = SIZE / max(width, height)
        drawn = (max(1, round(width * fit)), max(1, round(height * fit)))
        left, top = (SIZE - drawn[0]) // 2, (SIZE - drawn[1]) // 2
        square = Image.new("RGB", (SIZE, SIZE), PADDING)
        square.paste(photo.resize(drawn, Image.BILINEAR), (left, top))
        pixels = (np.asarray(square, np.float32) / 255.0).transpose(2, 0, 1)[None]

        with self._lock:
            interpreter = self.interpreter
            interpreter.set_tensor(interpreter.get_input_details()[0]["index"], pixels)
            interpreter.invoke()
            outputs = [interpreter.get_tensor(o["index"]) for o in interpreter.get_output_details()]
        points, mask, scale = self._outputs(outputs)

        inside = np.zeros((SIZE, SIZE), bool)
        inside[top:top + drawn[1], left:left + drawn[0]] = True
        valid = (mask > 0.5) & inside
        sample = np.zeros((SIZE, SIZE), bool)
        steps = np.linspace(0, SIZE - 1, FIT_GRID).round().astype(int)
        sample[np.ix_(steps, steps)] = True
        sample &= valid
        if sample.sum() < 2:
            shift, focal = 0.0, None
        else:
            shift, focal = recover_shift(points[sample], self.uv[sample])
        points = points.copy()
        points[..., 2] += shift
        distances = np.linalg.norm(points, axis=-1) * scale
        distances[~valid | (points[..., 2] <= 0)] = np.nan
        fov = None
        if focal and focal > 0:
            fov = math.degrees(2 * math.atan(self.uv[0, -1, 0] * drawn[0] / SIZE / focal))
        return DepthMap(distances[top:top + drawn[1], left:left + drawn[0]], fov)

    @staticmethod
    def _outputs(outputs) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        The point map, mask and scale. The outputs' order isn't fixed, so
        they're told apart by shape, and the normals by their unit length.
        """
        scale = next(o for o in outputs if o.size == 1)
        mask = next(o for o in outputs if o.ndim == 4 and o.shape[-1] == 1 and o.size > 1)
        maps = [o[0] for o in outputs if o.ndim == 4 and o.shape[-1] == 3]
        normals = min(maps, key=lambda o: abs(float(np.linalg.norm(o[::16, ::16], axis=-1).mean()) - 1))
        points = next(o for o in maps if o is not normals)
        return points, mask[0, ..., 0], float(scale.ravel()[0])

    @classmethod
    def setup(cls):
        if not cached_hf_file(REPO, FILE):
            from huggingface_hub import hf_hub_download
            print(f"[MoGe] Downloading {FILE} from {REPO}...")
            hf_hub_download(REPO, FILE)
