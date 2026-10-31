"""
Base interfaces and definitions for object detection engines, and the
letterboxing and non-maximum suppression they share.
"""
import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Detection:
    """ An object found in a photo. """
    label: str
    # How sure the model is, from 0 to 1
    score: float
    # Left, top, right and bottom, as fractions of the photo's width and height from its top left
    box: Tuple[float, float, float, float]
    # Named points on the object, such as a face's eyes, as (x, y) fractions like the box's
    points: Optional[Dict[str, Tuple[float, float]]] = None


class DetectionEngine:
    """
    Abstract base class for all object detection engines.
    """

    def get_model_id(self) -> str:
        """
        Returns the ID of the model this engine has loaded.
        """
        raise NotImplementedError

    def get_labels(self) -> List[str]:
        """ The kinds of object the model knows. """
        raise NotImplementedError

    def detect(self, image: bytes, threshold: Optional[float] = None) -> List[Detection]:
        """
        The objects in a photo, given as JPEG, PNG or another format Pillow
        reads, that the model is at least threshold sure of (by default, the
        engine's own threshold), surest first. Raises ValueError if it isn't
        an image.
        """
        raise NotImplementedError

    @classmethod
    def setup(cls):
        """
        Performs first-time setup, such as downloading default models.
        """
        pass


def letterbox(image: bytes, size: int, fill: Tuple[int, int, int]) -> Tuple[np.ndarray, float, int, int]:
    """
    A photo scaled to fit a size by size square at its top left, the rest
    filled with a color: the square as BGR pixels from 0 to 255, [size, size,
    3], the scale, and the photo's width and height. Raises ValueError if it
    isn't an image.
    """
    from PIL import Image, ImageOps, UnidentifiedImageError
    try:
        photo = ImageOps.exif_transpose(Image.open(io.BytesIO(image))).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"not an image Pillow can read: {e}")
    width, height = photo.size
    fit = size / max(width, height)
    square = Image.new("RGB", (size, size), fill)
    square.paste(photo.resize((max(1, round(width * fit)), max(1, round(height * fit))), Image.BILINEAR), (0, 0))
    return np.ascontiguousarray(np.asarray(square, np.float32)[..., ::-1]), fit, width, height


def suppress(boxes: np.ndarray, scores: np.ndarray, overlap: float = 0.45) -> List[int]:
    """ Non-maximum suppression: the indices of the boxes kept, best first. """
    order = list(np.argsort(-scores))
    kept = []
    while order:
        best = order.pop(0)
        kept.append(best)
        if not order:
            break
        rest = boxes[order]
        x0 = np.maximum(boxes[best, 0], rest[:, 0])
        y0 = np.maximum(boxes[best, 1], rest[:, 1])
        x1 = np.minimum(boxes[best, 2], rest[:, 2])
        y1 = np.minimum(boxes[best, 3], rest[:, 3])
        shared = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
        union = area(boxes[best]) + area(rest) - shared
        order = [i for i, iou in zip(order, shared / np.maximum(union, 1e-9)) if iou <= overlap]
    return kept
