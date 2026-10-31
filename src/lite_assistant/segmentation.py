"""
Where a segmentation engine found something in a photo: its box, how much
of the photo it covers and how far away it is, and its mask as an image.
"""
import io
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .depth import meters
from .detection import OBJECT_PERCENTILE

# A point is part of what was asked for when the engine is at least this sure
THRESHOLD = 0.4


def region(probability: np.ndarray, threshold: float = THRESHOLD) -> Optional[Dict[str, Any]]:
    """
    Where a probability map finds its thing: its box and center as
    fractions of the photo, the fraction of the photo it covers, and how
    sure the engine is at its surest point. None if it found nothing.
    """
    inside = probability >= threshold
    if not inside.any():
        return None
    height, width = probability.shape
    ys, xs = np.nonzero(inside)
    return {"score": round(float(probability.max()), 3),
            "box": [round(float(xs.min()) / width, 3), round(float(ys.min()) / height, 3),
                    round(float(xs.max() + 1) / width, 3), round(float(ys.max() + 1) / height, 3)],
            "center": [round(float(xs.mean() + 0.5) / width, 3), round(float(ys.mean() + 0.5) / height, 3)],
            "area": round(float(inside.mean()), 4)}


def resized(probability: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """ A probability map at another size, (width, height). """
    from PIL import Image
    return np.asarray(Image.fromarray(probability.astype(np.float32), "F").resize(size, Image.BILINEAR))


def distance(depth, probability: np.ndarray, threshold: float = THRESHOLD) -> Optional[float]:
    """ How far what a probability map finds is, in meters, as detection measures an object; None if unknown. """
    height, width = depth.distances.shape
    inside = resized(probability, (width, height)) >= threshold
    valid = depth.distances[inside & np.isfinite(depth.distances)]
    return meters(np.percentile(valid, OBJECT_PERCENTILE)) if valid.size else None


def mask(probability: np.ndarray, size: Tuple[int, int]) -> bytes:
    """ A probability map as a grayscale PNG of the photo's size, white where the engine is sure. """
    from PIL import Image
    out = io.BytesIO()
    Image.fromarray((np.clip(resized(probability, size), 0, 1) * 255).astype(np.uint8), "L").save(out, "PNG")
    return out.getvalue()


def found_mask(probability: np.ndarray, threshold: float = THRESHOLD):
    """ A probability map as a mask image, white where the engine is sure, or None if it found nothing. """
    from PIL import Image
    inside = probability >= threshold
    return Image.fromarray(inside.astype(np.uint8) * 255, "L") if inside.any() else None
