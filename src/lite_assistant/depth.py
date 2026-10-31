"""
Distances in a depth map, as /v1/depth reports them and the estimate_distance
tool tells the model: the distance at the center, the nearest and farthest
points, a coarse grid, and a depth image.
"""
import io
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .engines import DepthMap

# The center reading, like a rangefinder's: the median over this fraction of the width and height
CENTER = 0.1
# The nearest and farthest points are these percentiles, so a few stray pixels don't count
NEAR, FAR = 2, 98
# The most cells a grid may have across or down
MAX_GRID = 64


def meters(value: float) -> Optional[float]:
    """ A distance rounded to the centimeter, or None for no estimate. """
    return None if value is None or not np.isfinite(value) else round(float(value), 2)


def median(distances: np.ndarray) -> float:
    valid = distances[np.isfinite(distances)]
    return float(np.median(valid)) if valid.size else float("nan")


def box(distances: np.ndarray, left: float, top: float, right: float, bottom: float) -> np.ndarray:
    """ The part of the map between these fractions of its width and height. """
    height, width = distances.shape
    x0, x1 = int(left * width), max(int(left * width) + 1, int(round(right * width)))
    y0, y1 = int(top * height), max(int(top * height) + 1, int(round(bottom * height)))
    return distances[y0:y1, x0:x1]


def center(distances: np.ndarray) -> float:
    half = CENTER / 2
    return median(box(distances, 0.5 - half, 0.5 - half, 0.5 + half, 0.5 + half))


def extreme(distances: np.ndarray, percentile: float) -> Dict[str, Any]:
    """
    The distance at a percentile of the map's, and where it is: the middle
    of the points at least that near (or that far), as fractions of the
    width and height from the top left.
    """
    valid = np.isfinite(distances)
    if not valid.any():
        return {"distance": None, "x": None, "y": None}
    value = float(np.percentile(distances[valid], percentile))
    with np.errstate(invalid="ignore"):
        beyond = valid & ((distances <= value) if percentile < 50 else (distances >= value))
    ys, xs = np.nonzero(beyond)
    height, width = distances.shape
    return {"distance": meters(value), "x": round(float(xs.mean() + 0.5) / width, 3),
            "y": round(float(ys.mean() + 0.5) / height, 3)}


def grid(distances: np.ndarray, columns: int, rows: int) -> List[List[Optional[float]]]:
    """ The median distance in each cell of a grid over the map, row by row from the top. """
    return [[meters(median(box(distances, c / columns, r / rows, (c + 1) / columns, (r + 1) / rows)))
             for c in range(columns)] for r in range(rows)]


def summary(depth: DepthMap, columns: int = 8, rows: int = 8) -> Dict[str, Any]:
    distances = depth.distances
    height, width = distances.shape
    return {
        "width": width, "height": height,
        "horizontal_fov": None if depth.horizontal_fov is None else round(depth.horizontal_fov, 1),
        "center": meters(center(distances)),
        "nearest": extreme(distances, NEAR),
        "farthest": extreme(distances, FAR),
        "grid": grid(distances, columns, rows),
        "coverage": round(float(np.isfinite(distances).mean()), 3),
    }


def image(depth: DepthMap) -> Tuple[bytes, Optional[float], Optional[float]]:
    """
    The map as a grayscale PNG, near points bright and far ones dark, with
    no estimate black. Shading is by inverse distance, which shows near
    detail. Returns the PNG and the distances its brightest and darkest
    grays stand for.
    """
    from PIL import Image
    distances = depth.distances
    valid = np.isfinite(distances) & (distances > 0)
    gray = np.zeros(distances.shape, np.uint8)
    near = far = None
    if valid.any():
        near, far = (float(np.percentile(distances[valid], p)) for p in (NEAR, FAR))
        far = max(far, near * 1.01)
        inverse = 1 / np.clip(distances[valid], near, far)
        gray[valid] = np.round(1 + 254 * (inverse - 1 / far) / (1 / near - 1 / far)).astype(np.uint8)
    out = io.BytesIO()
    Image.fromarray(gray, "L").save(out, "PNG")
    return out.getvalue(), meters(near), meters(far)


# Words that pick a part of the photo, as thirds of its width and height
ROWS = {"top": 0, "upper": 0, "bottom": 2, "lower": 2}
COLUMNS = {"left": 0, "right": 2}


def at(depth: DepthMap, where: str = "") -> Tuple[str, float]:
    """
    The distance to a part of the photo named in words: "center", "left",
    "top right" and so on, each a third of the width and height. The center
    is the rangefinder reading. Returns the part's name and its distance.
    """
    words = re.findall(r"[a-z]+", (where or "").lower())
    row = next((ROWS[w] for w in words if w in ROWS), 1)
    column = next((COLUMNS[w] for w in words if w in COLUMNS), 1)
    if (row, column) == (1, 1):
        return "center", center(depth.distances)
    name = " ".join(filter(None, (["top", "", "bottom"][row], ["left", "", "right"][column])))
    return name, median(box(depth.distances, column / 3, row / 3, (column + 1) / 3, (row + 1) / 3))


def place(x: float, y: float) -> str:
    """ Where a point is in the photo, in words. """
    row = ["top", "", "bottom"][min(2, int(y * 3))]
    column = ["left", "", "right"][min(2, int(x * 3))]
    return " ".join(filter(None, (row, column))) or "center"
