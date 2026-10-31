"""
Lines to follow, as a board steering a car wants them: where the way
ahead is, from a line engine's points.
"""
import io
import math
from typing import Any, Dict, List, Optional, Tuple

from .engines.line_base import Point


def measure(path: Optional[List[Point]], size: Tuple[int, int]) -> Optional[Dict[str, Any]]:
    """
    Where a path is, from its points, nearest first: offset, how far its
    nearest point is from the photo's middle (-1 at the left edge, 1 at
    the right); angle, how many degrees it leans right of straight up the
    photo over its nearest half; and ahead, its farthest point's offset.
    """
    if not path:
        return None
    width, height = size
    near = path[:max(2, (len(path) + 1) // 2)]
    (x0, y0), (x1, y1) = near[0], near[-1]
    angle = math.degrees(math.atan2((x1 - x0) * width, (y0 - y1) * height)) if len(near) > 1 else 0.0
    return {"offset": round(path[0][0] * 2 - 1, 3), "angle": round(angle, 1),
            "ahead": round(path[-1][0] * 2 - 1, 3), "points": [list(p) for p in path]}


def drawn(photo, lines: List[List[Point]], path: Optional[List[Point]]) -> bytes:
    """ The photo as a JPEG with the lines drawn on it in yellow and the path in green. """
    from PIL import ImageDraw
    photo = photo.convert("RGB")
    draw = ImageDraw.Draw(photo)
    radius = max(2, photo.width // 160)
    for points, color in [(line, (255, 210, 0)) for line in lines] + ([(path, (0, 220, 90))] if path else []):
        xy = [(x * photo.width, y * photo.height) for x, y in points]
        draw.line(xy, fill=color, width=radius)
        for x, y in xy:
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
    out = io.BytesIO()
    photo.save(out, "JPEG", quality=85)
    return out.getvalue()
