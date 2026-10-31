"""
A line finder for a robot car: tape or paint on a floor, with no model.
"""
from typing import List, Optional

import numpy as np

from .line_base import LineEngine, Point

# The photo is looked at this many pixels wide, in this many bands up from its bottom
WIDTH, BANDS = 160, 16
# The bands cover this share of the photo, from the bottom up, where the floor is
LOOK = 0.7
# A line differs from the floor by at least this much color, 0 to 441
CONTRAST = 40
# A line is this wide in the band, as a share of the photo's width
NARROWEST, WIDEST = 0.02, 0.35
# From one band to the next, a line moves at most this share of the width
STEP = 0.12


def runs(on: np.ndarray):
    """ The (start, end) of each stretch where a row of booleans is on. """
    edges = np.diff(np.concatenate([[0], on.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


class LineFinderEngine(LineEngine):
    """
    Finds a line that stands out from the floor, darker, lighter or a
    different color, such as black tape on a light floor. Each band across
    the photo, from the bottom up, is compared with its own median color;
    the stretch that differs most is the line, followed up the photo while
    it stays near where it was. About 7 ms for a 640x480 photo.
    """
    MODEL_ID, TASK = "line-finder", "line"

    def find(self, photo) -> List[List[Point]]:
        height = max(1, round(photo.height * WIDTH / photo.width))
        pixels = np.asarray(photo.convert("RGB").resize((WIDTH, height)), np.float32)
        top = int(height * (1 - LOOK))
        edges = np.linspace(height, top, BANDS + 1).astype(int)
        line: List[Point] = []
        last: Optional[float] = None
        for low, high in zip(edges[:-1], edges[1:]):
            band = pixels[high:low]
            if not len(band):
                continue
            # How far each column is from the band's usual color
            profile = np.linalg.norm(band - np.median(band.reshape(-1, 3), axis=0), axis=2).mean(axis=0)
            threshold = max(CONTRAST, profile.max() / 2)
            best, best_mass = None, 0.0
            for start, end in runs(profile > threshold):
                width = (end - start) / WIDTH
                if not NARROWEST <= width <= WIDEST:
                    continue
                weights = profile[start:end]
                center = float((np.arange(start, end) + 0.5) @ weights / weights.sum()) / WIDTH
                if last is not None and abs(center - last) > STEP:
                    continue
                if weights.sum() > best_mass:
                    best, best_mass = center, float(weights.sum())
            if best is None:
                # The line ended, or there isn't one
                if line:
                    break
                continue
            line.append((round(best, 4), round((low + high) / 2 / height, 4)))
            last = best
        # A few points could be anything; a line runs through several bands
        return [line] if len(line) >= 3 else []
