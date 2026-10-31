"""
Base interface for line engines, which find lines to follow in a photo
from a camera looking ahead, such as tape on a floor or a road's lanes.
"""
from typing import List, Optional, Tuple

# A point as fractions of the photo's width and height, from the top left
Point = Tuple[float, float]


class LineEngine:
    """
    Abstract base class for all line engines. TASK says what it finds, such
    as "line" or "lanes".
    """
    MODEL_ID = ""
    TASK = ""

    def get_model_id(self) -> str:
        return self.MODEL_ID

    def find(self, photo) -> List[List[Point]]:
        """
        The lines in a PIL photo, left to right, each as points from the
        nearest (the bottom of the photo) to the farthest.
        """
        raise NotImplementedError

    def path(self, lines: List[List[Point]]) -> Optional[List[Point]]:
        """ The way to go, nearest first: by default the line that reaches nearest, or None. """
        return max(lines, key=lambda line: line[0][1], default=None)

    @classmethod
    def setup(cls):
        """
        Performs first-time setup, such as downloading default models.
        """
        pass
