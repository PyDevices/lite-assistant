"""
Base interfaces and definitions for OCR engines, which read the text in a photo.
"""
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TextLine:
    """
    A line of text read from a photo: its text, how sure the reader is of
    it from 0 to 1, and its box as fractions of the photo's width and
    height (left, top, right, bottom).
    """
    text: str
    score: float
    box: Tuple[float, float, float, float]


class OCREngine:
    """
    Abstract base class for all OCR engines.
    """
    MODEL_ID = ""

    def get_model_id(self) -> str:
        return self.MODEL_ID

    def read(self, image: bytes) -> List[TextLine]:
        """
        The lines of text in a photo, in reading order: top to bottom, and
        left to right within a row. Raises ValueError if it isn't an image.
        """
        raise NotImplementedError

    @classmethod
    def setup(cls):
        """
        Performs first-time setup, such as downloading default models.
        """
        pass


def as_text(lines: List[TextLine]) -> str:
    """ Lines read from a photo as one text: lines side by side in a row joined by spaces, rows by newlines. """
    rows: List[List[TextLine]] = []
    for line in lines:
        middle = (line.box[1] + line.box[3]) / 2
        if rows and rows[-1][-1].box[1] <= middle <= rows[-1][-1].box[3] and line.box[0] >= rows[-1][-1].box[2]:
            rows[-1].append(line)
        else:
            rows.append([line])
    return "\n".join(" ".join(line.text for line in row) for row in rows)
