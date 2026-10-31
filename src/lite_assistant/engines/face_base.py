"""
Base interface for face engines, which tell something about a face that a
detection engine found, such as its expression or where it's looking.
"""
from typing import TYPE_CHECKING, Any, Dict, Tuple

if TYPE_CHECKING:
    from PIL import Image


class FaceEngine:
    """
    Abstract base class for all face engines.
    """
    MODEL_ID = ""

    def get_model_id(self) -> str:
        return self.MODEL_ID

    def describe(self, photo: "Image.Image", box: Tuple[float, float, float, float]) -> Dict[str, Any]:
        """
        What the engine tells about the face in a box of a photo (left, top,
        right and bottom, as fractions of its width and height), as fields
        to add to the face's listing.
        """
        raise NotImplementedError

    @classmethod
    def setup(cls):
        """
        Performs first-time setup, such as downloading default models.
        """
        pass
