"""
Base interface for pose engines, which find the joints of a person or a
hand that a detection engine found.
"""
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from PIL import Image


class PoseEngine:
    """
    Abstract base class for all pose engines. TASK says what it finds the
    joints of: "body" or "hand".
    """
    MODEL_ID = ""
    TASK = ""
    # The joints, in the order keypoints gives them
    KEYPOINTS: List[str] = []

    def get_model_id(self) -> str:
        return self.MODEL_ID

    def keypoints(self, photo: "Image.Image", box: Tuple[float, float, float, float]
                  ) -> Dict[str, Tuple[float, float, float]]:
        """
        Each joint of the person or hand in a box of a photo (left, top,
        right and bottom, as fractions of its width and height): its x and y
        as fractions of the photo's, and how sure the engine is of it, from
        0 to 1.
        """
        raise NotImplementedError

    @classmethod
    def setup(cls):
        """
        Performs first-time setup, such as downloading default models.
        """
        pass
