"""
Base interface for image editing engines, which change a photo: brighten it,
remove its background, erase something, or upscale it.
"""
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from PIL import Image


class ImageEditEngine:
    """
    Abstract base class for all image editing engines. TASK names what an
    engine does, such as "brighten", so a request can ask for it by name.
    """
    MODEL_ID = ""
    TASK = ""
    # Whether the edit needs a mask: with a prompt instead, the server makes one with a segmentation engine
    MASKED = False

    def get_model_id(self) -> str:
        return self.MODEL_ID

    def edit(self, photo: "Image.Image", prompt: Optional[str] = None,
             mask: Optional["Image.Image"] = None) -> "Image.Image":
        """
        The edited photo. prompt says what to change, for the engines that
        need it, such as what to erase; mask, white where to change, can
        say it instead. Raises ValueError for an edit it can't make.
        """
        raise NotImplementedError

    @classmethod
    def setup(cls):
        """
        Performs first-time setup, such as downloading default models.
        """
        pass
