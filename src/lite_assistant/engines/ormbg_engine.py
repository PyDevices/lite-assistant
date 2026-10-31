"""
ormbg background removal engine, on LiteRT.
"""
from typing import TYPE_CHECKING, Optional

import numpy as np

from . import tflite
from .image_edit_base import ImageEditEngine

if TYPE_CHECKING:
    from PIL import Image

# schirrmacher's ormbg, converted by litert-community; Apache 2.0, 176 MB
REPO = "litert-community/ormbg-LiteRT"
FILE = "ormbg.tflite"
# The graph's fixed square input
SIZE = 1024


class OrmbgEngine(ImageEditEngine):
    """
    ormbg, which finds a photo's subject and makes the rest transparent. The
    graph sees 1024x1024; its matte is scaled back to the photo's size and
    becomes the photo's alpha channel.
    """
    MODEL_ID = "ormbg"
    TASK = "remove_background"

    def __init__(self):
        paths = tflite.files("ormbg", REPO, [FILE])
        loaded = tflite.load_graphs("ormbg", {"net": paths[0]}) if paths else None
        self.graph = loaded["net"] if loaded else None

    def matte(self, photo: "Image.Image") -> "Image.Image":
        """ How much of each pixel is the subject, as a grayscale image the photo's size. """
        from PIL import Image
        if not self.graph:
            raise RuntimeError("ormbg is not available.")
        small = np.asarray(photo.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255
        alpha = self.graph(small.transpose(2, 0, 1)[None])[0][0, 0]
        low, high = float(alpha.min()), float(alpha.max())
        alpha = (alpha - low) / max(high - low, 1e-6)
        return Image.fromarray(np.round(alpha * 255).astype(np.uint8)).resize(photo.size, Image.BILINEAR)

    def edit(self, photo: "Image.Image", prompt: Optional[str] = None, mask: Optional["Image.Image"] = None) -> "Image.Image":
        out = photo.convert("RGBA")
        out.putalpha(self.matte(photo))
        return out

    @classmethod
    def setup(cls):
        tflite.download("ormbg", REPO, [FILE])
