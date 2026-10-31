"""
CPGA-Net low-light enhancement engine, on LiteRT.
"""
from typing import TYPE_CHECKING, Optional

import numpy as np

from . import tflite
from .image_edit_base import ImageEditEngine

if TYPE_CHECKING:
    from PIL import Image

# Shyandram's CPGA-Net, converted by litert-community; MIT, 0.1 MB
REPO = "litert-community/CPGA-Net-LowLight-LiteRT"
FILE = "cpga_fp16.tflite"
# The graph's fixed square input
SIZE = 256


class CPGAEngine(ImageEditEngine):
    """
    CPGA-Net, which brightens a dark photo in about 30 ms on the CPU. The
    graph sees 256x256, so the brightening it does there is scaled up and
    applied to the photo, which keeps its resolution and detail.
    """
    MODEL_ID = "cpga-net"
    TASK = "brighten"

    def __init__(self):
        paths = tflite.files("CPGA", REPO, [FILE])
        loaded = tflite.load_graphs("CPGA", {"net": paths[0]}) if paths else None
        self.graph = loaded["net"] if loaded else None

    def edit(self, photo: "Image.Image", prompt: Optional[str] = None, mask: Optional["Image.Image"] = None) -> "Image.Image":
        from PIL import Image
        if not self.graph:
            raise RuntimeError("CPGA-Net is not available.")
        full = np.asarray(photo, np.float32) / 255
        small = np.asarray(photo.resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255
        out = self.graph(small.transpose(2, 0, 1)[None])[0][0].transpose(1, 2, 0)
        # Each pixel's gain, smooth enough at 256 to stretch over the full photo
        gain = (out + 1e-3) / (small + 1e-3)
        gain = np.stack([np.asarray(Image.fromarray(gain[..., c], "F").resize(photo.size, Image.BILINEAR))
                         for c in range(3)], axis=-1)
        return Image.fromarray(np.round((full * gain).clip(0, 1) * 255).astype(np.uint8))

    @classmethod
    def setup(cls):
        tflite.download("CPGA", REPO, [FILE])
