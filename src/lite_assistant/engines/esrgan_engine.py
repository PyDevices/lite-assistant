"""
Real-ESRGAN upscaling engine, on LiteRT.
"""
from typing import TYPE_CHECKING, Optional

import numpy as np

from . import tflite
from .image_edit_base import ImageEditEngine

if TYPE_CHECKING:
    from PIL import Image

# xinntao's Real-ESRGAN general x4v3, converted by litert-community; BSD 3-Clause, 5 MB
REPO = "litert-community/real-esrgan-x4v3-litert"
FILE = "realesr_general_x4v3.tflite"
# The graph's fixed square input, and how much bigger it makes it
TILE, SCALE = 128, 4
# Tiles overlap by this many pixels, so their edges, which see less around them, are cut off
OVERLAP = 8
# A photo with more pixels than this is too big to upscale: its result would be 16 times as many
MAX_PIXELS = 1024 * 1024


class ESRGANEngine(ImageEditEngine):
    """
    Real-ESRGAN, which makes a photo 4 times as wide and tall, filling in
    detail rather than blurring it. The graph sees 128x128 tiles, so the
    photo is upscaled tile by tile, the tiles overlapping so their seams
    don't show.
    """
    MODEL_ID = "real-esrgan-x4v3"
    TASK = "upscale"

    def __init__(self):
        paths = tflite.files("Real-ESRGAN", REPO, [FILE])
        loaded = tflite.load_graphs("Real-ESRGAN", {"net": paths[0]}) if paths else None
        self.graph = loaded["net"] if loaded else None

    def edit(self, photo: "Image.Image", prompt: Optional[str] = None, mask: Optional["Image.Image"] = None) -> "Image.Image":
        from PIL import Image
        if not self.graph:
            raise RuntimeError("Real-ESRGAN is not available.")
        if photo.width * photo.height > MAX_PIXELS:
            raise ValueError(f"The photo is {photo.width}x{photo.height}; upscaling takes up to {MAX_PIXELS} pixels.")
        pixels = np.asarray(photo.convert("RGB"), np.float32) / 255
        wide, tall = photo.size
        # A photo smaller than a tile is stretched to one with its edge pixels, so no black is upscaled into it
        pixels = np.pad(pixels, ((0, max(TILE - tall, 0)), (0, max(TILE - wide, 0)), (0, 0)), mode="edge")
        height, width = pixels.shape[:2]
        out = np.zeros((height * SCALE, width * SCALE, 3), np.float32)
        step = TILE - 2 * OVERLAP
        for top in range(0, height, step):
            for left in range(0, width, step):
                # The tile around this step's square, moved inside the photo where it can be
                y0, x0 = max(min(top - OVERLAP, height - TILE), 0), max(min(left - OVERLAP, width - TILE), 0)
                big = self.graph(pixels[y0:y0 + TILE, x0:x0 + TILE][None])[0][0].transpose(1, 2, 0)
                # Keep only the step's square of it
                y1, x1 = min(top + step, height), min(left + step, width)
                out[top * SCALE:y1 * SCALE, left * SCALE:x1 * SCALE] = \
                    big[(top - y0) * SCALE:(y1 - y0) * SCALE, (left - x0) * SCALE:(x1 - x0) * SCALE]
        return Image.fromarray(np.round(out[:tall * SCALE, :wide * SCALE].clip(0, 1) * 255).astype(np.uint8))

    @classmethod
    def setup(cls):
        tflite.download("Real-ESRGAN", REPO, [FILE])
