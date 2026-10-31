"""
MI-GAN inpainting engine, on LiteRT.
"""
from typing import TYPE_CHECKING, Optional

import numpy as np

from . import tflite
from .image_edit_base import ImageEditEngine

if TYPE_CHECKING:
    from PIL import Image

# Picsart's MI-GAN trained on Places2, converted by litert-community; MIT, 15 MB
REPO = "litert-community/MI-GAN-512-Places2-LiteRT"
FILE = "migan_fp16.tflite"
# The graph's fixed square input
SIZE = 512
# The mask is grown by this fraction of the region's size, so an object's edges and shadow go too
GROW = 0.04
# The region around the mask that the graph sees is this many times the mask's size, so it sees what to fill with
CONTEXT = 2.0


class MIGANEngine(ImageEditEngine):
    """
    MI-GAN, which erases what a mask covers and fills it in from what's
    around it. The graph sees 512x512, so it's given the mask's part of the
    photo, with room around it, and only the masked pixels of what it makes
    are put back, at the photo's resolution.
    """
    MODEL_ID = "mi-gan-512-places2"
    TASK = "erase"
    MASKED = True

    def __init__(self):
        paths = tflite.files("MI-GAN", REPO, [FILE])
        loaded = tflite.load_graphs("MI-GAN", {"net": paths[0]}) if paths else None
        self.graph = loaded["net"] if loaded else None

    def edit(self, photo: "Image.Image", prompt: Optional[str] = None, mask: Optional["Image.Image"] = None) -> "Image.Image":
        from PIL import Image, ImageFilter
        if not self.graph:
            raise RuntimeError("MI-GAN is not available.")
        if mask is None:
            raise ValueError("Erasing needs a mask, or a prompt saying what to erase.")
        photo = photo.convert("RGB")
        from scipy import ndimage
        # What an outline encloses goes too, such as a pair of glasses' lenses
        erase = ndimage.binary_fill_holes(np.asarray(mask.convert("L").resize(photo.size, Image.BILINEAR)) >= 128)
        if not erase.any():
            return photo
        ys, xs = np.nonzero(erase)
        left, top, right, bottom = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        grow = int(max(right - left, bottom - top) * GROW) * 2 + 1
        erase = np.asarray(Image.fromarray(erase.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(min(grow, 31)))) >= 128
        # The mask's box with room around it to see what to fill with, inside the photo
        half_w = max((right - left) * CONTEXT, SIZE / 2) / 2
        half_h = max((bottom - top) * CONTEXT, SIZE / 2) / 2
        cx, cy = (left + right) / 2, (top + bottom) / 2
        region = (int(max(cx - half_w, 0)), int(max(cy - half_h, 0)),
                  int(min(cx + half_w, photo.width)), int(min(cy + half_h, photo.height)))
        size = (region[2] - region[0], region[3] - region[1])
        cut = Image.fromarray(erase.astype(np.uint8) * 255).crop(region)
        rgb = np.asarray(photo.crop(region).resize((SIZE, SIZE), Image.BICUBIC), np.float32) / 127.5 - 1
        keep = 1 - (np.asarray(cut.resize((SIZE, SIZE), Image.NEAREST)) >= 128).astype(np.float32)
        out = self.graph(np.concatenate([keep[None] - 0.5, (rgb * keep[..., None]).transpose(2, 0, 1)])[None])[0][0]
        filled = Image.fromarray(np.round((out.transpose(1, 2, 0).clip(-1, 1) + 1) * 127.5).astype(np.uint8))
        # Back at the region's size, put in only where the mask erases
        canvas = photo.copy()
        canvas.paste(filled.resize(size, Image.BICUBIC), region[:2], cut)
        return canvas

    @classmethod
    def setup(cls):
        tflite.download("MI-GAN", REPO, [FILE])
