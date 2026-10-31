"""
PlantNet-300K plant identification engine, on LiteRT.
"""
import json
from typing import List, Tuple

import numpy as np

from . import tflite
from .image_classifier_base import ImageClassifierEngine

# Pl@ntNet-300K's ResNet18, by cpoisson, converted by litert-community; Apache-2.0, 47 MB
REPO = "litert-community/PlantNet-300K-ResNet18-LiteRT"
FILE, NAMES = "plantnet.tflite", "plantnet300K_species_id_2_name.json"
SIZE, RESIZE = 224, 256
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class PlantNetEngine(ImageClassifierEngine):
    """
    PlantNet-300K: a plant's species, one of 1081, by its Latin name, from a
    photo centered on it, in about 20 ms on the CPU.
    """
    MODEL_ID, TASK = "plantnet-300k-resnet18", "plant"

    def __init__(self):
        paths = tflite.files("PlantNet", REPO, [FILE, NAMES])
        self.graphs = None
        if paths:
            with open(paths[1], encoding="utf-8") as f:
                names = json.load(f)
            # The graph scores the species in the order of their ids, sorted as strings
            self.labels = [names[species] for species in sorted(names)]
            self.graphs = tflite.load_graphs("PlantNet", {"species": paths[0]})

    def classify(self, image: bytes, top: int = 5) -> List[Tuple[str, float]]:
        if not self.graphs:
            raise RuntimeError("PlantNet is not available.")
        from PIL import Image
        photo = tflite.open_photo(image)
        # As PlantNet-300K's evaluation: the short side scaled to RESIZE, and the center SIZE square
        scale = RESIZE / min(photo.size)
        photo = photo.resize((max(SIZE, round(photo.width * scale)), max(SIZE, round(photo.height * scale))),
                             Image.BILINEAR)
        left, top_edge = (photo.width - SIZE) // 2, (photo.height - SIZE) // 2
        photo = photo.crop((left, top_edge, left + SIZE, top_edge + SIZE))
        x = ((np.asarray(photo, np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
        logits = self.graphs["species"](x)[0][0].astype(np.float64)
        chances = np.exp(logits - logits.max())
        chances /= chances.sum()
        return [(self.labels[i], float(chances[i])) for i in np.argsort(-chances)[:top]]

    @classmethod
    def setup(cls):
        tflite.download("PlantNet", REPO, [FILE, NAMES])
