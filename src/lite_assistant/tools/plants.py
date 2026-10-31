"""
identify_plant: the species of the plant in the user's latest photo, from
an image classification engine that names plants.
"""
from typing import Any, Dict, List, Optional

from ..media import latest_image

# The likeliest species a reply gives, and how likely the first must be to be called a match
TOP, SURE = 3, 0.5


class PlantIdentifier:
    def __init__(self, engine):
        self.engine = engine

    def identify_plant(self, conversation: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Identifies the species of the plant, flower or tree in the user's latest photo, by its Latin
        name. Use it only when the user asks what a plant is, never for an animal, a person or a
        thing. Give its common name too, if you know it.
        """
        photo = latest_image(conversation or [])
        if photo is None:
            return {"error": "The user hasn't sent a photo. Ask them to send one."}
        ranked = self.engine.classify(photo, TOP)
        found: Dict[str, Any] = {"likeliest": [f"{name} ({score:.0%})" for name, score in ranked]}
        if ranked[0][1] < SURE:
            found["note"] = "Not a confident match: say it might be one of these, or ask for a closer photo of a leaf or flower."
        found["limits"] = ("It knows 1081 species, mostly of Europe, and names one even for a photo without a plant, "
                           "so check that against what you see.")
        return found
