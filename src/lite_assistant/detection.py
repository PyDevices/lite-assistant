"""
Objects in a photo, as /v1/detect reports them and the estimate_distance tool
tells the model: what each is, where, and with a depth map, how far.
"""
import re
from typing import Any, Dict, List, Optional

import numpy as np

from .depth import box, meters, place
from .engines import DepthMap, Detection

# An object's distance is this percentile of the middle half of its box:
# background seen around or through it is farther, so a low one reads the object
OBJECT_PERCENTILE = 30
INNER = 0.25


def distance(depth: DepthMap, found: Detection) -> Optional[float]:
    """ How far an object is, in meters, or None if the map has nothing there. """
    left, top, right, bottom = found.box
    width, height = right - left, bottom - top
    inside = box(depth.distances, left + width * INNER, top + height * INNER,
                 right - width * INNER, bottom - height * INNER)
    valid = inside[np.isfinite(inside)]
    return meters(np.percentile(valid, OBJECT_PERCENTILE)) if valid.size else None


def objects(found: List[Detection], depth: Optional[DepthMap] = None) -> List[Dict[str, Any]]:
    """ The objects as /v1/detect lists them, each with its distance if there's a depth map. """
    listed = []
    for d in found:
        left, top, right, bottom = d.box
        x, y = round((left + right) / 2, 3), round((top + bottom) / 2, 3)
        item = {"label": d.label, "score": d.score,
                "box": {"left": left, "top": top, "right": right, "bottom": bottom},
                "center": {"x": x, "y": y}, "place": place(x, y)}
        if d.points:
            item["points"] = {name: {"x": x, "y": y} for name, (x, y) in d.points.items()}
        if depth is not None:
            item["distance"] = distance(depth, d)
        listed.append(item)
    return listed


def singular(word: str) -> str:
    """ A word without a plural ending: "chairs" to "chair", "boxes" to "box", "buses" to "bus". """
    for ending, replacement in (("uses", "us"), ("ies", "y"), ("sses", "ss"), ("xes", "x"), ("ches", "ch"),
                                ("shes", "sh")):
        if word.endswith(ending):
            return word[:-len(ending)] + replacement
    return word[:-1] if word.endswith("s") and not word.endswith(("ss", "us")) else word


# Words that end the name of a thing and begin where it is: "the grass behind the dog"
PLACES = {"behind", "beside", "next", "near", "on", "in", "under", "over", "above", "below", "by", "at",
          "of", "with", "from", "to", "between", "that", "which", "where"}


def subject(thing: str) -> str:
    """ What a name is of, without where it is: "the grass behind the dog" is "the grass". """
    words = thing.split()
    for i, word in enumerate(words):
        if i and word.lower().strip(",") in PLACES:
            return " ".join(words[:i])
    return thing


def matching(found: List[Detection], thing: str) -> List[Detection]:
    """ The objects a name such as "the chairs", "TV" or "the cup on the table" refers to. """
    words = [singular(w) for w in re.findall(r"[a-z]+", subject(thing).lower()) if w not in ("the", "a", "an", "my")]
    found_words = lambda d: [singular(w) for w in d.label.split()]
    # "table" finds a dining table, and "the tv screen" a tv; labels lose
    # plural endings too, so "bus" and "skis" match whatever the user says
    return [d for d in found if words and (" ".join(words) in found_words(d) or set(found_words(d)) <= set(words))]
