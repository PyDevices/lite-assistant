"""
estimate_distance: how far things in the user's latest photo are from the
camera, from the depth engine, and with a detection engine, which objects
are where. A segmentation engine finds a named thing the detector doesn't
know, such as a tree or a door.
"""
from typing import Any, Dict, List, Optional

from ..depth import FAR, NEAR, at, extreme, place
from .. import segmentation
from ..detection import distance, matching, subject
from ..media import latest_image

FEET_PER_METER = 3.28084
# The most objects a reply lists
MAX_OBJECTS = 6


def spoken(distance: Optional[float]) -> Optional[str]:
    """ A distance in meters and feet, as precisely as a guess from one photo deserves. """
    if distance is None or distance != distance:
        return None
    feet = distance * FEET_PER_METER
    return (f"about {distance:.1f} meters, or {feet:.0f} feet" if distance < 10
            else f"about {distance:.0f} meters, or {feet:.0f} feet")


class Distances:
    def __init__(self, depth_engine, detection_engine=None, segment_engine=None):
        self.engine, self.detector, self.segmenter = depth_engine, detection_engine, segment_engine

    def estimate_distance(self, where: str = "center", thing: str = "",
                          conversation: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Estimates how far things in the user's latest photo are from the
        camera. Use it when the user asks how far away something in a photo is.

        Args:
            where: The part of the photo: center, left, right, top, bottom, or a corner such as top left.
            thing: The object to measure, if the user named one, such as "the chair" or "the dog".
        """
        photo = latest_image(conversation or [])
        if photo is None:
            return {"error": "The user hasn't sent a photo. Ask them to send one."}
        depth = self.engine.estimate(photo)
        seen = self.detector.detect(photo) if self.detector else []
        found = {}
        wanted = matching(seen, thing) if thing and self.detector else []
        # What the detector doesn't know, the segmentation engine looks for by name
        named = None
        if thing and not wanted and self.segmenter:
            probability = self.segmenter.segment(photo, [subject(thing)])[0]
            named = segmentation.region(probability)
        if wanted:
            found["measured"] = [{"object": d.label, "at": f"the {place(*center(d))} of the photo",
                                  "distance": spoken(distance(depth, d)) or "unknown"} for d in wanted[:MAX_OBJECTS]]
        elif named:
            found["measured"] = [{"object": thing, "at": f"the {place(*named['center'])} of the photo",
                                  "distance": spoken(segmentation.distance(depth, probability)) or "unknown"}]
        else:
            if thing and (self.detector or self.segmenter):
                found["note"] = (f"Nothing like \"{thing}\" was recognized in the photo, "
                                 f"so this measures the {where or 'center'}.")
            name, measured = at(depth, where)
            found["measured_at"] = f"the {name} of the photo"
            found["distance"] = spoken(measured) or "unknown: there's nothing there to measure, such as sky"
        for key, percentile in (("nearest", NEAR), ("farthest", FAR)):
            point = extreme(depth.distances, percentile)
            if point["distance"] is not None:
                found[key] = f"{spoken(point['distance'])}, at the {place(point['x'], point['y'])}"
        others = [d for d in seen if d not in wanted][:MAX_OBJECTS]
        if others:
            found["other_objects"] = [f"{d.label} at the {place(*center(d))}, "
                                      f"{spoken(distance(depth, d)) or 'distance unknown'}" for d in others]
        found["accuracy"] = "a rough guess from one photo; it can be off by a fifth or more"
        return found


def center(found) -> tuple:
    """ The middle of a detected object's box, as fractions of the photo's width and height. """
    left, top, right, bottom = found.box
    return (left + right) / 2, (top + bottom) / 2
