"""
What the face and pose engines tell about the people a detection engine
found: faces' expressions and gaze, and bodies' and hands' joints, with
the gestures they make, such as a raised hand or fingers held up.
"""
import math
from typing import Any, Dict, List, Optional, Tuple

# A joint the pose engine is less sure of than this isn't used for gestures
SURE = 0.3
# A hand is raised when its wrist is above its shoulder by this fraction of the shoulders' width
RAISED = 0.25
# A hand's box is centered this far past the wrist, along the forearm, and
# is this many forearms wide, as hand crops from a body pose usually are
PAST_WRIST, HAND_SIZE = 0.35, 1.0
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
# A hand smaller than this, in pixels, is too small to read its fingers
MIN_HAND = 32


def joint(keypoints: Dict[str, Tuple[float, float, float]], name: str, aspect: float) -> Optional[Tuple[float, float]]:
    """ A joint the engine is sure of, in units of the photo's height, so distances are true; None if it isn't. """
    x, y, score = keypoints.get(name, (0, 0, 0))
    return (x * aspect, y) if score >= SURE else None


def raised_hands(keypoints: Dict[str, Tuple[float, float, float]], aspect: float) -> List[str]:
    """ Which hands, left or right (the person's own), a body pose holds up above the shoulders. """
    shoulders = [joint(keypoints, f"{side}_shoulder", aspect) for side in ("left", "right")]
    if not all(shoulders):
        return []
    width = math.dist(*shoulders)
    raised = []
    for side, shoulder in zip(("left", "right"), shoulders):
        wrist = joint(keypoints, f"{side}_wrist", aspect)
        if wrist and shoulder[1] - wrist[1] > RAISED * width:
            raised.append(side)
    return raised


def hand_box(keypoints: Dict[str, Tuple[float, float, float]], side: str,
             aspect: float) -> Optional[Tuple[float, float, float, float]]:
    """ Where a body pose's hand is, as a box of fractions of the photo, from its forearm; None if unsure. """
    elbow, wrist = joint(keypoints, f"{side}_elbow", aspect), joint(keypoints, f"{side}_wrist", aspect)
    if not elbow or not wrist:
        return None
    forearm = math.dist(elbow, wrist)
    x = wrist[0] + (wrist[0] - elbow[0]) * PAST_WRIST
    y = wrist[1] + (wrist[1] - elbow[1]) * PAST_WRIST
    half = forearm * HAND_SIZE / 2
    return ((x - half) / aspect, y - half, (x + half) / aspect, y + half)


def fingers_up(keypoints: Dict[str, Tuple[float, float, float]], aspect: float) -> List[str]:
    """
    The fingers a hand holds out straight. A finger is out when its tip is
    farther from the wrist than its middle joint; the thumb, when its tip is
    farther from the little finger's knuckle than its own middle joint.
    """
    wrist, pinky = joint(keypoints, "wrist", aspect), joint(keypoints, "pinky_mcp", aspect)
    out = []
    for finger in FINGERS:
        middle, tip = (joint(keypoints, f"{finger}_ip" if finger == "thumb" else f"{finger}_pip", aspect),
                       joint(keypoints, f"{finger}_tip", aspect))
        base = pinky if finger == "thumb" else wrist
        if base and middle and tip and math.dist(base, tip) > math.dist(base, middle) * 1.1:
            out.append(finger)
    return out


def points(keypoints: Dict[str, Tuple[float, float, float]]) -> Dict[str, Dict[str, float]]:
    """ Joints as /v1/detect lists them. """
    return {name: {"x": x, "y": y, "score": score} for name, (x, y, score) in keypoints.items()}


def describe(found, photo, faces: List[Any], body=None, hand=None) -> List[Dict[str, Any]]:
    """
    For each detection, the fields the people engines add to its listing:
    what each face engine tells about a face, and with a body pose engine,
    a person's joints and raised hands, and with a hand pose engine too,
    each hand's joints and the fingers it holds up.
    """
    aspect = photo.width / photo.height
    extras = []
    for d in found:
        extra: Dict[str, Any] = {}
        if d.label == "face":
            for engine in faces:
                extra.update(engine.describe(photo, d.box))
        elif d.label == "person" and body is not None:
            keypoints = body.keypoints(photo, d.box)
            extra["keypoints"] = points(keypoints)
            extra["raised_hands"] = raised_hands(keypoints, aspect)
            if hand is not None:
                extra["hands"] = []
                for side in ("left", "right"):
                    box = hand_box(keypoints, side, aspect)
                    big = box and (box[3] - box[1]) * photo.height >= MIN_HAND
                    joints = hand.keypoints(photo, box) if big else {}
                    # A hand out of sight, or behind something, gets joints the engine isn't sure of
                    if joints and sum(score for _, _, score in joints.values()) / len(joints) >= SURE:
                        extra["hands"].append({"side": side, "fingers_up": fingers_up(joints, aspect),
                                               "keypoints": points(joints)})
        extras.append(extra)
    return extras
