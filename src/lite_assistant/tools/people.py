"""
describe_people: the people in the user's latest photo, from the detection,
face and pose engines: their faces' expressions and whether they're looking
at the camera, and raised hands, with the fingers they hold up.
"""
from typing import Any, Dict, List, Optional

from .. import people
from ..depth import place
from ..media import latest_image

# The most people a reply describes
MAX_PEOPLE = 6
# An expression is told only for a face at least this tall, in pixels, and when it's at least this likely
EXPRESSION_SIZE, EXPRESSION_CHANCE = 48, 0.5


class PeopleDescriber:
    def __init__(self, detector, face_detector=None, faces=(), body=None, hand=None):
        self.detector, self.face_detector = detector, face_detector
        self.faces, self.body, self.hand = list(faces), body, hand

    def describe_people(self, conversation: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Describes the people in the user's latest photo: how many there are, their facial expressions,
        whether they're looking at the camera, and raised hands with the fingers held up. Use it when the
        user asks about the people in a photo, how someone looks or feels, or what gesture they're making.
        """
        from ..engines.tflite import open_photo
        image = latest_image(conversation or [])
        if image is None:
            return {"error": "The user hasn't sent a photo. Ask them to send one."}
        photo = open_photo(image)
        persons = [d for d in self.detector.detect(image) if d.label == "person"] if self.detector else []
        found: Dict[str, Any] = {"people": len(persons)}
        if self.face_detector and self.faces:
            faces = self.face_detector.detect(image)[:MAX_PEOPLE]
            described = []
            for face, extra in zip(faces, people.describe(faces, photo, self.faces)):
                center = ((face.box[0] + face.box[2]) / 2, (face.box[1] + face.box[3]) / 2)
                parts = [f"a face at the {place(*center)}"]
                tall = (face.box[3] - face.box[1]) * photo.height >= EXPRESSION_SIZE
                if "expression" in extra and tall and extra["expressions"][extra["expression"]] >= EXPRESSION_CHANCE:
                    parts.append(f"looks {extra['expression']}")
                if "looking_at_camera" in extra:
                    parts.append("looking at the camera" if extra["looking_at_camera"] else "looking away")
                described.append(", ".join(parts))
            found["faces"] = described or "no faces seen clearly"
            if described and not any("looks " in d for d in described):
                found["note"] = "The faces are too small or unclear to read their expressions here; judge them from the photo yourself."
            if not persons:
                found["people"] = len(faces)
        if self.body and persons:
            gestures = []
            persons = persons[:MAX_PEOPLE]
            for person, extra in zip(persons, people.describe(persons, photo, [], self.body, self.hand)):
                center = ((person.box[0] + person.box[2]) / 2, (person.box[1] + person.box[3]) / 2)
                for side in extra["raised_hands"]:
                    gesture = f"the person at the {place(*center)} has their {side} hand raised"
                    held = next((h for h in extra.get("hands", []) if h["side"] == side), None)
                    if held is not None:
                        fingers = held["fingers_up"]
                        gesture += (f", holding up {len(fingers)} finger{'s' if len(fingers) != 1 else ''}"
                                    f" ({', '.join(fingers)})" if fingers else ", in a fist")
                    gestures.append(gesture)
            found["gestures"] = gestures or "no one has a hand raised"
        return found
