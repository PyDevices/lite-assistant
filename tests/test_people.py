"""
What the face and pose engines add about people: gestures from joints,
/v1/detect's describe, pose and hands options, and the describe_people tool,
with fake engines.
"""
import base64
import io
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from lite_assistant import people
from lite_assistant.app import Engines, create_app
from lite_assistant.engines import Detection, DetectionEngine, FaceEngine, PoseEngine
from lite_assistant.engines.rtmpose_engine import BODY, HAND
from lite_assistant.engines.tflite import crop
from lite_assistant.tools.people import PeopleDescriber


def jpeg(size=(400, 400)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, (128, 128, 128)).save(out, "JPEG")
    return out.getvalue()


def body(raised=(), score=0.9):
    """ A person standing in the middle of a square photo, raising the hands named. """
    joints = {name: (0.5, 0.5, score) for name in BODY}
    joints.update({"left_shoulder": (0.6, 0.3, score), "right_shoulder": (0.4, 0.3, score),
                   "left_elbow": (0.65, 0.45, score), "right_elbow": (0.35, 0.45, score),
                   "left_wrist": (0.65, 0.6, score), "right_wrist": (0.35, 0.6, score)})
    for side, x in (("left", 0.65), ("right", 0.35)):
        if side in raised:
            joints[f"{side}_elbow"] = (x, 0.2, score)
            joints[f"{side}_wrist"] = (x, 0.05, score)
    return joints


def hand(up=("index", "middle")):
    """ A hand, fingers up from a wrist at the bottom; the rest curled back toward it. """
    joints = {"wrist": (0.5, 0.9, 0.9)}
    for i, finger in enumerate(people.FINGERS):
        x = 0.3 + i * 0.1
        names = ["cmc", "mcp", "ip", "tip"] if finger == "thumb" else ["mcp", "pip", "dip", "tip"]
        ys = [0.7, 0.55, 0.4, 0.25] if finger in up else [0.7, 0.55, 0.6, 0.7]
        if finger == "thumb":
            # The thumb points away from the little finger, or folds across the palm toward it
            xs = [0.4, 0.3, 0.2, 0.1] if finger in up else [0.4, 0.45, 0.6, 0.7]
            joints.update({f"thumb_{n}": (x2, 0.7, 0.9) for n, x2 in zip(names, xs)})
        else:
            joints.update({f"{finger}_{n}": (x, y, 0.9) for n, y in zip(names, ys)})
    return joints


class FakeFaces(DetectionEngine):
    def get_model_id(self):
        return "fake-yunet"

    def get_labels(self):
        return ["face"]

    def detect(self, image, threshold=None):
        return [Detection("face", 0.9, (0.4, 0.1, 0.6, 0.3))]


class FakePeople(DetectionEngine):
    def get_model_id(self):
        return "fake-yolox"

    def get_labels(self):
        return ["person", "cup"]

    def detect(self, image, threshold=None):
        return [Detection("person", 0.9, (0.3, 0.0, 0.7, 1.0)), Detection("cup", 0.8, (0.0, 0.8, 0.1, 0.9))]


class FakeExpression(FaceEngine):
    MODEL_ID = "fake-hsemotion"

    def describe(self, photo, box):
        return {"expression": "happy", "expressions": {"happy": 0.9}}


class FakeGaze(FaceEngine):
    MODEL_ID = "fake-l2cs"

    def describe(self, photo, box):
        return {"gaze": {"yaw": 3.0, "pitch": -2.0}, "looking_at_camera": True}


class FakeBody(PoseEngine):
    MODEL_ID, TASK, KEYPOINTS = "fake-rtmpose", "body", BODY

    def keypoints(self, photo, box):
        return body(raised=("right",))


class FakeHand(PoseEngine):
    MODEL_ID, TASK, KEYPOINTS = "fake-rtmhand", "hand", HAND

    def __init__(self):
        self.boxes = []

    def keypoints(self, photo, box):
        self.boxes.append(box)
        return hand()


class TestGestures(unittest.TestCase):
    def test_raised_hands(self):
        self.assertEqual(people.raised_hands(body(), 1.0), [])
        self.assertEqual(people.raised_hands(body(raised=("left", "right")), 1.0), ["left", "right"])
        # Joints the engine isn't sure of say nothing
        self.assertEqual(people.raised_hands(body(raised=("left",), score=0.1), 1.0), [])

    def test_hand_box_is_past_the_wrist(self):
        left, top, right, bottom = people.hand_box(body(raised=("right",)), "right", 1.0)
        self.assertAlmostEqual((left + right) / 2, 0.35)
        self.assertLess((top + bottom) / 2, 0.05)
        self.assertAlmostEqual(bottom - top, 0.15)
        self.assertIsNone(people.hand_box(body(score=0.1), "left", 1.0))

    def test_fingers_up(self):
        self.assertEqual(people.fingers_up(hand(), 1.0), ["index", "middle"])
        self.assertEqual(people.fingers_up(hand(up=people.FINGERS), 1.0), people.FINGERS)
        self.assertEqual(people.fingers_up(hand(up=()), 1.0), [])

    def test_crop_grows_and_keeps_an_aspect(self):
        photo = Image.new("RGB", (200, 100))
        region, (left, top, width, height) = crop(photo, (0.25, 0.25, 0.75, 0.75), 1.0, aspect=1.0)
        self.assertEqual(region.size, (100, 100))
        self.assertEqual((left, top), (50, 0))


class TestDetect(unittest.TestCase):
    def setUp(self):
        self.hand = FakeHand()
        engines = Engines({}, {}, {}, detect={"yolox": FakePeople(), "yunet": FakeFaces()},
                          face={"hsemotion": FakeExpression(), "l2cs": FakeGaze()},
                          pose={"rtmposebody": FakeBody(), "rtmposehand": self.hand})
        self.http = TestClient(create_app(engines, None, "none"), client=("127.0.0.1", 50000))

    def detect(self, **options):
        return self.http.post("/v1/detect", params={"distances": "false", **options}, content=jpeg(),
                              headers={"Content-Type": "image/jpeg"}).json()["objects"]

    def test_faces_are_described(self):
        face = self.detect(model="yunet")[0]
        self.assertEqual((face["expression"], face["looking_at_camera"]), ("happy", True))
        self.assertNotIn("expression", self.detect(model="yunet", describe="false")[0])

    def test_pose_is_asked_for(self):
        person, cup = self.detect()
        self.assertNotIn("keypoints", person)
        person, cup = self.detect(pose="true")
        self.assertEqual(person["raised_hands"], ["right"])
        self.assertEqual(person["keypoints"]["nose"], {"x": 0.5, "y": 0.5, "score": 0.9})
        self.assertNotIn("hands", person)
        self.assertNotIn("keypoints", cup)

    def test_hands(self):
        person = self.detect(hands="true")[0]
        self.assertEqual([(h["side"], h["fingers_up"]) for h in person["hands"]],
                         [("left", ["index", "middle"]), ("right", ["index", "middle"])])
        self.assertEqual(len(person["hands"][0]["keypoints"]), 21)


class TestTool(unittest.TestCase):
    conversation = [{"role": "user", "content": [
        {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(jpeg()).decode()}]}]

    def test_describes_faces_and_gestures(self):
        tool = PeopleDescriber(FakePeople(), FakeFaces(), [FakeExpression(), FakeGaze()], FakeBody(), FakeHand())
        found = tool.describe_people(self.conversation)
        self.assertEqual(found["people"], 1)
        self.assertEqual(found["faces"], ["a face at the top, looks happy, looking at the camera"])
        self.assertEqual(found["gestures"], ["the person at the center has their right hand raised, "
                                             "holding up 2 fingers (index, middle)"])

    def test_small_faces_leave_expressions_to_the_model(self):
        tool = PeopleDescriber(FakePeople(), FakeFaces(), [FakeExpression()])
        found = tool.describe_people([{"role": "user", "content": [{"type": "input_image", "image_url":
                                       "data:image/jpeg;base64," + base64.b64encode(jpeg((100, 100))).decode()}]}])
        self.assertEqual(found["faces"], ["a face at the top"])
        self.assertIn("judge them from the photo yourself", found["note"])
        self.assertNotIn("gestures", found)

    def test_without_a_photo(self):
        self.assertIn("error", PeopleDescriber(FakePeople()).describe_people([]))


if __name__ == "__main__":
    unittest.main()
