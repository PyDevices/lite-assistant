"""
Naming what a photo shows from a list, such as a plant's species: the
endpoint and the identify_plant tool, with a fake engine.
"""
import base64
import io
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from lite_assistant.app import Engines, create_app
from lite_assistant.engines import ImageClassifierEngine
from lite_assistant.engines.tflite import open_photo
from lite_assistant.tools.plants import PlantIdentifier


def jpeg() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (64, 48), (40, 160, 40)).save(out, "JPEG")
    return out.getvalue()


class FakePlants(ImageClassifierEngine):
    MODEL_ID, TASK = "fake-plantnet", "plant"

    def __init__(self, first=0.7):
        self.first = first

    def classify(self, image, top=5):
        open_photo(image)
        found = [("Calendula officinalis L.", self.first), ("Tagetes erecta L.", 0.2), ("Papaver rhoeas L.", 0.05)]
        return found[:top]


class TestEndpoint(unittest.TestCase):
    def setUp(self):
        self.http = TestClient(create_app(Engines({}, {}, {}, recognize={"plantnet": FakePlants()}), None, "none"), client=("127.0.0.1", 50000))

    def post(self, **options):
        return self.http.post("/v1/images/classifications", params=options, content=jpeg(),
                              headers={"Content-Type": "image/jpeg"})

    def test_classifies(self):
        for model in (None, "plant", "plantnet", "fake-plantnet"):
            body = self.post(**({"model": model} if model else {}), top=2).json()
            self.assertEqual(body["model"], "fake-plantnet")
            self.assertEqual(body["data"], [{"label": "Calendula officinalis L.", "score": 0.7},
                                            {"label": "Tagetes erecta L.", "score": 0.2}])

    def test_bad_requests(self):
        self.assertEqual(self.post(model="birds").status_code, 404)
        self.assertEqual(self.post(top="0").status_code, 400)
        bad = self.http.post("/v1/images/classifications", content=b"x", headers={"Content-Type": "image/jpeg"})
        self.assertEqual(bad.status_code, 400)

    def test_listed_as_a_kind(self):
        kinds = {m["id"]: m.get("kind") for m in self.http.get("/v1/models").json()["data"]}
        self.assertEqual(kinds.get("plantnet"), "image_classification")


class TestTool(unittest.TestCase):
    conversation = [{"role": "user", "content": [
        {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(jpeg()).decode()}]}]

    def test_names_the_likeliest(self):
        found = PlantIdentifier(FakePlants()).identify_plant(self.conversation)
        self.assertEqual(found["likeliest"], ["Calendula officinalis L. (70%)", "Tagetes erecta L. (20%)",
                                              "Papaver rhoeas L. (5%)"])
        self.assertNotIn("note", found)

    def test_unsure(self):
        self.assertIn("note", PlantIdentifier(FakePlants(0.3)).identify_plant(self.conversation))

    def test_without_a_photo(self):
        self.assertIn("error", PlantIdentifier(FakePlants()).identify_plant([]))


if __name__ == "__main__":
    unittest.main()
