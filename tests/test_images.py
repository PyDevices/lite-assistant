"""
The image endpoints, editing and quality, and brightening dark photos
before detection, with fake engines.
"""
import base64
import io
import os
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from lite_assistant.app import Engines, create_app
from lite_assistant.engines import Detection, DetectionEngine, ImageEditEngine, LazyEngine, QualityEngine

TEST_IMAGE = os.path.join(os.path.dirname(__file__), "test_image.jpg")


def jpeg(brightness: int, size=(64, 48)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, (brightness,) * 3).save(out, "JPEG")
    return out.getvalue()


class FakeBrighten(ImageEditEngine):
    MODEL_ID, TASK = "fake-cpga", "brighten"

    def __init__(self):
        self.edits = []

    def edit(self, photo, prompt=None, mask=None):
        self.edits.append((photo.size, prompt, mask.size if mask else None))
        return photo.point(lambda v: min(255, v * 4))


class FakeEraser(ImageEditEngine):
    MODEL_ID, TASK = "fake-migan", "erase"

    def edit(self, photo, prompt=None, mask=None):
        if not prompt and mask is None:
            raise ValueError("say what to erase, in prompt or mask")
        return photo


class FakeQuality(QualityEngine):
    MODEL_ID = "fake-nima"

    def score(self, image):
        # Brighter fake photos score better
        level = Image.open(io.BytesIO(image)).convert("L").getpixel((0, 0))
        return {"technical": round(1 + level / 255 * 9, 2), "aesthetic": 5.0}


class SeesBrightness(DetectionEngine):
    """ Finds a "light" when the photo's top-left pixel is bright. """
    def get_model_id(self):
        return "fake-detector"

    def get_labels(self):
        return ["light"]

    def detect(self, image, threshold=None):
        level = Image.open(io.BytesIO(image)).convert("L").getpixel((0, 0))
        return [Detection("light", 0.9, (0.1, 0.1, 0.5, 0.5))] if level > 100 else []


class ImageTestCase(unittest.TestCase):
    def setUp(self):
        self.brighten = FakeBrighten()
        engines = Engines({}, {}, {}, edit={"cpga": self.brighten, "migan": FakeEraser()},
                          quality={"nima": FakeQuality()}, detect={"fake": SeesBrightness()})
        self.http = TestClient(create_app(engines, None, "none"), client=("127.0.0.1", 50000))
        self.http.__enter__()
        self.addCleanup(self.http.__exit__, None, None, None)


class TestEdits(ImageTestCase):
    def test_by_task_or_id_as_b64_json(self):
        for model in ("brighten", "cpga", "fake-cpga"):
            found = self.http.post("/v1/images/edits", files={"image": ("dark.jpg", jpeg(40))}, data={"model": model})
            self.assertEqual(found.status_code, 200, found.text)
            body = found.json()
            self.assertEqual(body["model"], "fake-cpga")
            edited = Image.open(io.BytesIO(base64.b64decode(body["data"][0]["b64_json"])))
            self.assertEqual(edited.size, (64, 48))
            self.assertGreater(edited.convert("L").getpixel((0, 0)), 120)

    def test_png_and_prompt_and_mask(self):
        mask = io.BytesIO()
        Image.new("L", (64, 48), 255).save(mask, "PNG")
        found = self.http.post("/v1/images/edits?response_format=png", files={
            "file": ("p.jpg", jpeg(40)), "mask": ("m.png", mask.getvalue())}, data={"model": "cpga", "prompt": "lamp"})
        self.assertEqual(found.headers["content-type"], "image/png")
        self.assertEqual(self.brighten.edits[-1], ((64, 48), "lamp", (64, 48)))

    def test_bad_requests(self):
        cases = [({"model": "nope"}, 404), ({}, 400), ({"model": "erase"}, 400),
                 ({"model": "cpga", "response_format": "url"}, 400)]
        for data, status in cases:
            found = self.http.post("/v1/images/edits", files={"image": ("p.jpg", jpeg(40))}, data=data)
            self.assertEqual(found.status_code, status, (data, found.text))
        found = self.http.post("/v1/images/edits", files={"image": ("p.jpg", b"bad")}, data={"model": "cpga"})
        self.assertEqual(found.status_code, 400)


class TestQuality(ImageTestCase):
    def test_a_burst_picks_the_best(self):
        files = [("file", (f"{i}.jpg", jpeg(level))) for i, level in enumerate((60, 200, 120))]
        body = self.http.post("/v1/images/quality", files=files).json()
        self.assertEqual(len(body["data"]), 3)
        self.assertEqual(body["best"], 1)
        self.assertEqual(body["model"], "fake-nima")

    def test_one_photo_as_the_body(self):
        body = self.http.post("/v1/images/quality", content=jpeg(200), headers={"Content-Type": "image/jpeg"}).json()
        self.assertEqual([d["index"] for d in body["data"]], [0])


class TestBrighten(ImageTestCase):
    def detect(self, level, **options):
        return self.http.post("/v1/detect", params=options, content=jpeg(level),
                              headers={"Content-Type": "image/jpeg"}).json()

    def test_a_dark_photo_is_brightened_first(self):
        found = self.detect(40)
        self.assertTrue(found["brightened"])
        self.assertEqual([o["label"] for o in found["objects"]], ["light"])

    def test_a_bright_photo_is_left_alone(self):
        self.assertNotIn("brightened", self.detect(150))

    def test_it_doesnt_need_an_editing_engine(self):
        self.assertEqual(self.brighten.edits, [])
        self.detect(40)
        self.assertEqual(self.brighten.edits, [])

    def test_the_option(self):
        self.assertNotIn("brightened", self.detect(40, brighten="false"))
        self.assertTrue(self.detect(150, brighten="true")["brightened"])


class TestLazyConstants(unittest.TestCase):
    def test_task_is_known_without_loading(self):
        engine = LazyEngine(FakeBrighten)
        self.assertEqual(engine.TASK, "brighten")
        self.assertFalse(engine.loaded)


if __name__ == "__main__":
    unittest.main()
