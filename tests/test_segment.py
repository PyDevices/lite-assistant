"""
Finding things by name: the region and distance of a probability map, the
/v1/segment endpoint, and estimate_distance finding what the detector
doesn't know, with fake engines.
"""
import base64
import io
import unittest

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from lite_assistant import segmentation
from lite_assistant.app import Engines, create_app
from lite_assistant.engines import SegmentEngine
from lite_assistant.engines.clipseg_engine import WORDS
from lite_assistant.tools.distance import Distances

from test_app import FakeDepth, FakeDetector


def jpeg(size=(60, 40)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, (128, 128, 128)).save(out, "JPEG")
    return out.getvalue()


class FakeSegmenter(SegmentEngine):
    """ Sees a tree in the bottom right, where FakeDepth's wall is 4 m away, and nothing else. A photo of b"bad" isn't one. """
    MODEL_ID = "fake-clipseg"

    def __init__(self):
        self.prompts = []

    def segment(self, image, prompts):
        if image == b"bad":
            raise ValueError("cannot identify image file")
        self.prompts.append(prompts)
        maps = []
        for prompt in prompts:
            probability = np.full((20, 30), 0.05, np.float32)
            if "tree" in prompt:
                probability[10:20, 15:30] = 0.9
            maps.append(probability)
        return maps


class TestRegion(unittest.TestCase):
    def test_box_center_and_area(self):
        probability = np.zeros((10, 20), np.float32)
        probability[2:6, 10:20] = 0.8
        found = segmentation.region(probability)
        self.assertEqual(found, {"score": 0.8, "box": [0.5, 0.2, 1.0, 0.6], "center": [0.75, 0.4], "area": 0.2})
        self.assertIsNone(segmentation.region(np.full((10, 20), 0.1, np.float32)))

    def test_distance_from_a_depth_map_of_another_size(self):
        probability = np.zeros((20, 30), np.float32)
        probability[10:, 15:] = 0.9
        self.assertEqual(segmentation.distance(FakeDepth().estimate(b"photo"), probability), 4.0)

    def test_mask_is_the_photos_size(self):
        png = segmentation.mask(np.ones((20, 30), np.float32), (64, 48))
        self.assertEqual(Image.open(io.BytesIO(png)).size, (64, 48))


class TestTokenizerPattern(unittest.TestCase):
    def test_clip_splits_words_digits_and_punctuation(self):
        import regex
        self.assertEqual(regex.findall(WORDS, "the cup's 12 handles!"), ["the", "cup", "'s", "1", "2", "handles", "!"])


class TestEndpoint(unittest.TestCase):
    def setUp(self):
        self.segmenter = FakeSegmenter()
        engines = Engines({}, {}, {}, depth={"moge": FakeDepth()}, segment={"clipseg": self.segmenter})
        self.http = TestClient(create_app(engines, None, "none"), client=("127.0.0.1", 50000))

    def post(self, **options):
        return self.http.post("/v1/segment", params=options, content=jpeg(), headers={"Content-Type": "image/jpeg"})

    def test_finds_with_distances(self):
        body = self.post(prompt="a tree, a cat").json()
        self.assertEqual(self.segmenter.prompts, [["a tree", "a cat"]])
        tree, cat = body["data"]
        self.assertEqual((tree["prompt"], tree["found"], tree["box"], tree["distance"]), ("a tree", True, [0.5, 0.5, 1.0, 1.0], 4.0))
        self.assertEqual(cat, {"prompt": "a cat", "found": False})
        self.assertEqual((body["model"], body["units"]), ("fake-clipseg", "meters"))

    def test_json_list_and_no_distances(self):
        body = self.http.post("/v1/segment?distances=false", json={
            "image": "data:image/jpeg;base64," + base64.b64encode(jpeg()).decode(),
            "prompt": ["the tree"]}).json()
        self.assertNotIn("distance", body["data"][0])

    def test_png_mask(self):
        found = self.post(prompt="tree", response_format="png")
        self.assertEqual(found.headers["content-type"], "image/png")
        mask = Image.open(io.BytesIO(found.content))
        self.assertEqual(mask.size, (60, 40))
        self.assertGreater(mask.getpixel((55, 35)), 200)
        self.assertLess(mask.getpixel((5, 5)), 30)

    def test_not_a_photo(self):
        found = self.http.post("/v1/segment?prompt=x", content=b"bad", headers={"Content-Type": "image/jpeg"})
        self.assertEqual(found.status_code, 400)

    def test_bad_requests(self):
        for options in ({}, {"prompt": " , "}, {"prompt": "x", "threshold": "2"}, {"prompt": "x", "response_format": "url"}):
            self.assertEqual(self.post(**options).status_code, 400, options)


class TestDistanceTool(unittest.TestCase):
    photo = [{"role": "user", "content": [{"type": "input_image", "image_url": "data:image/jpeg;base64,cGhvdG8="}]}]

    def test_finds_what_the_detector_doesnt_know(self):
        found = Distances(FakeDepth(), FakeDetector(), FakeSegmenter()).estimate_distance("center", "the tree", self.photo)
        self.assertEqual(found["measured"], [{"object": "the tree", "at": "the bottom right of the photo",
                                              "distance": "about 4.0 meters, or 13 feet"}])

    def test_the_detector_comes_first(self):
        segmenter = FakeSegmenter()
        found = Distances(FakeDepth(), FakeDetector(), segmenter).estimate_distance("center", "the chair", self.photo)
        self.assertEqual(found["measured"][0]["object"], "chair")
        self.assertEqual(segmenter.prompts, [])

    def test_nothing_found(self):
        found = Distances(FakeDepth(), None, FakeSegmenter()).estimate_distance("left", "the cat", self.photo)
        self.assertIn("Nothing like", found["note"])
        self.assertEqual(found["measured_at"], "the left of the photo")


if __name__ == "__main__":
    unittest.main()
