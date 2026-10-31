"""
Reading text in photos: the detector's box finding, joining lines into
text, and the endpoint and read_text tool, with a fake engine.
"""
import base64
import io
import json
import unittest

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from lite_assistant.app import Engines, create_app
from lite_assistant.engines import OCREngine, TextLine, as_text
from lite_assistant.engines.tflite import open_photo
from lite_assistant.engines.ppocr_engine import boxes, min_area_rect, ordered
from lite_assistant.tools.reading import TextReader


def jpeg() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (64, 48), (200, 200, 200)).save(out, "JPEG")
    return out.getvalue()


class FakeReader(OCREngine):
    MODEL_ID = "fake-ocr"

    def read(self, image):
        open_photo(image)
        return [TextLine("EXIT", 0.99, (0.1, 0.1, 0.3, 0.2)), TextLine("ONLY", 0.98, (0.35, 0.1, 0.5, 0.2)),
                TextLine("Open 9-5", 0.6, (0.1, 0.4, 0.6, 0.5))]


class TestGeometry(unittest.TestCase):
    def test_min_area_rect_of_a_tilted_bar(self):
        angle = np.radians(20)
        along, across = np.array([np.cos(angle), np.sin(angle)]), np.array([-np.sin(angle), np.cos(angle)])
        points = np.array([a * along + b * across for a in np.linspace(0, 100, 50) for b in np.linspace(0, 10, 6)])
        box = ordered(min_area_rect(points))
        self.assertAlmostEqual(np.linalg.norm(box[1] - box[0]), 100, delta=0.5)
        self.assertAlmostEqual(np.linalg.norm(box[3] - box[0]), 10, delta=0.5)

    def test_boxes_in_reading_order_grown_around_each_line(self):
        probability = np.zeros((100, 200), np.float32)
        probability[60:70, 20:120] = 0.9
        # Two words in one row, the right one a little higher
        probability[20:30, 110:180] = 0.9
        probability[22:32, 20:90] = 0.9
        # Too faint to keep
        probability[85:95, 20:60] = 0.3
        found = boxes(probability)
        self.assertEqual(len(found), 3)
        lefts = [round(b[0][0]) for b in found]
        self.assertLess(lefts[0], lefts[1])
        self.assertLess(found[1][0][1], found[2][0][1])
        # The kernel is grown by its area * 1.4 over its perimeter on every side
        grow = 100 * 10 * 1.4 / 220
        self.assertAlmostEqual(found[2][1][0] - found[2][0][0], 100 + 2 * grow, delta=1.5)

    def test_nothing_on_an_empty_map(self):
        self.assertEqual(boxes(np.zeros((64, 64), np.float32)), [])


class TestText(unittest.TestCase):
    def test_rows_join_with_spaces_and_newlines(self):
        self.assertEqual(as_text(FakeReader().read(jpeg())), "EXIT ONLY\nOpen 9-5")
        self.assertEqual(as_text([]), "")


class TestEndpoint(unittest.TestCase):
    def setUp(self):
        self.http = TestClient(create_app(Engines({}, {}, {}, ocr={"fake": FakeReader()}), None, "none"), client=("127.0.0.1", 50000))

    def test_reads_a_photo(self):
        body = self.http.post("/v1/ocr", content=jpeg(), headers={"Content-Type": "image/jpeg"}).json()
        self.assertEqual(body["model"], "fake-ocr")
        self.assertEqual(body["text"], "EXIT ONLY\nOpen 9-5")
        self.assertEqual(body["lines"][0], {"text": "EXIT", "score": 0.99, "box": [0.1, 0.1, 0.3, 0.2]})

    def test_not_a_photo(self):
        found = self.http.post("/v1/ocr", content=b"nope", headers={"Content-Type": "image/jpeg"})
        self.assertEqual(found.status_code, 400)

    def test_without_an_engine(self):
        http = TestClient(create_app(Engines({}, {}, {}), None, "none"), client=("127.0.0.1", 50000))
        self.assertEqual(http.post("/v1/ocr", content=jpeg(), headers={"Content-Type": "image/jpeg"}).status_code, 500)


class TestTool(unittest.TestCase):
    def conversation(self):
        url = "data:image/jpeg;base64," + base64.b64encode(jpeg()).decode()
        return [{"role": "user", "content": [{"type": "text", "text": "What does it say?"},
                                             {"type": "image_url", "image_url": {"url": url}}]}]

    def test_reads_the_latest_photo_and_marks_guesses(self):
        found = TextReader(FakeReader()).read_text(self.conversation())
        self.assertEqual(found, {"text": "EXIT ONLY\nOpen 9-5", "unsure": ["Open 9-5"]})
        json.dumps(found)

    def test_without_a_photo(self):
        self.assertIn("error", TextReader(FakeReader()).read_text([{"role": "user", "content": "hi"}]))
        # With no photo, an attached file's text is pointed to
        told = TextReader(FakeReader()).read_text([{"role": "user", "content": [
            {"type": "text", "text": "The attached file letter.docx:\n\nDear Alex,"}, {"type": "text", "text": "Who wrote it?"}]}])
        self.assertIn("attached letter.docx, whose text is already in their message", told["error"])


if __name__ == "__main__":
    unittest.main()
