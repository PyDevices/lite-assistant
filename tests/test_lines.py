"""
Lines to follow: the line finder on drawn floors, UFLD's lanes from a
fake graph, where the path is, and the /v1/lines endpoint.
"""
import io
import unittest

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from lite_assistant.app import Engines, create_app
from lite_assistant.engines.linefinder_engine import LineFinderEngine
from lite_assistant.engines.ufld_engine import CELLS, ROWS, UFLDEngine
from lite_assistant.lines import measure


def floor(*tape, width=40, color=(20, 20, 20)) -> Image.Image:
    """ A 640x480 light floor with tape from (x, y) to (x, y), in pixels. """
    photo = Image.new("RGB", (640, 480), (200, 195, 190))
    ImageDraw.Draw(photo).line(tape, fill=color, width=width)
    return photo


def jpeg(photo) -> bytes:
    out = io.BytesIO()
    photo.save(out, "JPEG")
    return out.getvalue()


class FakeUFLD(UFLDEngine):
    """ Sees lanes at given columns, as fractions of the width, on every row. """
    def __init__(self, *columns):
        def graph(x):
            scores = np.zeros((1, CELLS + 1, len(ROWS), 4), np.float32)
            scores[0, CELLS] = 5
            for lane, column in enumerate(columns):
                scores[0, round(column * (CELLS - 1)), :, lane] = 50
                scores[0, CELLS, :, lane] = 0
            return [scores]
        self.graph = graph


class TestLineFinder(unittest.TestCase):
    def test_finds_the_tape_and_where_it_goes(self):
        found = measure(*self.follow(floor((448, 480), (448, 0))))
        self.assertAlmostEqual(found["offset"], 0.4, delta=0.02)
        self.assertAlmostEqual(found["angle"], 0, delta=1)
        # Leaning right as it goes away
        found = measure(*self.follow(floor((320, 480), (520, 0))))
        self.assertAlmostEqual(found["offset"], 0, delta=0.05)
        self.assertGreater(found["angle"], 15)
        self.assertGreater(found["ahead"], 0.2)

    def test_light_and_colored_tape(self):
        dark = floor()
        ImageDraw.Draw(dark).rectangle([0, 0, 640, 480], fill=(40, 40, 40))
        ImageDraw.Draw(dark).line([(200, 480), (200, 0)], fill=(240, 240, 240), width=40)
        self.assertAlmostEqual(measure(*self.follow(dark))["offset"], -0.375, delta=0.02)
        self.assertTrue(measure(*self.follow(floor((320, 480), (320, 0), color=(30, 60, 200)))))

    def test_seams_and_bare_floors_arent_lines(self):
        self.assertIsNone(measure(*self.follow(floor((320, 480), (320, 0), width=4))))
        self.assertIsNone(measure(*self.follow(floor())))

    @staticmethod
    def follow(photo):
        engine = LineFinderEngine()
        return engine.path(engine.find(photo)), photo.size


class TestLanes(unittest.TestCase):
    def test_the_path_is_the_middle_of_the_cars_lane(self):
        engine = FakeUFLD(0.1, 0.4, 0.7)
        lanes = engine.find(floor())
        self.assertEqual([round(lane[0][0], 2) for lane in lanes], [0.1, 0.4, 0.7])
        # Nearest first
        self.assertGreater(lanes[0][0][1], lanes[0][-1][1])
        self.assertEqual({round(x, 2) for x, _ in engine.path(lanes)}, {0.55})

    def test_no_path_without_a_lane_each_side(self):
        engine = FakeUFLD(0.2)
        self.assertIsNone(engine.path(engine.find(floor())))


class TestEndpoint(unittest.TestCase):
    def setUp(self):
        engines = Engines({}, {}, {}, line={"linefinder": LineFinderEngine(), "ufld": FakeUFLD(0.3, 0.6)})
        self.http = TestClient(create_app(engines, None, "none"), client=("127.0.0.1", 50000))

    def post(self, photo, **params):
        return self.http.post("/v1/lines", params=params, content=jpeg(photo),
                              headers={"Content-Type": "image/jpeg"})

    def test_line_by_default_and_lanes_by_name(self):
        found = self.post(floor((448, 480), (448, 0))).json()
        self.assertEqual((found["model"], found["found"]), ("line-finder", True))
        self.assertAlmostEqual(found["path"]["offset"], 0.4, delta=0.02)
        lanes = self.post(floor(), model="lanes").json()
        self.assertEqual(len(lanes["lines"]), 2)
        self.assertAlmostEqual(lanes["path"]["offset"], -0.1, delta=0.01)
        self.assertFalse(self.post(floor()).json()["found"])

    def test_drawn_and_errors(self):
        drawn = self.post(floor((448, 480), (448, 0)), response_format="jpeg")
        self.assertEqual(Image.open(io.BytesIO(drawn.content)).size, (640, 480))
        self.assertEqual(self.post(floor(), model="rails").status_code, 404)
        self.assertEqual(self.post(floor(), response_format="gif").status_code, 400)


if __name__ == "__main__":
    unittest.main()
