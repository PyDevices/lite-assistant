"""
Actions across a video's frames: photos sent together or one at a time in
a session, a video decoded by ffmpeg, and MoViNet's memory with a fake graph.
"""
import io
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from lite_assistant import video
from lite_assistant.app import Engines, create_app
from lite_assistant.engines import ActionEngine
from lite_assistant.engines.movinet_engine import DEPTHS, POOLS, MoViNetEngine


def jpeg(brightness: int) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (64, 48), (brightness,) * 3).save(out, "JPEG")
    return out.getvalue()


class Brightness(ActionEngine):
    """ Knows two actions, "dark" and "bright", by the mean of every frame so far. """
    MODEL_ID = "fake-actions"
    labels = ["dark", "bright"]

    def start(self):
        return {"total": 0.0, "count": 0}

    def watch(self, memory, frame):
        memory["total"] += np.asarray(frame, np.float32).mean() / 255
        memory["count"] += 1
        bright = memory["total"] / memory["count"]
        return np.array([1 - bright, bright])


class TestEndpoint(unittest.TestCase):
    def setUp(self):
        self.http = TestClient(create_app(Engines({}, {}, {}, action={"fake": Brightness()}), None, "none"), client=("127.0.0.1", 50000))

    def test_photos_together(self):
        found = self.http.post("/v1/video/actions", files=[("file", ("1.jpg", jpeg(250), "image/jpeg")),
                                                           ("file", ("2.jpg", jpeg(250), "image/jpeg"))]).json()
        self.assertEqual(found["frames"], 2)
        self.assertEqual(found["actions"][0]["label"], "bright")

    def test_a_session_remembers_the_frames_before(self):
        def send(brightness, session="car"):
            return self.http.post(f"/v1/video/actions?session={session}", content=jpeg(brightness),
                                  headers={"Content-Type": "image/jpeg"}).json()
        send(250)
        send(250)
        found = send(0)
        # Two bright frames and a dark one: bright, on the whole
        self.assertEqual((found["frames"], found["actions"][0]["label"], found["session"]), (3, "bright", "car"))
        self.assertEqual(send(0, "other")["actions"][0]["label"], "dark")

    @unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
    def test_a_video(self):
        with tempfile.TemporaryDirectory() as folder:
            path = f"{folder}/white.webm"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=white:size=64x48:duration=2:rate=10",
                            path], check=True)
            with open(path, "rb") as f:
                found = self.http.post("/v1/video/actions", files={"file": ("white.webm", f, "video/webm")}).json()
        # 2 s at the engine's 5 frames a second
        self.assertEqual(found["frames"], 10)
        self.assertEqual(found["actions"][0]["label"], "bright")

    def test_bad_requests(self):
        self.assertEqual(self.http.post("/v1/video/actions").status_code, 400)
        self.assertEqual(self.http.post("/v1/video/actions?top=0", content=jpeg(9),
                                        headers={"Content-Type": "image/jpeg"}).status_code, 400)
        with mock.patch("shutil.which", return_value=None):
            found = self.http.post("/v1/video/actions", content=b"not a video", headers={"Content-Type": "video/mp4"})
        self.assertEqual(found.status_code, 400)
        self.assertIn("ffmpeg", found.json()["error"]["message"])


class TestSessions(unittest.TestCase):
    def test_idle_sessions_are_dropped(self):
        sessions = video.Sessions()
        sessions.keep("a", ("memory", 1))
        self.assertEqual(sessions.take("a"), ("memory", 1))
        self.assertIsNone(sessions.take("a"))
        sessions.keep("b", ("memory", 1))
        with mock.patch("time.monotonic", return_value=1e12):
            self.assertIsNone(sessions.take("b"))


class TestMoViNet(unittest.TestCase):
    def test_memory_shifts_and_sums(self):
        engine = MoViNetEngine.__new__(MoViNetEngine)
        engine.labels = ["a", "b"]
        given = []

        class Graph:
            def shape(self, index):
                return [1, 1, 1, 1]

            def __call__(self, *inputs):
                given.append(inputs)
                n = len(given)
                return [np.array([[0.0, 1.0]])] + [np.full((1, 1, 1, 1), n, np.float32)] * 11 + \
                    [np.ones((1, 1, 1, 1), np.float32)] * 16
        engine.graph = Graph()
        memory = engine.start()
        for _ in range(3):
            chances = engine.watch(memory, Image.new("RGB", (300, 200)))
        self.assertAlmostEqual(chances.sum(), 1)
        self.assertGreater(chances[1], chances[0])
        # The third frame sees the first two frames' outputs in the 2-deep memories, and sums of 2
        third = given[2]
        self.assertEqual(len(third), 1 + sum(DEPTHS) + POOLS + 2)
        self.assertEqual(third[0].shape, (1, 3, 172, 172))
        self.assertEqual([float(x.flat[0]) for x in third[1:3]], [1.0, 2.0])
        self.assertEqual(float(third[1 + sum(DEPTHS)].flat[0]), 2.0)
        self.assertAlmostEqual(float(third[-2].flat[0]), 1 / 3)


if __name__ == "__main__":
    unittest.main()
