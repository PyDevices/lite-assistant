"""
Tuning: notes, cents and strings from the pitches heard, CREPE's decoding
with a fake graph, /v1/audio/pitch and the tune_instrument tool.
"""
import base64
import unittest

import numpy as np
from fastapi.testclient import TestClient

from lite_assistant import pitch
from lite_assistant.app import Engines, create_app
from lite_assistant.engines import PitchEngine
from lite_assistant.engines.crepe_engine import CENTS_PER_BIN, FIRST_CENTS, CREPEEngine
from lite_assistant.engines.tflite import wav
from lite_assistant.tools.tuner import Tuner

RATE = 16000


class Steady(PitchEngine):
    """ Hears one frequency throughout, sure of it, or nothing for silence. """
    MODEL_ID = "fake-pitch"

    def __init__(self, hz):
        self.hz = hz

    def pitches(self, samples):
        frames = len(samples) // int(self.HOP * self.RATE)
        loud = float(np.abs(samples).max()) > 0.01
        return [(self.hz, 0.9 if loud else 0.1)] * frames


def tone(seconds=1.0, level=0.3):
    return wav(np.full(int(seconds * RATE), level, np.float32))


class TestTuning(unittest.TestCase):
    def test_notes_and_cents(self):
        self.assertEqual(pitch.name(69), "A4")
        self.assertEqual(pitch.number("E2"), 40)
        self.assertAlmostEqual(pitch.frequency(pitch.number("E2")), 82.41, places=2)
        found = pitch.tune([(440 * 2 ** (10 / 1200), 0.9)] * 5 + [(1000, 0.1)], 0.05)
        self.assertEqual((found["note"], found["cents"], found["in_tune"]), ("A4", 10, False))
        # The unsure frame is in the track, without a frequency
        self.assertEqual(found["track"][-1], [0.25, None, 0.1])
        self.assertEqual(pitch.advice(found), "10 cents sharp.")

    def test_a_strings_nearest_and_which_way_to_turn(self):
        flat_a = pitch.frequency(pitch.number("A2")) * 2 ** (-30 / 1200)
        found = pitch.tune([(flat_a, 0.9)] * 3, 0.05, instrument="guitar")
        self.assertEqual(found["string"]["note"], "A2")
        self.assertEqual(found["string"]["cents"], -30)
        self.assertEqual(pitch.advice(found), "30 cents flat: tighten the string.")
        # A different A: the strings are tuned from it
        found = pitch.tune([(432.0, 0.9)], 0.05, reference=432.0, instrument="violin")
        self.assertEqual((found["string"]["note"], found["string"]["cents"]), ("A4", 0))
        self.assertEqual(pitch.advice(found), "In tune.")

    def test_nothing_heard(self):
        found = pitch.tune([(200, 0.2)] * 3, 0.05)
        self.assertIsNone(found["note"])
        self.assertIn("No note", pitch.advice(found))


class TestCREPE(unittest.TestCase):
    def test_decodes_the_peak_between_bins(self):
        engine = CREPEEngine.__new__(CREPEEngine)
        # 440 Hz is between two bins: the fake graph's chances peak there
        target = (1200 * np.log2(440 / 10) - FIRST_CENTS) / CENTS_PER_BIN
        chances = np.exp(-0.5 * (np.arange(360) - target) ** 2).astype(np.float32)
        windows = []
        engine.graph = lambda x: windows.append(x) or [chances[None]]
        found = engine.pitches(np.random.default_rng(0).normal(size=RATE).astype(np.float32))
        self.assertEqual(len(found), 20)
        self.assertAlmostEqual(found[0][0], 440, delta=0.5)
        # Each window is normalized
        self.assertAlmostEqual(float(windows[3].std()), 1, places=3)


class TestEndpointAndTool(unittest.TestCase):
    def setUp(self):
        self.engine = Steady(pitch.frequency(pitch.number("D3")) * 2 ** (12 / 1200))
        self.http = TestClient(create_app(Engines({}, {}, {}, pitch={"fake": self.engine}), None, "none"), client=("127.0.0.1", 50000))

    def test_endpoint(self):
        found = self.http.post("/v1/audio/pitch", files={"file": ("a.wav", tone())}, data={"instrument": "guitar"}).json()
        self.assertEqual((found["note"], found["cents"], found["string"]["note"]), ("D3", 12, "D3"))
        self.assertIn("loosen", found["advice"])
        self.assertEqual(len(found["track"]), 20)
        self.assertEqual(self.http.post("/v1/audio/pitch", files={"file": ("a.wav", tone())},
                                        data={"instrument": "kazoo"}).status_code, 400)
        self.assertEqual(self.http.post("/v1/audio/pitch", files={"file": ("a.wav", tone(11))}).status_code, 400)

    def test_tool_hears_the_latest_recording(self):
        recording = base64.b64encode(tone()).decode()
        conversation = [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": recording, "format": "wav"}}]}]
        found = Tuner(self.engine).tune_instrument("guitar", conversation=conversation)
        self.assertEqual((found["note"], found["string"], found["cents_off_the_string"]), ("D3", "D3", 12))
        # An instrument it doesn't know still gets the note
        found = Tuner(self.engine).tune_instrument("string", conversation=conversation)
        self.assertEqual((found["note"], "string" in found), ("D3", False))
        self.assertIn("hasn't sent", Tuner(self.engine).tune_instrument(conversation=[])["error"])


if __name__ == "__main__":
    unittest.main()
