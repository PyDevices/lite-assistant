"""
Moonshine hears only the speech in a clip: a board's recording that starts
with seconds of room noise, which Moonshine alone turns into nothing, is cut
to the speech Silero VAD finds first.
"""
import os
import unittest

import numpy as np

from lite_assistant.engines.moonshine_engine import MARGIN, MoonshineEngine, REPO, WEIGHTS, _weights_folder
from lite_assistant.paths import cached_hf_file
from lite_assistant.vad import FILE, REPO as VAD_REPO, WINDOW, speech_span

RATE = 16000
HERE = os.path.dirname(os.path.abspath(__file__))
# The DEV-KIT's speaker saying "The quick brown fox jumps over the lazy dog.",
# recorded by a webcam's microphone, starting 1.7 s before the speech
BOARD_CLIP = os.path.join(HERE, "board_room_noise_first.wav")
DOWNLOADED = cached_hf_file(VAD_REPO, FILE) and all(
    cached_hf_file(REPO, f"{_weights_folder('moonshine/base')}/{name}") for name in WEIGHTS)


def loud_between(start, stop):
    """ A fake Silero: speech in the windows from start to stop seconds. """
    def probability(window, state):
        probability.at += WINDOW
        return (0.9 if start * RATE < probability.at <= stop * RATE else 0.1), state
    probability.at = 0
    return probability


class FakeVAD:
    def __init__(self, probability):
        self.probability = probability


class TestSpeechSpan(unittest.TestCase):
    def test_finds_the_first_and_last_speech(self):
        span = speech_span(loud_between(2, 3), np.zeros(5 * RATE, np.float32))
        self.assertEqual((span[0] // WINDOW, span[1] // WINDOW), (2 * RATE // WINDOW, 3 * RATE // WINDOW))

    def test_none_without_speech(self):
        self.assertIsNone(speech_span(loud_between(9, 10), np.zeros(5 * RATE, np.float32)))


class TestMoonshineTrims(unittest.TestCase):
    def engine(self, vad):
        engine = MoonshineEngine.__new__(MoonshineEngine)
        engine._vad = vad
        return engine

    def test_cuts_to_the_speech_with_a_margin(self):
        audio = np.arange(5 * RATE, dtype=np.float32)
        cut = self.engine(FakeVAD(loud_between(2, 3))).speech(audio)
        first, = np.where(audio == cut[0])[0]
        self.assertLess(abs(first - (2 * RATE - MARGIN)), WINDOW)
        self.assertLess(abs(len(cut) - (RATE + 2 * MARGIN)), 2 * WINDOW)

    def test_keeps_everything_when_it_hears_no_speech_or_has_no_vad(self):
        audio = np.ones(RATE, np.float32)
        self.assertIs(self.engine(FakeVAD(loud_between(9, 10))).speech(audio), audio)
        self.assertIs(self.engine(False).speech(audio), audio)

    @unittest.skipUnless(DOWNLOADED, "Moonshine or Silero VAD not downloaded")
    def test_hears_a_board_recording_that_starts_with_room_noise(self):
        with open(BOARD_CLIP, "rb") as f:
            text = MoonshineEngine().transcribe(f.read())
        self.assertIn("quick brown fox", text.lower())


if __name__ == "__main__":
    unittest.main()
