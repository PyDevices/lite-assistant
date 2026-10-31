"""
Whisper: its log-mel input, finding the language, greedy decoding and
long audio heard 30 s at a time, with a fake graph.
"""
import threading
import unittest

import numpy as np

from lite_assistant.engines.tflite import wav
from lite_assistant.engines.whisper_engine import (END, LANGUAGES, NO_TIMESTAMPS, START, TRANSCRIBE, WhisperEngine,
                                                   log_mel)

RATE = 16000
EN, FR = LANGUAGES[0], LANGUAGES[3]


class FakeTokenizer:
    def id_to_token(self, token):
        return {EN: "<|en|>", FR: "<|fr|>"}.get(token, f"<|x{token}|>")

    def decode(self, tokens):
        return "".join(f" w{t}" for t in tokens)


class FakeWhisper(WhisperEngine):
    """
    Hears French unless told otherwise, and says the tokens
    1, 2, 3 for each window, remembering every prompt it was given.
    """
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.languages = {self.tokenizer.id_to_token(t)[2:-2]: t for t in LANGUAGES}
        self.mask = None
        self.lock = threading.Lock()
        self.prompts = []
        self.encode = lambda args_0: {"output_0": np.zeros((1, 1500, 1280), np.float32)}
        self.decode = "fake"

    def logits(self, states, tokens):
        scores = np.zeros(51866, np.float32)
        if tokens == [START]:
            scores[FR] = 1
            return scores
        if len(tokens) == 4:
            self.prompts.append(list(tokens))
        said = len(tokens) - 4
        scores[said + 1 if said < 3 else END] = 1
        return scores


class TestWhisper(unittest.TestCase):
    def test_log_mel_is_whispers_shape_and_range(self):
        mel = log_mel(np.sin(np.arange(RATE) / 5).astype(np.float32) * 0.3)
        self.assertEqual(mel.shape, (128, 3000))
        # Clamped to 8 below the peak, then scaled: everything is within 2 of each other
        self.assertLessEqual(mel.max() - mel.min(), 2 + 1e-6)

    def test_finds_the_language_unless_told(self):
        engine = FakeWhisper()
        speech = wav(np.full(RATE, 0.1, np.float32))
        self.assertEqual(engine.transcribe(speech), "w1 w2 w3")
        self.assertEqual(engine.transcribe(speech, "en"), "w1 w2 w3")
        self.assertEqual([p[1] for p in engine.prompts], [FR, EN])
        self.assertEqual(engine.prompts[0], [START, FR, TRANSCRIBE, NO_TIMESTAMPS])

    def test_long_audio_is_heard_30_s_at_a_time(self):
        engine = FakeWhisper()
        self.assertEqual(engine.transcribe(wav(np.full(65 * RATE, 0.1, np.float32))), "w1 w2 w3 w1 w2 w3 w1 w2 w3")
        with self.assertRaises(ValueError):
            engine.transcribe(wav(np.zeros(121 * RATE, np.float32)))

    def test_unavailable_without_its_model(self):
        engine = WhisperEngine.__new__(WhisperEngine)
        engine.decode = None
        with self.assertRaises(RuntimeError):
            engine.transcribe(b"")


if __name__ == "__main__":
    unittest.main()
