"""
Voices: who spoke when, enrolled voices by name, the speaker endpoints and
noise removal, with fake engines.
"""
import io
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

from lite_assistant import speakers
from lite_assistant.app import Engines, create_app
from lite_assistant.engines import DenoiseEngine, SpeakerEngine
from lite_assistant.engines.cmgan_engine import CMGANEngine
from lite_assistant.engines.pyannote_engine import BINS, EMBED_SAMPLES, FRAMES, fbank, mel_banks
from lite_assistant.engines.tflite import wav
from lite_assistant.users import Users

RATE = 16000
# The fake engines hear a voice by its loudness: Ann speaks at 0.1, Bob at 0.3
ANN, BOB = 0.1, 0.3


def conversation(*turns) -> np.ndarray:
    """ Turns of (voice level, seconds), with silence for a level of 0. """
    return np.concatenate([np.full(int(seconds * RATE), level, np.float32) for level, seconds in turns])


class FakeSpeakers(SpeakerEngine):
    MODEL_ID = "fake-pyannote"
    FRAME = 0.1
    SAME = 0.5

    def embed(self, samples):
        level = float(np.median(samples))
        return np.array([1.0, 0.0]) if level < 0.2 else np.array([0.0, 1.0])

    def activity(self, window):
        frames = np.zeros(100, np.float32)
        heard = window[:len(window) // 1600 * 1600].reshape(-1, 1600).mean(axis=1)
        frames[:len(heard)] = heard
        return np.stack([abs(frames - ANN) < 0.05, abs(frames - BOB) < 0.05], axis=1)


class Quieter(DenoiseEngine):
    MODEL_ID = "fake-cmgan"

    def denoise(self, samples):
        return samples * 0.5


class TestTurns(unittest.TestCase):
    def test_who_spoke_when(self):
        audio = conversation((ANN, 4), (0, 1), (BOB, 6), (ANN, 3), (0, 2))
        found = speakers.diarize(FakeSpeakers(), audio)
        self.assertEqual([(s["speaker"], s["start"], s["end"]) for s in found["segments"]],
                         [("speaker 1", 0.0, 4.0), ("speaker 2", 5.0, 11.0), ("speaker 1", 11.0, 14.0)])
        self.assertEqual(found["speakers"], {"speaker 1": 7.0, "speaker 2": 6.0})
        self.assertEqual(found["duration"], 16.0)

    def test_short_pauses_join_and_blips_go(self):
        self.assertEqual(speakers.spans(np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1], bool), 0.1),
                         [(0.0, 0.6)])

    def test_enrolled_voices_are_named(self):
        with tempfile.TemporaryDirectory() as folder:
            voices = speakers.Voices(os.path.join(folder, "voices.json"))
            self.assertEqual(voices.add("Bob", np.array([0.0, 1.0])), 1)
            found = speakers.diarize(FakeSpeakers(), conversation((ANN, 3), (BOB, 3)), voices)
            self.assertEqual([s["speaker"] for s in found["segments"]], ["speaker 1", "Bob"])
            self.assertEqual(voices.names(), {"Bob": {"owner": "", "recordings": 1}})
            self.assertTrue(voices.remove("Bob"))
            self.assertFalse(voices.remove("Bob"))


class TestFeatures(unittest.TestCase):
    def test_kaldi_filter_bank(self):
        banks = mel_banks()
        self.assertEqual(banks.shape, (BINS, 256))
        # Every filter catches some bin, though the narrow low ones fall between bins, and they march up the spectrum
        self.assertTrue(np.all(banks.max(axis=1) > 0.4))
        self.assertTrue(np.all(np.diff(banks.argmax(axis=1)) >= 0))
        tone = np.sin(2 * np.pi * 1000 * np.arange(EMBED_SAMPLES) / RATE).astype(np.float32) * 0.5
        features = fbank(tone)
        self.assertEqual(features.shape, (FRAMES, BINS))
        # A 1 kHz tone is loudest in the filter centered nearest 1 kHz
        center = 1127 * np.log(1 + 1000 / 700)
        low, high = 1127 * np.log(1 + 20 / 700), 1127 * np.log(1 + 8000 / 700)
        self.assertAlmostEqual(features.mean(axis=0).argmax(), (center - low) / ((high - low) / 81) - 1, delta=1)


class TestDenoise(unittest.TestCase):
    def test_cmgan_chunks_join_up(self):
        def same(padded):
            """ A fake graph that returns its chunk's spectrogram, compressed, as CMGAN's does. """
            from lite_assistant.engines.cmgan_engine import FFT, HOP, WINDOW
            frames = np.lib.stride_tricks.sliding_window_view(padded[0], FFT)[::HOP]
            spec = np.fft.rfft(frames * WINDOW, axis=1)
            magnitude = np.abs(spec) ** 0.3
            phase = np.angle(spec)
            return [(magnitude * np.cos(phase))[None, None], (magnitude * np.sin(phase))[None, None]]
        engine = CMGANEngine.__new__(CMGANEngine)
        engine.graph = same
        speech = (np.sin(np.arange(int(4.5 * RATE)) / 9) * 0.3).astype(np.float32)
        cleaned = engine.denoise(speech)
        self.assertEqual(len(cleaned), len(speech))
        self.assertLess(np.abs(cleaned - speech).max(), 1e-3)

    def test_endpoint_and_transcription_option(self):
        class Hears:
            def transcribe(self, audio, language=None):
                return f"{len(audio)} bytes"
        engines = Engines({}, {"moonshine": Hears()}, {}, denoise={"cmgan": Quieter()})
        engines.default_stt = "moonshine"
        http = TestClient(create_app(engines, None, "none"), client=("127.0.0.1", 50000))
        clip = wav(np.full(24000, ANN, np.float32), 24000)
        found = http.post("/v1/audio/denoise", content=clip, headers={"Content-Type": "audio/wav"})
        self.assertEqual(found.headers["content-type"], "audio/wav")
        # It comes back at 16 kHz
        self.assertEqual(len(found.content), 44 + 16000 * 2)
        self.assertEqual(http.post("/v1/audio/denoise", content=b"nope",
                                   headers={"Content-Type": "audio/wav"}).status_code, 400)
        text = http.post("/v1/audio/transcriptions", files={"file": ("a.wav", clip)}, data={"denoise": "true"}).json()
        self.assertEqual(text["text"], f"{44 + 16000 * 2} bytes")


class TestEndpoints(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        patch = mock.patch.dict(os.environ, {"LITE_ASSISTANT_HOME": folder.name})
        patch.start()
        self.addCleanup(patch.stop)
        self.http = TestClient(create_app(Engines({}, {}, {}, speaker={"pyannote": FakeSpeakers()}), None, "none"), client=("127.0.0.1", 50000))

    def post(self, path, audio, **data):
        return self.http.post(path, files={"file": ("a.wav", wav(audio))}, data=data)

    def test_enroll_then_diarize(self):
        me = self.http.get("/me").json()["name"]
        self.assertEqual(self.post("/v1/audio/speakers", conversation((BOB, 4)), name="Bob").json(),
                         {"object": "speaker", "name": "Bob", "owner": me, "recordings": 1})
        self.assertEqual(self.http.get("/v1/audio/speakers").json()["data"],
                         [{"name": "Bob", "owner": me, "recordings": 1}])
        found = self.post("/v1/audio/diarization", conversation((ANN, 3), (BOB, 3))).json()
        self.assertEqual(found["speakers"], {"speaker 1": 3.0, "Bob": 3.0})
        self.assertEqual(self.http.delete("/v1/audio/speakers/Bob").json()["deleted"], True)
        self.assertEqual(self.http.delete("/v1/audio/speakers/Bob").status_code, 404)

    def test_identify_one_voice(self):
        self.post("/v1/audio/speakers", conversation((ANN, 4)), name="Ann")
        found = self.post("/v1/audio/speakers/identify", conversation((ANN, 2))).json()
        self.assertEqual(found["speaker"], "Ann")
        self.assertIsNone(self.post("/v1/audio/speakers/identify", conversation((BOB, 2))).json()["speaker"])

    def test_only_its_owner_or_an_admin_changes_a_voice(self):
        users = Users(os.path.join(os.environ["LITE_ASSISTANT_HOME"], "users.json"))
        app = create_app(Engines({}, {}, {}, speaker={"pyannote": FakeSpeakers()}), None, "none", users=users)
        def client(name):
            return TestClient(app, client=("192.168.1.20", 50000), headers={"Authorization": f"Bearer {users.add(name)}"})
        alice, bob, carol = client("alice"), client("bob"), client("carol")
        users.set_admin("carol", True)
        def enroll(http, name, voice=ANN):
            return http.post("/v1/audio/speakers", files={"file": ("a.wav", wav(conversation((voice, 4))))},
                             data={"name": name})
        self.assertEqual(enroll(alice, "Ann").json()["owner"], "alice")
        self.assertEqual(enroll(alice, "Ann").json()["recordings"], 2)
        refused = enroll(bob, "Ann")
        self.assertEqual((refused.status_code, refused.json()["error"]["code"]), (403, "not_your_speaker"))
        self.assertEqual(bob.delete("/v1/audio/speakers/Ann").status_code, 403)
        # Everyone's voices are recognized for everyone
        identified = bob.post("/v1/audio/speakers/identify", files={"file": ("a.wav", wav(conversation((ANN, 2))))})
        self.assertEqual(identified.json()["speaker"], "Ann")
        self.assertEqual(carol.delete("/v1/audio/speakers/Ann").json()["deleted"], True)

    def test_older_voices_are_read_and_only_admins_change_them(self):
        with open(os.path.join(os.environ["LITE_ASSISTANT_HOME"], "voices.json"), "w") as f:
            json.dump({"Bob": [[0.0, 1.0]]}, f)
        self.assertEqual(self.http.get("/v1/audio/speakers").json()["data"],
                         [{"name": "Bob", "owner": "", "recordings": 1}])
        self.assertEqual(self.post("/v1/audio/speakers/identify", conversation((BOB, 2))).json()["speaker"], "Bob")
        # This PC's owner is an admin
        self.assertEqual(self.http.delete("/v1/audio/speakers/Bob").json()["deleted"], True)

    def test_bad_requests(self):
        self.assertEqual(self.post("/v1/audio/speakers", conversation((ANN, 2))).status_code, 400)
        self.assertEqual(self.post("/v1/audio/speakers", conversation((ANN, 0.5)), name="Ann").status_code, 400)


if __name__ == "__main__":
    unittest.main()
