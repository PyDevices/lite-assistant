"""
Comprehensive unit test suite for every kind of engine.
"""
import unittest
import os

from lite_assistant.engines import load_engines

class TestEngines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\nLoading engines for testing...")
        # The models must be downloaded already ('lite-assistant setup')
        found = load_engines()
        (cls.tts_registry, cls.stt_registry, cls.llm_registry, cls.embedding_registry, cls.depth_registry,
         cls.detection_registry, cls.sound_registry, cls.classifier_registry) = (
            found[kind] for kind in ("tts", "stt", "llm", "embed", "depth", "detect", "sound", "classify"))

    def test_01_tts_engines(self):
        self.assertGreater(len(self.tts_registry), 0, "No TTS engines found")
        for name, engine in self.tts_registry.items():
            with self.subTest(engine=name):
                # Ensure it has voices
                voices = engine.get_available_voices()
                # Some engines might not have voices if models failed to load, 
                # but since we are testing "downloaded and usable", they should be > 0
                self.assertGreater(len(voices), 0, f"{name} has no available voices (model might not be loaded)")
                
                # Generate a small chunk
                default_voice = engine.get_default_voice()
                print(f"\nGenerating test PCM with {name}...")
                chunks = list(engine.generate_pcm("Test.", voice=default_voice))
                self.assertGreater(len(chunks), 0, f"{name} failed to generate audio chunks")
                self.assertIsInstance(chunks[0], bytes)

    def test_02_stt_engines(self):
        self.assertGreater(len(self.stt_registry), 0, "No STT engines found")
        test_wav_path = os.path.join(os.path.dirname(__file__), "test_audio.wav")
        self.assertTrue(os.path.exists(test_wav_path), "test_audio.wav not found")
        
        with open(test_wav_path, "rb") as f:
            audio_bytes = f.read()
            
        for name, engine in self.stt_registry.items():
            with self.subTest(engine=name):
                print(f"\nTranscribing with {name}...")
                text = engine.transcribe(audio_bytes)
                self.assertIsInstance(text, str)
                self.assertGreater(len(text.strip()), 0, f"{name} returned empty transcription")
                print(f"[{name}] Transcription: {text.strip()}")

    def test_03_llm_engines(self):
        self.assertGreater(len(self.llm_registry), 0, "No LLM engines found")
        for name, engine in self.llm_registry.items():
            with self.subTest(engine=name):
                print(f"\nGenerating a reply with {name}...")
                usage = {}
                chunks = list(engine.generate(
                    [{"role": "user", "content": "Say hello in three words."}],
                    max_tokens=16, usage=usage))
                self.assertGreater(len(chunks), 0, f"{name} generated no text")
                self.assertIsInstance(chunks[0], str)
                self.assertGreater(usage.get("completion_tokens", 0), 0)
                print(f"[{name}] Reply: {''.join(chunks).strip()}")

    def test_04_embedding_engines(self):
        self.assertGreater(len(self.embedding_registry), 0, "No embedding engines found")
        for name, engine in self.embedding_registry.items():
            with self.subTest(engine=name):
                vector = engine.embed("A cat sleeps in the sun.")
                self.assertEqual(len(vector), engine.get_dimensions()[-1])
                self.assertAlmostEqual(sum(x * x for x in vector), 1.0, places=3)
                self.assertEqual(len(engine.embed("A cat.", engine.get_dimensions()[0])),
                                 engine.get_dimensions()[0])

    def test_05_depth_engines(self):
        import math
        import numpy as np
        self.assertGreater(len(self.depth_registry), 0, "No depth engines found")
        with open(os.path.join(os.path.dirname(__file__), "test_image.jpg"), "rb") as f:
            photo = f.read()
        for name, engine in self.depth_registry.items():
            with self.subTest(engine=name):
                depth = engine.estimate(photo)
                height, width = depth.distances.shape
                # The 320x213 photo's shape, at the model's resolution
                self.assertAlmostEqual(width / height, 320 / 213, places=1)
                center = float(np.nanmedian(depth.distances[height // 3:2 * height // 3, width // 3:2 * width // 3]))
                print(f"[{name}] Center: {center:.2f} m, field of view {depth.horizontal_fov:.0f} degrees")
                self.assertTrue(0.1 < center < 20)
                self.assertTrue(math.isfinite(depth.horizontal_fov))
                with self.assertRaises(ValueError):
                    engine.estimate(b"not an image")

    def test_06_detection_engines(self):
        self.assertGreater(len(self.detection_registry), 0, "No detection engines found")
        with open(os.path.join(os.path.dirname(__file__), "test_image.jpg"), "rb") as f:
            photo = f.read()
        for name, engine in self.detection_registry.items():
            with self.subTest(engine=name):
                found = engine.detect(photo)
                print(f"[{name}] Found: {', '.join(f'{d.label} {d.score:.2f}' for d in found)}")
                # The test photo is of a cat, which a face detector rightly misses
                if "cat" not in engine.get_labels():
                    self.assertEqual(found, [])
                    continue
                self.assertEqual(found[0].label, "cat")
                left, top, right, bottom = found[0].box
                self.assertTrue(0 <= left < right <= 1 and 0 <= top < bottom <= 1)
                self.assertEqual(engine.detect(photo, threshold=0.99), [])
                with self.assertRaises(ValueError):
                    engine.detect(b"not an image")


    def test_07_sound_engines(self):
        self.assertGreater(len(self.sound_registry), 0, "No sound engines found")
        with open(os.path.join(os.path.dirname(__file__), "test_audio.wav"), "rb") as f:
            audio = f.read()
        for name, engine in self.sound_registry.items():
            with self.subTest(engine=name):
                sounds = engine.classify(audio)
                best = sounds.labels[int(sounds.scores.max(axis=0).argmax())]
                print(f"[{name}] Heard: {best}")
                # The test clip is speech
                self.assertEqual(best, "Speech")
                self.assertEqual(sounds.scores.shape, (len(sounds.windows), len(sounds.labels)))
                with self.assertRaises(ValueError):
                    engine.classify(b"not audio")


    def test_08_classifier_engines(self):
        self.assertGreater(len(self.classifier_registry), 0, "No classification engines found")
        labels = ["lights", "timer", "weather", "music"]
        for name, engine in self.classifier_registry.items():
            with self.subTest(engine=name):
                for text, intent in (("Turn on the kitchen light", "lights"), ("Set a timer for ten minutes", "timer"),
                                     ("Will it rain tomorrow?", "weather"), ("Play some jazz", "music")):
                    logits = engine.logits(text, labels)
                    self.assertEqual(len(logits), len(labels))
                    self.assertEqual(labels[logits.index(max(logits))], intent, text)
                with self.assertRaises(ValueError):
                    engine.logits("Too many labels", [f"label {i}" for i in range(26)])


class TestKokoroLanguages(unittest.TestCase):
    """ Each Kokoro voice speaks its own language, with a fake model in place of Kokoro's. """
    def setUp(self):
        from lite_assistant.engines.kokoro_engine import KokoroEngine
        import numpy as np
        self.calls = []

        class FakeKokoro:
            def get_voices(_):
                return ["af_heart", "bm_george", "ff_siwis", "jf_alpha", "zm_yunxi", "xx_odd"]

            def create(_, text, voice, speed, lang):
                self.calls.append((voice, lang))
                return np.zeros(240, dtype=np.float32), 24000

        self.engine = KokoroEngine.__new__(KokoroEngine)
        self.engine.kokoro = FakeKokoro()

    def test_phonemized_in_the_voices_language(self):
        for voice in ("af_heart", "bm_george", "ff_siwis", "jf_alpha", "zm_yunxi", "xx_odd"):
            list(self.engine.generate_pcm("Hello.", voice))
        self.assertEqual(self.calls, [("af_heart", "en-us"), ("bm_george", "en-gb"), ("ff_siwis", "fr-fr"),
                                      ("jf_alpha", "ja"), ("zm_yunxi", "cmn"), ("xx_odd", "en-us")])

    def test_voice_names_say_their_language(self):
        names = {v["id"]: v["name"] for v in self.engine.get_available_voices()}
        self.assertEqual(names["ff_siwis"], "Siwis (French Female)")
        self.assertEqual(names["bm_george"], "George (British English Male)")
        self.assertEqual(names["zm_yunxi"], "Yunxi (Mandarin Chinese Male)")
        self.assertEqual(names["xx_odd"], "Odd (Unknown Unknown)")

if __name__ == '__main__':
    unittest.main()
