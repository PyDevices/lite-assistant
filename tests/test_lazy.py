"""
Engines that load on first use and unload when idle.
"""
import unittest
from unittest import mock

from lite_assistant.app import Engines
from lite_assistant.engines import EAGER, KINDS, LazyEngine, engine_classes, load_engines


class Counted:
    """ An engine that counts how often it's created. """
    MODEL_ID = "counted-1"
    made = 0

    def __init__(self):
        Counted.made += 1

    def get_model_id(self):
        return self.MODEL_ID

    def detect(self, image):
        return [image]


class TestLazyEngine(unittest.TestCase):
    def setUp(self):
        Counted.made = 0
        self.engine = LazyEngine(Counted, idle=600)

    def test_loads_on_first_use_only(self):
        self.assertFalse(self.engine.loaded)
        self.assertEqual(self.engine.get_model_id(), "counted-1")
        self.assertIsNone(self.engine.backend)
        self.assertEqual(Counted.made, 0)
        self.assertEqual(self.engine.detect("photo"), ["photo"])
        self.assertEqual(self.engine.detect("photo"), ["photo"])
        self.assertTrue(self.engine.loaded)
        self.assertEqual(Counted.made, 1)

    def test_unloads_when_idle_and_reloads(self):
        self.engine.detect("photo")
        used = self.engine._used
        self.assertFalse(self.engine.unload_idle(now=used + 599))
        self.assertTrue(self.engine.unload_idle(now=used + 601))
        self.assertFalse(self.engine.loaded)
        self.engine.detect("photo")
        self.assertEqual(Counted.made, 2)

    def test_never_unloads_with_idle_zero(self):
        engine = LazyEngine(Counted, idle=0)
        engine.detect("photo")
        self.assertFalse(engine.unload_idle(now=engine._used + 1e9))

    def test_a_call_under_way_keeps_its_engine(self):
        method = self.engine.detect
        self.engine.unload_idle(now=self.engine._used + 1e9)
        self.assertEqual(method("photo"), ["photo"])

    def test_engines_unload_idle_lazy_ones(self):
        engines = Engines({}, {}, {}, detect={"counted": self.engine})
        self.engine.detect("photo")
        self.engine._used -= 1000
        engines.unload_idle()
        self.assertFalse(self.engine.loaded)


class TestDiscovery(unittest.TestCase):
    def test_every_engine_module_imports_without_loading_models(self):
        found = engine_classes()
        self.assertEqual(set(found), set(KINDS))
        for kind in KINDS:
            self.assertTrue(found[kind], f"no {kind} engine")
        self.assertLessEqual(EAGER, set(KINDS))

    def test_a_class_can_ask_to_load_lazily(self):
        class Big:
            LAZY = True

        class Small:
            pass
        with mock.patch("lite_assistant.engines.engine_classes", return_value={"stt": {"big": Big, "small": Small}}):
            found = load_engines()["stt"]
        self.assertIsInstance(found["big"], LazyEngine)
        self.assertIsInstance(found["small"], Small)


if __name__ == "__main__":
    unittest.main()
