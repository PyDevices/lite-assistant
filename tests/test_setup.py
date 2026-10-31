"""
Setup's choice of chat model: asked once in a terminal, given with --llm,
kept for next time, and today's default from a script; and the server
starting on whichever Gemma setup downloaded.
"""
import argparse
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

from lite_assistant import cli
from lite_assistant.engines import litertlm_engine
from lite_assistant.engines.litertlm_engine import _start_model


class FakeLLM:
    calls = []

    @classmethod
    def setup(cls):
        cls.calls.append("auto")

    @classmethod
    def setup_model(cls, key):
        cls.calls.append(key)


class TestSetup(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        patch = mock.patch.dict(os.environ, {"LITE_ASSISTANT_HOME": folder.name})
        patch.start()
        self.addCleanup(patch.stop)
        FakeLLM.calls = []

    def setup(self, llm=None, tty=False, answers=()):
        with mock.patch("lite_assistant.engines.engine_classes", return_value={"llm": {"litertlm": FakeLLM}}), \
                mock.patch("lite_assistant.vad.SileroVAD.setup"), \
                mock.patch("sys.stdin.isatty", return_value=tty), \
                mock.patch("builtins.input", side_effect=list(answers)), redirect_stdout(io.StringIO()) as out:
            cli.setup(argparse.Namespace(llm=llm))
        return out.getvalue()

    def test_a_script_gets_todays_default(self):
        self.setup()
        self.assertEqual(FakeLLM.calls, ["auto"])
        # Nothing was chosen, so nothing is kept, and a terminal would still be asked
        self.setup(tty=True, answers=["2"])
        self.assertEqual(FakeLLM.calls, ["auto", "e4b"])

    def test_the_terminal_is_asked_once(self):
        with mock.patch.object(cli, "memory", return_value=8e9):
            out = self.setup(tty=True, answers=["5", ""])
        self.assertIn("E2B, 2.6 GB (recommended)", out)
        self.assertIn("lite-assistant models", out)
        self.assertEqual(FakeLLM.calls, ["e2b"])
        # run.sh runs setup every time; the answer is kept
        self.setup(tty=True)
        self.assertEqual(FakeLLM.calls, ["e2b", "e2b"])

    def test_e4b_is_recommended_with_room_for_it(self):
        with mock.patch.object(cli, "memory", return_value=16e9):
            out = self.setup(tty=True, answers=[""])
        self.assertIn("E4B, 3.7 GB (recommended)", out)
        self.assertEqual(FakeLLM.calls, ["e4b"])
        self.assertGreater(cli.memory(), 1e9)

    def test_none_is_for_boards(self):
        out = self.setup(llm="none")
        self.assertEqual(FakeLLM.calls, [])
        self.assertIn("No chat model", out)
        self.setup(tty=True)
        self.assertEqual(FakeLLM.calls, [])
        # --llm changes the answer
        self.setup(llm="e4b")
        self.setup()
        self.assertEqual(FakeLLM.calls, ["e4b", "e4b"])


class TestStartModel(unittest.TestCase):
    def test_whichever_gemma_was_downloaded(self):
        e2b, e4b = "gemma-4-E2B-it", "gemma-4-E4B-it"
        with mock.patch.object(litertlm_engine, "MODEL", ""):
            self.assertEqual(_start_model("cpu", {e2b: "a", e4b: "b"}), e2b)
            self.assertEqual(_start_model("gpu", {e2b: "a", e4b: "b"}), e4b)
            self.assertEqual(_start_model("cpu", {e4b: "b"}), e4b)
            self.assertEqual(_start_model("gpu", {e2b: "a"}), e2b)
            self.assertEqual(_start_model("cpu", {"Qwen3-0.6B": "c"}), "Qwen3-0.6B")
            # With none, it's the backend's own, which the log says to download
            self.assertEqual(_start_model("cpu", {}), e2b)
        with mock.patch.object(litertlm_engine, "MODEL", "e4b"):
            self.assertEqual(_start_model("cpu", {e2b: "a", e4b: "b"}), e4b)


if __name__ == "__main__":
    unittest.main()


class TestServeLlm(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        patch = mock.patch.dict(os.environ, {"LITE_ASSISTANT_HOME": folder.name})
        patch.start()
        self.addCleanup(patch.stop)

    def serve(self, llm=None):
        args = argparse.Namespace(host="127.0.0.1", port=5000, tools_dir=[], behind_proxy=False, allow_public=False,
                                  certfile=None, keyfile=None, forget_chats_after=None, llm=llm)
        with mock.patch("uvicorn.run") as run, mock.patch("lite_assistant.app.create_app") as create, \
                redirect_stdout(io.StringIO()):
            cli.serve(args)
        return create.call_args.kwargs["no_llm"]

    def test_none_loads_no_chat_model(self):
        self.assertTrue(self.serve("none"))
        self.assertFalse(self.serve("e2b"))
        self.assertFalse(self.serve())

    def test_what_setup_kept_is_served(self):
        with redirect_stdout(io.StringIO()):
            cli.llm_choice(argparse.Namespace(llm="none"))
        self.assertTrue(self.serve())
        self.assertFalse(self.serve("e4b"))

    def test_no_llm_leaves_that_kind_out(self):
        from lite_assistant.engines import load_engines
        with mock.patch("lite_assistant.engines.engine_classes", return_value={"llm": {"x": FakeLLM}, "tts": {}}):
            self.assertEqual(load_engines(skip=["llm"]), {"llm": {}, "tts": {}})
