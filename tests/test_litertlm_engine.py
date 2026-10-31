"""
Unit tests for the LiteRT-LM (Gemma) LLM engine. Loads the model once; skips
if it hasn't been downloaded (run 'lite-assistant setup' first).
"""
import unittest
from unittest import mock
import subprocess
import threading

from lite_assistant.engines import ToolCall
from lite_assistant.engines.litertlm_engine import LiteRTLMEngine, MAX_NUM_TOKENS, MODELS, _TextToolCalls
from lite_assistant.paths import cached_hf_file

CPU_MODEL = cached_hf_file(MODELS["e2b"]["repo"], MODELS["e2b"]["file"])

COLOR_PROMPT = [{"role": "user", "content": "Invent a name for a new color. Reply with only the name."}]

class TestMessageConversion(unittest.TestCase):
    def test_string_content_passes_through(self):
        self.assertEqual(LiteRTLMEngine._to_litert_message({"role": "system", "content": "Be brief."}),
                         {"role": "system", "content": "Be brief."})

    def test_missing_role_and_content(self):
        self.assertEqual(LiteRTLMEngine._to_litert_message({}), {"role": "user", "content": ""})

    def test_content_parts(self):
        message = {"role": "user", "content": [
            {"type": "text", "text": "What is said here?"},
            {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
        ]}
        self.assertEqual(LiteRTLMEngine._to_litert_message(message), {"role": "user", "content": [
            {"type": "text", "text": "What is said here?"},
            {"type": "audio", "blob": "UklGRg=="},
            {"type": "image", "blob": "iVBORw0KGgo="},
        ]})
    def test_tool_calls_and_results(self):
        turns = LiteRTLMEngine._to_litert_turns([
            {"role": "user", "content": "Weather in Paris and Tokyo?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"call_{i}", "type": "function",
                 "function": {"name": "weather", "arguments": f'{{"city": "{city}"}}'}}
                for i, city in ((1, "Paris"), (2, "Tokyo"))]},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"sky": "cloudy"}'},
            {"role": "tool", "tool_call_id": "call_2", "content": "sunny"},
        ])
        self.assertEqual(turns[1:], [
            {"role": "assistant", "tool_calls": [
                {"type": "function", "function": {"name": "weather", "arguments": {"city": "Paris"}}},
                {"type": "function", "function": {"name": "weather", "arguments": {"city": "Tokyo"}}}]},
            {"role": "tool", "content": [
                {"type": "tool_response", "name": "weather", "response": {"sky": "cloudy"}},
                {"type": "tool_response", "name": "weather", "response": "sunny"}]},
        ])


class TestTextToolCalls(unittest.TestCase):
    """ Tool calls a model writes as text, as Qwen3 and Qwen3.5 do, become calls. """
    TOOLS = [{"type": "function", "function": {"name": "set_timer", "parameters": {"properties": {
        "seconds": {"type": "integer"}, "label": {"type": "string"}}}}},
             {"type": "function", "function": {"name": "get_time", "parameters": {"properties": {}}}}]

    def stream(self, text, size=3):
        """ text fed in pieces of size, as a model streams it, with the calls' ids left out. """
        calls, out = _TextToolCalls(self.TOOLS), []
        for i in range(0, len(text), size):
            out += calls.feed(text[i:i + size])
        out += calls.flush()
        joined = []
        for piece in out:
            if isinstance(piece, ToolCall):
                joined.append((piece.name, piece.arguments))
            elif joined and isinstance(joined[-1], str):
                joined[-1] += piece
            else:
                joined.append(piece)
        return joined

    def test_tagged_calls(self):
        self.assertEqual(self.stream(
            "Setting it. <tool_call>\n<function=set_timer>\n<parameter=seconds>\n90\n</parameter>\n"
            "<parameter=label>\n007\n</parameter>\n</function>\n</tool_call>"),
            ["Setting it. ", ("set_timer", '{"seconds": 90, "label": "007"}')])
        self.assertEqual(self.stream("<tool_call>\n<function=get_time>\n</function>\n</tool_call>", 1),
                         [("get_time", "{}")])

    def test_json_calls(self):
        self.assertEqual(self.stream('<tool_call>\n{"name": "set_timer", "arguments": {"seconds": 5}}\n</tool_call>'
                                     '<tool_call>{"name": "get_time", "arguments": "{}"}</tool_call>'),
                         [("set_timer", '{"seconds": 5}'), ("get_time", "{}")])

    def test_unclosed_call_at_the_end(self):
        self.assertEqual(self.stream("<tool_call>\n<function=get_time>\n</function>\n"), [("get_time", "{}")])

    def test_text_is_left_alone(self):
        self.assertEqual(self.stream("Use <b>bold</b> or a < sign."), ["Use <b>bold</b> or a < sign."])
        self.assertEqual(self.stream("It ends <tool_"), ["It ends <tool_"])
        self.assertEqual(self.stream("<tool_call>not a call</tool_call> after"), ["<tool_call>not a call</tool_call> after"])


class TestModelChoice(unittest.TestCase):
    """ use() and the switch it makes generate do, without loading a model. """
    FACTS = {"gemma-4-E2B-it": ["text", "image", "audio"], "gemma-4-E4B-it": ["text", "image", "audio"],
             "Qwen3-0.6B": ["text"]}

    def setUp(self):
        self.engine = LiteRTLMEngine.__new__(LiteRTLMEngine)
        self.engine.engine, self.engine.backend = object(), "cpu"
        self.engine.model_id, self.engine.modalities = "gemma-4-E2B-it", self.FACTS["gemma-4-E2B-it"]
        self.engine._lock = threading.Lock()
        self.downloaded = dict(self.FACTS)
        for name, fake in (("downloaded", lambda: {i: f"/models/{i}" for i in self.downloaded}),
                           ("facts", lambda path: {"modalities": self.FACTS[path.split("/")[-1]]})):
            patch = mock.patch(f"lite_assistant.litert_models.{name}", side_effect=fake)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch("os.path.getsize", lambda path: len(path))
        patch.start()
        self.addCleanup(patch.stop)
        self.switches = []

        def switch(model_id):
            self.switches.append(model_id)
            self.engine.model_id, self.engine.modalities = model_id, self.FACTS[model_id]
        self.engine._switch = switch
        self.engine._reply = lambda messages, *args: iter(["Hi"])

    def test_names(self):
        for name in ("gemma-4-E4B-it", "GEMMA-4-e4b-IT", "e4b", "E4B"):
            self.assertEqual(self.engine.use(name).get_model_id(), "gemma-4-E4B-it")
        self.assertEqual(self.engine.use("qwen3-0.6b").get_model_id(), "Qwen3-0.6B")
        for name in (None, "", "litertlm", "gpt-4o", "gemma-4-E4B"):
            self.assertIsNone(self.engine.use(name))

    def test_models_are_the_downloaded_ones(self):
        self.assertEqual(self.engine.get_models(), {i: {"modalities": m, "size": len(f"/models/{i}"),
                                                         "context": MAX_NUM_TOKENS, "thinking": True, "tools": True}
                                                     for i, m in self.FACTS.items()})
        self.assertEqual(self.engine.use("Qwen3-0.6B").get_modalities(), ["text"])
        del self.downloaded["gemma-4-E4B-it"]
        self.assertNotIn("gemma-4-E4B-it", self.engine.get_models())
        with self.assertRaises(LookupError):
            self.engine.use("e4b")

    def test_generate_switches_once(self):
        messages = [{"role": "user", "content": "Hi"}]
        choice = self.engine.use("e4b")
        self.assertEqual(list(choice.generate(messages)), ["Hi"])
        self.assertEqual(list(choice.generate(messages)), ["Hi"])
        # The engine itself stays on whatever is loaded
        self.assertEqual(list(self.engine.generate(messages)), ["Hi"])
        self.assertEqual(self.switches, ["gemma-4-E4B-it"])
        self.assertEqual(choice.backend, "cpu")

    def test_an_empty_reply_to_a_tool_result_is_asked_again(self):
        asked = []

        def reply(messages, max_tokens, sampler, usage, *args):
            asked.append(messages)
            usage["prompt_tokens"], usage["completion_tokens"] = 10, 1
            return iter([] if len(asked) == 1 else ["It's noon."])
        self.engine._reply = reply
        messages = [{"role": "user", "content": "Time?"}, {"role": "tool", "content": "12:00"}]
        usage = {}
        self.assertEqual(list(self.engine.generate(messages, usage=usage)), ["It's noon."])
        self.assertEqual(asked[1][-1]["role"], "user")
        self.assertEqual(usage, {"prompt_tokens": 20, "completion_tokens": 2})
        # Anything else that says nothing isn't
        asked.clear()
        self.engine._reply = lambda messages, *args: asked.append(messages) or iter([])
        self.assertEqual(list(self.engine.generate([{"role": "user", "content": "Hi"}])), [])
        self.assertEqual(len(asked), 1)

    def test_image_for_a_model_that_cant_see(self):
        image = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}]
        with self.assertRaisesRegex(ValueError, "Qwen3-0.6B can't take images"):
            list(self.engine.use("Qwen3-0.6B").generate(image))

    def test_requests_wait_for_a_switch(self):
        """ While a switch has no engine loaded, a request waits for it, still naming its model. """
        engine = self.engine.engine
        with self.engine._lock:
            self.engine.engine = None
            self.assertEqual(len(self.engine.get_models()), 3)
            choice = self.engine.use("e4b")
            replies = []
            thread = threading.Thread(target=lambda: replies.append(list(choice.generate([{"role": "user", "content": "Hi"}]))))
            thread.start()
            thread.join(0.2)
            self.assertTrue(thread.is_alive())
            self.engine.engine = engine
        thread.join(5)
        self.assertEqual((replies, self.switches), ([["Hi"]], ["gemma-4-E4B-it"]))


class TestGPUCheck(unittest.TestCase):
    def check(self, returncode=0, log=""):
        result = subprocess.CompletedProcess([], returncode, b"", log.encode())
        with mock.patch("subprocess.run", return_value=result):
            return LiteRTLMEngine._gpu_works("model.litertlm")

    def test_crashed_backend_is_not_usable(self):
        # The WSL GPU driver segfaults the process (exit -11, or 139 from a shell)
        self.assertFalse(self.check(returncode=-11))

    def test_software_adapter_is_not_usable(self):
        self.assertFalse(self.check(log="environment.cc:532] Selected adapter: llvmpipe (LLVM 20.1.2, 256 bits), "
                                        "arch=software, vendor=mesa, backend=Vulkan, adapterType=CPU / Software"))

    def test_hardware_adapter_is_usable(self):
        self.assertTrue(self.check(log="environment.cc:532] Selected adapter: NVIDIA GeForce RTX 4090, "
                                       "arch=ada, vendor=nvidia, backend=Vulkan, adapterType=Discrete GPU"))

    def test_hung_backend_is_not_usable(self):
        with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired("probe", 300)):
            self.assertFalse(LiteRTLMEngine._gpu_works("model.litertlm"))

    @unittest.skipUnless(CPU_MODEL, "CPU model not downloaded")
    def test_real_check_returns_without_crashing_the_caller(self):
        # On a machine whose GPU backend crashes, reaching the assertion at all
        # is the point: the crash stays in the child process
        self.assertIn(LiteRTLMEngine._gpu_works(CPU_MODEL), (True, False))

class TestLiteRTLMEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = LiteRTLMEngine()
        if not cls.engine.engine:
            raise unittest.SkipTest("No LiteRT-LM model loaded")

    def reply(self, messages=COLOR_PROMPT, **kwargs):
        return "".join(self.engine.generate(messages, **kwargs)).strip()

    def test_streams_text_chunks_and_reports_usage(self):
        usage = {}
        chunks = list(self.engine.generate(
            [{"role": "user", "content": "Count from one to ten in words."}], max_tokens=40, usage=usage))
        self.assertGreater(len(chunks), 1, "reply arrived in one piece, not streamed")
        self.assertTrue(all(isinstance(c, str) for c in chunks))
        self.assertGreater(usage["prompt_tokens"], 0)
        self.assertGreater(usage["completion_tokens"], 0)

    def test_max_tokens_caps_the_reply(self):
        usage = {}
        self.reply([{"role": "user", "content": "Write a long story about a dragon."}], max_tokens=5, usage=usage)
        self.assertLessEqual(usage["completion_tokens"], 5)

    def test_earlier_turns_are_context(self):
        reply = self.reply([
            {"role": "system", "content": "Answer in as few words as possible."},
            {"role": "user", "content": "My name is Zorblax."},
            {"role": "assistant", "content": "Nice to meet you, Zorblax."},
            {"role": "user", "content": "What is my name?"},
        ], max_tokens=16)
        self.assertIn("zorblax", reply.lower())

    def test_temperature_samples(self):
        # Regression test: with only a
        # temperature set, every reply used to be identical
        replies = {self.reply(max_tokens=12, temperature=1.0) for _ in range(4)}
        self.assertGreater(len(replies), 1, f"temperature=1.0 gave the same reply 4 times: {replies}")

    def test_temperature_zero_is_deterministic(self):
        self.assertEqual(self.reply(max_tokens=12, temperature=0.0), self.reply(max_tokens=12, temperature=0.0))

    def test_no_sampling_options_is_deterministic(self):
        self.assertEqual(self.reply(max_tokens=12), self.reply(max_tokens=12))

    def test_empty_messages_rejected(self):
        with self.assertRaises(ValueError):
            next(self.engine.generate([]))

    def test_abandoned_generation_frees_the_engine(self):
        # A client that disconnects mid-reply closes the generator; the next
        # request must not wait on the first one's lock
        chunks = self.engine.generate([{"role": "user", "content": "Write a long story about a dragon."}])
        next(chunks)
        chunks.close()

        result = []
        worker = threading.Thread(target=lambda: result.append(self.reply(max_tokens=4)), daemon=True)
        worker.start()
        worker.join(timeout=60)
        self.assertFalse(worker.is_alive(), "second generation is stuck behind the abandoned one")
        self.assertTrue(result[0])

    def test_upstream_partial_sampler_config_still_greedy(self):
        # Canary for the upstream bug the engine works around. If this fails,
        # LiteRT-LM fixed it, and the full SamplerConfig built in generate()
        # may no longer be needed.
        import litert_lm
        replies = set()
        for seed in (1, 2, 3, 4):
            conv = self.engine.engine.create_conversation(
                sampler_config=litert_lm.SamplerConfig(temperature=1.0, seed=seed))
            try:
                replies.add("".join(str(c) for c in conv.send_message_async(
                    COLOR_PROMPT[0], max_output_tokens=12)).strip())
            finally:
                conv.close()
        self.assertEqual(len(replies), 1, "LiteRT-LM now samples a partial SamplerConfig; "
                                          "the full SamplerConfig may no longer be needed")

if __name__ == '__main__':
    unittest.main()
