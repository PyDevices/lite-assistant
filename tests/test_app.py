"""
Tests for the HTTP API with fake engines, so they need no models and run in
seconds. The OpenAI-compatible endpoints are driven through the official
openai client, which is the compatibility that matters: if the SDK can parse
a reply, so can the apps built on it.
"""
import base64
import http.server
import io
import itertools
import json
import os
import re
import tempfile
import threading
import time
import unittest
import urllib.parse
from unittest import mock

import anyio
import numpy as np
import openai
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from lite_assistant import litert_models
from lite_assistant.app import Engines, create_app, in_threadpool
from lite_assistant import depth
from lite_assistant import detection
from lite_assistant.engines import (ClassifierEngine, DepthEngine, DepthMap, Detection, DetectionEngine,
                                    EmbeddingEngine, LLMEngine, SoundEngine, Sounds, STTEngine, Thought, ToolCall,
                                    TTSEngine)
from lite_assistant.function_tools import ToolError
from lite_assistant.tools import MAX_ROUNDS, BuiltinLLM, Toolbox, definition
from lite_assistant.tools.clock import Timers, get_time
from lite_assistant.tools.distance import Distances
from lite_assistant.tools.notes import Notes
from lite_assistant.users import User
from lite_assistant.tools.sounds import SoundTagger
from lite_assistant.vad import VADStream

TEST_WAV = os.path.join(os.path.dirname(__file__), "test_audio.wav")
TEST_IMAGE = os.path.join(os.path.dirname(__file__), "test_image.jpg")

# One fake second of 16-bit 24kHz mono silence per sentence spoken
PCM_SECOND = b"\0\0" * 24000


class FakeTTS(TTSEngine):
    def __init__(self):
        self.spoken = []

    def get_available_voices(self):
        return [{"id": "bob", "name": "Bob (en_US, Medium)"}, {"id": "amy", "name": "Amy (en_US, Medium)"}]

    def get_default_voice(self):
        return "amy"

    def generate_pcm(self, text, voice):
        self.spoken.append((text, voice))
        yield PCM_SECOND

    @classmethod
    def setup(cls):
        pass


class FakeSTT(STTEngine):
    TRANSCRIPT = "Hello, café."

    def transcribe(self, audio_bytes, language=None):
        return self.TRANSCRIPT

    @classmethod
    def setup(cls):
        pass


class FakeLLM(LLMEngine):
    """
    Replies with REPLY one word at a time, cut off at max_tokens words. Given
    tools, it says "Checking." and calls the first, unless it was just given
    a tool's result. Asked to think, it thinks THOUGHTS first.
    """
    REPLY = "One two three. Four five six."
    THOUGHTS = ["Hmm, ", "six words."]

    def __init__(self):
        self.requests = []
        self.closed = 0

    def get_model_id(self):
        return "fake-gemma"

    def generate(self, messages, max_tokens=None, temperature=None, top_p=None, usage=None, tools=None,
                 thinking=None, response_schema=None):
        self.requests.append({"messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                              "top_p": top_p, "tools": tools, "thinking": thinking, "schema": response_schema})
        if thinking:
            for thought in self.THOUGHTS:
                yield Thought(thought)
        if tools and messages[-1]["role"] != "tool":
            yield "Checking."
            yield ToolCall(tools[0]["function"]["name"], '{"city": "Paris"}', "call_1")
            return
        words = self.REPLY.split(" ")[:max_tokens]
        try:
            for i, word in enumerate(words):
                yield word if i == 0 else " " + word
            if usage is not None:
                usage["prompt_tokens"] = 7
                usage["completion_tokens"] = len(words)
        finally:
            self.closed += 1

    @classmethod
    def setup(cls):
        pass


class SwitchingLLM(FakeLLM):
    """
    A FakeLLM that switches between models, as the LiteRT-LM engine does, by
    the inputs each takes. It knows one more, "fake-huge", that isn't downloaded.
    """
    MODELS = {"fake-small": ["text", "image", "audio"], "fake-big": ["text", "image", "audio"],
              "fake-text": ["text"]}

    def __init__(self):
        super().__init__()
        self.model = "fake-small"
        self.switches = []

    def get_model_id(self):
        return self.model

    def get_modalities(self):
        return self.MODELS[self.model]

    def get_models(self):
        return {model: {"modalities": modalities} for model, modalities in self.MODELS.items()}

    def generate(self, messages, **options):
        parts = [part for m in messages if isinstance(m.get("content"), list) for part in m["content"]]
        if "image" not in self.get_modalities() and any(p.get("type") == "image_url" for p in parts):
            raise ValueError(f"{self.model} can't take images")
        return super().generate(messages, **options)

    def use(self, model):
        if model == "fake-huge":
            raise LookupError("fake-huge isn't downloaded.")
        if model not in self.MODELS:
            return None
        engine = self

        class Choice(LLMEngine):
            def get_model_id(self):
                return model

            def get_modalities(self):
                return engine.MODELS[model]

            def generate(self, messages, **options):
                if engine.model != model:
                    engine.switches.append(model)
                    engine.model = model
                return engine.generate(messages, **options)

        return Choice()


class FakeEmbedder(EmbeddingEngine):
    """ Embeds an input as [number of items embedded so far, 1, 0.5, 0.25], cut to the requested size. """
    def __init__(self):
        self.items = []

    def get_model_id(self):
        return "fake-embeddinggemma"

    def get_dimensions(self):
        return [2, 4]

    def embed(self, item, dimensions=None):
        if isinstance(item, dict) and item.get("type") not in ("image_url", "input_audio"):
            raise ValueError(f"can't embed a {item.get('type')!r} part")
        self.items.append(item)
        return [float(len(self.items)), 1.0, 0.5, 0.25][:dimensions or 4]

    @classmethod
    def setup(cls):
        pass


class FakeDepth(DepthEngine):
    """
    Sees every photo as the same 40x60 map: sky (no estimate) in the top
    quarter, a wall 1 m away in the left third and 4 m away in the rest, and
    a ball 0.5 m away in the bottom left corner. A photo of b"bad" isn't one.
    """
    def __init__(self):
        self.photos = []

    def get_model_id(self):
        return "fake-moge"

    def estimate(self, image):
        if image == b"bad":
            raise ValueError("cannot identify image file")
        self.photos.append(image)
        distances = np.full((40, 60), 4.0)
        distances[:10] = np.nan
        distances[10:, :20] = 1.0
        distances[32:, :8] = 0.5
        return DepthMap(distances, 60.0)


class FakeDetector(DetectionEngine):
    """
    Sees every photo the way FakeDepth does: a chair against the near wall
    on the left, a ball in the bottom left corner and a dining table on the
    right. A photo of b"bad" isn't one.
    """
    FOUND = [Detection("dining table", 0.8, (0.5, 0.5, 1.0, 1.0)),
             Detection("chair", 0.6, (0.05, 0.3, 0.3, 0.7)),
             Detection("sports ball", 0.4, (0.0, 0.8, 0.133, 1.0))]

    def __init__(self):
        self.photos = []

    def get_model_id(self):
        return "fake-yolox"

    def get_labels(self):
        return ["chair", "dining table", "sports ball"]

    def detect(self, image, threshold=None):
        if image == b"bad":
            raise ValueError("cannot identify image file")
        self.photos.append(image)
        return [d for d in self.FOUND if d.score >= (0.35 if threshold is None else threshold)]


class FakeFaces(FakeDetector):
    """ Sees one face, looking out from the near wall on the left. """
    FOUND = [Detection("face", 0.9, (0.1, 0.4, 0.2, 0.6),
                       {"right_eye": (0.13, 0.45), "left_eye": (0.17, 0.45), "nose": (0.15, 0.5)})]

    def get_model_id(self):
        return "fake-yunet"


class FakeSound(SoundEngine):
    """
    Hears every clip as 14.5 s: speech and a dog in the first 10 s window, a
    doorbell in the last. Audio of b"bad" isn't audio.
    """
    def __init__(self):
        self.clips = []

    def get_model_id(self):
        return "fake-panns"

    def classify(self, audio):
        if audio == b"bad":
            raise ValueError("Format not recognised.")
        self.clips.append(audio)
        return Sounds(["Speech", "Doorbell", "Dog"], np.array([[0.8, 0.05, 0.3], [0.2, 0.6, 0.0]]),
                      [(0.0, 10.0), (4.5, 14.5)], 14.5)


class FakeClassifier(ClassifierEngine):
    """ Scores a label 2 if the text or prompt names it, else 0, and takes at most 3 labels. """
    def __init__(self):
        self.requests = []

    def get_model_id(self):
        return "fake-gliclass"

    def logits(self, text, labels, prompt=None):
        if len(labels) > 3:
            raise ValueError(f"{len(labels)} labels; the model takes at most 3")
        self.requests.append((text, list(labels), prompt))
        return [2.0 if label in f"{prompt or ''}{text}".lower() else 0.0 for label in labels]


class FakeVAD:
    """ Hears speech in any window louder than a tenth of full scale. """
    def stream(self):
        return VADStream(lambda window, state: (float(abs(window).max() > 0.1), state))


def tone(ms):
    """ ms of a loud square wave, which FakeVAD hears as speech. """
    return (b"\x00\x40" * 12 + b"\x00\xc0" * 12) * ms


def silence(ms):
    return b"\0\0" * 24 * ms


class APITestCase(unittest.TestCase):
    VAD = None
    # The built-in tools on by default; off unless a test case is about them
    BUILTIN_TOOLS = "none"
    LLM = FakeLLM

    def setUp(self):
        self.tts, self.stt, self.llm, self.embedder = FakeTTS(), FakeSTT(), self.LLM(), FakeEmbedder()
        self.depth, self.detector, self.sound = FakeDepth(), FakeDetector(), FakeSound()
        self.classifier = FakeClassifier()
        engines = Engines({"kokoro": self.tts}, {"moonshine": self.stt}, {"litertlm": self.llm},
                          {"embeddinggemma": self.embedder}, {"moge": self.depth},
                          detect={"yunet": FakeFaces(), "yolox": self.detector}, sound={"panns": self.sound},
                          classify={"gliclass": self.classifier})
        self.http = TestClient(create_app(engines, self.VAD, self.BUILTIN_TOOLS), client=("127.0.0.1", 50000))
        self.http.__enter__()
        self.addCleanup(self.http.__exit__, None, None, None)
        self.client = openai.OpenAI(base_url=f"{self.http.base_url}/v1", api_key="unused",
                                    http_client=self.http, max_retries=0)


class TestOpenAIChat(APITestCase):
    def test_completion(self):
        reply = self.client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "Count."}], temperature=0.5, top_p=0.9)
        self.assertEqual(reply.object, "chat.completion")
        self.assertTrue(reply.id.startswith("chatcmpl-"))
        self.assertEqual(reply.model, "fake-gemma")
        self.assertEqual(reply.choices[0].message.role, "assistant")
        self.assertEqual(reply.choices[0].message.content, FakeLLM.REPLY)
        self.assertEqual(reply.choices[0].finish_reason, "stop")
        self.assertEqual((reply.usage.prompt_tokens, reply.usage.completion_tokens, reply.usage.total_tokens),
                         (7, 6, 13))
        self.assertEqual(self.llm.requests[0]["temperature"], 0.5)
        self.assertEqual(self.llm.requests[0]["top_p"], 0.9)
        self.assertEqual(self.llm.closed, 1)

    def test_max_tokens_truncates_with_length(self):
        for option in ("max_tokens", "max_completion_tokens"):
            with self.subTest(option=option):
                reply = self.client.chat.completions.create(
                    model="litertlm", messages=[{"role": "user", "content": "Count."}], **{option: 2})
                self.assertEqual(reply.choices[0].message.content, "One two")
                self.assertEqual(reply.choices[0].finish_reason, "length")

    def test_stream(self):
        stream = self.client.chat.completions.create(
            model="litertlm", messages=[{"role": "user", "content": "Count."}], stream=True,
            stream_options={"include_usage": True})
        chunks = list(stream)
        self.assertEqual(len({c.id for c in chunks}), 1, "chunks don't share one completion id")
        self.assertEqual(chunks[0].choices[0].delta.role, "assistant")
        text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
        self.assertEqual(text, FakeLLM.REPLY)
        self.assertEqual(chunks[-2].choices[0].finish_reason, "stop")
        self.assertEqual(chunks[-1].choices, [])
        self.assertEqual(chunks[-1].usage.completion_tokens, 6)
        self.assertEqual(self.llm.closed, 1)

    def test_stream_without_usage(self):
        chunks = list(self.client.chat.completions.create(
            model="litertlm", messages=[{"role": "user", "content": "Count."}], stream=True))
        self.assertTrue(all(c.usage is None for c in chunks))
        self.assertEqual(chunks[-1].choices[0].finish_reason, "stop")

    def test_history_and_content_parts_reach_the_engine(self):
        messages = [{"role": "system", "content": "Be brief."},
                    {"role": "user", "content": [{"type": "text", "text": "Hi"},
                                                 {"type": "input_audio",
                                                  "input_audio": {"data": "UklGRg==", "format": "wav"}}]}]
        self.client.chat.completions.create(model="litertlm", messages=messages)
        self.assertEqual(self.llm.requests[0]["messages"], messages)

    def test_an_attached_file_reaches_the_engine_as_text(self):
        notes = "data:text/markdown;base64," + base64.b64encode(b"# Shed\nThe key is under the pot.").decode()
        self.client.chat.completions.create(model="litertlm", messages=[{"role": "user", "content": [
            {"type": "text", "text": "Where's the key?"},
            {"type": "file", "file": {"filename": "shed.md", "file_data": notes}}]}])
        self.assertEqual(self.llm.requests[-1]["messages"][0]["content"][1],
                         {"type": "text", "text": "The attached file shed.md:\n\n# Shed\nThe key is under the pot."})
        # The Responses API's input_file, with bare base64
        self.client.responses.create(model="litertlm", input=[{"role": "user", "content": [
            {"type": "input_text", "text": "Where's the key?"},
            {"type": "input_file", "filename": "shed.txt", "file_data": base64.b64encode(b"Under the pot.").decode()}]}])
        self.assertEqual(self.llm.requests[-1]["messages"][0]["content"][1]["text"],
                         "The attached file shed.txt:\n\nUnder the pot.")
        for bad in ({"filename": "x.bin", "file_data": base64.b64encode(b"\0\1\2").decode()},
                    {"file_id": "file-abc"}):
            with self.assertRaises(openai.BadRequestError):
                self.client.chat.completions.create(model="litertlm", messages=[{"role": "user", "content": [
                    {"type": "file", "file": bad}]}])

    def test_rejects_empty_messages(self):
        with self.assertRaises(openai.BadRequestError) as error:
            self.client.chat.completions.create(model="litertlm", messages=[])
        self.assertIn("messages", error.exception.message)

    def test_engine_failure_is_a_server_error(self):
        def broken(*args, **kwargs):
            raise RuntimeError("model fell over")
            yield
        self.llm.generate = broken
        with self.assertRaises(openai.InternalServerError) as error:
            self.client.chat.completions.create(model="litertlm", messages=[{"role": "user", "content": "Hi"}])
        self.assertIn("model fell over", error.exception.message)


class ImageServer(http.server.ThreadingHTTPServer):
    """ Serves the test image at /cat.jpg and some text at /page.html, on a free local port. """
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/cat.jpg":
                with open(TEST_IMAGE, "rb") as f:
                    body, media_type = f.read(), "image/jpeg"
            elif self.path == "/page.html":
                body, media_type = b"<p>not an image</p>", "text/html"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", media_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    def __init__(self):
        super().__init__(("127.0.0.1", 0), self.Handler)
        threading.Thread(target=self.serve_forever, daemon=True).start()

    def url(self, path):
        return f"http://127.0.0.1:{self.server_address[1]}{path}"


class TestImages(APITestCase):
    def setUp(self):
        super().setUp()
        with open(TEST_IMAGE, "rb") as f:
            self.image = f.read()
        self.data_url = "data:image/jpeg;base64," + base64.b64encode(self.image).decode("ascii")

    def ask(self, url):
        return self.client.chat.completions.create(model="litertlm", messages=[{"role": "user", "content": [
            {"type": "text", "text": "What is this?"}, {"type": "image_url", "image_url": {"url": url}}]}])

    def sent_url(self):
        return self.llm.requests[0]["messages"][0]["content"][1]["image_url"]["url"]

    def test_data_url_reaches_the_engine(self):
        self.ask(self.data_url)
        self.assertEqual(self.sent_url(), self.data_url)

    def test_http_url_is_fetched_into_a_data_url(self):
        server = ImageServer()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.ask(server.url("/cat.jpg"))
        self.assertEqual(self.sent_url(), self.data_url)

    def test_unusable_images_are_bad_requests(self):
        server = ImageServer()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        for url in (server.url("/page.html"), server.url("/missing.jpg"), "ftp://example.com/x.jpg",
                    "data:image/jpeg;base64,not*base64"):
            with self.subTest(url=url), self.assertRaises(openai.BadRequestError):
                self.ask(url)
        self.assertEqual(self.llm.requests, [])


class TestEmbeddings(APITestCase):
    def test_one_string(self):
        # The SDK asks for base64 and decodes it
        result = self.client.embeddings.create(model="text-embedding-3-small", input="Hello.")
        self.assertEqual(result.object, "list")
        self.assertEqual(result.model, "fake-embeddinggemma")
        self.assertEqual(result.data[0].embedding, [1.0, 1.0, 0.5, 0.25])
        self.assertEqual(self.embedder.items, ["Hello."])

    def test_list_and_dimensions(self):
        result = self.client.embeddings.create(model="embeddinggemma", input=["a", "b"], dimensions=2)
        self.assertEqual([(d.index, d.embedding) for d in result.data], [(0, [1.0, 1.0]), (1, [2.0, 1.0])])

    def test_float_format(self):
        response = self.http.post("/v1/embeddings", json={"input": "Hi", "encoding_format": "float"})
        self.assertEqual(response.json()["data"][0]["embedding"], [1.0, 1.0, 0.5, 0.25])

    def test_image_part(self):
        with open(TEST_IMAGE, "rb") as f:
            url = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("ascii")
        part = {"type": "image_url", "image_url": {"url": url}}
        response = self.http.post("/v1/embeddings", json={"input": ["a cat", part], "encoding_format": "float"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["data"]), 2)
        self.assertEqual(self.embedder.items[1], part)

    def test_bad_requests(self):
        for body in ({"input": []}, {"input": ""}, {"input": [1, 2, 3]}, {"input": [[1, 2]]},
                     {"input": "Hi", "dimensions": 3}, {"input": "Hi", "encoding_format": "int8"},
                     {"input": {"type": "text_file"}},
                     {"input": {"type": "image_url", "image_url": {"url": "ftp://example.com/x.jpg"}}}):
            with self.subTest(body=body):
                response = self.http.post("/v1/embeddings", json=body)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(response.json()["error"]["type"], "invalid_request_error")


class TestResponses(APITestCase):
    def test_create(self):
        response = self.client.responses.create(model="gpt-4o", input="Count.", instructions="Be brief.",
                                                temperature=0.5)
        self.assertTrue(response.id.startswith("resp_"))
        self.assertEqual(response.object, "response")
        self.assertEqual(response.status, "completed")
        self.assertEqual(response.model, "fake-gemma")
        self.assertEqual(response.output_text, FakeLLM.REPLY)
        self.assertEqual((response.usage.input_tokens, response.usage.output_tokens, response.usage.total_tokens),
                         (7, 6, 13))
        self.assertEqual(self.llm.requests[0]["messages"],
                         [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Count."}])
        self.assertEqual(self.llm.requests[0]["temperature"], 0.5)
        self.assertEqual(self.llm.closed, 1)

    def test_max_output_tokens_is_incomplete(self):
        response = self.client.responses.create(model="litertlm", input="Count.", max_output_tokens=2)
        self.assertEqual(response.status, "incomplete")
        self.assertEqual(response.incomplete_details.reason, "max_output_tokens")
        self.assertEqual(response.output_text, "One two")

    def test_input_items(self):
        image = "data:image/png;base64,iVBORw0KGgo="
        self.client.responses.create(model="litertlm", input=[
            {"role": "developer", "content": "Be brief."},
            {"role": "user", "content": "Hi."},
            {"role": "assistant", "content": "Hello!"},
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "What's this?"},
                {"type": "input_image", "image_url": image, "detail": "auto"}]},
        ])
        self.assertEqual(self.llm.requests[0]["messages"], [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi."},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": [{"type": "text", "text": "What's this?"},
                                         {"type": "image_url", "image_url": {"url": image}}]},
        ])

    def test_stream(self):
        with self.client.responses.stream(model="litertlm", input="Count.") as stream:
            deltas = [e.delta for e in stream if e.type == "response.output_text.delta"]
            final = stream.get_final_response()
        self.assertGreater(len(deltas), 1)
        self.assertEqual("".join(deltas), FakeLLM.REPLY)
        self.assertEqual(final.output_text, FakeLLM.REPLY)
        self.assertEqual(final.status, "completed")
        self.assertEqual(final.usage.output_tokens, 6)
        self.assertEqual(self.llm.closed, 1)

    def test_stream_event_order(self):
        events = list(self.client.responses.create(model="litertlm", input="Count.", stream=True))
        types = [e.type for e in events]
        self.assertEqual(types[:4], ["response.created", "response.in_progress", "response.output_item.added",
                                     "response.content_part.added"])
        self.assertEqual(types[-4:], ["response.output_text.done", "response.content_part.done",
                                      "response.output_item.done", "response.completed"])
        self.assertEqual([e.sequence_number for e in events], list(range(len(events))))

    def test_bad_requests(self):
        for input in ([], [{"type": "item_reference", "id": "msg_1"}], [{"type": "function_call", "name": "f"}],
                      [{"role": "user", "content": [{"type": "input_file", "file_id": "file-1"}]}],
                      [{"role": "user", "content": [{"type": "input_image", "image_url": "ftp://x/y.png"}]}]):
            with self.subTest(input=input), self.assertRaises(openai.BadRequestError):
                self.client.responses.create(model="litertlm", input=input)
        self.assertEqual(self.llm.requests, [])


class TestStoredResponses(APITestCase):
    def wait_for(self, response_id, statuses=("completed", "incomplete", "cancelled", "failed")):
        for _ in range(200):
            response = self.client.responses.retrieve(response_id)
            if response.status in statuses:
                return response
            time.sleep(0.01)
        self.fail(f"response stayed {response.status}")

    def test_chaining_carries_the_history_but_not_the_instructions(self):
        first = self.client.responses.create(model="litertlm", input="My name is Ada.", instructions="Be brief.")
        self.client.responses.create(model="litertlm", input="What's my name?", previous_response_id=first.id)
        self.assertEqual(self.llm.requests[1]["messages"], [
            {"role": "user", "content": "My name is Ada."},
            {"role": "assistant", "content": FakeLLM.REPLY},
            {"role": "user", "content": "What's my name?"},
        ])

    def test_retrieve_and_delete(self):
        created = self.client.responses.create(model="litertlm", input="Count.")
        retrieved = self.client.responses.retrieve(created.id)
        self.assertEqual((retrieved.id, retrieved.output_text, retrieved.store), (created.id, FakeLLM.REPLY, True))
        self.client.responses.delete(created.id)
        with self.assertRaises(openai.NotFoundError):
            self.client.responses.retrieve(created.id)
        with self.assertRaises(openai.NotFoundError):
            self.client.responses.create(model="litertlm", input="More.", previous_response_id=created.id)

    def test_store_false_keeps_nothing(self):
        created = self.client.responses.create(model="litertlm", input="Count.", store=False)
        with self.assertRaises(openai.NotFoundError):
            self.client.responses.retrieve(created.id)

    def test_streamed_response_is_stored(self):
        with self.client.responses.stream(model="litertlm", input="Count.") as stream:
            final = stream.get_final_response()
        self.assertEqual(self.client.responses.retrieve(final.id).output_text, FakeLLM.REPLY)

    def test_input_items_pages(self):
        created = self.client.responses.create(model="litertlm", input=[
            {"role": "user", "content": "One."}, {"role": "assistant", "content": "Two."},
            {"role": "user", "content": "Three."}])
        first = self.client.responses.input_items.list(created.id, order="asc", limit=2)
        self.assertEqual([item.content[0].text for item in first.data], ["One.", "Two."])
        self.assertTrue(first.has_more)
        rest = self.client.responses.input_items.list(created.id, order="asc", after=first.data[-1].id)
        self.assertEqual([item.content[0].text for item in rest.data], ["Three."])
        self.assertFalse(rest.has_more)
        newest = self.client.responses.input_items.list(created.id)
        self.assertEqual(newest.data[0].content[0].text, "Three.")

    def test_background(self):
        created = self.client.responses.create(model="litertlm", input="Count.", background=True)
        self.assertIn(created.status, ("queued", "in_progress"))
        self.assertTrue(created.background)
        done = self.wait_for(created.id)
        self.assertEqual((done.status, done.output_text), ("completed", FakeLLM.REPLY))
        self.assertEqual(self.llm.closed, 1)

    def test_cancel_background(self):
        def slow(messages, usage=None, **options):
            try:
                for i in range(1000):
                    time.sleep(0.01)
                    yield f" {i}"
            finally:
                self.llm.closed += 1
        self.llm.generate = slow
        created = self.client.responses.create(model="litertlm", input="Count.", background=True)
        cancelled = self.client.responses.cancel(created.id)
        self.assertEqual(cancelled.status, "cancelled")
        self.wait_for(created.id)
        self.assertEqual(self.llm.closed, 1)

    def test_only_background_responses_cancel(self):
        created = self.client.responses.create(model="litertlm", input="Count.")
        with self.assertRaises(openai.BadRequestError):
            self.client.responses.cancel(created.id)

    def test_background_rejects_stream_and_store_false(self):
        for options in ({"stream": True}, {"store": False}):
            with self.subTest(**options), self.assertRaises(openai.BadRequestError):
                self.client.responses.create(model="litertlm", input="Count.", background=True, **options)


class TestConversations(APITestCase):
    def test_lifecycle(self):
        conversation = self.client.conversations.create(
            items=[{"role": "user", "content": "Hi."}], metadata={"topic": "greeting"})
        self.assertTrue(conversation.id.startswith("conv_"))
        self.assertEqual(conversation.metadata, {"topic": "greeting"})
        self.assertEqual(self.client.conversations.retrieve(conversation.id).id, conversation.id)
        updated = self.client.conversations.update(conversation.id, metadata={"topic": "farewell"})
        self.assertEqual(updated.metadata, {"topic": "farewell"})
        deleted = self.client.conversations.delete(conversation.id)
        self.assertTrue(deleted.deleted)
        with self.assertRaises(openai.NotFoundError):
            self.client.conversations.retrieve(conversation.id)

    def test_items(self):
        conversation = self.client.conversations.create()
        added = self.client.conversations.items.create(conversation.id, items=[
            {"role": "user", "content": "One."}, {"role": "assistant", "content": "Two."}])
        self.assertEqual(len(added.data), 2)
        listed = self.client.conversations.items.list(conversation.id, order="asc")
        self.assertEqual([item.content[0].text for item in listed.data], ["One.", "Two."])
        first = added.data[0].id
        self.assertEqual(self.client.conversations.items.retrieve(first, conversation_id=conversation.id).id, first)
        self.client.conversations.items.delete(first, conversation_id=conversation.id)
        listed = self.client.conversations.items.list(conversation.id)
        self.assertEqual([item.content[0].text for item in listed.data], ["Two."])
        with self.assertRaises(openai.NotFoundError):
            self.client.conversations.items.retrieve(first, conversation_id=conversation.id)

    def test_responses_continue_and_extend_a_conversation(self):
        conversation = self.client.conversations.create(items=[{"role": "user", "content": "My name is Ada."}])
        first = self.client.responses.create(model="litertlm", input="Hello?", conversation=conversation.id)
        self.assertEqual(first.conversation.id, conversation.id)
        self.client.responses.create(model="litertlm", input="What's my name?",
                                     conversation={"id": conversation.id})
        self.assertEqual(self.llm.requests[1]["messages"], [
            {"role": "user", "content": "My name is Ada."},
            {"role": "user", "content": "Hello?"},
            {"role": "assistant", "content": FakeLLM.REPLY},
            {"role": "user", "content": "What's my name?"},
        ])
        items = self.client.conversations.items.list(conversation.id, order="asc", limit=100)
        self.assertEqual([item.role for item in items.data], ["user", "user", "assistant", "user", "assistant"])

    def test_streamed_and_background_responses_extend_it_too(self):
        conversation = self.client.conversations.create()
        with self.client.responses.stream(model="litertlm", input="One.", conversation=conversation.id) as stream:
            stream.get_final_response()
        created = self.client.responses.create(model="litertlm", input="Two.", conversation=conversation.id,
                                               background=True)
        for _ in range(200):
            if self.client.responses.retrieve(created.id).status == "completed":
                break
            time.sleep(0.01)
        items = self.client.conversations.items.list(conversation.id, order="asc")
        self.assertEqual(len(items.data), 4)

    def test_bad_requests(self):
        conversation = self.client.conversations.create()
        first = self.client.responses.create(model="litertlm", input="Hi.")
        with self.assertRaises(openai.BadRequestError):
            self.client.responses.create(model="litertlm", input="Hi.", conversation=conversation.id,
                                         previous_response_id=first.id)
        with self.assertRaises(openai.NotFoundError):
            self.client.responses.create(model="litertlm", input="Hi.", conversation="conv_missing")
        with self.assertRaises(openai.BadRequestError):
            self.client.conversations.items.create(conversation.id, items=[])
        with self.assertRaises(openai.BadRequestError):
            self.client.conversations.create(items=[{"role": "user", "content": "x"}] * 21)


class TestRealtime(APITestCase):
    def connect(self):
        ws = self.http.websocket_connect("/v1/realtime?model=gpt-realtime")
        ws.__enter__()
        self.addCleanup(ws.__exit__, None, None, None)
        self.assertEqual(ws.receive_json()["type"], "session.created")
        return ws

    def until(self, ws, kind):
        """ The events received up to and including the first of this type. """
        events = []
        while True:
            events.append(ws.receive_json())
            if events[-1]["type"] == kind:
                return events
            self.assertNotEqual(events[-1]["type"], "error", events[-1])

    def speak(self, ws):
        with open(TEST_WAV, "rb") as f:
            pcm = f.read()[44:]  # the WAV's PCM, past its header
        for i in range(0, len(pcm), 9600):
            ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[i:i + 9600]).decode()})
        ws.send_json({"type": "input_audio_buffer.commit"})
        return self.until(ws, "conversation.item.done")

    def test_spoken_turn(self):
        ws = self.connect()
        events = self.speak(ws)
        self.assertEqual([e["type"] for e in events],
                         ["input_audio_buffer.committed", "conversation.item.added", "conversation.item.done"])
        ws.send_json({"type": "response.create"})
        events = self.until(ws, "response.done")
        audio = b"".join(base64.b64decode(e["delta"]) for e in events if e["type"] == "response.output_audio.delta")
        self.assertEqual(audio, PCM_SECOND * 2)
        transcript = "".join(e["delta"] for e in events if e["type"] == "response.output_audio_transcript.delta")
        self.assertEqual(transcript, FakeLLM.REPLY)
        done = events[-1]["response"]
        self.assertEqual(done["status"], "completed")
        self.assertEqual(done["output"][0]["content"][0], {"type": "output_audio", "transcript": FakeLLM.REPLY})
        self.assertEqual(done["usage"]["output_tokens"], 6)
        system, user = self.llm.requests[0]["messages"]
        self.assertEqual(system["role"], "system")
        heard = base64.b64decode(user["content"][0]["input_audio"]["data"])
        self.assertEqual(heard[:4], b"RIFF")

    def test_text_turns_keep_the_conversation(self):
        ws = self.connect()
        ws.send_json({"type": "session.update", "session": {"type": "realtime", "output_modalities": ["text"],
                                                            "instructions": "Be brief."}})
        self.assertEqual(self.until(ws, "session.updated")[-1]["session"]["output_modalities"], ["text"])
        for text in ("Hi.", "Again."):
            ws.send_json({"type": "conversation.item.create", "item": {
                "type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}})
            ws.send_json({"type": "response.create"})
            events = self.until(ws, "response.done")
            self.assertEqual("".join(e["delta"] for e in events if e["type"] == "response.output_text.delta"),
                             FakeLLM.REPLY)
        self.assertEqual(self.llm.requests[1]["messages"], [
            {"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi."},
            {"role": "assistant", "content": FakeLLM.REPLY}, {"role": "user", "content": "Again."}])
        self.assertEqual(self.tts.spoken, [])

    def test_transcription(self):
        ws = self.connect()
        ws.send_json({"type": "session.update", "session": {
            "type": "realtime", "audio": {"input": {"transcription": {"model": "whisper-1"}}}}})
        self.until(ws, "session.updated")
        self.speak(ws)
        event = self.until(ws, "conversation.item.input_audio_transcription.completed")[-1]
        self.assertEqual(event["transcript"], "Hello, café.")

    def test_cancel(self):
        def slow(messages, usage=None, **options):
            try:
                for i in range(1000):
                    time.sleep(0.01)
                    yield f" {i}."
            finally:
                self.llm.closed += 1
        self.llm.generate = slow
        ws = self.connect()
        ws.send_json({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": "Count."}]}})
        ws.send_json({"type": "response.create"})
        self.until(ws, "response.output_audio.delta")
        ws.send_json({"type": "response.cancel"})
        done = self.until(ws, "response.done")[-1]["response"]
        self.assertEqual(done["status"], "cancelled")
        self.assertEqual(self.llm.closed, 1)

    def test_truncate(self):
        ws = self.connect()
        self.speak(ws)
        ws.send_json({"type": "response.create"})
        item_id = self.until(ws, "response.done")[-1]["response"]["output"][0]["id"]
        # Each fake sentence is a second of audio: cut halfway through the first
        ws.send_json({"type": "conversation.item.truncate", "item_id": item_id, "content_index": 0,
                      "audio_end_ms": 500})
        event = self.until(ws, "conversation.item.truncated")[-1]
        self.assertEqual((event["item_id"], event["audio_end_ms"]), (item_id, 500))
        ws.send_json({"type": "conversation.item.retrieve", "item_id": item_id})
        item = self.until(ws, "conversation.item.retrieved")[-1]["item"]
        self.assertEqual(item["content"][0]["transcript"], "One two three.")
        for end_ms in (2001, "500"):
            ws.send_json({"type": "conversation.item.truncate", "item_id": item_id, "content_index": 0,
                          "audio_end_ms": end_ms})
            self.assertEqual(ws.receive_json()["error"]["code"], "invalid_value")
        ws.send_json({"type": "response.create", "response": {"output_modalities": ["text"]}})
        self.until(ws, "response.done")
        self.assertEqual(self.llm.requests[1]["messages"][2], {"role": "assistant", "content": "One two three."})

    def test_errors(self):
        ws = self.connect()
        for event, code in (({"type": "input_audio_buffer.commit"}, "input_audio_buffer_commit_empty"),
                            ({"type": "nonsense"}, "invalid_event"),
                            ({"type": "response.cancel"}, "response_cancel_not_active"),
                            ({"type": "conversation.item.delete", "item_id": "nope"}, "item_not_found"),
                            ({"type": "session.update", "session": {"audio": {"output": {"voice": "nobody"}}}},
                             "invalid_value"),
                            ({"type": "session.update", "session": {"audio": {"input": {
                                "turn_detection": {"type": "server_vad"}}}}}, "unsupported_turn_detection")):
            with self.subTest(event=event["type"]):
                ws.send_json({**event, "event_id": "evt_1"})
                error = ws.receive_json()
                self.assertEqual(error["type"], "error")
                self.assertEqual((error["error"]["code"], error["error"]["event_id"]), (code, "evt_1"))



WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "The weather in a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
# The same tool as the Responses and Realtime APIs define it
FLAT_WEATHER = {"type": "function", **WEATHER["function"]}


class TestFunctionTools(APITestCase):
    def test_chat_tool_call(self):
        reply = self.client.chat.completions.create(
            model="litertlm", messages=[{"role": "user", "content": "Weather in Paris?"}], tools=[WEATHER])
        choice = reply.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        self.assertEqual(choice.message.content, "Checking.")
        call = choice.message.tool_calls[0]
        self.assertEqual((call.id, call.function.name, call.function.arguments),
                         ("call_1", "get_weather", '{"city": "Paris"}'))
        self.assertEqual(self.llm.requests[0]["tools"], [WEATHER])

    def test_chat_tool_result(self):
        messages = [{"role": "user", "content": "Weather in Paris?"},
                    {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
                    {"role": "tool", "tool_call_id": "call_1", "content": "cloudy"}]
        reply = self.client.chat.completions.create(model="litertlm", messages=messages, tools=[WEATHER])
        self.assertEqual((reply.choices[0].message.content, reply.choices[0].finish_reason), (FakeLLM.REPLY, "stop"))
        self.assertEqual(self.llm.requests[0]["messages"], messages)

    def test_chat_stream(self):
        chunks = list(self.client.chat.completions.create(
            model="litertlm", messages=[{"role": "user", "content": "Weather?"}], tools=[WEATHER], stream=True))
        calls = [call for c in chunks if c.choices and c.choices[0].delta.tool_calls
                 for call in c.choices[0].delta.tool_calls]
        self.assertEqual([(c.index, c.id, c.function.name) for c in calls], [(0, "call_1", "get_weather")])
        self.assertEqual(chunks[-1].choices[0].finish_reason, "tool_calls")

    def test_tool_choice(self):
        reply = self.client.chat.completions.create(
            model="litertlm", messages=[{"role": "user", "content": "Hi"}], tools=[WEATHER], tool_choice="none")
        self.assertEqual(reply.choices[0].message.content, FakeLLM.REPLY)
        self.assertIsNone(self.llm.requests[0]["tools"])
        for tools, choice in (([WEATHER], {"type": "function", "function": {"name": "other"}}),
                              ([{"type": "web_search"}], "auto"), ([{"type": "function", "function": {}}], "auto")):
            with self.subTest(tools=tools), self.assertRaises(openai.BadRequestError):
                self.client.chat.completions.create(model="litertlm", messages=[{"role": "user", "content": "Hi"}],
                                                    tools=tools, tool_choice=choice)

    def test_responses_function_call_and_output(self):
        first = self.client.responses.create(model="litertlm", input="Weather in Paris?", tools=[FLAT_WEATHER])
        self.assertEqual([item.type for item in first.output], ["message", "function_call"])
        self.assertEqual(first.output_text, "Checking.")
        call = first.output[1]
        self.assertEqual((call.call_id, call.name, call.arguments), ("call_1", "get_weather", '{"city": "Paris"}'))
        self.assertEqual(self.llm.requests[0]["tools"], [WEATHER])
        second = self.client.responses.create(
            model="litertlm", previous_response_id=first.id, tools=[FLAT_WEATHER],
            input=[{"type": "function_call_output", "call_id": call.call_id, "output": "cloudy"}])
        self.assertEqual(second.output_text, FakeLLM.REPLY)
        self.assertEqual(self.llm.requests[1]["messages"], [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": "Checking.", "tool_calls": [{"id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "cloudy"}])

    def test_responses_stream(self):
        with self.client.responses.stream(model="litertlm", input="Weather?", tools=[FLAT_WEATHER]) as stream:
            events = list(stream)
            final = stream.get_final_response()
        self.assertEqual([item.type for item in final.output], ["message", "function_call"])
        done = next(e for e in events if e.type == "response.function_call_arguments.done")
        self.assertEqual((done.output_index, done.arguments), (1, '{"city": "Paris"}'))

    def test_responses_call_without_text(self):
        def call_only(messages, tools=None, **options):
            yield ToolCall("get_weather", "{}", "call_9")
        self.llm.generate = call_only
        with self.client.responses.stream(model="litertlm", input="Weather?", tools=[FLAT_WEATHER]) as stream:
            final = stream.get_final_response()
        self.assertEqual([item.type for item in final.output], ["function_call"])

    def test_realtime_function_call_and_output(self):
        ws = TestRealtime.connect(self)
        until = lambda kind: TestRealtime.until(self, ws, kind)
        ws.send_json({"type": "session.update", "session": {"output_modalities": ["text"], "tools": [FLAT_WEATHER]}})
        until("session.updated")
        ws.send_json({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": "Weather?"}]}})
        ws.send_json({"type": "response.create"})
        events = until("response.done")
        done = next(e for e in events if e["type"] == "response.function_call_arguments.done")
        self.assertEqual((done["name"], done["call_id"], done["output_index"]), ("get_weather", "call_1", 1))
        self.assertEqual([item["type"] for item in events[-1]["response"]["output"]], ["message", "function_call"])
        ws.send_json({"type": "conversation.item.create", "item": {
            "type": "function_call_output", "call_id": "call_1", "output": "cloudy"}})
        ws.send_json({"type": "response.create"})
        self.assertEqual(until("response.done")[-1]["response"]["output"][0]["content"][0]["text"], FakeLLM.REPLY)
        messages = self.llm.requests[1]["messages"]
        self.assertEqual(messages[-2]["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertEqual(messages[-1], {"role": "tool", "tool_call_id": "call_1", "content": "cloudy"})

    def test_spoken_realtime_call(self):
        ws = TestRealtime.connect(self)
        ws.send_json({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": "Weather?"}]}})
        ws.send_json({"type": "response.create", "response": {"tools": [FLAT_WEATHER]}})
        output = TestRealtime.until(self, ws, "response.done")[-1]["response"]["output"]
        self.assertEqual([item["type"] for item in output], ["message", "function_call"])
        self.assertEqual(output[0]["content"][0]["transcript"], "Checking.")
        self.assertEqual(self.tts.spoken, [("Checking.", "amy")])


class KeywordEmbedder(FakeEmbedder):
    """ Embeds text as how often it says each of a few words, scaled to unit length. """
    WORDS = ["milk", "eggs", "dentist", "tuesday"]

    def embed(self, item, dimensions=None):
        self.items.append(item)
        words = re.findall(r"\w+", item.lower())
        vector = [words.count(w) for w in self.WORDS] + [0.1]
        size = sum(v * v for v in vector) ** 0.5
        return [v / size for v in vector]


class TestDepth(APITestCase):
    def setUp(self):
        super().setUp()
        with open(TEST_IMAGE, "rb") as f:
            self.image = f.read()

    def check(self, response, columns=8, rows=8):
        self.assertEqual(response.status_code, 200, response.text)
        found = response.json()
        self.assertEqual((found["object"], found["model"], found["units"]), ("depth", "fake-moge", "meters"))
        self.assertEqual((found["width"], found["height"], found["horizontal_fov"]), (60, 40, 60.0))
        self.assertEqual(found["center"], 4.0)
        self.assertEqual(found["nearest"], {"distance": 0.5, "x": 0.067, "y": 0.9})
        self.assertEqual(found["farthest"]["distance"], 4.0)
        self.assertEqual(found["coverage"], 0.75)
        self.assertEqual((len(found["grid"]), len(found["grid"][0])), (rows, columns))
        self.assertEqual(self.depth.photos[-1], self.image)
        return found

    def test_image_body(self):
        found = self.check(self.http.post("/v1/depth", content=self.image, headers={"Content-Type": "image/jpeg"}))
        # The sky has no estimate
        self.assertEqual(found["grid"][0], [None] * 8)
        # Cells 7.5 pixels wide: the third is mostly the near wall
        self.assertEqual(found["grid"][4], [1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0, 4.0])

    def test_multipart(self):
        found = self.check(self.http.post("/v1/depth", files={"file": ("photo.jpg", self.image, "image/jpeg")},
                                          data={"columns": "3", "rows": "1"}), 3, 1)
        self.assertEqual(found["grid"], [[1.0, 4.0, 4.0]])

    def test_json_data_url(self):
        url = "data:image/jpeg;base64," + base64.b64encode(self.image).decode("ascii")
        found = self.check(self.http.post("/v1/depth?columns=2", json={"image": url, "rows": 2}), 2, 2)
        self.assertEqual(found["grid"], [[1.0, 4.0], [1.0, 4.0]])
        self.check(self.http.post("/v1/depth", json={"image": {"url": url}}))

    def test_png(self):
        from PIL import Image
        response = self.http.post("/v1/depth?response_format=png", content=self.image,
                                  headers={"Content-Type": "image/jpeg"})
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual((response.headers["x-depth-near"], response.headers["x-depth-far"]), ("0.5", "4.0"))
        gray = np.asarray(Image.open(io.BytesIO(response.content)))
        self.assertEqual(gray.shape, (40, 60))
        # No estimate is black, the nearest white, the farthest the darkest gray
        self.assertEqual((gray[0, 0], gray[39, 0], gray[20, 59]), (0, 255, 1))
        self.assertTrue(1 < gray[20, 0] < 255)

    def test_bad_requests(self):
        jpeg = {"Content-Type": "image/jpeg"}
        for response in (self.http.post("/v1/depth?columns=0", content=self.image, headers=jpeg),
                         self.http.post("/v1/depth?rows=many", content=self.image, headers=jpeg),
                         self.http.post("/v1/depth?response_format=gif", content=self.image, headers=jpeg),
                         self.http.post("/v1/depth", content=b"", headers=jpeg),
                         self.http.post("/v1/depth", content=b"bad", headers=jpeg),
                         self.http.post("/v1/depth", content=b"a photo", headers={"Content-Type": "text/plain"}),
                         self.http.post("/v1/depth", json={"image": "ftp://example.com/x.jpg"}),
                         self.http.post("/v1/depth", data={"columns": "2"})):
            with self.subTest(body=response.request.content[:30]):
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()["error"]["type"], "invalid_request_error")

    def test_listed_as_a_model(self):
        model = self.client.models.retrieve("moge")
        self.assertEqual((model.kind, model.model), ("depth", "fake-moge"))


class TestDetect(APITestCase):
    def setUp(self):
        super().setUp()
        with open(TEST_IMAGE, "rb") as f:
            self.image = f.read()

    def detect(self, query="", **kwargs):
        response = self.http.post("/v1/detect" + query, content=self.image, headers={"Content-Type": "image/jpeg"},
                                  **kwargs)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_objects_with_distances(self):
        found = self.detect()
        self.assertEqual((found["object"], found["model"], found["depth_model"], found["units"]),
                         ("detection", "fake-yolox", "fake-moge", "meters"))
        self.assertEqual([(o["label"], o["place"], o["distance"]) for o in found["objects"]],
                         [("dining table", "bottom right", 4.0), ("chair", "left", 1.0),
                          ("sports ball", "bottom left", 0.5)])
        self.assertEqual(found["objects"][1]["box"], {"left": 0.05, "top": 0.3, "right": 0.3, "bottom": 0.7})
        self.assertEqual(found["objects"][1]["center"], {"x": 0.175, "y": 0.5})
        self.assertEqual((self.detector.photos, self.depth.photos), ([self.image], [self.image]))

    def test_options(self):
        found = self.detect("?threshold=0.5&distances=false")
        self.assertEqual([o["label"] for o in found["objects"]], ["dining table", "chair"])
        self.assertNotIn("distance", found["objects"][0])
        self.assertNotIn("depth_model", found)
        self.assertEqual(self.depth.photos, [])
        # Nothing found, so nothing to measure
        self.assertEqual(self.detect("?threshold=0.9")["objects"], [])
        self.assertEqual(self.depth.photos, [])

    def test_faces(self):
        found = self.detect("?model=yunet")
        self.assertEqual(found["model"], "fake-yunet")
        face = found["objects"][0]
        self.assertEqual((face["label"], face["place"], face["distance"]), ("face", "left", 1.0))
        self.assertEqual(face["points"]["right_eye"], {"x": 0.13, "y": 0.45})
        self.assertNotIn("points", self.detect()["objects"][0])

    def test_unknown_model(self):
        response = self.http.post("/v1/detect?model=nope", content=self.image, headers={"Content-Type": "image/jpeg"})
        self.assertEqual((response.status_code, response.json()["error"]["code"]), (404, "model_not_found"))

    def test_bad_requests(self):
        jpeg = {"Content-Type": "image/jpeg"}
        for response in (self.http.post("/v1/detect?threshold=2", content=self.image, headers=jpeg),
                         self.http.post("/v1/detect?threshold=some", content=self.image, headers=jpeg),
                         self.http.post("/v1/detect", content=b"bad", headers=jpeg),
                         self.http.post("/v1/detect", json={"threshold": 0.5})):
            with self.subTest(body=response.request.content[:30]):
                self.assertEqual(response.status_code, 400)

    def test_listed_as_a_model(self):
        model = self.client.models.retrieve("yolox")
        self.assertEqual((model.kind, model.model), ("detection", "fake-yolox"))


class TestSounds(APITestCase):
    SPEECH = [{"label": "Speech", "score": 0.8}, {"label": "Dog", "score": 0.3}]

    def test_multipart(self):
        response = self.http.post("/v1/audio/classifications", files={"file": ("clip.wav", b"RIFF", "audio/wav")})
        self.assertEqual(response.status_code, 200, response.text)
        found = response.json()
        self.assertEqual((found["object"], found["model"], found["duration"]),
                         ("audio.classification", "fake-panns", 14.5))
        # Each kind's best score in any window
        self.assertEqual(found["sounds"], [{"label": "Speech", "score": 0.8}, {"label": "Doorbell", "score": 0.6},
                                           {"label": "Dog", "score": 0.3}])
        self.assertEqual(found["segments"], [
            {"start": 0.0, "end": 10.0, "sounds": self.SPEECH},
            {"start": 4.5, "end": 14.5,
             "sounds": [{"label": "Doorbell", "score": 0.6}, {"label": "Speech", "score": 0.2}]}])
        self.assertEqual(self.sound.clips, [b"RIFF"])

    def test_options(self):
        response = self.http.post("/v1/audio/classifications?threshold=0.5", content=b"RIFF",
                                  headers={"Content-Type": "audio/wav"})
        self.assertEqual(response.json()["sounds"],
                         [{"label": "Speech", "score": 0.8}, {"label": "Doorbell", "score": 0.6}])
        response = self.http.post("/v1/audio/classifications", files={"file": ("clip.wav", b"RIFF", "audio/wav")},
                                  data={"top": "1"})
        self.assertEqual(response.json()["segments"][0]["sounds"], [{"label": "Speech", "score": 0.8}])

    def test_bad_requests(self):
        wav = {"Content-Type": "audio/wav"}
        for response in (self.http.post("/v1/audio/classifications?threshold=2", content=b"RIFF", headers=wav),
                         self.http.post("/v1/audio/classifications?top=0", content=b"RIFF", headers=wav),
                         self.http.post("/v1/audio/classifications?top=many", content=b"RIFF", headers=wav),
                         self.http.post("/v1/audio/classifications", content=b"", headers=wav),
                         self.http.post("/v1/audio/classifications", content=b"bad", headers=wav),
                         self.http.post("/v1/audio/classifications", json={"file": "clip.wav"}),
                         self.http.post("/v1/audio/classifications", data={"top": "2"})):
            with self.subTest(body=response.request.content[:30]):
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()["error"]["type"], "invalid_request_error")

    def test_listed_as_a_model(self):
        model = self.client.models.retrieve("panns")
        self.assertEqual((model.kind, model.model), ("sound", "fake-panns"))


class TestClassify(APITestCase):
    def classify(self, **body):
        response = self.http.post("/v1/classify", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_one_label_wins(self):
        found = self.classify(input="Turn on the lights", labels=["timer", "lights", "music"])
        self.assertEqual((found["object"], found["model"]), ("list", "fake-gliclass"))
        result = found["data"][0]
        # e^2 / (e^2 + 2): the scores add up to 1
        self.assertEqual((result["label"], result["score"]), ("lights", 0.787))
        self.assertEqual([r["label"] for r in result["labels"]], ["lights", "timer", "music"])
        self.assertAlmostEqual(sum(r["score"] for r in result["labels"]), 1, places=3)
        self.assertNotIn("chosen", result)

    def test_several_texts_and_a_prompt(self):
        found = self.classify(input=["Play music", "What time is it?"], labels=["music", "timer"], prompt="A timer? ")
        self.assertEqual([(r["index"], r["label"]) for r in found["data"]], [(0, "music"), (1, "timer")])
        self.assertEqual(self.classifier.requests[1], ("What time is it?", ["music", "timer"], "A timer? "))

    def test_multi_label(self):
        result = self.classify(input="Lights and music, please", labels=["timer", "lights", "music"],
                               multi_label=True, threshold=0.8)["data"][0]
        # Each label's sigmoid: 1 / (1 + e^-2) for a fit, 0.5 for none
        self.assertEqual(result["chosen"], ["lights", "music"])
        self.assertEqual(result["labels"][-1], {"label": "timer", "score": 0.5})
        self.assertNotIn("label", result)

    def test_bad_requests(self):
        for body in ({"input": "Hi", "labels": ["a", "b", "c", "d"]},
                     {"input": "", "labels": ["a"]},
                     {"input": [], "labels": ["a"]},
                     {"input": "Hi", "labels": ["a", " "]},
                     {"input": "Hi", "labels": ["a"], "threshold": 2},
                     {"input": "Hi"}):
            with self.subTest(body=body):
                response = self.http.post("/v1/classify", json=body)
                self.assertIn(response.status_code, (400, 422))

    def test_listed_as_a_model(self):
        model = self.client.models.retrieve("gliclass")
        self.assertEqual((model.kind, model.model), ("classification", "fake-gliclass"))


class TestSoundTool(unittest.TestCase):
    def clip(self, data):
        return {"type": "input_audio", "input_audio": {"data": base64.b64encode(data).decode("ascii"), "format": "wav"}}

    def test_identify_sounds(self):
        engine = FakeSound()
        found = SoundTagger(engine).identify_sounds(conversation=[
            {"role": "user", "content": [self.clip(b"old")]},
            {"role": "assistant", "content": "That's a dog."},
            {"role": "user", "content": [{"type": "text", "text": "And this?"}, self.clip(b"new")]}])
        self.assertEqual(engine.clips, [b"new"])
        self.assertEqual(found["best_guess"], "Speech")
        self.assertEqual(found["heard"], ["Speech (very likely)", "Doorbell (very likely)", "Dog (likely)"])
        self.assertEqual(found["over_time"], ["0 to 10 seconds: Speech, Dog", "4 to 14 seconds: Doorbell, Speech"])
        self.assertIn("error", SoundTagger(engine).identify_sounds(conversation=[{"role": "user", "content": "Hi"}]))

    def test_on_by_default_with_a_sound_engine(self):
        with mock.patch.dict(os.environ, clear=False) as environ:
            environ.pop("LITE_ASSISTANT_TOOLS", None)
            toolbox = Toolbox.create(Engines({}, {}, {}, sound={"panns": FakeSound()}))
            self.assertEqual(toolbox.default, ["get_time", "identify_sounds", "load_skill", "search_web", "read_web_page"])


class TestDistanceTool(APITestCase):
    BUILTIN_TOOLS = "estimate_distance"

    def test_chat_with_a_photo(self):
        with open(TEST_IMAGE, "rb") as f:
            image = f.read()
        url = "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")
        self.client.chat.completions.create(model="litertlm", messages=[{"role": "user", "content": [
            {"type": "text", "text": "How far away is that?"}, {"type": "image_url", "image_url": {"url": url}}]}])
        self.assertEqual(self.depth.photos, [image])
        result = json.loads(self.llm.requests[1]["messages"][-1]["content"])
        self.assertEqual((result["measured_at"], result["distance"]),
                         ("the center of the photo", "about 4.0 meters, or 13 feet"))


class TestDistances(unittest.TestCase):
    def setUp(self):
        self.map = FakeDepth().estimate(b"photo")

    def conversation(self, *contents):
        return [{"role": "user", "content": content} for content in contents]

    def test_parts_of_the_photo(self):
        self.assertEqual(depth.at(self.map, ""), ("center", 4.0))
        self.assertEqual(depth.at(self.map, "on the left"), ("left", 1.0))
        self.assertEqual(depth.at(self.map, "Bottom Left"), ("bottom left", 1.0))
        self.assertEqual(depth.at(self.map, "upper right")[0], "top right")
        self.assertEqual(depth.place(0.067, 0.9), "bottom left")
        self.assertEqual(depth.place(0.5, 0.5), "center")

    def test_estimate_distance(self):
        photo = "data:image/jpeg;base64," + base64.b64encode(b"photo").decode("ascii")
        engine = FakeDepth()
        found = Distances(engine).estimate_distance("left", conversation=self.conversation(
            [{"type": "image_url", "image_url": {"url": "data:image/png;base64,b2xk"}}],
            [{"type": "text", "text": "How far is the wall?"}, {"type": "input_image", "image_url": photo}],
            "And on the left?"))
        self.assertEqual(engine.photos, [b"photo"])
        self.assertEqual(found["measured_at"], "the left of the photo")
        self.assertEqual(found["distance"], "about 1.0 meters, or 3 feet")
        self.assertEqual(found["nearest"], "about 0.5 meters, or 2 feet, at the bottom left")
        self.assertEqual(found["farthest"], "about 4.0 meters, or 13 feet, at the right")
        missing = Distances(engine).estimate_distance("center", conversation=self.conversation("How far?"))
        self.assertIn("error", missing)

    def test_estimate_the_distance_to_a_thing(self):
        photo = [{"type": "input_image", "image_url": "data:image/jpeg;base64,cGhvdG8="}]
        tool = Distances(FakeDepth(), FakeDetector()).estimate_distance
        found = tool("center", "the chairs", conversation=self.conversation(photo))
        self.assertEqual(found["measured"], [{"object": "chair", "at": "the left of the photo",
                                              "distance": "about 1.0 meters, or 3 feet"}])
        self.assertNotIn("distance", found)
        self.assertEqual(found["other_objects"], ["dining table at the bottom right, about 4.0 meters, or 13 feet",
                                                  "sports ball at the bottom left, about 0.5 meters, or 2 feet"])
        table = tool("center", "table", conversation=self.conversation(photo))
        self.assertEqual(table["measured"][0]["object"], "dining table")
        missing = tool("left", "the dog", conversation=self.conversation(photo))
        self.assertEqual((missing["measured_at"], missing["distance"]), ("the left of the photo",
                                                                        "about 1.0 meters, or 3 feet"))
        self.assertIn('Nothing like "the dog" was recognized', missing["note"])

    def test_matching(self):
        found = FakeDetector.FOUND
        names = lambda thing: [d.label for d in detection.matching(found, thing)]
        self.assertEqual(names("the ball"), ["sports ball"])
        self.assertEqual(names("Sports Balls"), ["sports ball"])
        self.assertEqual(names("table"), ["dining table"])
        self.assertEqual(names("chairs"), ["chair"])
        self.assertEqual(names("hair"), [])
        self.assertEqual(names("the"), [])
        self.assertEqual(names("the bus"), [])
        # Where a thing is isn't what it is
        self.assertEqual(names("the grass behind the chair"), [])
        self.assertEqual(names("the ball next to the table"), ["sports ball"])
        self.assertEqual(detection.subject("tall grass behind the dog"), "tall grass")
        self.assertEqual([d.label for d in detection.matching([Detection("bus", 0.5, (0, 0, 1, 1))], "the buses")],
                         ["bus"])
        self.assertEqual([detection.singular(w) for w in ("boxes", "glasses", "puppies")], ["box", "glass", "puppy"])

    def test_on_by_default_with_a_depth_engine(self):
        engines = Engines({}, {}, {}, {}, {"moge": FakeDepth()})
        with mock.patch.dict(os.environ, clear=False) as environ:
            environ.pop("LITE_ASSISTANT_TOOLS", None)
            self.assertEqual(Toolbox.create(engines).default, ["get_time", "estimate_distance", "load_skill", "search_web", "read_web_page"])
            self.assertEqual(Toolbox.create(Engines({}, {}, {})).default, ["get_time", "load_skill", "search_web", "read_web_page"])


class TestBuiltinToolFunctions(unittest.TestCase):
    def test_definition(self):
        self.assertEqual(definition(Timers().set_timer), {"type": "function", "function": {
            "name": "set_timer", "description": "Sets a timer.",
            "parameters": {"type": "object", "properties": {
                "seconds": {"type": "integer", "description": "How long the timer runs, in seconds."},
                "label": {"type": "string", "description": "What the timer is for, if the user said."}},
                "required": ["seconds"]}}})

    def test_get_time(self):
        now = get_time()
        self.assertEqual(set(now), {"date", "time", "time_12h", "timezone", "utc_offset"})
        self.assertRegex(now["time"], r"^\d\d:\d\d$")

    def test_timers(self):
        timers = Timers()
        self.assertEqual(timers.list_timers()["timers"], [])
        self.assertEqual(timers.set_timer(90, "tea"), {"set": "tea", "duration": "1 minute 30 seconds"})
        timers.set_timer(0.01)
        time.sleep(0.02)
        listed = timers.list_timers()["timers"]
        self.assertEqual(listed[0], {"label": "tea", "left": "1 minute 30 seconds"})
        self.assertIn("finished", listed[1])
        # A finished timer is listed once
        self.assertEqual(len(timers.list_timers()["timers"]), 1)
        with self.assertRaises(ValueError):
            timers.set_timer(-5)

    def test_search_notes(self):
        with tempfile.TemporaryDirectory() as folder:
            notes = Notes(KeywordEmbedder(), folder)
            self.assertIn("no notes", notes.search_notes("milk")["note"])
            with open(os.path.join(folder, "shopping.md"), "w") as f:
                f.write("Buy milk and eggs.\n\nMore milk.")
            with open(os.path.join(folder, "health.txt"), "w") as f:
                f.write("The dentist is on Tuesday.")
            results = notes.search_notes("When is the dentist?")["results"]
            self.assertEqual((results[0]["note"], results[0]["text"]), ("health.txt", "The dentist is on Tuesday."))
            self.assertEqual(len(results), 3)
            # Paragraphs are embedded with the document prefix, once
            embedded = len(notes.embedder.items)
            notes.search_notes("milk")
            self.assertEqual(len(notes.embedder.items), embedded + 1)
            self.assertIn("title: health | text: The dentist is on Tuesday.", notes.embedder.items)

    def test_each_user_searches_their_own_notes(self):
        with tempfile.TemporaryDirectory() as home, mock.patch.dict(os.environ, {"LITE_ASSISTANT_HOME": home}):
            alice, bob = User("alice"), User("bob")
            os.makedirs(alice.folder("notes"))
            with open(alice.folder("notes", "health.txt"), "w") as f:
                f.write("The dentist is on Tuesday.")
            notes = Notes(KeywordEmbedder(), os.path.join(home, "notes"))
            self.assertEqual(notes.search_notes("dentist", alice)["results"][0]["note"], "health.txt")
            self.assertEqual(notes.search_notes("dentist", bob)["results"], [])
            self.assertIn(bob.folder("notes"), notes.search_notes("dentist", bob)["note"])
            # Bob's search leaves Alice's notes indexed
            self.assertIn(alice.folder("notes", "health.txt"), notes.cache)

    def test_select(self):
        toolbox = Toolbox({"get_time": get_time, "list_timers": Timers().list_timers}, "get_time")
        self.assertEqual(list(toolbox.select(None)), ["get_time"])
        self.assertEqual(list(toolbox.select(True)), ["get_time"])
        self.assertEqual(toolbox.select(False), {})
        self.assertEqual(toolbox.select("false"), {})
        self.assertEqual(list(toolbox.select(["list_timers"])), ["list_timers"])
        self.assertEqual(list(toolbox.select("get_time, list_timers")), ["get_time", "list_timers"])
        with self.assertRaises(ToolError):
            toolbox.select(["nope"])
        self.assertEqual(Toolbox({"get_time": get_time}, "all").default, ["get_time"])
        self.assertEqual(Toolbox({"get_time": get_time}, "nope,get_time").default, ["get_time"])

    def test_a_failing_tool_reports_its_error(self):
        llm = FakeLLM()
        replies = list(BuiltinLLM(llm, {"get_time": get_time}).generate([{"role": "user", "content": "Hi"}]))
        self.assertEqual("".join(replies), "Checking. One two three. Four five six.")

        def call_with_bad_arguments(messages, tools=None, **options):
            llm.requests.append(messages)
            if messages[-1]["role"] != "tool":
                yield ToolCall("get_time", "not JSON", "call_2")
        llm.generate = call_with_bad_arguments
        list(BuiltinLLM(llm, {"get_time": get_time}).generate([{"role": "user", "content": "Hi"}]))
        self.assertIn("JSONDecodeError", json.loads(llm.requests[-1][-1]["content"])["error"])

    def test_rounds_are_limited(self):
        offered = []

        def always_call(messages, tools=None, **options):
            offered.append(tools)
            if tools:
                yield ToolCall("get_time", "{}")
            else:
                yield "Done."
        llm = FakeLLM()
        llm.generate = always_call
        self.assertEqual(list(BuiltinLLM(llm, {"get_time": get_time}).generate([{"role": "user", "content": "Hi"}])),
                         ["Done."])
        self.assertEqual(len(offered), MAX_ROUNDS)
        self.assertIsNone(offered[-1])

    def test_a_call_to_a_tool_not_offered_gets_an_error(self):
        lamp = [{"type": "function", "function": {"name": "set_lamp", "parameters": {"type": "object"}}}]
        requests = []

        def call_a_fan(messages, tools=None, **options):
            requests.append(messages)
            if messages[-1]["role"] == "tool":
                yield "I can't run a fan."
            else:
                yield ToolCall("set_fan", '{"on": true}')
        llm = FakeLLM()
        llm.generate = call_a_fan
        # Even with no built-ins, the client never sees the call
        replies = list(Toolbox({}).llm(llm, {}).generate([{"role": "user", "content": "Fan on"}], tools=lamp))
        self.assertEqual(replies, ["I can't run a fan."])
        error = json.loads(requests[-1][-1]["content"])["error"]
        self.assertIn("'set_fan'", error)
        self.assertIn("set_lamp", error)



class TestBuiltinReplies(unittest.TestCase):
    """ A built-in can say what to reply, which ends the reply in place of the model's. """
    class CallingLLM:
        def __init__(self, calls):
            self.calls, self.rounds = calls, 0

        def generate(self, messages, max_tokens=None, usage=None, tools=None, **options):
            self.rounds += 1
            yield "Let me see."
            yield from (ToolCall(name, "{}") for name in self.calls)

    def reply(self, *calls):
        def refuse():
            """ Says no. """
            return {"reply_with": "Ask me nicely."}
        llm = self.CallingLLM(calls)
        return list(BuiltinLLM(llm, {"refuse": refuse, "get_time": get_time}).generate([])), llm.rounds

    def test_reply_with(self):
        self.assertEqual(self.reply("refuse", "refuse"), (["Let me see.", " Ask me nicely."], 1))

    def test_only_when_every_call_says_one(self):
        chunks, rounds = self.reply("refuse", "get_time")
        self.assertNotIn(" Ask me nicely.", chunks)
        self.assertEqual(rounds, MAX_ROUNDS)


class TestBuiltinTools(APITestCase):
    BUILTIN_TOOLS = "get_time"
    REPLY = "Checking. One two three. Four five six."

    def chat(self, **options):
        return self.client.chat.completions.create(model="litertlm", messages=[
            {"role": "user", "content": "What time is it?"}], **options)

    def test_chat_runs_the_tool(self):
        reply = self.chat()
        message = reply.choices[0].message
        self.assertEqual((message.content, message.tool_calls), (self.REPLY, None))
        self.assertEqual(reply.choices[0].finish_reason, "stop")
        self.assertEqual(len(self.llm.requests), 2)
        self.assertEqual(self.llm.requests[0]["tools"][0]["function"]["name"], "get_time")
        call, result = self.llm.requests[1]["messages"][-2:]
        self.assertEqual((call["content"], call["tool_calls"][0]["function"]["name"]), ("Checking.", "get_time"))
        self.assertEqual(result["tool_call_id"], call["tool_calls"][0]["id"])
        self.assertIn("time", json.loads(result["content"]))
        self.assertEqual(reply.usage.completion_tokens, 6)

    def test_chat_stream(self):
        chunks = list(self.chat(stream=True))
        self.assertEqual("".join(c.choices[0].delta.content or "" for c in chunks), self.REPLY)
        self.assertFalse(any(c.choices[0].delta.tool_calls for c in chunks))
        self.assertEqual(chunks[-1].choices[0].finish_reason, "stop")

    def test_turned_off_or_unknown(self):
        self.assertEqual(self.chat(extra_body={"builtin_tools": False}).choices[0].message.content,
                         FakeLLM.REPLY)
        self.assertIsNone(self.llm.requests[-1]["tools"])
        with self.assertRaises(openai.BadRequestError):
            self.chat(extra_body={"builtin_tools": ["nope"]})

    def test_client_tools_still_go_to_the_client(self):
        reply = self.chat(tools=[WEATHER])
        self.assertEqual(reply.choices[0].message.tool_calls[0].function.name, "get_weather")
        self.assertEqual([t["function"]["name"] for t in self.llm.requests[0]["tools"]], ["get_weather", "get_time"])

    def test_response_formats(self):
        schema = {"type": "object", "properties": {"on": {"type": "boolean"}}}
        # A reply held to a schema can't call tools, so the built-ins are left out
        self.chat(response_format={"type": "json_schema", "json_schema": {"name": "lamp", "schema": schema}})
        self.assertEqual((self.llm.requests[-1]["schema"], self.llm.requests[-1]["tools"]), (schema, None))
        self.chat(response_format={"type": "json_object"})
        self.assertEqual(self.llm.requests[-1]["schema"], {"type": "object"})
        self.chat(response_format={"type": "text"})
        self.assertIsNone(self.llm.requests[-1]["schema"])
        self.client.responses.create(model="litertlm", input="Hi", text={"format": {
            "type": "json_schema", "name": "lamp", "schema": schema}})
        self.assertEqual(self.llm.requests[-1]["schema"], schema)
        with self.assertRaisesRegex(openai.BadRequestError, "can't be used with tools"):
            self.chat(tools=[WEATHER], response_format={"type": "json_object"})
        with self.assertRaises(openai.BadRequestError):
            self.chat(response_format={"type": "xml"})

    def test_responses_report_mcp_calls(self):
        response = self.client.responses.create(model="litertlm", input="What time is it?")
        self.assertEqual([item.type for item in response.output], ["message", "mcp_call", "message"])
        self.assertEqual(response.output_text, self.REPLY)
        call = response.output[1]
        self.assertEqual((call.name, call.server_label), ("get_time", "lite-assistant"))
        self.assertIn("time", json.loads(call.output))
        # The call and its output are context for the next turn
        self.client.responses.create(model="litertlm", input="Thanks.", previous_response_id=response.id,
                                     extra_body={"builtin_tools": False})
        roles = [m["role"] for m in self.llm.requests[-1]["messages"]]
        self.assertEqual(roles, ["user", "assistant", "tool", "assistant", "user"])

    def test_responses_stream(self):
        with self.client.responses.stream(model="litertlm", input="What time is it?") as stream:
            events = list(stream)
            final = stream.get_final_response()
        self.assertEqual([item.type for item in final.output], ["message", "mcp_call", "message"])
        self.assertEqual(final.output_text, self.REPLY)
        added = [e.output_index for e in events if e.type == "response.output_item.added"]
        self.assertEqual(added, [0, 1, 2])
        self.assertIn("response.mcp_call.completed", [e.type for e in events])

    def test_realtime(self):
        ws = TestRealtime.connect(self)
        ws.send_json({"type": "session.update", "session": {"output_modalities": ["text"]}})
        TestRealtime.until(self, ws, "session.updated")
        ws.send_json({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": "Time?"}]}})
        ws.send_json({"type": "response.create"})
        output = TestRealtime.until(self, ws, "response.done")[-1]["response"]["output"]
        self.assertEqual([item["type"] for item in output], ["message"])
        self.assertEqual(output[0]["content"][0]["text"], self.REPLY)
        ws.send_json({"type": "session.update", "session": {"builtin_tools": ["nope"]}})
        self.assertEqual(TestRealtime.until(self, ws, "error")[-1]["error"]["code"], "invalid_value")

    def test_voice(self):
        with open(TEST_WAV, "rb") as f:
            self.http.post("/v1/voice", files={"file": ("turn.wav", f, "audio/wav")})
        self.assertEqual([text for text, voice in self.tts.spoken], ["Checking.", "One two three.", "Four five six."])
        with open(TEST_WAV, "rb") as f:
            response = self.http.post("/v1/voice", files={"file": ("turn.wav", f, "audio/wav")},
                                      data={"builtin_tools": "nope"})
        self.assertEqual(response.status_code, 400)


class TestRealtimeVAD(APITestCase):
    VAD = FakeVAD()
    connect, until = TestRealtime.connect, TestRealtime.until

    def append(self, ws, pcm):
        for i in range(0, len(pcm), 4800):
            ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[i:i + 4800]).decode()})

    def test_server_vad_is_the_default(self):
        ws = self.http.websocket_connect("/v1/realtime")
        with ws:
            turn_detection = ws.receive_json()["session"]["audio"]["input"]["turn_detection"]
        self.assertEqual((turn_detection["type"], turn_detection["silence_duration_ms"]), ("server_vad", 500))

    def test_turn_is_committed_and_answered(self):
        ws = self.connect()
        self.append(ws, silence(1000) + tone(1000) + silence(700))
        events = self.until(ws, "response.done")
        kinds = [e["type"] for e in events]
        self.assertEqual(kinds[:5], ["input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped",
                                     "input_audio_buffer.committed", "conversation.item.added",
                                     "conversation.item.done"])
        started, stopped, committed = events[:3]
        self.assertAlmostEqual(started["audio_start_ms"], 1000, delta=40)
        self.assertAlmostEqual(stopped["audio_end_ms"], 2000, delta=40)
        self.assertEqual({started["item_id"], stopped["item_id"], committed["item_id"]}, {committed["item_id"]})
        self.assertEqual(events[-1]["response"]["status"], "completed")
        # The turn's audio starts prefix_padding_ms before the speech and runs past it
        heard = base64.b64decode(self.llm.requests[0]["messages"][1]["content"][0]["input_audio"]["data"])
        self.assertAlmostEqual((len(heard) - 44) / 48, 300 + 1000 + 500, delta=150)

    def test_speech_interrupts_the_reply(self):
        def slow(messages, usage=None, **options):
            for i in range(1000):
                time.sleep(0.01)
                yield f" {i}."
        self.llm.generate = slow
        ws = self.connect()
        self.append(ws, tone(500) + silence(600))
        self.until(ws, "response.output_audio.delta")
        self.append(ws, tone(200))
        events = self.until(ws, "response.done")
        self.assertIn("input_audio_buffer.speech_started", [e["type"] for e in events])
        self.assertEqual(events[-1]["response"]["status_details"], {"type": "cancelled", "reason": "turn_detected"})

    def test_settings(self):
        ws = self.connect()
        ws.send_json({"type": "session.update", "session": {"audio": {"input": {"turn_detection": {
            "type": "server_vad", "create_response": False, "silence_duration_ms": 200}}}}})
        turn_detection = self.until(ws, "session.updated")[-1]["session"]["audio"]["input"]["turn_detection"]
        self.assertEqual((turn_detection["create_response"], turn_detection["threshold"]), (False, 0.5))
        self.append(ws, tone(300) + silence(300))
        self.until(ws, "conversation.item.done")
        ws.send_json({"type": "session.update", "session": {"audio": {"input": {"turn_detection": {
            "type": "semantic_vad", "eagerness": "high"}}}}})
        turn_detection = self.until(ws, "session.updated")[-1]["session"]["audio"]["input"]["turn_detection"]
        self.assertEqual(set(turn_detection), {"type", "eagerness", "create_response", "interrupt_response"})
        for bad in ({"type": "push_to_talk"}, {"type": "server_vad", "threshold": 2}):
            ws.send_json({"type": "session.update", "session": {"audio": {"input": {"turn_detection": bad}}}})
            self.assertEqual(ws.receive_json()["error"]["code"], "invalid_value")
        ws.send_json({"type": "session.update", "session": {"audio": {"input": {"turn_detection": None}}}})
        self.until(ws, "session.updated")
        self.append(ws, tone(300) + silence(600))
        ws.send_json({"type": "input_audio_buffer.commit"})
        self.assertEqual(ws.receive_json()["type"], "input_audio_buffer.committed")
        self.assertEqual(self.llm.requests, [])

class TestOpenAIAudio(APITestCase):
    def test_speech(self):
        with self.client.audio.speech.with_streaming_response.create(
                model="tts-1", voice="bob", input="Hello there.") as response:
            self.assertEqual(response.headers["content-type"], "audio/raw")
            self.assertEqual(response.read(), PCM_SECOND)
        self.assertEqual(self.tts.spoken, [("Hello there.", "bob")])

    def test_speech_rejects_empty_input(self):
        with self.assertRaises(openai.BadRequestError):
            self.client.audio.speech.create(model="tts-1", voice="bob", input="")

    def test_transcription(self):
        with open(TEST_WAV, "rb") as audio:
            transcript = self.client.audio.transcriptions.create(model="whisper-1", file=audio)
        self.assertEqual(transcript.text, "Hello, café.")

    def test_transcription_requires_a_file(self):
        response = self.http.post("/v1/audio/transcriptions", data={"model": "whisper-1"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["type"], "invalid_request_error")

    def test_translation(self):
        with open(TEST_WAV, "rb") as audio:
            translation = self.client.audio.translations.create(model="whisper-1", file=audio, prompt="A greeting.")
        self.assertEqual(translation.text, FakeLLM.REPLY)
        system, user = self.llm.requests[0]["messages"]
        self.assertEqual(system["role"], "system")
        self.assertIn("English", system["content"])
        self.assertEqual(user["content"][0]["type"], "input_audio")
        self.assertEqual(user["content"][1], {"type": "text", "text": "Context: A greeting."})

    def test_translation_as_text(self):
        with open(TEST_WAV, "rb") as audio:
            text = self.client.audio.translations.create(model="whisper-1", file=audio, response_format="text")
        self.assertEqual(text, FakeLLM.REPLY)

    def test_translation_rejects_other_formats(self):
        with open(TEST_WAV, "rb") as audio, self.assertRaises(openai.BadRequestError):
            self.client.audio.translations.create(model="whisper-1", file=audio, response_format="srt")

    def test_models(self):
        self.assertEqual({m.id for m in self.client.models.list()},
                         {"litertlm", "kokoro", "moonshine", "embeddinggemma", "moge", "yolox", "yunet", "panns", "gliclass"})

    def test_model(self):
        model = self.client.models.retrieve("kokoro")
        self.assertEqual((model.id, model.object, model.owned_by), ("kokoro", "model", "lite-assistant"))
        # Beyond OpenAI's fields: what kind of engine it is, and the model it loaded where it says
        self.assertEqual(model.kind, "tts")
        llm = self.client.models.retrieve("litertlm")
        self.assertEqual((llm.kind, llm.model), ("llm", "fake-gemma"))

    def test_unknown_model(self):
        with self.assertRaises(openai.NotFoundError) as error:
            self.client.models.retrieve("gpt-4o")
        self.assertEqual(error.exception.code, "model_not_found")

    def test_api_index(self):
        # OpenAI's /v1 is a 404; ours lists the endpoints and where the docs are
        for path in ("/v1", "/v1/"):
            index = self.http.get(path).json()
            self.assertEqual((index["object"], index["docs"], index["chat"]), ("api", "/llms.txt", True))
            endpoints = {e["path"]: e["methods"] for e in index["endpoints"]}
            self.assertEqual(endpoints["/v1/audio/speech"], ["POST"])
            self.assertEqual(endpoints["/v1/realtime"], ["WEBSOCKET"])
            # Routers' endpoints too, not only the app's own
            self.assertEqual(endpoints["/v1/detect"], ["POST"])
            self.assertEqual(endpoints["/v1/audio/speakers"], ["GET", "POST"])
            self.assertIn("GET", endpoints["/v1/models"])

    def test_llms_txt(self):
        reply = self.http.get("/llms.txt")
        self.assertTrue(reply.headers["content-type"].startswith("text/markdown"))
        self.assertTrue(reply.text.startswith("# lite-assistant\n"))
        self.assertIn("/v1/audio/speech", reply.text)


class TestModelChoice(APITestCase):
    """ A request's model picks one of the models the LLM engine can switch between. """
    LLM = SwitchingLLM

    def chat(self, model):
        return self.client.chat.completions.create(model=model, messages=[{"role": "user", "content": "Hi"}])

    def test_models_listed(self):
        models = {m.id: m for m in self.client.models.list()}
        self.assertEqual(models["litertlm"].model, "fake-small")
        self.assertEqual({(models[i].kind, models[i].engine, models[i].loaded) for i in SwitchingLLM.MODELS},
                         {("llm", "litertlm", True), ("llm", "litertlm", False)})
        self.assertTrue(models["fake-small"].loaded)
        self.assertEqual(self.client.models.retrieve("fake-big").engine, "litertlm")
        self.assertEqual(models["fake-text"].modalities, ["text"])
        self.assertEqual(models["litertlm"].modalities, ["text", "image", "audio"])

    def test_voice_clips_are_transcribed_for_a_model_that_cant_hear(self):
        with open(TEST_WAV, "rb") as f:
            clip = base64.b64encode(f.read()).decode()
        content = [{"type": "input_audio", "input_audio": {"data": clip, "format": "wav"}},
                   {"type": "text", "text": "Answer that."}]
        self.client.chat.completions.create(model="fake-text", messages=[{"role": "user", "content": content}])
        self.assertEqual(self.llm.requests[-1]["messages"][0]["content"],
                         [{"type": "text", "text": f"(Voice message) {FakeSTT.TRANSCRIPT}"},
                          {"type": "text", "text": "Answer that."}])
        # A model that can hear gets the clip itself
        self.client.chat.completions.create(model="fake-big", messages=[{"role": "user", "content": content}])
        self.assertEqual(self.llm.requests[-1]["messages"][0]["content"][0]["type"], "input_audio")

    def test_image_for_a_model_that_cant_see(self):
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]
        with self.assertRaises(openai.BadRequestError) as error:
            self.client.chat.completions.create(model="fake-text", messages=[{"role": "user", "content": content}])
        self.assertIn("can't take images", error.exception.message)

    def test_switch(self):
        self.assertEqual(self.chat("fake-big").model, "fake-big")
        self.assertEqual(self.llm.switches, ["fake-big"])
        # Any other name gets whatever is loaded now, without switching back
        self.assertEqual(self.chat("gpt-4o").model, "fake-big")
        self.assertEqual(self.chat("litertlm").model, "fake-big")
        self.assertEqual(self.llm.switches, ["fake-big"])
        models = {m.id: m for m in self.client.models.list()}
        self.assertEqual(models["litertlm"].model, "fake-big")
        self.assertTrue(models["fake-big"].loaded)
        self.assertFalse(models["fake-small"].loaded)

    def test_responses(self):
        response = self.client.responses.create(model="fake-big", input="Hi")
        self.assertEqual((response.model, self.llm.switches), ("fake-big", ["fake-big"]))
        with self.client.responses.stream(model="fake-small", input="Hi") as stream:
            final = stream.get_final_response()
        self.assertEqual((final.model, self.llm.switches), ("fake-small", ["fake-big", "fake-small"]))

    def test_not_downloaded(self):
        with self.assertRaises(openai.NotFoundError) as error:
            self.chat("fake-huge")
        self.assertEqual(error.exception.code, "model_not_found")
        self.assertEqual(self.llm.switches, [])

    def test_realtime(self):
        with self.http.websocket_connect("/v1/realtime?model=fake-big") as ws:
            self.assertEqual(ws.receive_json()["session"]["model"], "fake-big")
            ws.send_json({"type": "conversation.item.create",
                          "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi"}]}})
            ws.send_json({"type": "response.create", "response": {"output_modalities": ["text"]}})
            while ws.receive_json()["type"] != "response.done":
                pass
        self.assertEqual(self.llm.switches, ["fake-big"])

    def test_realtime_not_downloaded(self):
        with self.assertRaises(WebSocketDisconnect) as error:
            with self.http.websocket_connect("/v1/realtime?model=fake-huge") as ws:
                ws.receive_json()
        self.assertEqual(error.exception.code, 1008)


class TestModelHub(APITestCase):
    """ Finding, adding and deleting LiteRT-LM models, with Hugging Face faked. """
    LLM = SwitchingLLM
    FOUND = [{"repo": "litert-community/Qwen3-0.6B", "file": "Qwen3-0.6B.litertlm", "id": "Qwen3-0.6B",
              "size": 600, "fits": True, "downloaded": False, "downloads": 5, "likes": 1, "other_files": []}]

    def patch(self, name, replacement):
        patch = mock.patch.object(litert_models, name, replacement)
        patch.start()
        self.addCleanup(patch.stop)

    def events(self, response):
        return [json.loads(line[len("data: "):]) for line in response.text.splitlines() if line.startswith("data: ")]

    def test_search(self):
        searched = []
        self.patch("search", lambda query, limit, sort: searched.append((query, limit, sort)) or self.FOUND)
        response = self.http.get("/hub/models", params={"search": "qwen3", "limit": 5})
        self.assertEqual(response.json(), {"object": "list", "data": self.FOUND})
        self.http.get("/hub/models", params={"sort": "trending"})
        self.assertEqual(searched, [("qwen3", 5, "downloads"), (None, 30, "trending")])

    def test_search_for_a_repo_it_cant_add(self):
        def gated(query, limit, sort):
            raise ValueError("a/b is gated")
        self.patch("search", gated)
        response = self.http.get("/hub/models", params={"search": "a/b"})
        self.assertEqual((response.status_code, response.json()["error"]["message"]), (400, "a/b is gated"))

    def test_search_offline(self):
        def offline(query, limit, sort):
            raise OSError("no network")
        self.patch("search", offline)
        response = self.http.get("/hub/models")
        self.assertEqual(response.status_code, 502)
        self.assertIn("no network", response.json()["error"]["message"])

    def test_add_streams_progress(self):
        def add(repo, file, progress):
            for done in (0, 300, 600):
                progress(done, 600)
                time.sleep(0.15)
            return "Qwen3-0.6B"
        self.patch("add", add)
        events = self.events(self.http.post("/hub/models", json={"repo": "litert-community/Qwen3-0.6B"}))
        self.assertEqual(events[-1], {"type": "done", "id": "Qwen3-0.6B"})
        self.assertEqual([e["downloaded"] for e in events[:-1]], [0, 300, 600])
        self.assertEqual({e["total"] for e in events[:-1]}, {600})

    def test_add_says_how_fast_and_how_long(self):
        def add(repo, file, progress):
            for done in (0, 100, 200, 300):
                progress(done, 600)
            return "Qwen3-0.6B"
        self.patch("add", add)
        # Each look at the clock is a second later
        clock = itertools.count(1000)
        with mock.patch("lite_assistant.app.time.monotonic", lambda: next(clock)):
            events = self.events(self.http.post("/hub/models", json={"repo": "litert-community/Qwen3-0.6B"}))
        progress = [e for e in events if e["type"] == "progress"]
        self.assertNotIn("rate", progress[0])
        self.assertGreater(progress[-1]["rate"], 0)
        self.assertEqual(progress[-1]["left"], round(300 / progress[-1]["rate"]))

    def test_add_fails(self):
        def add(repo, file, progress):
            raise ValueError(f"{repo} has no LiteRT-LM file named {file}.")
        self.patch("add", add)
        events = self.events(self.http.post("/hub/models", json={"repo": "a/b", "file": "c.litertlm"}))
        self.assertEqual(events, [{"type": "error", "message": "a/b has no LiteRT-LM file named c.litertlm."}])

    def test_delete(self):
        removed = []
        self.patch("remove", removed.append)
        deleted = self.client.models.delete("fake-big")
        self.assertEqual((deleted.id, deleted.deleted), ("fake-big", True))
        self.assertEqual(removed, ["fake-big"])
        with self.assertRaises(openai.ConflictError):
            self.client.models.delete("fake-small")
        for model in ("litertlm", "kokoro", "nope"):
            with self.assertRaises(openai.NotFoundError):
                self.client.models.delete(model)
        self.assertEqual(removed, ["fake-big"])


class TestThinking(APITestCase):
    """ reasoning_effort and reasoning turn thinking on or off; the thoughts come back apart from the reply. """
    THOUGHT = "".join(FakeLLM.THOUGHTS)

    def chat(self, **options):
        return self.client.chat.completions.create(model="litertlm", messages=[{"role": "user", "content": "Hi"}],
                                                   **options)

    def test_efforts(self):
        for effort, thinking in (("none", False), ("minimal", False), ("low", 512), ("medium", 2048),
                                 ("high", True), ("xhigh", True)):
            self.chat(reasoning_effort=effort)
            self.assertEqual(self.llm.requests[-1]["thinking"], thinking, effort)
        # Unset, the model does as it does
        self.chat()
        self.assertIsNone(self.llm.requests[-1]["thinking"])
        with self.assertRaises(openai.BadRequestError):
            self.chat(reasoning_effort="extreme")

    def test_chat(self):
        reply = self.chat(reasoning_effort="high")
        message = reply.choices[0].message
        self.assertEqual((message.content, message.reasoning_content), (FakeLLM.REPLY, self.THOUGHT))
        self.assertEqual(reply.usage.completion_tokens_details.reasoning_tokens, 2)
        self.assertFalse(hasattr(self.chat().choices[0].message, "reasoning_content"))

    def test_chat_stream(self):
        chunks = list(self.chat(reasoning_effort="high", stream=True))
        deltas = [c.choices[0].delta for c in chunks if c.choices]
        self.assertEqual("".join(getattr(d, "reasoning_content", None) or "" for d in deltas), self.THOUGHT)
        self.assertEqual("".join(d.content or "" for d in deltas), FakeLLM.REPLY)

    def test_responses(self):
        response = self.client.responses.create(model="litertlm", input="Hi", reasoning={"effort": "medium"})
        self.assertEqual([item.type for item in response.output], ["reasoning", "message"])
        self.assertEqual(response.output[0].content[0].text, self.THOUGHT)
        self.assertEqual(response.output_text, FakeLLM.REPLY)
        self.assertEqual(response.usage.output_tokens_details.reasoning_tokens, 2)
        self.assertEqual(response.reasoning.effort, "medium")
        self.assertEqual(self.llm.requests[-1]["thinking"], 2048)

    def test_responses_stream(self):
        with self.client.responses.stream(model="litertlm", input="Hi", reasoning={"effort": "high"}) as stream:
            events = list(stream)
            final = stream.get_final_response()
        self.assertEqual("".join(e.delta for e in events if e.type == "response.reasoning_text.delta"), self.THOUGHT)
        done = [e.text for e in events if e.type == "response.reasoning_text.done"]
        self.assertEqual(done, [self.THOUGHT])
        self.assertEqual([item.type for item in final.output], ["reasoning", "message"])
        self.assertEqual(final.output_text, FakeLLM.REPLY)
        # The thoughts are done before the reply starts
        kinds = [(e.type, getattr(e, "item", None) and e.item.type) for e in events if e.type.startswith("response.output_item")]
        self.assertEqual(kinds, [("response.output_item.added", "reasoning"), ("response.output_item.done", "reasoning"),
                                 ("response.output_item.added", "message"), ("response.output_item.done", "message")])

    def test_thoughts_stay_out_of_the_next_turn(self):
        conversation = self.client.conversations.create()
        self.client.responses.create(model="litertlm", input="Hi", conversation=conversation.id,
                                     reasoning={"effort": "high"})
        items = self.client.conversations.items.list(conversation.id, order="asc").data
        self.assertEqual([item.type for item in items], ["message", "reasoning", "message"])
        self.client.responses.create(model="litertlm", input="Again", conversation=conversation.id)
        self.assertEqual([m["role"] for m in self.llm.requests[-1]["messages"]], ["user", "assistant", "user"])
        self.assertNotIn(self.THOUGHT, json.dumps(self.llm.requests[-1]["messages"]))

    def test_spoken_replies_dont_think(self):
        with open(TEST_WAV, "rb") as f:
            self.http.post("/v1/voice", files={"file": ("a.wav", f, "audio/wav")})
        self.assertIs(self.llm.requests[-1]["thinking"], False)
        with open(TEST_WAV, "rb") as f:
            self.client.audio.translations.create(model="litertlm", file=f)
        self.assertIs(self.llm.requests[-1]["thinking"], False)


class TestVoice(APITestCase):
    def voice(self, **fields):
        with open(TEST_WAV, "rb") as f:
            return self.http.post("/v1/voice", files={"file": ("turn.wav", f, "audio/wav")}, data=fields)

    def test_gemma_hears_the_audio_by_default(self):
        response = self.voice()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/raw")
        self.assertEqual(response.headers["x-transcript"], "")
        # Two sentences, each spoken as soon as it's written
        self.assertEqual(response.content, PCM_SECOND * 2)
        self.assertEqual([text for text, voice in self.tts.spoken], ["One two three.", "Four five six."])
        user = self.llm.requests[0]["messages"][-1]
        self.assertEqual(user["content"][0]["type"], "input_audio")
        self.assertEqual(self.llm.closed, 1)

    def test_stt_engine_transcribes_first(self):
        response = self.voice(stt_model="moonshine", voice="bob", max_tokens="3")
        self.assertEqual(urllib.parse.unquote(response.headers["x-transcript"]), "Hello, café.")
        self.assertEqual(self.llm.requests[0]["messages"][-1], {"role": "user", "content": "Hello, café."})
        self.assertEqual(self.llm.requests[0]["max_tokens"], 3)
        self.assertEqual(self.tts.spoken, [("One two three.", "bob")])

    def test_rejects_unknown_stt_model(self):
        self.assertEqual(self.voice(stt_model="nope").status_code, 400)

    def test_rejects_missing_audio(self):
        self.assertEqual(self.http.post("/v1/voice", data={"voice": "bob"}).status_code, 400)


class TestStreamClosing(unittest.TestCase):
    def test_abandoned_stream_closes_the_generator(self):
        # What a client disconnect does to the response's iterator
        closed = []
        def chunks():
            try:
                yield from (b"a", b"b", b"c")
            finally:
                closed.append(True)

        async def take_one():
            stream = in_threadpool(chunks())
            self.assertEqual(await stream.__anext__(), b"a")
            await stream.aclose()
        anyio.run(take_one)
        self.assertEqual(closed, [True])


class TestPage(APITestCase):
    def test_index(self):
        response = self.http.get("/")
        self.assertEqual(response.status_code, 200)
        # The chat page uses the server's own APIs
        for path in ("/v1/responses", "/v1/realtime", "/v1/audio/speech", "/v1/conversations", "/v1/embeddings",
                     "/hub/models", "/v1/models/"):
            self.assertIn(path, response.text)

    def test_voices_sorted_with_engine(self):
        self.assertEqual(self.http.get("/voices").json(), [
            {"id": "amy", "name": "Amy (en_US, Medium)", "engine": "kokoro"},
            {"id": "bob", "name": "Bob (en_US, Medium)", "engine": "kokoro"},
        ])

    def test_cors(self):
        response = self.http.get("/voices", headers={"Origin": "http://example.com"})
        self.assertEqual(response.headers["access-control-allow-origin"], "*")


if __name__ == "__main__":
    unittest.main()
