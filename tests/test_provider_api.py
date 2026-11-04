# SPDX-FileCopyrightText: 2026 Brad Barnett
#
# SPDX-License-Identifier: MIT

"""
Tests that a client can't tell this server from a provider: the same requests,
through the openai client and through examples/board_client.py, on fake engines.
"""
import base64
import importlib.util
import json
import os
import re
import socket
import threading
import time
import unittest

import uvicorn

from lite_assistant import vision_tasks
from lite_assistant.engines import OCREngine, TextLine
from test_app import APITestCase, FakeLLM, FakeSTT

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PHOTO = os.path.join(ROOT, "tests", "test_image.jpg")


def load(name, *path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, *path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bc = load("board_client", "examples", "board_client.py")


def schema_format(schema):
    return {"type": "json_schema", "json_schema": schema}


class FakeOCR(OCREngine):
    MODEL_ID = "fake-ocr"

    def read(self, image):
        return [TextLine("STOP", 0.9, (0.1, 0.2, 0.3, 0.4))]


class JsonLLM(FakeLLM):
    """ A FakeLLM that keeps to a schema as a model does: only the one check_provider asks for. """
    def generate(self, messages, response_schema=None, **options):
        if response_schema:
            options.get("usage", {}).update(prompt_tokens=1, completion_tokens=1)
            yield '{"action": "stop"}'
            return
        yield from super().generate(messages, **options)


class VisionCase(APITestCase):
    """ The fake engines with an OCR engine, and a real photo. """
    LLM = JsonLLM

    def setUp(self):
        super().setUp()
        self.engines = self.http.app.state.engines
        self.engines.ocr = {"ppocr": FakeOCR()}
        with open(PHOTO, "rb") as file:
            self.photo = file.read()
        self.data_url = "data:image/jpeg;base64," + base64.b64encode(self.photo).decode()

class TestVisionTasks(VisionCase):
    def ask(self, model, schema, **options):
        content = [{"type": "text", "text": "What is here?"},
                   {"type": "image_url", "image_url": {"url": self.data_url}}]
        return self.client.chat.completions.create(model=model, messages=[{"role": "user", "content": content}],
                                                   response_format=schema_format(schema), **options)

    def test_detector_answers_a_chat_completion(self):
        reply = self.ask("yolox", vision_tasks.OBJECT_DETECTION)
        self.assertEqual(reply.object, "chat.completion")
        self.assertEqual(reply.model, "fake-yolox")
        self.assertEqual(reply.choices[0].finish_reason, "stop")
        objects = json.loads(reply.choices[0].message.content)["objects"]
        self.assertEqual([o["label"] for o in objects], ["dining table", "chair", "sports ball"])
        self.assertEqual(objects[0]["box"], {"left": 0.5, "top": 0.5, "right": 1.0, "bottom": 1.0})
        self.assertEqual(set(objects[0]), {"label", "score", "box", "distance"})
        self.assertIsInstance(objects[0]["distance"], float)

    def test_no_depth_engine_gives_a_null_distance(self):
        self.engines.depth = {}
        objects = json.loads(self.ask("yolox", vision_tasks.OBJECT_DETECTION).choices[0].message.content)["objects"]
        self.assertIsNone(objects[0]["distance"])

    def test_reader_answers_a_chat_completion(self):
        reply = self.ask("ppocr", vision_tasks.OCR_EXTRACTION)
        self.assertEqual(json.loads(reply.choices[0].message.content), {
            "text": "STOP", "lines": [{"text": "STOP", "box": {"left": 0.1, "top": 0.2, "right": 0.3, "bottom": 0.4}}]})

    def test_streamed(self):
        chunks = list(self.ask("ppocr", vision_tasks.OCR_EXTRACTION, stream=True))
        text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
        self.assertEqual(json.loads(text)["text"], "STOP")

    def test_works_with_no_chat_model(self):
        self.engines.llm = {}
        self.assertTrue(json.loads(self.ask("yolox", vision_tasks.OBJECT_DETECTION).choices[0].message.content)["objects"])

    def test_a_chat_model_takes_the_same_schema(self):
        # The model chooses who answers, never the schema's name
        self.ask("gpt-4o", vision_tasks.OBJECT_DETECTION)
        self.assertEqual(self.detector.photos, [])

    def test_engine_refuses_another_schema(self):
        altered = {"name": "object_detection", "schema": {"type": "object"}}
        for model, schema in (("yolox", vision_tasks.OCR_EXTRACTION), ("ppocr", vision_tasks.OBJECT_DETECTION),
                              ("yolox", altered)):
            with self.subTest(model=model), self.assertRaises(Exception) as caught:
                self.ask(model, schema)
            self.assertEqual(caught.exception.status_code, 400)

    def test_engine_needs_a_photo(self):
        with self.assertRaises(Exception) as caught:
            self.client.chat.completions.create(model="yolox", messages=[{"role": "user", "content": "Hi"}],
                                                response_format=schema_format(vision_tasks.OBJECT_DETECTION))
        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("photo", caught.exception.message)

    def test_board_client_holds_the_same_schemas(self):
        self.assertEqual(bc.OBJECT_DETECTION, vision_tasks.OBJECT_DETECTION)
        self.assertEqual(bc.OCR_EXTRACTION, vision_tasks.OCR_EXTRACTION)

    def test_reference_page_holds_the_same_schemas(self):
        with open(os.path.join(ROOT, "docs", "reference", "provider-api.md"), encoding="utf-8") as file:
            blocks = re.findall(r"```json\n(.*?)```", file.read(), re.S)
        self.assertEqual([json.loads(b) for b in blocks], [vision_tasks.OBJECT_DETECTION, vision_tasks.OCR_EXTRACTION])


class TestReasoningForms(APITestCase):
    def chat(self, **options):
        return self.client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "Hi"}], extra_body=options)

    def test_reasoning_object_sets_thinking(self):
        for reasoning, thinking in (({"enabled": False}, False), ({"enabled": True}, True),
                                    ({"effort": "low"}, 512), ({"effort": "none"}, False)):
            with self.subTest(reasoning=reasoning):
                self.chat(reasoning=reasoning)
                self.assertEqual(self.llm.requests[-1]["thinking"], thinking)

    def test_reasoning_effort_wins(self):
        self.chat(reasoning_effort="none", reasoning={"effort": "high"})
        self.assertIs(self.llm.requests[-1]["thinking"], False)

    def test_bad_effort_is_a_400(self):
        with self.assertRaises(Exception) as caught:
            self.chat(reasoning={"effort": "lots"})
        self.assertEqual(caught.exception.status_code, 400)

    def test_thoughts_come_under_both_names(self):
        message = self.chat(reasoning={"enabled": True}).choices[0].message
        self.assertEqual(message.reasoning, "".join(FakeLLM.THOUGHTS))
        self.assertEqual(message.reasoning_content, message.reasoning)
        stream = self.client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}],
                                                     stream=True, extra_body={"reasoning": {"enabled": True}})
        deltas = [c.choices[0].delta for c in stream if c.choices]
        self.assertEqual("".join(getattr(d, "reasoning", None) or "" for d in deltas), "".join(FakeLLM.THOUGHTS))


class TestSpeechOptions(APITestCase):
    def speak(self, **options):
        return self.http.post("/v1/audio/speech", json={"input": "Hello.", **options})

    def test_pcm_unless_asked(self):
        response = self.speak()
        self.assertEqual(response.headers["content-type"], "audio/raw")
        self.assertNotEqual(response.content[:4], b"RIFF")
        self.assertEqual(self.speak(response_format="pcm").headers["content-type"], "audio/raw")

    def test_wav_holds_the_same_samples(self):
        pcm = self.speak().content
        response = self.speak(response_format="wav")
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content[:4], b"RIFF")
        self.assertEqual(int.from_bytes(response.content[24:28], "little"), 24000)
        self.assertEqual(response.content[44:], pcm)

    def test_other_formats_are_a_400(self):
        response = self.speak(response_format="mp3")
        self.assertEqual(response.status_code, 400)
        self.assertIn("pcm, wav", response.json()["error"]["message"])

    def test_speed(self):
        self.tts.SPEEDS = (0.5, 2.0)
        self.assertEqual(self.speak(speed=1.0).status_code, 200)
        self.assertEqual(self.speak(speed=3.0).status_code, 400)
        self.tts.SPEEDS = (1.0, 1.0)
        self.assertEqual(self.speak(speed=1.5).status_code, 400)


class WhisperSTT(FakeSTT):
    TRANSCRIPT = "From whisper."


class TestProviderModelIds(APITestCase):
    def setUp(self):
        super().setUp()
        self.http.app.state.engines.stt["whisper"] = WhisperSTT()

    def heard(self, model):
        response = self.http.post("/v1/audio/transcriptions", data={"model": model},
                                  files={"file": ("a.wav", b"RIFF", "audio/wav")})
        return response.json()["text"]

    def test_a_provider_id_names_the_engine(self):
        self.assertEqual(self.heard("openai/whisper-large-v3"), WhisperSTT.TRANSCRIPT)

    def test_openai_names_keep_the_default(self):
        self.assertEqual(self.heard("whisper-1"), FakeSTT.TRANSCRIPT)
        self.assertEqual(self.heard("someone/else"), FakeSTT.TRANSCRIPT)


class FakeSockets:
    """ Stands in for the socket module, keeping what each request sent. """
    AF_INET = socket.AF_INET

    def __init__(self):
        self.sent = b""
        self.addresses = []

    def getaddrinfo(self, host, port, family=0):
        self.addresses.append((host, port))
        return [(family, 1, 0, "", ("1.2.3.4", port))]

    def socket(self):
        return self

    def settimeout(self, seconds):
        pass

    def connect(self, address):
        pass

    def send(self, data):
        self.sent += bytes(data)
        return len(data)


class TestBaseUrl(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, bc, "socket", bc.socket)
        bc.socket = self.sockets = FakeSockets()

    def test_paths_go_under_the_base_url(self):
        client = bc.Client(base_url="http://api.example.com:8000/v1/openai/", key="k")
        client._open("GET", "/v1/models")
        self.assertEqual(self.sockets.addresses, [("api.example.com", 8000)])
        self.assertTrue(self.sockets.sent.startswith(b"GET /v1/openai/models HTTP/1.0\r\nHost: api.example.com\r\n"))
        self.assertIn(b"Authorization: Bearer k\r\n", self.sockets.sent)

    def test_a_provider_never_gets_the_servers_key_file(self):
        import sys
        key_file = type(sys)("lite_assistant_key")
        key_file.KEY = "the-servers-key"
        self.addCleanup(sys.modules.pop, "lite_assistant_key", None)
        sys.modules["lite_assistant_key"] = key_file
        bc.Client(base_url="http://api.example.com/v1")._open("GET", "/v1/models")
        self.assertNotIn(b"Authorization", self.sockets.sent)
        self.sockets.sent = b""
        bc.Client("192.168.1.10")._open("GET", "/v1/models")
        self.assertIn(b"the-servers-key", self.sockets.sent)

    def test_a_host_still_works(self):
        client = bc.Client("192.168.1.10")
        client._open("GET", "/v1/models")
        self.assertEqual(self.sockets.addresses, [("192.168.1.10", 5000)])
        self.assertTrue(self.sockets.sent.startswith(b"GET /v1/models "))

    def test_bad_urls(self):
        for url in ("api.example.com/v1", "ftp://example.com"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                bc.Client(base_url=url)


class TestCheckProvider(VisionCase):
    """ scripts/check_provider.py's whole run, over a real socket. """

    def setUp(self):
        super().setUp()
        import sys
        sys.path.insert(0, os.path.join(ROOT, "examples"))
        self.addCleanup(sys.path.remove, os.path.join(ROOT, "examples"))
        self.check = load("check_provider", "scripts", "check_provider.py")
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        self.port = probe.getsockname()[1]
        probe.close()
        self.server = uvicorn.Server(uvicorn.Config(self.http.app, host="127.0.0.1", port=self.port, lifespan="off",
                                                    log_level="warning"))
        thread = threading.Thread(target=self.server.run, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(setattr, self.server, "should_exit", True)
        for _ in range(100):
            if self.server.started:
                break
            time.sleep(0.05)

    def board(self):
        return self.check.board_client.Client(base_url=f"http://127.0.0.1:{self.port}/v1", models={
            "chat": "gpt-4o", "speech": "kokoro", "transcribe": "moonshine", "detect": "yolox", "read": "ppocr"})

    def test_every_check_passes(self):
        lines = []
        failed = self.check.run(self.board(), self.photo, out=lines.append)
        self.assertEqual(failed, [], lines)
        self.assertEqual(len(lines), len(self.check.CHECKS))

    def test_a_check_catches_a_wrong_shape(self):
        self.assertFalse(self.check.matches({"objects": [{"label": "x"}]}, vision_tasks.OBJECT_DETECTION["schema"]))
        self.assertFalse(self.check.matches({"action": "fly"}, self.check.COMMAND["schema"]))
        lines = []
        self.engines.detect = {}
        self.assertEqual(self.check.run(self.board(), self.photo, only=["objects"], out=lines.append), ["objects"])
