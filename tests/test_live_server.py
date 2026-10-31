"""
Integration tests against a running server with real models
(LITE_ASSISTANT_URL, default http://localhost:5000), through the official
openai client where OpenAI has the endpoint. Skips if no server is reachable.
"""
import base64
import io
import json
import os
import re
import threading
import time
import unittest
import urllib.parse
import wave
from datetime import datetime

import httpx
import openai

BASE_URL = os.environ.get("LITE_ASSISTANT_URL", "http://localhost:5000")
TEST_WAV = os.path.join(os.path.dirname(__file__), "test_audio.wav")
# "Domestic shorthair cat portrait in grass", Wikimedia Commons, CC0
TEST_IMAGE = os.path.join(os.path.dirname(__file__), "test_image.jpg")


def setUpModule():
    try:
        httpx.get(f"{BASE_URL}/v1/models", timeout=5).raise_for_status()
    except httpx.HTTPError as e:
        raise unittest.SkipTest(f"No lite-assistant server at {BASE_URL}: {e}")


def client():
    return openai.OpenAI(base_url=f"{BASE_URL}/v1", api_key="unused", timeout=300, max_retries=0)


class TestChat(unittest.TestCase):
    def chat(self, content, **options):
        return client().chat.completions.create(
            model="litertlm", messages=[{"role": "user", "content": content}], **options)

    def test_completion(self):
        reply = self.chat("Say hello in three words.", max_tokens=16)
        self.assertTrue(reply.choices[0].message.content.strip())
        self.assertIn(reply.choices[0].finish_reason, ("stop", "length"))
        self.assertGreater(reply.usage.prompt_tokens, 0)
        self.assertGreater(reply.usage.completion_tokens, 0)

    def test_truncated_reply_finishes_with_length(self):
        reply = self.chat("Write a long story about a dragon.", max_tokens=5)
        self.assertEqual(reply.choices[0].finish_reason, "length")
        self.assertEqual(reply.usage.completion_tokens, 5)

    def test_stream(self):
        chunks = list(self.chat("Count from one to five in words.", max_tokens=32, stream=True,
                                stream_options={"include_usage": True}))
        contents = [c.choices[0].delta.content for c in chunks if c.choices and c.choices[0].delta.content]
        self.assertGreater(len(contents), 1, "stream carried the reply in one piece")
        self.assertIn(chunks[-2].choices[0].finish_reason, ("stop", "length"))
        self.assertGreater(chunks[-1].usage.completion_tokens, 0)

    def test_rejects_empty_messages(self):
        with self.assertRaises(openai.BadRequestError):
            client().chat.completions.create(model="litertlm", messages=[])

    def test_sees_an_image(self):
        with open(TEST_IMAGE, "rb") as f:
            url = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("ascii")
        reply = self.chat([{"type": "text", "text": "What animal is this? Answer in one word."},
                           {"type": "image_url", "image_url": {"url": url}}], max_tokens=8)
        self.assertIn("cat", reply.choices[0].message.content.lower())

    def test_disconnect_frees_the_engine(self):
        # A client that hangs up mid-reply must not leave Gemma generating, or
        # holding the lock the next request waits on
        stream = self.chat("Write a long story about a dragon.", max_tokens=2000, stream=True)
        next(iter(stream))
        stream.close()
        started = time.time()
        result = []
        worker = threading.Thread(target=lambda: result.append(self.chat("Say hi.", max_tokens=4)), daemon=True)
        worker.start()
        worker.join(timeout=60)
        self.assertFalse(worker.is_alive(), "the next request is stuck behind the abandoned one")
        self.assertLess(time.time() - started, 30)


class TestModels(unittest.TestCase):
    def test_every_listed_model_can_be_retrieved(self):
        for model in client().models.list():
            self.assertEqual(client().models.retrieve(model.id).id, model.id)

    def test_unknown_model(self):
        with self.assertRaises(openai.NotFoundError):
            client().models.retrieve("gpt-4o")


class TestModelSwitch(unittest.TestCase):
    def test_switch_and_back(self):
        """ Asking for the other Gemma loads it for every client, until something asks for the first. """
        choices = {m.id: m for m in client().models.list() if getattr(m, "engine", None)}
        if len(choices) < 2:
            self.skipTest("Only one Gemma model is downloaded.")
        loaded = next(i for i, m in choices.items() if m.loaded)
        other = next(i for i in choices if i != loaded)
        chat = lambda model: client().chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Say hello in three words."}], max_tokens=16)
        try:
            reply = chat(other)
            self.assertEqual(reply.model, other)
            self.assertTrue(reply.choices[0].message.content.strip())
            self.assertEqual(client().models.retrieve("litertlm").model, other)
            self.assertTrue(client().models.retrieve(other).loaded)
            # Any other name gets whatever is loaded
            self.assertEqual(chat("gpt-4o").model, other)
        finally:
            self.assertEqual(chat(loaded).model, loaded)


class TestThinking(unittest.TestCase):
    QUESTION = [{"role": "user", "content": "What is 17 * 23?"}]

    def ask(self, model="litertlm", **options):
        reply = client().chat.completions.create(model=model, messages=self.QUESTION, max_tokens=1500, **options)
        message = reply.choices[0].message
        self.assertIn("391", message.content)
        return getattr(message, "reasoning_content", None), reply.usage.completion_tokens_details.reasoning_tokens

    def test_gemma_thinks_when_asked(self):
        thoughts, tokens = self.ask(reasoning_effort="high")
        self.assertTrue(thoughts.strip())
        self.assertGreater(tokens, 0)
        self.assertEqual(self.ask(reasoning_effort="none"), (None, 0))

    def test_responses_reasoning_item(self):
        response = client().responses.create(model="litertlm", input="What is 17 * 23?",
                                             reasoning={"effort": "high"}, max_output_tokens=1500)
        reasoning = [item for item in response.output if item.type == "reasoning"]
        # One for each stretch of thought: a model that calls a tool, such as calculate, thinks again after
        self.assertGreaterEqual(len(reasoning), 1)
        self.assertTrue(all(item.content[0].text.strip() for item in reasoning))
        self.assertIn("391", response.output_text)

    def test_qwen3_thinks_by_default(self):
        """ Qwen3 thinks unless told not to; asking for it loads it, so the loaded model comes back after. """
        models = {m.id: m for m in client().models.list() if getattr(m, "engine", None)}
        qwen = next((i for i in models if i.lower().startswith("qwen3")), None)
        if not qwen:
            self.skipTest("No Qwen3 model is downloaded.")
        loaded = next(i for i, m in models.items() if m.loaded)
        try:
            thoughts, tokens = self.ask(qwen)
            self.assertTrue(thoughts.strip())
            self.assertGreater(tokens, 0)
            self.assertEqual(self.ask(qwen, reasoning_effort="none"), (None, 0))
        finally:
            self.ask(loaded, reasoning_effort="none")


class TestResponses(unittest.TestCase):
    def test_create(self):
        response = client().responses.create(model="litertlm", input="Say hello in three words.",
                                             instructions="Be brief.", max_output_tokens=16)
        self.assertIn(response.status, ("completed", "incomplete"))
        self.assertTrue(response.output_text.strip())
        self.assertGreater(response.usage.output_tokens, 0)

    def test_stream(self):
        with client().responses.stream(model="litertlm", input="Count from one to five in words.",
                                       max_output_tokens=32) as stream:
            deltas = [e.delta for e in stream if e.type == "response.output_text.delta"]
            final = stream.get_final_response()
        self.assertGreater(len(deltas), 1, "stream carried the reply in one piece")
        self.assertEqual("".join(deltas), final.output_text)

    def test_sees_an_image(self):
        with open(TEST_IMAGE, "rb") as f:
            url = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("ascii")
        response = client().responses.create(model="litertlm", max_output_tokens=8, input=[{"role": "user", "content": [
            {"type": "input_text", "text": "What animal is this? Answer in one word."},
            {"type": "input_image", "image_url": url, "detail": "auto"}]}])
        self.assertIn("cat", response.output_text.lower())


    def test_chained_response_remembers(self):
        first = client().responses.create(model="litertlm", input="My name is Zorblat. Just say OK.",
                                          max_output_tokens=8)
        second = client().responses.create(model="litertlm", input="What is my name? Answer in one word.",
                                           previous_response_id=first.id, max_output_tokens=8)
        self.assertIn("zorblat", second.output_text.lower())
        client().responses.delete(first.id)
        with self.assertRaises(openai.NotFoundError):
            client().responses.retrieve(first.id)

    def test_background(self):
        created = client().responses.create(model="litertlm", input="Say hi.", max_output_tokens=8, background=True)
        for _ in range(300):
            response = client().responses.retrieve(created.id)
            if response.status not in ("queued", "in_progress"):
                break
            time.sleep(0.2)
        self.assertIn(response.status, ("completed", "incomplete"))
        self.assertTrue(response.output_text.strip())


    def test_conversation_remembers(self):
        conversation = client().conversations.create()
        client().responses.create(model="litertlm", input="My name is Zorblat. Just say OK.",
                                  conversation=conversation.id, max_output_tokens=8)
        second = client().responses.create(model="litertlm", input="What is my name? Answer in one word.",
                                           conversation=conversation.id, max_output_tokens=8)
        self.assertIn("zorblat", second.output_text.lower())
        self.assertEqual(len(client().conversations.items.list(conversation.id).data), 4)
        client().conversations.delete(conversation.id)


def dot(a, b):
    # The vectors are unit length, so this is their cosine similarity
    return sum(x * y for x, y in zip(a, b))


class TestEmbeddings(unittest.TestCase):
    def embed(self, texts, **options):
        return [d.embedding for d in client().embeddings.create(model="embeddinggemma", input=texts, **options).data]

    def test_related_texts_are_closer(self):
        cat, kitten, tax = self.embed(["The cat sleeps on the sofa.", "A kitten naps on the couch.",
                                       "Quarterly tax filings are due in April."])
        self.assertGreater(dot(cat, kitten), dot(cat, tax))

    def test_dimensions(self):
        self.assertEqual(len(self.embed("Hello.", dimensions=256)[0]), 256)

    def test_image_sits_near_its_description(self):
        with open(TEST_IMAGE, "rb") as f:
            url = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("ascii")
        response = httpx.post(f"{BASE_URL}/v1/embeddings", timeout=300, json={
            "model": "embeddinggemma", "encoding_format": "float",
            "input": [{"type": "image_url", "image_url": {"url": url}}, "a photo of a cat", "a photo of a car"]})
        response.raise_for_status()
        image, cat, car = [d["embedding"] for d in response.json()["data"]]
        self.assertGreater(dot(image, cat), dot(image, car))


class TestRerank(unittest.TestCase):
    def test_the_answer_comes_first(self):
        response = httpx.post(f"{BASE_URL}/v1/rerank", timeout=300, json={
            "query": "How long is the warranty?", "top_n": 2,
            "documents": ["Clean the filter every month.", "The warranty covers parts for two years.",
                          "Basil wilts quickly."]})
        response.raise_for_status()
        results = response.json()["results"]
        self.assertEqual([r["index"] for r in results][:1], [1])
        self.assertGreater(results[0]["relevance_score"], 0.5)
        self.assertLess(results[1]["relevance_score"], 0.5)


class TestAudio(unittest.TestCase):
    def test_speech(self):
        with client().audio.speech.with_streaming_response.create(
                model="tts-1", voice="af_heart", input="Hello there.") as response:
            pcm = response.read()
        # Half a second of 16-bit 24kHz mono is 24000 bytes; any real phrase is longer
        self.assertGreater(len(pcm), 24000)
        self.assertEqual(len(pcm) % 2, 0, "PCM isn't whole 16-bit samples")

    def test_transcription(self):
        with open(TEST_WAV, "rb") as audio:
            self.assertTrue(client().audio.transcriptions.create(model="whisper-1", file=audio).text.strip())

    def test_translation(self):
        with open(TEST_WAV, "rb") as audio:
            self.assertTrue(client().audio.translations.create(model="whisper-1", file=audio).text.strip())


class TestRealtime(unittest.TestCase):
    def turn(self, connection):
        """ The events of one response, up to response.done. """
        connection.response.create()
        events = []
        for event in connection:
            self.assertNotEqual(event.type, "error", event)
            events.append(event)
            if event.type == "response.done":
                return events

    def test_spoken_turn(self):
        with open(TEST_WAV, "rb") as f:
            pcm = f.read()[44:]  # 24 kHz 16-bit mono, past the WAV header
        with client().realtime.connect(model="gpt-realtime") as connection:
            connection.session.update(session={"type": "realtime", "max_output_tokens": 48, "audio": {"input": {
                "transcription": {"model": "whisper-1"}, "turn_detection": None}}})
            for i in range(0, len(pcm), 4800):
                connection.input_audio_buffer.append(audio=base64.b64encode(pcm[i:i + 4800]).decode())
            connection.input_audio_buffer.commit()
            events = self.turn(connection)
        audio = b"".join(base64.b64decode(e.delta) for e in events if e.type == "response.output_audio.delta")
        # A second of 24 kHz 16-bit mono is 48000 bytes; any real reply is longer
        self.assertGreater(len(audio), 48000)
        transcript = "".join(e.delta for e in events if e.type == "response.output_audio_transcript.delta")
        self.assertTrue(transcript.strip())
        self.assertIn(events[-1].response.status, ("completed", "incomplete"))
        self.assertTrue(any(e.type == "conversation.item.input_audio_transcription.completed" for e in events))

    def test_server_vad_turn(self):
        """ Speech then silence, streamed in real time with no commit: the server finds the turn and answers. """
        with open(TEST_WAV, "rb") as f:
            pcm = f.read()[44:] + b"\0\0" * 24000
        with client().realtime.connect(model="gpt-realtime") as connection:
            connection.session.update(session={"type": "realtime", "max_output_tokens": 32})
            for i in range(0, len(pcm), 4800):
                connection.input_audio_buffer.append(audio=base64.b64encode(pcm[i:i + 4800]).decode())
                time.sleep(0.1)
            events = []
            for event in connection:
                self.assertNotEqual(event.type, "error", event)
                events.append(event)
                if event.type == "response.done":
                    break
        kinds = [e.type for e in events]
        for kind in ("input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped",
                     "input_audio_buffer.committed", "response.output_audio.delta"):
            self.assertIn(kind, kinds)
        started = next(e for e in events if e.type == "input_audio_buffer.speech_started")
        self.assertLess(started.audio_start_ms, 1000)
        self.assertIn(events[-1].response.status, ("completed", "incomplete"))

    def test_text_conversation(self):
        with client().realtime.connect(model="gpt-realtime") as connection:
            connection.session.update(session={"type": "realtime", "output_modalities": ["text"],
                                               "max_output_tokens": 16})
            for text in ("My name is Zorblat. Just say OK.", "What is my name? Answer in one word."):
                connection.conversation.item.create(item={
                    "type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]})
                events = self.turn(connection)
        reply = "".join(e.delta for e in events if e.type == "response.output_text.delta")
        self.assertIn("zorblat", reply.lower())


WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather in a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city"}},
                   "required": ["city"]}}}
FLAT_WEATHER = {"type": "function", **WEATHER["function"]}
QUESTION = "What's the weather in Paris right now?"


class TestFunctionTools(unittest.TestCase):
    def test_chat(self):
        messages = [{"role": "user", "content": QUESTION}]
        reply = client().chat.completions.create(model="litertlm", messages=messages, tools=[WEATHER])
        self.assertEqual(reply.choices[0].finish_reason, "tool_calls")
        call = reply.choices[0].message.tool_calls[0]
        self.assertEqual(call.function.name, "get_weather")
        self.assertIn("paris", json.loads(call.function.arguments)["city"].lower())
        messages += [reply.choices[0].message.model_dump(exclude_none=True),
                     {"role": "tool", "tool_call_id": call.id, "content": '{"sky": "cloudy", "celsius": 18}'}]
        answer = client().chat.completions.create(model="litertlm", messages=messages, tools=[WEATHER],
                                                  max_tokens=48)
        self.assertRegex(answer.choices[0].message.content.lower(), "cloud|18")

    def test_responses(self):
        first = client().responses.create(model="litertlm", input=QUESTION, tools=[FLAT_WEATHER])
        call = next(item for item in first.output if item.type == "function_call")
        self.assertEqual(call.name, "get_weather")
        answer = client().responses.create(
            model="litertlm", previous_response_id=first.id, tools=[FLAT_WEATHER], max_output_tokens=48,
            input=[{"type": "function_call_output", "call_id": call.call_id, "output": "cloudy, 18 C"}])
        self.assertRegex(answer.output_text.lower(), "cloud|18")

    def test_realtime(self):
        with client().realtime.connect(model="gpt-realtime") as connection:
            connection.session.update(session={"type": "realtime", "output_modalities": ["text"],
                                               "tools": [FLAT_WEATHER]})
            connection.conversation.item.create(item={
                "type": "message", "role": "user", "content": [{"type": "input_text", "text": QUESTION}]})
            connection.response.create()
            for event in connection:
                self.assertNotEqual(event.type, "error", event)
                if event.type == "response.done":
                    break
        calls = [item for item in event.response.output if item.type == "function_call"]
        self.assertEqual([call.name for call in calls], ["get_weather"])


class TestVoice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(TEST_WAV, "rb") as f:
            cls.audio = f.read()

    def voice(self, **fields):
        return httpx.post(f"{BASE_URL}/v1/voice", files={"file": ("turn.wav", self.audio, "audio/wav")},
                          data=fields, timeout=300)

    def assert_pcm_reply(self, response):
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/raw")
        # A second of 16-bit 24kHz mono is 48000 bytes; any real reply is longer
        self.assertGreater(len(response.content), 48000)
        self.assertEqual(len(response.content) % 2, 0, "PCM isn't whole 16-bit samples")

    def test_gemma_hears_the_audio_directly(self):
        response = self.voice(max_tokens="48")
        self.assertEqual(response.headers["x-transcript"], "")
        self.assert_pcm_reply(response)

    def test_moonshine_transcribes_then_replies_in_audio(self):
        response = self.voice(stt_model="moonshine", max_tokens="48")
        self.assertTrue(urllib.parse.unquote(response.headers["x-transcript"]).strip())
        self.assert_pcm_reply(response)

    def test_rejects_unknown_stt_model(self):
        self.assertEqual(self.voice(stt_model="nope").status_code, 400)


# Each hour in words, with what Moonshine may hear in its place
HOUR_WORDS = ["twelve", "one|won", "two|to|too", "three", "four|for", "five", "six", "seven", "eight|ate", "nine",
              "ten", "eleven"]


def says_the_hour(text: str, hours) -> bool:
    """ Whether text gives one of these hours (0-23), as a 24-hour or 12-hour number, or a word. """
    text = text.lower()
    for hour in hours:
        if re.search(rf"\b0?({hour}|{hour % 12 or 12})(:\d\d|\b)", text) \
                or re.search(rf"\b({HOUR_WORDS[hour % 12]})\b", text):
            return True
    return False


def wav(pcm: bytes) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm)
    return out.getvalue()


class TestBuiltinTools(unittest.TestCase):
    """ The server's default built-in tools include get_time. """
    QUESTION = "What time is it right now?"

    def answer_hours(self, ask):
        # The hour may tick over while Gemma answers
        before = datetime.now().hour
        text = ask()
        return text, {before, datetime.now().hour}

    def test_chat(self):
        text, hours = self.answer_hours(lambda: client().chat.completions.create(
            model="litertlm", messages=[{"role": "user", "content": self.QUESTION}],
            max_tokens=64).choices[0].message.content)
        self.assertTrue(says_the_hour(text, hours), f"{text!r} doesn't give the hour, {hours}")

    def test_responses_report_the_call(self):
        response = client().responses.create(model="litertlm", input=self.QUESTION, max_output_tokens=64)
        self.assertIn("get_time", [item.name for item in response.output if item.type == "mcp_call"])

    def test_turned_off(self):
        reply = client().chat.completions.create(model="litertlm", messages=[
            {"role": "user", "content": self.QUESTION}], max_tokens=64, extra_body={"builtin_tools": False})
        self.assertTrue(reply.choices[0].message.content)

    def test_voice(self):
        speech = httpx.post(f"{BASE_URL}/v1/audio/speech", json={"input": self.QUESTION}, timeout=120)

        def ask():
            reply = httpx.post(f"{BASE_URL}/v1/voice", files={"file": ("turn.wav", wav(speech.content), "audio/wav")},
                               data={"stt_model": "moonshine", "max_tokens": "64"}, timeout=300)
            self.assertIn("time", urllib.parse.unquote(reply.headers["x-transcript"]).lower())
            # What the reply said, heard back through Moonshine
            heard = httpx.post(f"{BASE_URL}/v1/audio/transcriptions",
                               files={"file": ("reply.wav", wav(reply.content), "audio/wav")}, timeout=120)
            return heard.json()["text"]
        text, hours = self.answer_hours(ask)
        self.assertTrue(says_the_hour(text, hours), f"{text!r} doesn't give the hour, {hours}")



class TestDepth(unittest.TestCase):
    """ MoGe-2's distances in the cat photo, and the estimate_distance tool, on by default. """
    def setUp(self):
        with open(TEST_IMAGE, "rb") as f:
            self.image = f.read()

    def test_depth(self):
        response = httpx.post(f"{BASE_URL}/v1/depth?columns=3&rows=2", content=self.image,
                              headers={"Content-Type": "image/jpeg"}, timeout=60)
        response.raise_for_status()
        found = response.json()
        self.assertEqual(found["model"], "moge-2-vits")
        self.assertTrue(0.1 < found["center"] < 20, found)
        self.assertLessEqual(found["nearest"]["distance"], found["center"])
        self.assertEqual((len(found["grid"]), len(found["grid"][0])), (2, 3))
        self.assertAlmostEqual(found["width"] / found["height"], 320 / 213, places=1)

    def test_png(self):
        response = httpx.post(f"{BASE_URL}/v1/depth", params={"response_format": "png"}, content=self.image,
                              headers={"Content-Type": "image/jpeg"}, timeout=60)
        response.raise_for_status()
        self.assertTrue(response.content.startswith(b"\x89PNG"))
        self.assertLess(float(response.headers["x-depth-near"]), float(response.headers["x-depth-far"]))

    def test_chat_tool(self):
        url = "data:image/jpeg;base64," + base64.b64encode(self.image).decode("ascii")
        response = client().responses.create(model="litertlm", max_output_tokens=128, input=[
            {"role": "user", "content": [{"type": "input_text", "text": "How far away is the cat from the camera?"},
                                         {"type": "input_image", "image_url": url}]}])
        self.assertIn("estimate_distance", [item.name for item in response.output if item.type == "mcp_call"])
        self.assertRegex(response.output_text, r"\d")


class TestBoardClient(unittest.TestCase):
    """ examples/board_client.py, the sockets-only client for boards, on the chat endpoints. """

    @classmethod
    def setUpClass(cls):
        import importlib.util
        path = os.path.join(os.path.dirname(__file__), "..", "examples", "board_client.py")
        spec = importlib.util.spec_from_file_location("board_client", path)
        cls.bc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.bc)
        url = urllib.parse.urlparse(BASE_URL)
        cls.la = cls.bc.Client(url.hostname, url.port or 80, timeout=300)
        with open(TEST_WAV, "rb") as f:
            cls.pcm = f.read()[44:]  # 24 kHz 16-bit mono, past the WAV header

    def test_chat(self):
        self.assertTrue(self.la.chat("Say hello in three words.", max_tokens=16).strip())
        self.assertIn(self.la.reply["choices"][0]["finish_reason"], ("stop", "length"))

    def test_chat_stream(self):
        pieces = list(self.la.chat_stream("Count from one to five in words.", max_tokens=32))
        self.assertGreater(len(pieces), 1, "stream carried the reply in one piece")
        self.assertIsNotNone(self.la.first_ms)

    def test_respond(self):
        self.assertTrue(self.la.respond("Say hello in three words.", max_output_tokens=16).strip())
        streamed = "".join(self.la.respond("Say hello in three words.", stream=True, max_output_tokens=16))
        self.assertTrue(streamed.strip())
        self.assertIn(self.la.reply["status"], ("completed", "incomplete"))

    def test_voice(self):
        pcm = b"".join(bytes(p) for p in self.la.voice_stream(self.pcm, 24000, max_tokens=48, stt_model="moonshine"))
        self.assertIn("test audio", self.la.heard.lower())
        # A second of 24 kHz 16-bit mono is 48000 bytes; any real reply is longer
        self.assertGreater(len(pcm), 48000)

    def test_realtime(self):
        rt = self.la.realtime(max_output_tokens=48, audio={"input": {"turn_detection": None}})
        try:
            rt.append(self.pcm)
            rt.send({"type": "input_audio_buffer.commit"})
            audio = b"".join(rt.reply())
        finally:
            rt.close()
        self.assertGreater(len(audio), 48000)
        self.assertTrue(rt.transcript.strip())
        self.assertIn(rt.done["response"]["status"], ("completed", "incomplete"))


if __name__ == "__main__":
    unittest.main()
