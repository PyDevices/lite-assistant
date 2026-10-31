"""
The Realtime API (/v1/realtime): a WebSocket session in which the client
streams microphone audio in and gets the reply back as streamed 24 kHz PCM
with its transcript, or as text. Gemma hears the audio itself; the reply is
spoken a sentence at a time as it is written. With server VAD, the server
finds where each turn ends and answers it, and speech that starts during a
reply interrupts it; without, the client commits each turn itself.
"""
import asyncio
import base64
import json
import threading
import uuid
from typing import Any, Dict, Iterator, List, Optional

from fastapi import WebSocket, WebSocketDisconnect
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from .engines import ToolCall
from .function_tools import ToolError, chat_tools
from .media import MediaError, resolve_files, resolve_images
from .tools import Toolbox
from .vad import TurnDetector
from .voice import SAMPLE_RATE, spoken_reply, wav
from .voice_text import VOICE_SYSTEM_PROMPT

PCM_FORMAT = {"type": "audio/pcm", "rate": SAMPLE_RATE}
BYTES_PER_MS = SAMPLE_RATE * 2 // 1000

SERVER_VAD = {"type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300, "silence_duration_ms": 500,
              "idle_timeout_ms": None, "create_response": True, "interrupt_response": True}
SEMANTIC_VAD = {"type": "semantic_vad", "eagerness": "auto", "create_response": True, "interrupt_response": True}
# semantic_vad waits for the end of a thought, which Silero can't judge, so
# its eagerness becomes how much silence ends a turn
EAGERNESS_SILENCE_MS = {"low": 1000, "medium": 500, "auto": 500, "high": 300}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


class ClientError(Exception):
    """ A client event the session can't act on, answered with an error event. """
    def __init__(self, message: str, code: str = "invalid_value"):
        super().__init__(message)
        self.message, self.code = message, code


def merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """ base with update's values laid over it, nested dicts merged rather than replaced. """
    merged = dict(base)
    for key, value in update.items():
        merged[key] = merge(base[key], value) if isinstance(value, dict) and isinstance(base.get(key), dict) \
            else value
    return merged


class RealtimeSession:
    def __init__(self, websocket: WebSocket, engines, vad=None, toolbox: Optional[Toolbox] = None, llm=None):
        self.ws = websocket
        self.e = engines
        self.vad = vad
        self.toolbox = toolbox or Toolbox(default="none")
        self.llm = llm or next(iter(engines.llm.values()))
        default_tts = engines.tts[engines.default_tts]
        self.session = {
            "type": "realtime", "object": "realtime.session", "id": new_id("sess"),
            "model": self.llm.get_model_id(), "output_modalities": ["audio"],
            "instructions": VOICE_SYSTEM_PROMPT, "max_output_tokens": "inf",
            "tools": [], "tool_choice": "auto",
            "audio": {
                "input": {"format": PCM_FORMAT, "transcription": None,
                          "turn_detection": dict(SERVER_VAD) if vad else None,
                          "noise_reduction": None},
                "output": {"format": PCM_FORMAT, "voice": default_tts.get_default_voice(), "speed": 1.0},
            },
        }
        self.input_audio = bytearray()
        # Where input_audio starts, in bytes of all the audio appended this session
        self.input_start = 0
        self.detector: Optional[TurnDetector] = None
        self.speech_item: Optional[str] = None
        self.turn_detection(self.session["audio"]["input"]["turn_detection"])
        # The conversation, oldest first, as Realtime items; audio parts keep their PCM here
        self.items: List[Dict[str, Any]] = []
        self.audio: Dict[str, bytes] = {}
        # Each spoken assistant item's sentences with where their audio starts
        # in ms, and the audio's length, for truncating it
        self.spoken: Dict[str, Any] = {}
        self.response: Optional[asyncio.Task] = None
        self.cancelled = threading.Event()
        self.cancel_reason = "client_cancelled"
        self.send_lock = asyncio.Lock()

    # --- Sending --------------------------------------------------------

    async def send(self, kind: str, **fields):
        async with self.send_lock:
            await self.ws.send_text(json.dumps({"type": kind, "event_id": new_id("event"), **fields}))

    async def send_error(self, message: str, code: str, client_event_id: Optional[str] = None):
        await self.send("error", error={"type": "invalid_request_error", "code": code, "message": message,
                                        "param": None, "event_id": client_event_id})

    # --- The session loop -----------------------------------------------

    async def run(self):
        await self.send("session.created", session=self.session)
        try:
            while True:
                text = await self.ws.receive_text()
                try:
                    event = json.loads(text)
                    kind = event.get("type", "")
                except (ValueError, AttributeError):
                    await self.send_error("Events must be JSON objects.", "invalid_json")
                    continue
                handler = getattr(self, "on_" + kind.replace(".", "_"), None)
                if not handler:
                    await self.send_error(f"Unknown event type {kind!r}.", "invalid_event", event.get("event_id"))
                    continue
                try:
                    await handler(event)
                except ClientError as exc:
                    await self.send_error(exc.message, exc.code, event.get("event_id"))
        except WebSocketDisconnect:
            pass
        finally:
            await self.cancel_response()

    # --- Session ---------------------------------------------------------

    def tts_for(self, voice: str):
        """ The TTS engine that has this voice. """
        for engine in self.e.tts.values():
            if any(v["id"] == voice for v in engine.get_available_voices()):
                return engine
        raise ClientError(f"Unknown voice {voice!r}; GET /voices lists them.")

    async def on_session_update(self, event):
        update = event.get("session") or {}
        session = merge(self.session, {k: v for k, v in update.items() if k not in ("type", "object", "id")})
        audio = session["audio"]
        for direction in ("input", "output"):
            fmt = audio[direction].get("format") or PCM_FORMAT
            if fmt.get("type") != "audio/pcm" or fmt.get("rate", 24000) != 24000:
                raise ClientError(f"Only 24 kHz audio/pcm is supported for {direction}.", "unsupported_format")
        turn_detection = audio["input"].get("turn_detection")
        if turn_detection:
            if not self.vad:
                raise ClientError("Server turn detection needs Silero VAD; run 'lite-assistant setup', or commit "
                                  "the audio buffer yourself.", "unsupported_turn_detection")
            defaults = {"server_vad": SERVER_VAD, "semantic_vad": SEMANTIC_VAD}.get(turn_detection.get("type"))
            if not defaults:
                raise ClientError("turn_detection's type must be server_vad or semantic_vad.")
            given = (update.get("audio") or {}).get("input", {}).get("turn_detection") or {}
            # A change of type starts again from that type's defaults
            base = turn_detection if given.get("type") in (None, turn_detection["type"]) else {}
            turn_detection = {**defaults, **{k: v for k, v in base.items() if k in defaults}, **given}
            if not 0 <= turn_detection.get("threshold", 0.5) <= 1:
                raise ClientError("turn_detection's threshold must be between 0 and 1.")
            if turn_detection.get("eagerness", "auto") not in EAGERNESS_SILENCE_MS:
                raise ClientError("turn_detection's eagerness must be low, medium, high or auto.")
            audio["input"]["turn_detection"] = turn_detection
        if audio["input"].get("transcription") and not self.e.stt:
            raise ClientError("No STT engine is loaded to transcribe input audio.", "unsupported_transcription")
        self.tts_for(audio["output"]["voice"])
        if not set(session.get("output_modalities") or ["audio"]) <= {"audio", "text"}:
            raise ClientError("output_modalities must be audio or text.")
        self.tools(session.get("tools"), session.get("tool_choice"))
        self.builtins(session.get("builtin_tools"))
        if turn_detection != self.session["audio"]["input"]["turn_detection"]:
            self.turn_detection(turn_detection)
        self.session = session
        await self.send("session.updated", session=self.session)

    @staticmethod
    def tools(tools, tool_choice) -> Optional[List[Dict[str, Any]]]:
        """ The function tools a response may call, as the LLM engine takes them. """
        try:
            return chat_tools(tools, tool_choice, flat=True)
        except ToolError as exc:
            raise ClientError(str(exc))

    def turn_detection(self, config: Optional[Dict[str, Any]]):
        """ Starts detecting turns with this turn_detection config, or stops for None. """
        self.speech_item = None
        if not config:
            self.detector = None
            return
        silence_ms = config.get("silence_duration_ms") or EAGERNESS_SILENCE_MS[config.get("eagerness", "auto")]
        elapsed_ms = (self.input_start + len(self.input_audio)) // BYTES_PER_MS
        self.detector = TurnDetector(self.vad.stream(), config.get("threshold", 0.5), silence_ms, elapsed_ms)

    # --- Input audio -----------------------------------------------------

    async def on_input_audio_buffer_append(self, event):
        try:
            pcm = base64.b64decode(event.get("audio", ""), validate=True)
        except ValueError:
            raise ClientError("audio must be base64-encoded PCM.")
        self.input_audio += pcm
        if not self.detector:
            return
        for change, ms in await run_in_threadpool(self.detector.feed, pcm):
            await (self.speech_started if change == "speech_started" else self.speech_stopped)(ms)
        if not self.detector.speaking:
            # Between turns, keep only the audio that may become the next turn's prefix padding
            self.trim_input(self.detector.elapsed_ms - self.prefix_padding_ms())

    def prefix_padding_ms(self) -> int:
        return self.session["audio"]["input"]["turn_detection"].get("prefix_padding_ms", 300)

    def trim_input(self, from_ms: int):
        """ Drops the buffered input audio from before from_ms. """
        drop = min(max(0, from_ms * BYTES_PER_MS - self.input_start), len(self.input_audio))
        del self.input_audio[:drop]
        self.input_start += drop

    async def speech_started(self, ms: int):
        self.speech_item = new_id("item")
        self.trim_input(ms - self.prefix_padding_ms())
        await self.send("input_audio_buffer.speech_started", audio_start_ms=ms, item_id=self.speech_item)
        if self.session["audio"]["input"]["turn_detection"].get("interrupt_response", True):
            await self.cancel_response("turn_detected")

    async def speech_stopped(self, ms: int):
        item_id, self.speech_item = self.speech_item, None
        await self.send("input_audio_buffer.speech_stopped", audio_end_ms=ms, item_id=item_id)
        await self.commit(item_id)
        if self.session["audio"]["input"]["turn_detection"].get("create_response", True) and not self.responding():
            await self.start_response({})

    def end_turn(self):
        """ Forgets the speech in progress, as the client has taken the buffer's audio. """
        self.speech_item = None
        if self.detector:
            self.detector.speaking = False

    async def on_input_audio_buffer_clear(self, event):
        self.input_start += len(self.input_audio)
        self.input_audio.clear()
        self.end_turn()
        await self.send("input_audio_buffer.cleared")

    async def on_input_audio_buffer_commit(self, event):
        if len(self.input_audio) < 2:
            raise ClientError("The input audio buffer is empty.", "input_audio_buffer_commit_empty")
        item_id = self.speech_item
        self.end_turn()
        await self.commit(item_id)

    async def commit(self, item_id: Optional[str] = None):
        """ Makes the buffered input audio a user item. """
        pcm, self.input_audio = bytes(self.input_audio), bytearray()
        self.input_start += len(pcm)
        item = {"id": item_id or new_id("item"), "type": "message", "role": "user",
                "content": [{"type": "input_audio", "transcript": None}]}
        self.audio[item["id"]] = pcm
        previous = self.items[-1]["id"] if self.items else None
        await self.send("input_audio_buffer.committed", item_id=item["id"], previous_item_id=previous)
        await self.add_item(item, previous)
        if self.session["audio"]["input"].get("transcription"):
            asyncio.create_task(self.transcribe(item, pcm))

    async def transcribe(self, item: Dict[str, Any], pcm: bytes):
        stt = self.e.stt[self.e.default_stt]
        try:
            transcript = await run_in_threadpool(stt.transcribe, wav(pcm))
        except Exception as exc:
            await self.send("conversation.item.input_audio_transcription.failed", item_id=item["id"],
                            content_index=0, error={"type": "transcription_error", "message": str(exc)})
            return
        item["content"][0]["transcript"] = transcript
        await self.send("conversation.item.input_audio_transcription.completed", item_id=item["id"],
                        content_index=0, transcript=transcript)

    # --- Conversation items ----------------------------------------------

    def find(self, item_id: str) -> int:
        for index, item in enumerate(self.items):
            if item["id"] == item_id:
                return index
        raise ClientError(f"Item with id {item_id!r} not found.", "item_not_found")

    async def add_item(self, item: Dict[str, Any], previous_item_id: Optional[str] = None):
        item = {"object": "realtime.item", "status": "completed", **item}
        index = self.find(previous_item_id) + 1 if previous_item_id and previous_item_id != "root" \
            else (0 if previous_item_id == "root" else len(self.items))
        self.items.insert(index, item)
        previous = self.items[index - 1]["id"] if index else None
        await self.send("conversation.item.added", item=item, previous_item_id=previous)
        await self.send("conversation.item.done", item=item, previous_item_id=previous)

    async def on_conversation_item_create(self, event):
        item = dict(event.get("item") or {})
        kind = item.get("type", "message")
        if kind == "function_call" and item.get("call_id") and item.get("name"):
            await self.add_item({"id": item.get("id") or new_id("item"), "type": kind, "call_id": item["call_id"],
                                 "name": item["name"], "arguments": item.get("arguments") or "{}"},
                                event.get("previous_item_id"))
            return
        if kind == "function_call_output" and item.get("call_id"):
            await self.add_item({"id": item.get("id") or new_id("item"), "type": kind, "call_id": item["call_id"],
                                 "output": item.get("output", "")}, event.get("previous_item_id"))
            return
        if kind in ("function_call", "function_call_output"):
            raise ClientError(f"A {kind} item needs a call_id{' and a name' if kind == 'function_call' else ''}.")
        if kind != "message":
            raise ClientError(f"Items of type {kind!r} aren't supported.")
        if item.get("role") not in ("user", "assistant", "system"):
            raise ClientError("A message's role must be user, assistant or system.")
        item["id"] = item.get("id") or new_id("item")
        item["type"] = "message"
        content = []
        for part in item.get("content") or []:
            part = dict(part)
            if part.get("type") == "input_audio" and part.get("audio"):
                try:
                    self.audio[item["id"]] = base64.b64decode(part.pop("audio"), validate=True)
                except ValueError:
                    raise ClientError("input_audio audio must be base64-encoded PCM.")
            elif part.get("type") not in ("input_text", "input_image", "input_file", "output_text", "output_audio", "text"):
                raise ClientError(f"Content parts of type {part.get('type')!r} aren't supported.")
            content.append(part)
        item["content"] = content
        await self.add_item(item, event.get("previous_item_id"))

    async def on_conversation_item_delete(self, event):
        item_id = event.get("item_id", "")
        del self.items[self.find(item_id)]
        self.audio.pop(item_id, None)
        self.spoken.pop(item_id, None)
        await self.send("conversation.item.deleted", item_id=item_id)

    async def on_conversation_item_truncate(self, event):
        """
        Cuts an assistant item's audio to what the client played, so the
        transcript the LLM sees next ends where the user stopped hearing it.
        Whole sentences are kept, those whose audio started before the cut.
        """
        item_id = event.get("item_id", "")
        item = self.items[self.find(item_id)]
        index = event.get("content_index", 0)
        parts = item.get("content") or []
        if item.get("role") != "assistant" or not (isinstance(index, int) and 0 <= index < len(parts)) \
                or parts[index].get("type") != "output_audio":
            raise ClientError("Only an assistant item's output_audio can be truncated.")
        sentences, length_ms = self.spoken.get(item_id, ([], 0))
        end_ms = event.get("audio_end_ms")
        if not isinstance(end_ms, int) or not 0 <= end_ms <= length_ms:
            raise ClientError(f"audio_end_ms must be between 0 and the audio's length, {length_ms} ms.")
        kept = [(start, sentence) for start, sentence in sentences if start < end_ms]
        self.spoken[item_id] = (kept, end_ms)
        parts[index] = {**parts[index], "transcript": " ".join(sentence for _, sentence in kept)}
        await self.send("conversation.item.truncated", item_id=item_id, content_index=index, audio_end_ms=end_ms)

    async def on_conversation_item_retrieve(self, event):
        await self.send("conversation.item.retrieved", item=self.items[self.find(event.get("item_id", ""))])

    def to_messages(self, instructions: str) -> List[Dict[str, Any]]:
        """ The conversation as chat messages for the LLM. """
        messages = [{"role": "system", "content": instructions}] if instructions else []
        for item in self.items:
            if item["type"] == "function_call":
                call = {"id": item["call_id"], "type": "function",
                        "function": {"name": item["name"], "arguments": item["arguments"]}}
                # Calls made together belong to one assistant message, with any text before them
                if messages and messages[-1]["role"] == "assistant":
                    messages[-1].setdefault("tool_calls", []).append(call)
                else:
                    messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
                continue
            if item["type"] == "function_call_output":
                messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": item["output"]})
                continue
            parts = []
            for part in item["content"]:
                kind = part.get("type")
                if kind in ("input_text", "output_text", "text"):
                    parts.append({"type": "text", "text": part.get("text", "")})
                elif kind == "output_audio":
                    parts.append({"type": "text", "text": part.get("transcript") or ""})
                elif kind == "input_audio" and item["id"] in self.audio:
                    parts.append({"type": "input_audio", "input_audio": {
                        "data": base64.b64encode(wav(self.audio[item["id"]])).decode("ascii"), "format": "wav"}})
                elif kind == "input_audio" and part.get("transcript"):
                    parts.append({"type": "text", "text": part["transcript"]})
                elif kind == "input_image":
                    parts.append({"type": "image_url", "image_url": {"url": part.get("image_url", "")}})
                elif kind == "input_file":
                    parts.append({"type": "file", "file": {"filename": part.get("filename") or "file",
                                                           "file_data": part.get("file_data", "")}})
            if all(p["type"] == "text" for p in parts):
                messages.append({"role": item["role"], "content": "".join(p["text"] for p in parts)})
            else:
                messages.append({"role": item["role"], "content": parts})
        return messages

    # --- Responses -------------------------------------------------------

    def builtins(self, value):
        """ The LLM engine, running the built-in tools that value names, as a request's builtin_tools does. """
        try:
            return self.toolbox.llm(self.llm, self.toolbox.select(value), user=self.ws.scope.get("user"))
        except ToolError as exc:
            raise ClientError(str(exc))

    def responding(self) -> bool:
        return bool(self.response and not self.response.done())

    async def on_response_create(self, event):
        await self.start_response(event.get("response") or {})

    async def start_response(self, options: Dict[str, Any]):
        if self.responding():
            raise ClientError("The conversation already has an active response.",
                              "conversation_already_has_active_response")
        tools = self.tools(options.get("tools", self.session.get("tools")),
                           options.get("tool_choice", self.session.get("tool_choice")))
        modalities = options.get("output_modalities") or self.session["output_modalities"]
        voice = ((options.get("audio") or {}).get("output") or {}).get("voice") \
            or self.session["audio"]["output"]["voice"]
        tts = self.tts_for(voice) if "audio" in modalities else None
        llm = self.builtins(options.get("builtin_tools", self.session.get("builtin_tools")))
        self.cancelled.clear()
        self.cancel_reason = "client_cancelled"
        self.response = asyncio.create_task(self.respond(options, modalities, tts, voice, tools, llm))

    async def on_response_cancel(self, event):
        if not self.responding():
            raise ClientError("There is no active response to cancel.", "response_cancel_not_active")
        await self.cancel_response()

    async def cancel_response(self, reason: str = "client_cancelled"):
        if self.responding():
            self.cancel_reason = reason
            self.cancelled.set()
            await asyncio.wait([self.response])

    async def respond(self, options: Dict[str, Any], modalities: List[str], tts, voice: str,
                      tools: Optional[List[Dict[str, Any]]], llm):
        session = self.session
        audio_out = tts is not None
        instructions = options.get("instructions", session["instructions"])
        limit = options.get("max_output_tokens", session["max_output_tokens"])
        max_tokens = None if limit in (None, "inf") else int(limit)
        try:
            messages = await run_in_threadpool(
                lambda: resolve_files(resolve_images(self.to_messages(instructions)), next(iter(self.e.ocr.values()), None)))
        except MediaError as exc:
            await self.send_error(str(exc), "invalid_image")
            return

        response = {"id": new_id("resp"), "object": "realtime.response", "status": "in_progress",
                    "status_details": None, "output": [], "conversation_id": "conv_default",
                    "output_modalities": modalities, "max_output_tokens": limit, "usage": None,
                    "audio": {"output": {"format": PCM_FORMAT, "voice": voice}}, "metadata": options.get("metadata")}
        await self.send("response.created", response=response)
        # The finished output items, and the assistant message being written, if any
        output: List[Dict[str, Any]] = []
        message: Optional[Dict[str, Any]] = None

        async def open_message() -> Dict[str, Any]:
            item = {"id": new_id("item"), "object": "realtime.item", "type": "message", "role": "assistant",
                    "status": "in_progress", "content": []}
            where = {"response_id": response["id"], "item_id": item["id"], "output_index": len(output),
                     "content_index": 0}
            previous = self.items[-1]["id"] if self.items else None
            await self.send("response.output_item.added", response_id=response["id"], output_index=len(output),
                            item=item)
            await self.send("conversation.item.added", item=item, previous_item_id=previous)
            await self.send("response.content_part.added", **where,
                            part={"type": "audio", "transcript": ""} if audio_out else {"type": "text", "text": ""})
            # sentences: each with where its audio starts, in ms, for truncating
            return {"item": item, "where": where, "previous": previous, "text": "", "sentences": [],
                    "audio_bytes": 0}

        async def close_message(status: str):
            where, text = message["where"], message["text"]
            if audio_out:
                await self.send("response.output_audio.done", **where)
                await self.send("response.output_audio_transcript.done", **where, transcript=text)
                part = {"type": "output_audio", "transcript": text}
                await self.send("response.content_part.done", **where, part={"type": "audio", "transcript": text})
            else:
                await self.send("response.output_text.done", **where, text=text)
                part = {"type": "output_text", "text": text}
                await self.send("response.content_part.done", **where, part={"type": "text", "text": text})
            item = {**message["item"], "status": "completed" if status == "completed" else "incomplete",
                    "content": [part]}
            self.items.append(item)
            if audio_out:
                self.spoken[item["id"]] = (message["sentences"], message["audio_bytes"] // BYTES_PER_MS)
            output.append(item)
            await self.send("response.output_item.done", response_id=response["id"],
                            output_index=where["output_index"], item=item)
            await self.send("conversation.item.done", item=item, previous_item_id=message["previous"])

        usage: Dict[str, int] = {}
        status = "completed"
        try:
            async for kind, value in iterate_in_threadpool(self.reply_events(llm, tts, voice, messages, max_tokens,
                                                                               usage, tools)):
                if self.cancelled.is_set():
                    continue
                if kind == "tool_call":
                    if message:
                        await close_message("completed")
                        message = None
                    output.append(await self.function_call(response["id"], len(output), value))
                    continue
                if not message:
                    message = await open_message()
                where = message["where"]
                if kind == "audio":
                    message["audio_bytes"] += len(value)
                    await self.send("response.output_audio.delta", **where,
                                    delta=base64.b64encode(value).decode("ascii"))
                elif audio_out:
                    # A sentence: the transcript of the audio about to follow
                    message["sentences"].append((message["audio_bytes"] // BYTES_PER_MS, value))
                    delta = (" " if message["text"] else "") + value
                    message["text"] += delta
                    await self.send("response.output_audio_transcript.delta", **where, delta=delta)
                else:
                    message["text"] += value
                    await self.send("response.output_text.delta", **where, delta=value)
        except Exception as exc:
            print(f"\n[!] Realtime response error: {exc}")
            status = "failed"
        if self.cancelled.is_set():
            status = "cancelled"
        elif status == "completed" and max_tokens and usage.get("completion_tokens", 0) >= max_tokens:
            status = "incomplete"

        # A reply with no text and no calls is still an (empty) message
        if not message and not output:
            message = await open_message()
        if message:
            await close_message(status)

        input_tokens, output_tokens = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        details = {"cancelled": {"type": "cancelled", "reason": self.cancel_reason},
                   "incomplete": {"type": "incomplete", "reason": "max_output_tokens"},
                   "failed": {"type": "failed", "error": {"type": "server_error", "code": "generation_failed"}}}
        response = {**response, "status": status, "status_details": details.get(status), "output": output,
                    "usage": {"total_tokens": input_tokens + output_tokens, "input_tokens": input_tokens,
                              "output_tokens": output_tokens}}
        await self.send("response.done", response=response)

    async def function_call(self, response_id: str, index: int, call) -> Dict[str, Any]:
        """ Sends a call the LLM made as a function_call item, and adds it to the conversation. """
        item = {"id": new_id("item"), "object": "realtime.item", "type": "function_call", "status": "in_progress",
                "call_id": call.id, "name": call.name, "arguments": ""}
        previous = self.items[-1]["id"] if self.items else None
        await self.send("response.output_item.added", response_id=response_id, output_index=index, item=item)
        await self.send("conversation.item.added", item=item, previous_item_id=previous)
        where = {"response_id": response_id, "item_id": item["id"], "output_index": index, "call_id": call.id}
        # A call arrives whole, so its arguments are one delta
        await self.send("response.function_call_arguments.delta", **where, delta=call.arguments)
        await self.send("response.function_call_arguments.done", **where, name=call.name, arguments=call.arguments)
        item = {**item, "status": "completed", "arguments": call.arguments}
        self.items.append(item)
        await self.send("response.output_item.done", response_id=response_id, output_index=index, item=item)
        await self.send("conversation.item.done", item=item, previous_item_id=previous)
        return item

    def reply_events(self, llm, tts, voice: str, messages: List[Dict[str, Any]], max_tokens: Optional[int],
                     usage: Dict[str, int], tools: Optional[List[Dict[str, Any]]] = None) -> Iterator:
        """
        The reply as ("sentence" or "text", str), ("audio", pcm) and
        ("tool_call", call) events, until it ends or is cancelled.
        """
        events = spoken_reply(llm, tts, voice, messages, max_tokens, usage, cancel=self.cancelled,
                              tools=tools) if tts else self.text_reply(llm, messages, max_tokens, usage, tools)
        try:
            for event in events:
                # Checked between audio chunks too, so a cancel stops the speech within a chunk
                if self.cancelled.is_set():
                    return
                yield event
        finally:
            events.close()

    @staticmethod
    def text_reply(llm, messages: List[Dict[str, Any]], max_tokens: Optional[int], usage: Dict[str, int],
                   tools: Optional[List[Dict[str, Any]]]) -> Iterator:
        # Realtime replies don't wait for the model to think, written or spoken
        chunks = llm.generate(messages, max_tokens=max_tokens, usage=usage, thinking=False,
                              **({"tools": tools} if tools else {}))
        try:
            for chunk in chunks:
                if isinstance(chunk, str):
                    yield "text", chunk
                elif isinstance(chunk, ToolCall):
                    yield "tool_call", chunk
        finally:
            chunks.close()


async def serve(websocket: WebSocket, engines, vad=None, toolbox: Optional[Toolbox] = None,
                model: Optional[str] = None):
    """
    Runs one Realtime session on an accepted WebSocket, detecting turns with
    vad if given, and running toolbox's built-in tools. model picks the LLM,
    as a request's model does.
    """
    if not engines.llm or not engines.tts:
        await websocket.close(code=1011, reason="Realtime needs an LLM engine and a TTS engine loaded.")
        return
    try:
        llm = engines.pick_llm(model)
    except Exception as exc:
        await websocket.close(code=1008, reason=getattr(exc, "message", str(exc))[:120])
        return
    await RealtimeSession(websocket, engines, vad, toolbox, llm).run()
