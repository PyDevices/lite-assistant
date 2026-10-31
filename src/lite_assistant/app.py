"""
The lite-assistant HTTP server: OpenAI-compatible speech, transcription and
chat endpoints, a spoken-turn endpoint, and a web page for trying the voices.

Engine calls block, so the endpoints are plain functions, which FastAPI runs
in its thread pool, and streamed replies wrap the engines' generators.
"""
import asyncio
import base64
import collections
import json
import queue
import threading
import time
import urllib.parse
import uuid
from contextlib import asynccontextmanager
from importlib import resources
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Sequence, Union

import anyio
import numpy as np
from fastapi import FastAPI, File, Form, Query, Request, UploadFile, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIWebSocketRoute
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import iterate_in_threadpool

from . import audio, images, litert_models, perception, realtime, video
from .engines import KINDS, LazyEngine, Thought, ToolCall, load_engines
from .files import MAX_UPLOAD, Files, from_files
from .api_common import APIError
from .function_tools import ToolError, chat_tools
from .media import MediaError, is_image_part, resolve_files, resolve_image_part, resolve_images
from .responses import (InputError, ResponseBuilder, ResponseRequest, input_items, new_id, schema_option,
                        thinking_option, to_messages)
from .store import MAX_ITEMS_PER_ADD, Conversation, Store, StoredResponse, page
from .tools import Toolbox
from .users import Auth, Users, admin_only
from .vad import SileroVAD
from .voice import spoken_reply
from .voice_text import VOICE_SYSTEM_PROMPT


# What /v1/models calls the kinds of engine whose names in KINDS are short
KIND_NAMES = {"embed": "embedding", "detect": "detection", "classify": "classification", "edit": "image_edit",
              "quality": "image_quality", "recognize": "image_classification", "denoise": "noise_removal"}


class Engines:
    """
    The loaded engines, by engine id: an attribute for each kind in KINDS,
    such as tts or detect, which is empty if none of that kind loaded.
    """
    def __init__(self, tts, stt, llm, embed=None, depth=None, **others):
        unknown = set(others) - set(KINDS)
        if unknown:
            raise TypeError(f"Unknown kinds of engine: {', '.join(sorted(unknown))}")
        self.tts, self.stt, self.llm, self.embed, self.depth = tts, stt, llm, embed or {}, depth or {}
        for kind in KINDS:
            if kind not in ("tts", "stt", "llm", "embed", "depth"):
                setattr(self, kind, others.get(kind) or {})
        self.default_tts = "kokoro" if "kokoro" in tts else next(iter(tts), None)
        self.default_stt = "moonshine" if "moonshine" in stt else next(iter(stt), None)
        # Objects by default; faces when asked for
        self.default_detect = "yolox" if "yolox" in self.detect else next(iter(self.detect), None)

    def registries(self) -> Dict[str, Dict[str, Any]]:
        """ Each kind's engines, by the kind /v1/models gives it. """
        return {KIND_NAMES.get(kind, kind): getattr(self, kind) for kind in KINDS}

    def unload_idle(self) -> None:
        """ Unloads the engines that load on first use and have gone unused for a while. """
        for registry in self.registries().values():
            for engine in registry.values():
                if isinstance(engine, LazyEngine):
                    engine.unload_idle()

    def pick_llm(self, model: Optional[str]):
        """
        The LLM engine for a request's model. A model an engine can switch to,
        such as "gemma-4-E4B-it" or "e4b", gets that engine on that model. Any
        other name the registry doesn't know (e.g. "gpt-4o") gets the default
        engine with whatever model it has loaded. Raises APIError for a model
        an engine knows but can't load. Voice clips sent to a model that can't
        hear them are transcribed first, by the default STT engine.
        """
        chosen = self.llm.get(model)
        for engine in self.llm.values() if not chosen else []:
            try:
                chosen = engine.use(model)
            except LookupError as exc:
                raise APIError(404, str(exc), code="model_not_found")
            if chosen:
                break
        chosen = chosen or next(iter(self.llm.values()))
        return TranscribingLLM(chosen, self.stt[self.default_stt]) if self.default_stt else chosen


class TranscribingLLM:
    """
    An LLM that, while its model can't hear, gets voice clips as their
    transcripts, from an STT engine. Anything else it passes to the LLM.
    """
    # A conversation resends its clips each turn, so the latest transcripts are kept
    transcripts: Dict[str, str] = {}
    KEPT_TRANSCRIPTS = 64

    def __init__(self, llm, stt):
        self.llm, self.stt = llm, stt

    def __getattr__(self, name):
        return getattr(self.llm, name)

    def generate(self, messages: List[Dict[str, Any]], **options) -> Iterator[Any]:
        # A model named by the request says what it takes before it's loaded
        if "audio" not in self.llm.get_modalities():
            messages = [self.transcribed(message) for message in messages]
        yield from self.llm.generate(messages, **options)

    def transcribed(self, message: Dict[str, Any]) -> Dict[str, Any]:
        content = message.get("content")
        if not isinstance(content, list) or not any(part.get("type") == "input_audio" for part in content):
            return message
        parts = []
        for part in content:
            if part.get("type") == "input_audio":
                data = part["input_audio"]["data"]
                transcript = self.transcripts.get(data)
                if transcript is None:
                    transcript = self.stt.transcribe(base64.b64decode(data)).strip()
                    self.transcripts[data] = transcript
                    while len(self.transcripts) > self.KEPT_TRANSCRIPTS:
                        del self.transcripts[next(iter(self.transcripts))]
                parts.append({"type": "text", "text": f"(Voice message) {transcript or '(nothing heard)'}"})
            else:
                parts.append(part)
        return {**message, "content": parts}


def generation_error(exc: Exception, what: str) -> APIError:
    """ The APIError for an LLM that failed to start a reply. """
    print(f"{what} error: {exc}")
    # A request the model can't take, such as an image for one that can't see
    if isinstance(exc, ValueError):
        return APIError(400, str(exc))
    return APIError(500, str(exc), "server_error")


def error_body(status: int, message: str, kind: str, code: Optional[str] = None) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind, "code": code}}, status_code=status)


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    input: str = Field(min_length=1)
    model: Optional[str] = None
    voice: Optional[str] = None


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    # A string, a list of strings, or, beyond OpenAI's API, image_url and
    # input_audio content parts, alone or in a list
    input: Union[str, Dict[str, Any], List[Any]]
    model: Optional[str] = None
    dimensions: Optional[int] = None
    encoding_format: str = "float"


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    query: str
    # Strings, or objects with a text field, as Cohere and Jina take them
    documents: List[Union[str, Dict[str, Any]]]
    model: Optional[str] = None
    top_n: Optional[int] = None
    return_documents: bool = True
    # Beyond Cohere's API: what relevant means, if not answering the query
    instruction: Optional[str] = None


class ConversationRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    items: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, str]] = None


class ConversationItemsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    items: List[Dict[str, Any]]


class AddModelRequest(BaseModel):
    repo: str
    file: Optional[str] = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    messages: List[Dict[str, Any]] = Field(min_length=1)
    model: Optional[str] = None
    stream: bool = False
    stream_options: Optional[Dict[str, Any]] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    # Beyond OpenAI's API: the server's own tools to offer, by name, or false for none
    builtin_tools: Optional[Union[bool, List[str]]] = None
    # Turns the model's thinking on or off (see thinking_option); its thoughts
    # come back as reasoning_content, as DeepSeek's and vLLM's APIs give them
    reasoning_effort: Optional[str] = None
    # json_object or json_schema constrains the reply to JSON (see schema_option)
    response_format: Optional[Dict[str, Any]] = None


TRANSLATION_PROMPT = ("Translate the speech in this recording into English. Reply with the English "
                      "translation only, with no notes or commentary. If the speech is already in "
                      "English, reply with exactly what was said.")


def audio_message(audio: bytes, text: Optional[str] = None) -> Dict[str, Any]:
    """ A user message carrying audio for Gemma to hear, and optionally some text with it. """
    parts = [{"type": "input_audio",
              "input_audio": {"data": base64.b64encode(audio).decode("ascii"), "format": "wav"}}]
    if text:
        parts.append({"type": "text", "text": text})
    return {"role": "user", "content": parts}


def streamed(chunks: Iterator, what: str) -> Iterator:
    """
    Passes chunks through to a streamed response, logging a failure instead of
    raising it into a response whose headers are already sent. Closing chunks
    when the stream ends, or the client goes away, stops the engine early.
    """
    try:
        yield from chunks
    except Exception as e:
        print(f"\n[!] {what} error: {e}")
    finally:
        close = getattr(chunks, "close", None)
        if close:
            close()


async def in_threadpool(chunks: Iterator) -> AsyncIterator:
    """
    Runs a blocking generator for a StreamingResponse. Starlette would run it
    in the thread pool itself, but when the client disconnects it just drops
    the generator, which then stops only whenever it's garbage collected;
    closing it here frees the engine right away.
    """
    try:
        async for chunk in iterate_in_threadpool(chunks):
            yield chunk
    finally:
        # The disconnect arrives as a cancellation, which would cut this short too
        with anyio.CancelScope(shield=True):
            await anyio.to_thread.run_sync(chunks.close)


def tools_option(tools: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """ generate's tools argument, given only when there are tools, so engines without tool support still work. """
    return {"tools": tools} if tools else {}


def create_app(engines: Optional[Engines] = None, vad=None, builtin_tools: Optional[str] = None,
               tool_dirs: Sequence[str] = (), users: Optional[Users] = None,
               behind_proxy: bool = False, allow_public: bool = False, database: str = "",
               forget_chats_after: Optional[float] = None, no_llm: bool = False) -> FastAPI:
    """
    Builds the app. Without engines, it loads every engine at startup, and
    Silero VAD for the Realtime API's turn detection if it's downloaded.
    builtin_tools names the built-in tools on by default, as
    LITE_ASSISTANT_TOOLS does, which it reads if not given, and tool_dirs are
    folders of Python files that add built-in tools (see tools.load_folders). users are who may
    use the server, those in the data directory's users.json if not given;
    behind_proxy makes requests from this PC need a key too, and allow_public
    lets requests from public addresses in (see users.Auth). database is the
    SQLite file conversations are kept in, in memory if not given, and
    forget_chats_after how many days unchanged one is kept (see store.Store).
    With no_llm, no chat model loads, and the endpoints that need one answer with an error.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.engines is None:
            print("Initializing TTS, STT and LLM engines...")
            app.state.engines = Engines(**load_engines(skip=["llm"] if no_llm else []))
            app.state.vad = SileroVAD.load()
            app.state.toolbox = Toolbox.create(app.state.engines, builtin_tools, tool_dirs)
        e = app.state.engines
        for kind, registry in e.registries().items():
            lazy = [engine_id for engine_id, engine in registry.items() if isinstance(engine, LazyEngine)]
            print(f"{kind} engines: {[i for i in registry if i not in lazy]}" +
                  (f", loaded on first use: {lazy}" if lazy else ""))
        print(f"Built-in tools on by default: {app.state.toolbox.default}")

        async def unload_idle():
            while True:
                await asyncio.sleep(60)
                await anyio.to_thread.run_sync(e.unload_idle)
        reaper = asyncio.create_task(unload_idle())
        yield
        reaper.cancel()

    app = FastAPI(title="lite-assistant", lifespan=lifespan)
    app.state.engines = engines
    app.state.vad = vad
    app.state.toolbox = Toolbox.create(engines, builtin_tools, tool_dirs) if engines else None
    app.state.store = Store(database, forget_chats_after)
    app.state.users = users or Users()
    app.include_router(perception.router)
    app.include_router(images.router)
    app.include_router(audio.router)
    app.include_router(video.router)
    # Added first, so CORS wraps it and a refusal carries CORS headers too
    app.add_middleware(Auth, users=app.state.users, behind_proxy=behind_proxy, allow_public=allow_public)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                       expose_headers=["X-Transcript", "X-Depth-Near", "X-Depth-Far"])

    @app.exception_handler(APIError)
    async def api_error(request: Request, exc: APIError):
        return error_body(exc.status, exc.message, exc.kind, exc.code)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        # OpenAI answers a malformed request with 400, not FastAPI's 422
        problems = "; ".join(f"{'.'.join(str(p) for p in err['loc'][1:]) or 'body'}: {err['msg']}"
                             for err in exc.errors())
        return error_body(400, problems, "invalid_request_error")

    def loaded(request: Request) -> Engines:
        return request.app.state.engines

    @app.get("/", response_class=HTMLResponse)
    def index():
        return resources.files(__package__).joinpath("static/index.html").read_text(encoding="utf-8")

    @app.get("/llms.txt")
    def llms_txt():
        """ What this server is and how to call it, for agents: the llms.txt convention. """
        return PlainTextResponse(resources.files(__package__).joinpath("static/llms.txt").read_text(encoding="utf-8"),
                                 media_type="text/markdown; charset=utf-8")

    @app.get("/v1")
    @app.get("/v1/", include_in_schema=False)
    def api_index(request: Request):
        """
        Every /v1 endpoint, read from the OpenAPI schema so it never goes
        stale, with where the docs are and whether a chat model is loaded.
        OpenAI's own /v1 is a 404; an index lets a client find its way from
        the address alone.
        """
        # The schema flattens the included routers; it leaves out WebSockets, which are added here
        endpoints = {path: {m.upper() for m in ops} for path, ops in request.app.openapi()["paths"].items()
                     if path.startswith("/v1/")}
        for route in request.app.routes:
            if isinstance(route, APIWebSocketRoute) and route.path.startswith("/v1/"):
                endpoints.setdefault(route.path, set()).add("WEBSOCKET")
        e = loaded(request)
        return {"object": "api", "name": "lite-assistant", "docs": "/llms.txt", "openapi": "/openapi.json",
                "models": "/v1/models", "voices": "/voices", "chat": bool(e and e.llm),
                "endpoints": [{"path": p, "methods": sorted(m)} for p, m in sorted(endpoints.items())]}

    @app.get("/me")
    def me(request: Request):
        """ Who the request comes from: their name, and whether they're an admin. """
        return {"object": "user", "name": request.user.name, "admin": request.user.admin}

    @app.get("/voices")
    def voices(request: Request):
        voices_list = []
        for engine_name, engine in loaded(request).tts.items():
            for voice in engine.get_available_voices():
                voices_list.append({**voice, "engine": engine_name})
        return sorted(voices_list, key=lambda v: v["name"])

    def model_objects(request: Request) -> Dict[str, Dict[str, Any]]:
        """
        Each loaded engine as an OpenAI model object, by id. Beyond OpenAI's
        fields: its kind (llm, tts, stt, embedding, depth, detection, sound
        or classification), and where the engine knows them, the model it loaded and the
        backend it runs on.
        """
        e = loaded(request)
        objects = {}
        for kind, registry in e.registries().items():
            for engine_id, engine in registry.items():
                obj = {"id": engine_id, "object": "model", "created": 0, "owned_by": "lite-assistant", "kind": kind}
                # One that loads on first use says whether it's in memory now
                if isinstance(engine, LazyEngine):
                    obj["loaded"] = engine.loaded
                try:
                    obj["model"] = engine.get_model_id()
                except (AttributeError, NotImplementedError):
                    pass
                if isinstance(getattr(engine, "backend", None), str):
                    obj["backend"] = engine.backend
                if kind == "llm":
                    obj["modalities"] = engine.get_modalities()
                objects.setdefault(engine_id, obj)
                # The models it can switch to, which a request's model can name
                for model_id, about in (engine.get_models() if kind == "llm" else {}).items():
                    objects.setdefault(model_id, {**obj, **about, "id": model_id, "model": model_id,
                                                  "engine": engine_id,
                                                  "loaded": model_id == obj.get("model")})
        return objects

    @app.get("/v1/models")
    def models(request: Request):
        return {"object": "list", "data": list(model_objects(request).values())}

    @app.get("/v1/models/{model}")
    def model(model: str, request: Request):
        obj = model_objects(request).get(model)
        if not obj:
            raise APIError(404, f"The model '{model}' does not exist", code="model_not_found")
        return obj

    @app.delete("/v1/models/{model}")
    def delete_model(model: str, request: Request):
        """ Deletes a downloaded LiteRT-LM model, unless it's loaded. """
        admin_only(request)
        obj = model_objects(request).get(model)
        if not obj or "engine" not in obj:
            raise APIError(404, f"The model '{model}' isn't a downloaded model that can be deleted",
                           code="model_not_found")
        if obj["loaded"]:
            raise APIError(409, f"{model} is loaded; pick another model first, then delete it.",
                           code="model_in_use")
        litert_models.remove(model)
        return {"id": model, "object": "model", "deleted": True}

    @app.get("/hub/models")
    def hub_models(request: Request, search: Optional[str] = None, limit: int = Query(30, ge=1, le=100), sort: str = "downloads"):
        """
        LiteRT-LM chat models on Hugging Face that can be added, most
        downloaded first, or by sort: downloads, likes, trending or recent.
        A search that's a Hub address or owner/repo gives that repo.
        """
        admin_only(request)
        try:
            return {"object": "list", "data": litert_models.search(search, limit, sort)}
        except ValueError as e:
            raise APIError(400, str(e))
        except Exception as e:
            raise APIError(502, f"Couldn't search Hugging Face: {e}", "server_error")

    @app.post("/hub/models")
    def add_hub_model(body: AddModelRequest, request: Request):
        """
        Downloads a model from Hugging Face, streaming its progress as server-sent
        events: {"type": "progress", "downloaded", "total", "rate", "left"} as it
        comes, rate in bytes a second and left in seconds once they're known,
        then {"type": "done", "id"} or {"type": "error", "message"}. repo may be
        a Hub address. A client that goes away doesn't stop the download.
        """
        admin_only(request)
        events: "queue.Queue" = queue.Queue()
        # The rate is measured over the last few seconds: (time, bytes) samples
        samples: "collections.deque" = collections.deque(maxlen=50)

        def progress(done, total):
            now = time.monotonic()
            samples.append((now, done))
            while len(samples) > 2 and now - samples[0][0] > 5:
                samples.popleft()
            event = {"type": "progress", "downloaded": done, "total": total}
            seconds, got = now - samples[0][0], done - samples[0][1]
            if seconds >= 1 and got > 0:
                event["rate"] = round(got / seconds)
                if total:
                    event["left"] = round((total - done) / event["rate"])
            events.put(event)

        def download():
            try:
                model_id = litert_models.add(body.repo, body.file, progress)
                events.put({"type": "done", "id": model_id})
            except Exception as e:
                events.put({"type": "error", "message": str(e)})

        def stream():
            threading.Thread(target=download, daemon=True).start()
            sent = 0.0
            while True:
                event = events.get()
                # A tenth of a second between progress reports is plenty
                if event["type"] == "progress" and time.monotonic() - sent < 0.1:
                    continue
                sent = time.monotonic()
                yield f"data: {json.dumps(event)}\n\n"
                if event["type"] != "progress":
                    return

        return StreamingResponse(in_threadpool(stream()), media_type="text/event-stream")

    @app.post("/v1/audio/speech")
    def speech(body: SpeechRequest, request: Request):
        e = loaded(request)
        if not e.tts:
            raise APIError(500, "No TTS engines loaded.", "server_error")
        # Any model the registry doesn't know, e.g. OpenAI's "tts-1", goes to the default engine
        engine = e.tts.get(body.model) or e.tts[e.default_tts]
        voice = body.voice or engine.get_default_voice()
        return StreamingResponse(in_threadpool(streamed(engine.generate_pcm(body.input, voice), "Speech")),
                                 media_type="audio/raw")

    @app.post("/v1/audio/transcriptions")
    def transcriptions(request: Request, file: UploadFile = File(...), model: Optional[str] = Form(None),
                       language: Optional[str] = Form(None), denoise: Optional[str] = Form(None)):
        """
        Speech to text. model "whisper" is slower than the default but more
        accurate, and hears other languages. Beyond OpenAI's fields,
        denoise=true takes the background noise out first.
        """
        e = loaded(request)
        if not e.stt:
            raise APIError(500, "No STT engines loaded.", "server_error")
        engine = e.stt.get(model) or e.stt[e.default_stt]
        speech = file.file.read()
        if str(denoise).lower() in ("true", "1", "yes"):
            from .engines.tflite import wav
            cleaner = audio.denoise_engine(e)
            try:
                speech = wav(audio.denoised(cleaner, speech), cleaner.RATE)
            except ValueError as exc:
                raise APIError(400, f"Can't read the audio: {exc}")
        try:
            return {"text": engine.transcribe(speech, language)}
        except Exception as exc:
            print(f"Transcription error: {exc}")
            raise APIError(500, str(exc), "server_error")

    @app.post("/v1/audio/translations")
    def translations(request: Request, file: UploadFile = File(...), model: Optional[str] = Form(None),
                     prompt: Optional[str] = Form(None), response_format: str = Form("json"),
                     temperature: Optional[float] = Form(None)):
        """ Speech in any language to English text. Gemma hears the audio itself. """
        e = loaded(request)
        if not e.llm:
            raise APIError(500, "No chat model is loaded; this server runs with --llm none.", "server_error")
        if response_format not in ("json", "text"):
            raise APIError(400, f"response_format {response_format!r} isn't supported; use json or text")
        audio = file.file.read()
        if not audio:
            raise APIError(400, "The audio file is empty.")
        engine = e.pick_llm(model)
        # OpenAI's prompt guides the style or carries context from earlier audio
        messages = [{"role": "system", "content": TRANSLATION_PROMPT},
                    audio_message(audio, prompt and f"Context: {prompt}")]
        try:
            text = "".join(chunk for chunk in engine.generate(messages, temperature=temperature, thinking=False)
                           if isinstance(chunk, str)).strip()
        except Exception as exc:
            raise generation_error(exc, "Translation")
        return PlainTextResponse(text) if response_format == "text" else {"text": text}

    @app.post("/v1/embeddings")
    def embeddings(body: EmbeddingRequest, request: Request):
        e = loaded(request)
        if not e.embed:
            raise APIError(500, "No embedding engines loaded.", "server_error")
        engine = e.embed.get(body.model) or next(iter(e.embed.values()))
        items = body.input if isinstance(body.input, list) else [body.input]
        if not items:
            raise APIError(400, "input must not be empty")
        for item in items:
            if isinstance(item, (int, list)):
                raise APIError(400, "Token arrays aren't supported; send text.")
            if not isinstance(item, (str, dict)):
                raise APIError(400, "Each input must be a string or a content part.")
            if item == "":
                raise APIError(400, "input must not contain empty strings")
        if body.dimensions is not None and body.dimensions not in engine.get_dimensions():
            raise APIError(400, f"dimensions must be one of {engine.get_dimensions()}")
        if body.encoding_format not in ("float", "base64"):
            raise APIError(400, "encoding_format must be float or base64")
        try:
            items = [resolve_image_part(item) if is_image_part(item) else item for item in items]
        except MediaError as exc:
            raise APIError(400, str(exc))

        data = []
        for index, item in enumerate(items):
            try:
                vector = engine.embed(item, body.dimensions)
            except ValueError as exc:
                raise APIError(400, f"input[{index}]: {exc}")
            except Exception as exc:
                print(f"Embedding error: {exc}")
                raise APIError(500, str(exc), "server_error")
            if body.encoding_format == "base64":
                # OpenAI's base64 is the vector's little-endian float32 bytes
                vector = base64.b64encode(np.asarray(vector, dtype="<f4").tobytes()).decode("ascii")
            data.append({"object": "embedding", "index": index, "embedding": vector})
        # The engines don't report token counts
        return {"object": "list", "data": data, "model": engine.get_model_id(),
                "usage": {"prompt_tokens": 0, "total_tokens": 0}}

    # The most documents one request may rerank: each is a pass of the model
    MAX_RERANK = 100

    @app.post("/v1/rerank")
    def rerank(body: RerankRequest, request: Request):
        """
        The documents in order of how well each answers the query, as
        Cohere's and Jina's rerank APIs give them: not an OpenAI endpoint.
        """
        e = loaded(request)
        if not e.rerank:
            raise APIError(500, "No reranking engines loaded.", "server_error")
        engine = e.rerank.get(body.model) or next(iter(e.rerank.values()))
        texts = [d if isinstance(d, str) else d.get("text") for d in body.documents]
        if not body.query.strip() or not texts:
            raise APIError(400, "query and documents must not be empty")
        if not all(isinstance(t, str) for t in texts):
            raise APIError(400, "Each document must be a string or an object with a text field.")
        if len(texts) > MAX_RERANK:
            raise APIError(400, f"At most {MAX_RERANK} documents can be reranked at once.")
        if body.top_n is not None and body.top_n < 1:
            raise APIError(400, "top_n must be at least 1")
        try:
            scores = engine.rank(body.query, texts, body.instruction)
        except ValueError as exc:
            raise APIError(400, str(exc))
        except Exception as exc:
            print(f"Rerank error: {exc}")
            raise APIError(500, str(exc), "server_error")
        order = sorted(range(len(texts)), key=lambda i: -scores[i])[:body.top_n]
        return {"object": "list", "model": engine.get_model_id(),
                "results": [{"index": i, "relevance_score": round(scores[i], 4),
                             **({"document": {"text": texts[i]}} if body.return_documents else {})}
                            for i in order]}

    @app.post("/v1/chat/completions")
    def chat_completions(body: ChatRequest, request: Request):
        e = loaded(request)
        if not e.llm:
            raise APIError(500, "No chat model is loaded; this server runs with --llm none.", "server_error")
        engine = e.pick_llm(body.model)

        try:
            messages = resolve_files(resolve_images(body.messages), next(iter(e.ocr.values()), None))
            tools = chat_tools(body.tools, body.tool_choice)
            thinking = thinking_option(body.reasoning_effort)
            schema = schema_option(body.response_format)
            # Built-in tools run here; the client sees only the reply they lead to
            engine = request.app.state.toolbox.llm(engine, request.app.state.toolbox.select(body.builtin_tools),
                                                   user=request.user)
        except (InputError, MediaError, ToolError) as exc:
            raise APIError(400, str(exc))
        max_tokens = body.max_completion_tokens if body.max_completion_tokens is not None else body.max_tokens
        usage = {}
        chunks = engine.generate(messages, max_tokens=max_tokens, temperature=body.temperature,
                                 top_p=body.top_p, usage=usage, **tools_option(tools), **thinking, **schema)
        # Wait for the first chunk before answering, so a failed load or bad
        # request still gets an error status instead of a broken stream
        try:
            first = next(chunks, "")
        except Exception as exc:
            raise generation_error(exc, "Chat completion")

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        model_id = engine.get_model_id()
        calls: List[ToolCall] = []
        # Each thought chunk is about a token, so they count the reasoning tokens
        thoughts = []

        def reply():
            if first:
                yield first
            yield from chunks

        def finish():
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            reason = "tool_calls" if calls else "length" if max_tokens and completion_tokens >= max_tokens \
                else "stop"
            counts = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens,
                      "completion_tokens_details": {"reasoning_tokens": min(len(thoughts), completion_tokens)}}
            # Beyond OpenAI's API: how much of the model's context the reply filled
            if "context_tokens" in usage:
                counts["context_tokens"] = usage["context_tokens"]
            return reason, counts

        def tool_call(call: ToolCall) -> Dict[str, Any]:
            return {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}

        if not body.stream:
            text = ""
            try:
                for chunk in reply():
                    if isinstance(chunk, ToolCall):
                        calls.append(chunk)
                    elif isinstance(chunk, Thought):
                        thoughts.append(chunk.text)
                    else:
                        text += chunk
            except Exception as exc:
                print(f"Chat completion error: {exc}")
                raise APIError(500, str(exc), "server_error")
            finally:
                chunks.close()
            reason, usage_body = finish()
            message = {"role": "assistant", "content": text if text or not calls else None}
            if thoughts:
                message["reasoning_content"] = "".join(thoughts)
            if calls:
                message["tool_calls"] = [tool_call(call) for call in calls]
            return {
                "id": completion_id, "object": "chat.completion", "created": created, "model": model_id,
                "choices": [{"index": 0, "message": message, "finish_reason": reason}],
                "usage": usage_body,
            }

        def event(choices, **extra) -> bytes:
            payload = {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                       "model": model_id, "choices": choices, **extra}
            return f"data: {json.dumps(payload)}\n\n".encode("utf-8")

        def delta(content) -> bytes:
            return event([{"index": 0, "delta": {"content": content}, "finish_reason": None}])

        def events():
            yield event([{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}])
            for chunk in reply():
                if isinstance(chunk, ToolCall):
                    calls.append(chunk)
                    # Each call arrives whole, so one chunk carries all of it
                    yield event([{"index": 0, "delta": {"tool_calls": [{"index": len(calls) - 1, **tool_call(chunk)}]},
                                  "finish_reason": None}])
                elif isinstance(chunk, Thought):
                    thoughts.append(chunk.text)
                    yield event([{"index": 0, "delta": {"reasoning_content": chunk.text}, "finish_reason": None}])
                else:
                    yield delta(chunk)
            reason, usage_body = finish()
            yield event([{"index": 0, "delta": {}, "finish_reason": reason}])
            if (body.stream_options or {}).get("include_usage"):
                yield event([], usage=usage_body)
            yield b"data: [DONE]\n\n"

        def stream():
            # Closing chunks, not just this generator, is what ends generation
            # when the client disconnects mid-reply
            try:
                yield from streamed(events(), "Chat stream")
            finally:
                chunks.close()

        return StreamingResponse(in_threadpool(stream()), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    def stored_response(request: Request, response_id: str) -> StoredResponse:
        stored = request.app.state.store.get_response(response_id)
        # Another user's response is as good as missing
        if not stored or stored.user != request.user.name:
            raise APIError(404, f"Response with id '{response_id}' not found.", code="not_found")
        return stored

    @app.post("/v1/responses")
    def create_response(body: ResponseRequest, request: Request):
        e = loaded(request)
        if not e.llm:
            raise APIError(500, "No chat model is loaded; this server runs with --llm none.", "server_error")
        if body.background and body.stream:
            raise APIError(400, "Streaming a background response isn't supported.")
        if body.background and not body.store:
            raise APIError(400, "A background response must be stored.")
        if body.conversation and body.previous_response_id:
            raise APIError(400, "Use conversation or previous_response_id, not both.")
        engine = e.pick_llm(body.model)
        store = request.app.state.store
        context = []
        if body.previous_response_id:
            context = stored_response(request, body.previous_response_id).chained_context()
        conversation = None
        if body.conversation:
            conversation_id = body.conversation.get("id") if isinstance(body.conversation, dict) \
                else body.conversation
            conversation = stored_conversation(request, conversation_id)
            context = list(conversation.items)
        try:
            items = input_items(body.input)
            # A saved conversation's photos, clips and documents, and any file_id input, are read from files
            files = Files(request.user)
            try:
                read = from_files(context, files) + from_files(items, files, missing="raise")
            except FileNotFoundError as missing:
                raise APIError(400, f"No file {missing} was found.", code="file_not_found")
            messages = resolve_files(resolve_images(to_messages(body.instructions, read)),
                                     next(iter(e.ocr.values()), None))
            tools = chat_tools(body.tools, body.tool_choice, flat=True)
            thinking = thinking_option((body.reasoning or {}).get("effort"))
            schema = schema_option((body.text or {}).get("format"), flat=True)
            # Built-in tools run here, and are reported as mcp_call items
            toolbox = request.app.state.toolbox
            engine = toolbox.llm(engine, toolbox.select(body.builtin_tools), report=True, user=request.user)
        except (InputError, MediaError, ToolError) as exc:
            raise APIError(400, str(exc))

        builder = ResponseBuilder(body, engine.get_model_id())
        stored = StoredResponse(builder.object(with_output=False), items, context + items, request.user.name)

        def save(response: Dict[str, Any]):
            stored.response = response
            if body.store:
                store.put_response(stored)

        def finished(response: Dict[str, Any]):
            """ Keeps the finished response, and adds this turn to its conversation. """
            save(response)
            if conversation and response["status"] in ("completed", "incomplete"):
                store.add_items(conversation, items + response["output"])

        usage = {}
        chunks = engine.generate(messages, max_tokens=body.max_output_tokens, temperature=body.temperature,
                                 top_p=body.top_p, usage=usage, **tools_option(tools), **thinking, **schema)

        if body.background:
            builder.status = "queued"
            # The worker replaces stored.response as it goes, so answer with this snapshot
            queued = builder.object(with_output=False)
            save(queued)

            def run():
                try:
                    builder.status = "in_progress"
                    stored.response = builder.object(with_output=False)
                    for chunk in chunks:
                        if stored.cancel.is_set():
                            break
                        builder.add(chunk)
                    if stored.cancel.is_set():
                        builder.status = "cancelled"
                    else:
                        builder.finish(usage)
                    finished(builder.object())
                except Exception as exc:
                    print(f"\n[!] Background response error: {exc}")
                    stored.response = {**builder.object(), "status": "failed",
                                       "error": {"code": "server_error", "message": str(exc)}}
                finally:
                    chunks.close()

            threading.Thread(target=run, daemon=True).start()
            return queued

        # As in chat completions: wait for the first chunk, so a failure is still an error status
        try:
            first = next(chunks, "")
        except Exception as exc:
            raise generation_error(exc, "Response")

        def reply() -> Iterator[Union[str, ToolCall]]:
            if first:
                yield first
            yield from chunks

        if not body.stream:
            try:
                for chunk in reply():
                    builder.add(chunk)
            except Exception as exc:
                print(f"Response error: {exc}")
                raise APIError(500, str(exc), "server_error")
            finally:
                chunks.close()
            builder.finish(usage)
            finished(builder.object())
            return builder.object()

        def stream():
            try:
                yield from streamed(builder.events(reply(), usage, on_done=finished), "Response stream")
            finally:
                chunks.close()

        return StreamingResponse(in_threadpool(stream()), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    @app.get("/v1/responses/{response_id}")
    def get_response(response_id: str, request: Request):
        return stored_response(request, response_id).response

    @app.delete("/v1/responses/{response_id}")
    def delete_response(response_id: str, request: Request):
        stored = stored_response(request, response_id)
        stored.cancel.set()
        request.app.state.store.delete_response(response_id)
        return {"id": response_id, "object": "response", "deleted": True}

    @app.post("/v1/responses/{response_id}/cancel")
    def cancel_response(response_id: str, request: Request):
        stored = stored_response(request, response_id)
        if not stored.response.get("background"):
            raise APIError(400, "Only background responses can be cancelled.")
        stored.cancel.set()
        # The worker notices between chunks; give it a moment so the reply shows the outcome
        for _ in range(50):
            if stored.response["status"] not in ("queued", "in_progress"):
                break
            time.sleep(0.02)
        return stored.response

    @app.get("/v1/responses/{response_id}/input_items")
    def response_input_items(response_id: str, request: Request, after: Optional[str] = None,
                             limit: int = Query(20, ge=1, le=100), order: str = Query("desc", pattern="^(asc|desc)$")):
        return page(stored_response(request, response_id).input_items, after, limit, order)

    def stored_conversation(request: Request, conversation_id: str) -> Conversation:
        conversation = request.app.state.store.get_conversation(conversation_id)
        if not conversation or conversation.user != request.user.name:
            raise APIError(404, f"Conversation with id '{conversation_id}' not found.", code="not_found")
        return conversation

    def new_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            return input_items(items)
        except InputError as exc:
            raise APIError(400, str(exc))

    @app.post("/v1/conversations")
    def create_conversation(body: ConversationRequest, request: Request):
        if len(body.items or []) > MAX_ITEMS_PER_ADD:
            raise APIError(400, f"A conversation can start with at most {MAX_ITEMS_PER_ADD} items.")
        conversation = Conversation(new_id("conv"), body.metadata, request.user.name)
        conversation.items = new_items(body.items or [])
        request.app.state.store.put_conversation(conversation)
        return conversation.object

    @app.get("/v1/conversations")
    def list_conversations(request: Request, after: Optional[str] = None, limit: int = Query(20, ge=1, le=100)):
        """
        The user's conversations, the last changed first, each with when, as
        updated_at. OpenAI's API has no such list; the chat page's comes from it.
        """
        found = request.app.state.store.list_conversations(request.user.name)
        return page([{**c.object, "updated_at": c.updated_at} for c in found], after, limit, "asc")

    @app.get("/v1/conversations/{conversation_id}")
    def get_conversation(conversation_id: str, request: Request):
        return stored_conversation(request, conversation_id).object

    @app.post("/v1/conversations/{conversation_id}")
    def update_conversation(conversation_id: str, body: ConversationRequest, request: Request):
        conversation = stored_conversation(request, conversation_id)
        request.app.state.store.set_metadata(conversation, body.metadata or {})
        return conversation.object

    @app.delete("/v1/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str, request: Request):
        stored_conversation(request, conversation_id)
        request.app.state.store.delete_conversation(conversation_id)
        return {"id": conversation_id, "object": "conversation.deleted", "deleted": True}

    @app.get("/v1/conversations/{conversation_id}/items")
    def list_conversation_items(conversation_id: str, request: Request, after: Optional[str] = None,
                                limit: int = Query(20, ge=1, le=100),
                                order: str = Query("desc", pattern="^(asc|desc)$")):
        return page(stored_conversation(request, conversation_id).items, after, limit, order)

    @app.post("/v1/conversations/{conversation_id}/items")
    def add_conversation_items(conversation_id: str, body: ConversationItemsRequest, request: Request):
        conversation = stored_conversation(request, conversation_id)
        if not 1 <= len(body.items) <= MAX_ITEMS_PER_ADD:
            raise APIError(400, f"Add between 1 and {MAX_ITEMS_PER_ADD} items at a time.")
        items = new_items(body.items)
        request.app.state.store.add_items(conversation, items)
        return page(items, None, len(items), "asc")

    @app.get("/v1/conversations/{conversation_id}/items/{item_id}")
    def get_conversation_item(conversation_id: str, item_id: str, request: Request):
        item = stored_conversation(request, conversation_id).item(item_id)
        if not item:
            raise APIError(404, f"Item with id '{item_id}' not found.", code="not_found")
        return item

    @app.delete("/v1/conversations/{conversation_id}/items/{item_id}")
    def delete_conversation_item(conversation_id: str, item_id: str, request: Request):
        conversation = stored_conversation(request, conversation_id)
        if not conversation.item(item_id):
            raise APIError(404, f"Item with id '{item_id}' not found.", code="not_found")
        request.app.state.store.delete_item(conversation, item_id)
        return conversation.object

    # --- Files: the user's notes, and their chats' media (see files.py) ---

    def user_file(request: Request, file_id: str):
        found = Files(request.user).find(file_id)
        if not found:
            raise APIError(404, f"No such File object: {file_id}", code="not_found")
        return found

    @app.post("/v1/files")
    def upload_file(request: Request, file: UploadFile = File(...), purpose: str = Form("user_data")):
        """ Adds a file: to the user's notes, which search_notes searches, or of purpose vision, to their media. """
        data = file.file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise APIError(400, f"A file can be at most {MAX_UPLOAD // 1048576} MB.")
        return Files(request.user).add(file.filename or "file", data, purpose)

    @app.get("/v1/files")
    def list_files(request: Request, purpose: Optional[str] = None, after: Optional[str] = None,
                   limit: int = Query(10000, ge=1, le=10000), order: str = Query("desc", pattern="^(asc|desc)$")):
        return page(Files(request.user).list(purpose), after, limit, order)

    @app.get("/v1/files/{file_id}")
    def get_file(file_id: str, request: Request):
        return user_file(request, file_id)[0]

    @app.get("/v1/files/{file_id}/content")
    def file_content(file_id: str, request: Request):
        found, path = user_file(request, file_id)
        with open(path, "rb") as f:
            return Response(f.read(), media_type=Files(request.user).media_type(path))

    @app.delete("/v1/files/{file_id}")
    def delete_file(file_id: str, request: Request):
        Files(request.user).delete(user_file(request, file_id)[0]["id"])
        return {"id": file_id, "object": "file", "deleted": True}

    @app.websocket("/v1/realtime")
    async def realtime_session(websocket: WebSocket):
        # The model query parameter picks the LLM, as a request's model does. A browser that
        # sends its key as a subprotocol needs one it offered chosen, as OpenAI chooses "realtime"
        offered = websocket.scope.get("subprotocols", [])
        await websocket.accept(subprotocol="realtime" if "realtime" in offered else None)
        state = websocket.app.state
        await realtime.serve(websocket, state.engines, state.vad, state.toolbox,
                             websocket.query_params.get("model"))

    @app.post("/v1/voice")
    def voice_turn(request: Request, file: UploadFile = File(...), voice: Optional[str] = Form(None),
                   tts_model: Optional[str] = Form(None), stt_model: str = Form("gemma"),
                   system_prompt: Optional[str] = Form(None), max_tokens: int = Form(256),
                   builtin_tools: Optional[str] = Form(None), model: Optional[str] = Form(None)):
        """
        Spoken conversation turn: audio in, raw 16-bit 24kHz mono PCM of the reply out.
        stt_model "gemma" (the default) hands the audio straight to Gemma; an STT
        engine id, such as "moonshine", transcribes it first. The transcript, if
        any, comes back percent-encoded in the X-Transcript header.
        builtin_tools names the built-in tools to offer, comma-separated, or is
        "false" for none. model picks the LLM, as in chat completions.
        """
        e = loaded(request)
        if not e.llm or not e.tts:
            raise APIError(500, "Voice needs an LLM engine and a TTS engine loaded.", "server_error")
        try:
            toolbox = request.app.state.toolbox
            llm = toolbox.llm(e.pick_llm(model), toolbox.select(builtin_tools), user=request.user)
        except ToolError as exc:
            raise APIError(400, str(exc))
        tts = e.tts.get(tts_model) or e.tts[e.default_tts]
        voice = voice or tts.get_default_voice()
        audio = file.file.read()
        if not audio:
            raise APIError(400, "The audio file is empty.")

        start = time.time()
        transcript = ""
        if stt_model == "gemma":
            user_message = audio_message(audio)
        else:
            stt = e.stt.get(stt_model)
            if not stt:
                raise APIError(400, f"Unknown stt_model {stt_model!r}")
            try:
                transcript = stt.transcribe(audio)
            except Exception as exc:
                print(f"Transcription error: {exc}")
                raise APIError(500, str(exc), "server_error")
            user_message = {"role": "user", "content": transcript}
        stt_done = time.time()
        messages = [{"role": "system", "content": system_prompt or VOICE_SYSTEM_PROMPT}, user_message]

        def pcm():
            first_sentence = first_audio = None
            reply = []
            try:
                for kind, value in spoken_reply(llm, tts, voice, messages, max_tokens):
                    if kind == "sentence":
                        first_sentence = first_sentence or time.time()
                        reply.append(value)
                    else:
                        first_audio = first_audio or time.time()
                        yield value
            finally:
                timing = lambda t: f"{t - start:.2f}s" if t else "n/a"
                print(f"[Voice] heard: {transcript!r} | said: {' '.join(reply)!r}")
                print(f"[Voice] stt {timing(stt_done)}, first sentence {timing(first_sentence)}, "
                      f"first audio {timing(first_audio)}, done {timing(time.time())}")

        # Headers are latin-1, so the transcript travels percent-encoded
        return StreamingResponse(in_threadpool(streamed(pcm(), "Voice")), media_type="audio/raw",
                                 headers={"X-Transcript": urllib.parse.quote(transcript)})

    return app
