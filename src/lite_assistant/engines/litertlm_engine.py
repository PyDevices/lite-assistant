"""
LiteRT-LM LLM engine implementation plugin.
"""
import base64
import json
import os
import re
import time
import random
import dataclasses
import threading
from typing import Iterator, Dict, Any, List, Optional, Union
from .llm_base import LLMEngine, Thought, ToolCall
from ..media import decode_data_url, image_url
from .. import litert_models, model_facts
from ..paths import cache_dir, cached_hf_file

# LiteRT-LM compiles each model for its backend on first load and keeps the
# result here; it can be deleted and is rebuilt as needed
CACHE_DIR = cache_dir("litertlm")

# Gemma 4's two small models, which setup downloads. "e2b" and "e4b" name
# them wherever a model can be named; any other downloaded LiteRT-LM chat
# model is named by its id (see litert_models).
MODELS = {
    "e2b": {"repo": "litert-community/gemma-4-E2B-it-litert-lm", "file": "gemma-4-E2B-it.litertlm",
            "size": 2588147712},
    "e4b": {"repo": "litert-community/gemma-4-E4B-it-litert-lm", "file": "gemma-4-E4B-it.litertlm",
            "size": 3659530240},
}
# The model each backend loads unless LITERTLM_MODEL says otherwise: the
# larger E4B on a GPU, E2B on the CPU, where E4B decodes at half the speed
BACKEND_MODELS = {"gpu": "e4b", "cpu": "e2b"}
# Speculative decoding, for models whose file carries a drafter. On the CPU
# it showed no clear gain with Gemma (docs/how-it-works/benchmarks.md), so it stays off there.
SPECULATIVE_DECODING = {"gpu": True, "cpu": False}

# LITERTLM_BACKEND picks "gpu", "cpu", or "auto" (GPU, falling back to CPU).
# LITERTLM_MODEL picks the model to start with, whichever backend loads:
# "e2b", "e4b", or any downloaded model's id.
# LITERTLM_MAX_TOKENS is the context length the KV cache is sized for.
# LITERTLM_THREADS sets the CPU backend's thread count. LiteRT-LM's own
# default is fastest on Linux; on Windows, 2-3 threads decode faster
# (docs/how-it-works/benchmarks.md).
BACKEND = os.environ.get("LITERTLM_BACKEND", "auto").lower()
MODEL = os.environ.get("LITERTLM_MODEL", "").lower()
MAX_NUM_TOKENS = int(os.environ.get("LITERTLM_MAX_TOKENS", "4096"))
THREADS = int(os.environ["LITERTLM_THREADS"]) if os.environ.get("LITERTLM_THREADS") else None


def _backends_to_try() -> List[str]:
    return {"gpu": ["gpu"], "cpu": ["cpu"]}.get(BACKEND, ["gpu", "cpu"])


def _model_config(backend: str) -> Dict[str, str]:
    """ The Gemma model setup downloads for backend. """
    return MODELS[MODEL if MODEL in MODELS else BACKEND_MODELS[backend]]


def _gemma_id(key: str) -> str:
    """ The id of one of MODELS, e.g. "gemma-4-E2B-it" for "e2b". """
    return MODELS[key]["file"][:-len(litert_models.SUFFIX)]


def _find_model(name: Optional[str], models: Dict[str, str]) -> Optional[str]:
    """ The id that name gives, ignoring case: one of models' ids, or "e2b" or "e4b". None if neither. """
    name = (name or "").lower()
    if name in MODELS:
        return _gemma_id(name)
    return next((model_id for model_id in models if model_id.lower() == name), None)


def _unsupported_input(messages: List[Dict[str, Any]], modalities: List[str]) -> Optional[str]:
    """ "images" or "audio" if a message carries one the model can't take. """
    for message in messages:
        for part in message.get("content") if isinstance(message.get("content"), list) else []:
            if part.get("type") == "image_url" and "image" not in modalities:
                return "images"
            if part.get("type") == "input_audio" and "audio" not in modalities:
                return "audio"
    return None


def _context(facts: Dict[str, Any]) -> int:
    """ The context length a model loads with: LITERTLM_MAX_TOKENS, but no more than it was built for. """
    return min(MAX_NUM_TOKENS, facts.get("context") or MAX_NUM_TOKENS)


def _start_model(backend: str, models: Dict[str, str]) -> str:
    """
    The model to load at startup on backend: LITERTLM_MODEL's, or that
    backend's Gemma, or if setup downloaded only the other, that one, or
    else any model that's downloaded.
    """
    chosen = _find_model(MODEL, models)
    if chosen:
        return chosen
    gemmas = [_gemma_id(BACKEND_MODELS[backend])] + [_gemma_id(key) for key in MODELS]
    return next((model_id for model_id in gemmas if model_id in models), next(iter(models), gemmas[0]))


def _parse_json(text: Any) -> Any:
    """ text as the JSON value it holds, or as it is if it isn't JSON. """
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def _client_tool(definition: Dict[str, Any]):
    """ A tool the client runs: Gemma sees its definition, and the call goes back to the client. """
    import litert_lm

    class ClientTool(litert_lm.interfaces.Tool):
        def get_tool_description(self):
            return {"type": "function", "function": definition["function"]}

        def execute(self, param):
            raise RuntimeError("Client tools are run by the client.")

    return ClientTool()


# Some models write their tool calls as text, which LiteRT-LM leaves in the
# reply: Qwen3's JSON, <tool_call>{"name": ..., "arguments": {...}}</tool_call>,
# and Qwen3.5's tags, <tool_call><function=name><parameter=p>value</parameter></function></tool_call>
CALL_START, CALL_END = "<tool_call>", "</tool_call>"
# Gemma's own call tag. LiteRT-LM turns a whole call into a tool call, but
# one cut off by max_tokens is left in the text, which the reply drops.
GEMMA_CALL_START = "<|tool_call>"
# What a reply that said nothing after a tool's result is asked again with.
# "(Now answer my question.)" made Gemma ask what the question was.
ANSWER_NUDGE = {"role": "user", "content": "(Answer me using the tool results.)"}


class _TextToolCalls:
    """
    Picks the tool calls out of a model's streamed text. feed() takes each
    chunk and yields the text and calls in it, holding back what may be a
    call until it's whole; flush() yields what's left at the end.
    """
    def __init__(self, tools: List[Dict[str, Any]]):
        # Each tool's parameter types: tagged values are text, which only strings keep as they are
        self.types = {t["function"]["name"]: {name: schema.get("type") for name, schema in
                                              t["function"].get("parameters", {}).get("properties", {}).items()}
                      for t in tools}
        self.held = ""

    def feed(self, text: str) -> Iterator[Union[str, ToolCall]]:
        self.held += text
        while True:
            start = self.held.find(CALL_START)
            if start < 0:
                # The end may be the start of a call's tag
                keep = next((n for n in range(len(CALL_START) - 1, 0, -1) if self.held.endswith(CALL_START[:n])), 0)
                text, self.held = self.held[:len(self.held) - keep], self.held[len(self.held) - keep:]
                if text:
                    yield text
                return
            if start:
                yield self.held[:start]
                self.held = self.held[start:]
            end = self.held.find(CALL_END)
            if end < 0:
                return
            body, self.held = self.held[len(CALL_START):end], self.held[end + len(CALL_END):]
            yield self.parse(body) or CALL_START + body + CALL_END

    def flush(self) -> Iterator[Union[str, ToolCall]]:
        held, self.held = self.held, ""
        # A model may stop without closing its last call
        call = self.parse(held[len(CALL_START):]) if held.startswith(CALL_START) else None
        if call or held:
            yield call or held

    def parse(self, body: str) -> Optional[ToolCall]:
        """ A call in either form, or None if it's neither. """
        tagged = re.match(r"\s*<function=([^>\s]+)>(.*?)(?:</function>|$)", body, re.S)
        if tagged:
            name, types = tagged.group(1), self.types.get(tagged.group(1), {})
            arguments = {}
            for parameter, value in re.findall(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", tagged.group(2), re.S):
                arguments[parameter] = value if types.get(parameter, "string") == "string" else _parse_json(value.strip())
            return ToolCall(name, json.dumps(arguments))
        found = _parse_json(body.strip())
        if isinstance(found, dict) and isinstance(found.get("name"), str):
            arguments = found.get("arguments", found.get("parameters", {}))
            arguments = _parse_json(arguments) if isinstance(arguments, str) else arguments
            return ToolCall(found["name"], json.dumps(arguments if isinstance(arguments, dict) else {}))
        return None


class _ModelChoice:
    """
    The LiteRT-LM engine on one of its models, switching to it as each reply
    starts. It isn't an LLMEngine subclass, so engine discovery passes it by.
    """
    def __init__(self, engine: "LiteRTLMEngine", model_id: str, path: str):
        self.engine, self.model_id, self.path = engine, model_id, path

    @property
    def backend(self) -> Optional[str]:
        return self.engine.backend

    def get_model_id(self) -> str:
        return self.model_id

    def get_modalities(self) -> List[str]:
        return (litert_models.facts(self.path) or {}).get("modalities", ["text"])

    def generate(self, messages, **options):
        return self.engine.generate(messages, model=self.model_id, **options)


class LiteRTLMEngine(LLMEngine):
    """
    LiteRT-LM Engine Implementation.
    Runs Google's Gemma models on-device, on the GPU when one is usable, and
    can switch to any other LiteRT-LM chat model that's downloaded.
    """
    def __init__(self):
        self.engine = None
        self.backend = None
        self.model_id = None
        self.model_path = None
        self.modalities = []
        # One generation at a time: the engine shares its weights and
        # accelerator between conversations.
        self._lock = threading.Lock()
        self._initialize_litertlm()

    def _initialize_litertlm(self):
        try:
            import litert_lm
        except ImportError:
            print("[LiteRT-LM] litert-lm package is not installed. Skipping LiteRTLMEngine.")
            return
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

        models = litert_models.downloaded()
        for backend in _backends_to_try():
            model_id = _start_model(backend, models)
            model_path = models.get(model_id)
            if not model_path:
                print(f"[LiteRT-LM] {model_id} for the {backend.upper()} backend "
                      "isn't downloaded; run 'lite-assistant setup'.")
                continue
            # A GPU failure can kill the process instead of raising, which the
            # fallback below can't catch, so try the GPU in a child process first
            if backend == "gpu" and not self._gpu_works(model_path):
                continue
            try:
                self._load(litert_lm, backend, model_id, model_path)
                return
            except Exception as e:
                print(f"[LiteRT-LM] {backend.upper()} backend failed: {e}")
                if self.engine:
                    self.engine.close()
                self.engine = None
        print("[LiteRT-LM] No model could be loaded. LiteRTLMEngine will be disabled.")

    def _load(self, litert_lm, backend: str, model_id: str, model_path: str):
        print(f"[LiteRT-LM] Loading {model_id} on the {backend.upper()} backend...")
        start = time.time()
        facts = litert_models.facts(model_path) or {"modalities": ["text"], "context": 0, "speculative": False}
        # LiteRT-LM refuses a cache directory that doesn't exist yet
        os.makedirs(CACHE_DIR, exist_ok=True)
        # The audio and vision encoders, for input_audio and image_url message
        # parts, of a model that has them. Each loads the first time a request
        # needs it; Gemma's files only support them on the CPU.
        encoders = {}
        if "audio" in facts["modalities"]:
            encoders["audio_backend"] = litert_lm.Backend.CPU()
        if "image" in facts["modalities"]:
            encoders["vision_backend"] = litert_lm.Backend.CPU()
        self.engine = litert_lm.Engine(
            model_path,
            backend=litert_lm.Backend.GPU() if backend == "gpu" else litert_lm.Backend.CPU(thread_count=THREADS),
            max_num_tokens=_context(facts),
            cache_dir=CACHE_DIR,
            enable_speculative_decoding=SPECULATIVE_DECODING[backend] and facts["speculative"],
            enable_benchmark=True,
            **encoders,
        )
        self.backend = backend
        self.model_id = model_id
        self.model_path = model_path
        self.modalities = facts["modalities"]
        with litert_lm.ModelInfo(model_path) as model_info:
            stored = model_info.llm.default_sampler_params
        # Fields the model file leaves unset (the Gemma 4 files set none) come
        # from the model's makers' recommendations
        self.default_sampler = litert_lm.SamplerConfig(**model_facts.sampler(model_id, {
            "top_k": stored.top_k, "top_p": stored.top_p, "temperature": stored.temperature}))
        # A first generation compiles kernels and fills caches; do it now so the
        # first real request doesn't pay for it. A backend that loads but can't
        # run also fails here, while there's still a fallback.
        for _ in self._reply([{"role": "user", "content": "Hi"}], max_tokens=1):
            pass
        print(f"[LiteRT-LM] Model loaded and warm in {time.time() - start:.1f}s.")

    def _switch(self, model_id: str):
        """
        Swaps the loaded model for model_id on the same backend. The machine
        may not hold both, so the old one goes first; if the new one fails to
        load, the old one is loaded again. Called with the lock held.
        """
        import litert_lm
        models = litert_models.downloaded()
        if model_id not in models:
            raise LookupError(f"{model_id} isn't downloaded any more.")
        old, old_path = self.model_id, self.model_path
        print(f"[LiteRT-LM] Switching from {old} to {model_id}...")
        self.engine.close()
        self.engine = None
        try:
            self._load(litert_lm, self.backend, model_id, models[model_id])
        except Exception as e:
            print(f"[LiteRT-LM] {model_id} failed to load: {e}")
            if self.engine:
                self.engine.close()
                self.engine = None
            self._load(litert_lm, self.backend, old, old_path)
            raise

    def get_model_id(self) -> str:
        return self.model_id

    def get_modalities(self) -> List[str]:
        return self.modalities

    def get_models(self) -> Dict[str, Dict[str, Any]]:
        # model_id stays set while a switch has no engine loaded
        if not self.model_id:
            return {}
        models = {}
        for model_id, path in litert_models.downloaded().items():
            facts = litert_models.facts(path) or {}
            models[model_id] = {"modalities": facts.get("modalities", ["text"]), "size": os.path.getsize(path),
                                "context": _context(facts), **model_facts.abilities(model_id, facts)}
        return models

    def use(self, model: Optional[str]) -> Optional["_ModelChoice"]:
        if not self.model_id:
            return None
        models = litert_models.downloaded()
        model_id = _find_model(model, models)
        if model_id is None:
            return None
        if model_id not in models:
            key = (model or "").lower()
            raise LookupError(f"{model_id} isn't downloaded; set LITERTLM_MODEL={key} "
                              "and run 'lite-assistant setup'.")
        return _ModelChoice(self, model_id, models[model_id])

    @staticmethod
    def _to_litert_message(message: Dict[str, Any]) -> Dict[str, Any]:
        content = message.get("content") or ""
        if isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type") == "text":
                    parts.append({"type": "text", "text": part.get("text", "")})
                elif part.get("type") == "input_audio":
                    # OpenAI sends base64 audio; LiteRT-LM takes it as a blob
                    parts.append({"type": "audio", "blob": part["input_audio"]["data"]})
                elif part.get("type") == "image_url":
                    # The app has already turned any http(s) URL into a data: URL
                    _, image = decode_data_url(image_url(part))
                    parts.append({"type": "image", "blob": base64.b64encode(image).decode("ascii")})
            content = parts
        turn = {"role": message.get("role", "user"), "content": content}
        if message.get("tool_calls"):
            # LiteRT-LM takes a call's arguments as an object, not OpenAI's JSON string
            turn["tool_calls"] = [
                {"type": "function", "function": {"name": call["function"]["name"],
                                                  "arguments": _parse_json(call["function"].get("arguments") or "{}")}}
                for call in message["tool_calls"]]
            if not content:
                del turn["content"]
        return turn

    @classmethod
    def _to_litert_turns(cls, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        The messages as LiteRT-LM turns. OpenAI's "tool" messages, one per
        result, become one tool turn of tool_responses named after their calls.
        """
        names = {}
        turns = []
        for message in messages:
            if message.get("role") != "tool":
                turns.append(cls._to_litert_message(message))
                for call in message.get("tool_calls") or []:
                    names[call.get("id")] = call["function"]["name"]
                continue
            result = message.get("content") or ""
            if isinstance(result, list):
                result = "".join(part.get("text", "") for part in result if isinstance(part, dict))
            name = names.get(message.get("tool_call_id"), message.get("name", ""))
            response = {"type": "tool_response", "name": name, "response": _parse_json(result)}
            if turns and turns[-1]["role"] == "tool":
                turns[-1]["content"].append(response)
            else:
                turns.append({"role": "tool", "content": [response]})
        return turns

    def generate(self, messages: List[Dict[str, Any]], max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None, top_p: Optional[float] = None,
                 usage: Optional[Dict[str, int]] = None,
                 tools: Optional[List[Dict[str, Any]]] = None, thinking: Union[bool, int, None] = None,
                 response_schema: Optional[Dict[str, Any]] = None,
                 model: Optional[str] = None) -> Iterator[Any]:
        """ model is the id of a model to switch to first, if another is loaded; use() sets it. """
        if not messages:
            raise ValueError("messages must not be empty")

        overrides = {name: value for name, value in (("temperature", temperature), ("top_p", top_p))
                     if value is not None}
        # LiteRT-LM fills any field left unset in a SamplerConfig with its own
        # fallback rather than the model's (top_k=1, which is greedy and ignores
        # the temperature; seed=0, which repeats every reply), so pass every
        # field.
        # A request that comes while the model is being switched waits here for it
        with self._lock:
            if not self.engine:
                raise RuntimeError("LiteRT-LM engine is not available.")
            if model and model != self.model_id:
                self._switch(model)
            unsupported = _unsupported_input(messages, self.modalities)
            if unsupported:
                raise ValueError(f"{self.model_id} can't take {unsupported}; pick a model that can.")
            sampler_config = None
            if overrides:
                sampler_config = dataclasses.replace(self.default_sampler, seed=random.randrange(2**31), **overrides)
            said = False
            for piece in self._reply(messages, max_tokens, sampler_config, usage, tools, thinking, response_schema):
                said = True
                yield piece
            # Given a tool's result, Gemma sometimes ends its turn without a
            # word ("What's today's date?" after get_time); asked once more, it answers
            if not said and messages[-1].get("role") == "tool":
                print("[LiteRT-LM] Empty reply to a tool's result; asking again.")
                first = dict(usage or {})
                yield from self._reply(messages + [ANSWER_NUDGE], max_tokens, sampler_config, usage, tools, thinking,
                                       response_schema)
                if usage is not None:
                    for key in ("prompt_tokens", "completion_tokens"):
                        usage[key] = usage.get(key, 0) + first.get(key, 0)

    def _reply(self, messages: List[Dict[str, Any]], max_tokens: Optional[int] = None,
               sampler_config=None, usage: Optional[Dict[str, int]] = None,
               tools: Optional[List[Dict[str, Any]]] = None,
               thinking: Union[bool, int, None] = None,
               response_schema: Optional[Dict[str, Any]] = None) -> Iterator[Any]:
        """ generate's work, with the lock held. """
        import litert_lm
        turns = self._to_litert_turns(messages)
        options, send_options = {}, {}
        if response_schema is not None:
            # llguidance masks every token that would break the schema
            options["constrained_decoding_config"] = litert_lm.interfaces.ConstrainedDecodingConfig(
                enable=True, provider=litert_lm._ffi.LiteRtLmConstraintProviderType.LL_GUIDANCE)
            send_options["response_format"] = litert_lm.interfaces.ResponseFormat.json(response_schema)
        if thinking is not None:
            # A number is a token budget for the thinking
            budget = -1 if isinstance(thinking, bool) else thinking
            options["thinking_config"] = litert_lm.ThinkingConfig(enable_thinking=bool(thinking),
                                                                   thinking_token_budget=budget)
        # A fresh conversation per request: OpenAI clients resend the whole
        # history, so the earlier turns become its preface.
        # Tool calls come back to the caller rather than LiteRT-LM running them
        conv = self.engine.create_conversation(messages=turns[:-1], sampler_config=sampler_config,
                                               tools=[_client_tool(t) for t in tools or []],
                                               automatic_tool_calling=False, **options)
        started = cut = False
        written = _TextToolCalls(tools) if tools else None

        def reply(pieces):
            nonlocal started, cut
            for piece in pieces:
                if cut:
                    continue
                if isinstance(piece, str) and GEMMA_CALL_START in piece:
                    piece, _, call = piece.partition(GEMMA_CALL_START)
                    print(f"[LiteRT-LM] Dropped a tool call cut off by max_tokens: {GEMMA_CALL_START}{call[:80]}")
                    cut = True
                # Some models, such as Qwen3, start the reply after their
                # thinking with blank lines
                if isinstance(piece, str) and not started:
                    piece = piece.lstrip()
                    started = bool(piece)
                if piece:
                    yield piece

        try:
            for chunk in conv.send_message_async(turns[-1], max_output_tokens=max_tokens, **send_options):
                # A model that reasons writes its thoughts on a channel of their own
                thought = chunk.channels.get("thought")
                if thought:
                    yield Thought(thought)
                text = str(chunk)
                yield from reply(written.feed(text) if written else [text])
                # Each call arrives whole, in a chunk of its own
                for call in chunk.tool_calls:
                    yield ToolCall(call.name, json.dumps(call.arguments))
            if written:
                yield from reply(written.flush())
            if usage is not None:
                info = conv.get_benchmark_info()
                usage["prompt_tokens"] = info.last_prefill_token_count
                usage["completion_tokens"] = info.last_decode_token_count
                # How much of the context this reply's conversation filled
                usage["context_tokens"] = info.last_prefill_token_count + info.last_decode_token_count
        finally:
            # Stops decoding if the client went away mid-reply.
            conv.cancel_process()
            conv.close()

    @classmethod
    def setup(cls):
        # The CPU model is the fallback, so it is always fetched. The GPU model
        # is only worth its download if this machine's GPU actually works.
        backends = _backends_to_try()
        cpu_model = cls._download("cpu") if "cpu" in backends else None
        if "gpu" in backends and (BACKEND == "gpu" or cls._gpu_works(cpu_model)):
            cls._download("gpu")

    @classmethod
    def _gpu_works(cls, model_path: str) -> bool:
        """
        Runs one token of model_path on the GPU backend in a child process, so
        a broken driver that crashes the process (WSL's does) can't take the
        server with it. A software GPU such as llvmpipe counts as not working:
        it is slower than the CPU backend.
        """
        import subprocess
        import sys
        # The probe is expected to crash on some machines, so it marks itself
        # non-dumpable first: WSL saves every core dump (about 3 GB here) to
        # the Windows drive, and ulimit -c doesn't stop that. It shares the
        # engine's cache directory: without one, LiteRT-LM writes a GPU weight
        # cache (2.2 GB for E4B) next to the model, and the real load can't reuse it
        probe = (
            "import os, sys\n"
            "if sys.platform == 'linux':\n"
            "    import ctypes\n"
            "    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE\n"
            "import litert_lm\n"
            "litert_lm.set_min_log_severity(litert_lm.LogSeverity.INFO)\n"
            "os.makedirs(sys.argv[2], exist_ok=True)\n"
            "engine = litert_lm.Engine(sys.argv[1], backend=litert_lm.Backend.GPU(), max_num_tokens=256,\n"
            "                          cache_dir=sys.argv[2])\n"
            "conv = engine.create_conversation()\n"
            "for _ in conv.send_message_async('Hi', max_output_tokens=1): pass\n"
        )
        print("[LiteRT-LM] Checking whether the GPU backend works...")
        try:
            result = subprocess.run([sys.executable, "-c", probe, model_path, CACHE_DIR],
                                    capture_output=True, timeout=300)
        except subprocess.TimeoutExpired:
            print("[LiteRT-LM] GPU check timed out; using the CPU.")
            return False
        log = result.stderr.decode("utf-8", errors="replace")
        # WebGPU logs the adapter it picked, e.g. "Selected adapter: llvmpipe
        # (...), arch=software, ..., adapterType=CPU / Software"
        adapter = re.search(r"Selected adapter: (.*?), arch=", log)
        if result.returncode != 0:
            print(f"[LiteRT-LM] GPU backend failed (exit code {result.returncode}); using the CPU.")
            return False
        if "adapterType=CPU" in log:
            print(f"[LiteRT-LM] Only a software GPU is available ({adapter.group(1) if adapter else 'unknown'}); using the CPU.")
            return False
        print(f"[LiteRT-LM] GPU backend works{f' on {adapter.group(1)}' if adapter else ''}.")
        return True

    @classmethod
    def setup_model(cls, key: str) -> str:
        """ Downloads one of MODELS, "e2b" or "e4b", unless it's cached already, and returns its path. """
        return cls._fetch(MODELS[key])

    @classmethod
    def _download(cls, backend: str) -> str:
        """ Downloads the backend's model unless it's cached already, and returns its path. """
        return cls._fetch(_model_config(backend))

    @staticmethod
    def _fetch(config: Dict[str, Any]) -> str:
        from huggingface_hub import hf_hub_download
        path = cached_hf_file(config["repo"], config["file"])
        if not path:
            print(f"[LiteRT-LM] Downloading {config['file']} from {config['repo']}...")
            path = hf_hub_download(config["repo"], config["file"])
        return path
