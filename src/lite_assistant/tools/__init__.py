"""
Built-in tools: plain Python functions the server runs itself when the
model calls them, unlike function tools, which go back to the client. Each
function's signature and docstring are its tool definition.

More come from tool folders (~/.config/lite-assistant/tools, and serve
--tools-dir): each Python file in them whose tools(engines) function
returns a list of functions. A function that takes a "conversation"
parameter gets the conversation so far, and one that takes "user" gets the
user asking, whose folder("notes") is their notes folder. One whose output
has a "reply_with" key ends the reply with its text, and one whose output
has an "images" key shows those PNGs or JPEGs, base64, to the user rather
than the model. See docs/guides/write-a-tool-file.md.

LITE_ASSISTANT_TOOLS picks which are on: comma-separated names, "all" or
"none". If it's unset, get_time, estimate_distance, identify_sounds,
identify_speakers, tune_instrument, read_text, describe_people, identify_plant, edit_photo,
load_skill, the web tools and every tool folder's tools are on. A request's builtin_tools overrides it.
"""
import importlib.util
import inspect
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Sequence, Tuple, Union, get_args, get_origin

from ..engines import ToolCall
from ..function_tools import ToolError
from .clock import Timers, get_time
from .distance import Distances
from .editing import IMAGES, PhotoEditor
from .notes import Notes
from .people import PeopleDescriber
from .plants import PlantIdentifier
from .reading import TextReader
from .skills import user_skills
from .sounds import SoundTagger
from .tuner import Tuner
from .voices import SpeakerFinder
from .web import read_web_page, search_web

# The tools on when LITE_ASSISTANT_TOOLS is unset, with those from tool folders
DEFAULT = ("get_time,estimate_distance,identify_sounds,identify_speakers,tune_instrument,read_text,describe_people,"
           "identify_plant,edit_photo,load_skill,search_web,read_web_page")
# Parameters the server fills, with the conversation so far and the user asking, which the model never sees
CONVERSATION, USER = "conversation", "user"
# A built-in's output with this key ends the reply with its text; the model doesn't write one
REPLY_WITH = "reply_with"
# Rounds of built-in calls one reply may make before it must answer
MAX_ROUNDS = 5

JSON_TYPES = {int: "integer", float: "number", bool: "boolean", str: "string"}


@dataclass
class BuiltinCall:
    """ A built-in tool call the server ran, with its output as JSON. """
    name: str
    arguments: str
    output: str
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:24]}")
    # PNGs or JPEGs, base64, it made to show the user, such as an edited photo
    images: List[str] = field(default_factory=list)


def definition(function: Callable) -> Dict[str, Any]:
    """
    The function as a tool definition in chat completions form: its
    docstring's first paragraph describes it, and its Args section and type
    hints describe the parameters. Parameters without defaults are required.
    A function with a refresh attribute, such as load_skill, whose list of
    skills changes, is refreshed first.
    """
    if callable(getattr(function, "refresh", None)):
        function.refresh()
    doc = inspect.getdoc(function) or ""
    described = dict(re.findall(r"^\s+(\w+):\s*(.+)$", doc.split("Args:", 1)[1], re.M)) if "Args:" in doc else {}
    properties, required = {}, []
    for name, parameter in inspect.signature(function).parameters.items():
        if name in (CONVERSATION, USER):
            continue
        properties[name] = {"type": JSON_TYPES.get(parameter.annotation, "string")}
        if get_origin(parameter.annotation) is Literal:
            properties[name]["enum"] = list(get_args(parameter.annotation))
        if name in described:
            properties[name]["description"] = described[name]
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "function", "function": {
        "name": function.__name__, "description": " ".join(doc.split("\n\n")[0].split()),
        "parameters": {"type": "object", "properties": properties, "required": required}}}


class Toolbox:
    """ The built-in tools this server has, and which are on by default. """
    def __init__(self, functions: Optional[Dict[str, Callable]] = None, default: Optional[str] = None,
                 added: Sequence[str] = ()):
        self.functions = functions or {}
        default = os.environ.get("LITE_ASSISTANT_TOOLS") if default is None else default
        if default is None:
            default = [name for name in DEFAULT.split(",") if name in self.functions] + list(added)
        self.default = self.names(default, strict=False)

    @classmethod
    def create(cls, engines, default: Optional[str] = None, folders: Sequence[str] = ()) -> "Toolbox":
        """
        Every built-in tool the loaded engines allow (search_notes needs an
        embedding engine, and reads photos of pages with an OCR engine and
        reranks with a reranking engine, estimate_distance a depth engine, which detection
        and segmentation engines add objects to, identify_sounds a sound
        engine, identify_speakers a speaker engine, tune_instrument a pitch engine, read_text an OCR engine, describe_people face or pose
        engines, identify_plant a plant classifier and edit_photo image editing
        engines; load_skill is offered only to a user with skills), and those
        the tool folders give.
        """
        timers = Timers()
        functions = [get_time, timers.set_timer, timers.list_timers, search_web, read_web_page]
        if engines.embed:
            functions.append(Notes(next(iter(engines.embed.values())), ocr=next(iter(engines.ocr.values()), None),
                                   reranker=next(iter(engines.rerank.values()), None)).search_notes)
        if engines.depth:
            detector = engines.detect.get(engines.default_detect)
            segmenter = next(iter(engines.segment.values()), None)
            functions.append(Distances(next(iter(engines.depth.values())), detector, segmenter).estimate_distance)
        if engines.sound:
            functions.append(SoundTagger(next(iter(engines.sound.values()))).identify_sounds)
        if engines.speaker:
            functions.append(SpeakerFinder(next(iter(engines.speaker.values()))).identify_speakers)
        if engines.pitch:
            functions.append(Tuner(next(iter(engines.pitch.values()))).tune_instrument)
        if engines.face or engines.pose:
            poses = {getattr(engine, "TASK", None): engine for engine in engines.pose.values()}
            functions.append(PeopleDescriber(engines.detect.get("yolox"), engines.detect.get("yunet"), engines.face.values(),
                                             poses.get("body"), poses.get("hand")).describe_people)
        plants = next((engine for engine in engines.recognize.values() if getattr(engine, "TASK", None) == "plant"), None)
        if plants:
            functions.append(PlantIdentifier(plants).identify_plant)
        if engines.edit:
            edits = {getattr(engine, "TASK", None): engine for engine in engines.edit.values()}
            functions.append(PhotoEditor(edits, next(iter(engines.segment.values()), None)).edit_photo)
        if engines.ocr:
            functions.append(TextReader(next(iter(engines.ocr.values()))).read_text)
        functions.append(user_skills())
        added = load_folders(folders, engines, {f.__name__ for f in functions})
        return cls({f.__name__: f for f in functions + added}, default, [f.__name__ for f in added])

    def names(self, value: Union[str, List[str]], strict: bool = True) -> List[str]:
        """
        The names in a list or a comma-separated string, or "all" or "none".
        An unknown name is an error, or with strict off, a warning.
        """
        if isinstance(value, str):
            value = {"all": list(self.functions), "none": []}.get(value.strip().lower(), value.split(","))
        names = []
        for name in filter(None, (str(n).strip() for n in value)):
            if name in self.functions:
                names.append(name)
            elif strict:
                raise ToolError(f"Unknown built-in tool {name!r}; the server has {', '.join(self.functions)}.")
            else:
                print(f"[!] Unknown built-in tool {name!r} in LITE_ASSISTANT_TOOLS, skipped.")
        return names

    def select(self, value: Union[bool, str, List[str], None]) -> Dict[str, Callable]:
        """
        The tools a request uses: given builtin_tools, false or "false" for
        none, true or None for the server's default, or which ones by name.
        """
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            value = value.strip().lower() == "true"
        if value is None or value is True:
            names = self.default
        elif value is False:
            names = []
        elif isinstance(value, (str, list)):
            names = self.names(value)
        else:
            raise ToolError("builtin_tools must be true, false, or a list of tool names.")
        return {name: self.functions[name] for name in names}

    def llm(self, llm, functions: Dict[str, Callable], report: bool = False, user=None):
        """
        llm, running these built-ins itself for user; with report, it yields
        a BuiltinCall for each. With no built-ins, it still catches calls to
        tools the request didn't offer.
        """
        return BuiltinLLM(llm, functions, report, user)


def load(path: str, engines) -> List[Callable]:
    """ The tools a tool file gives: what its tools(engines) function returns. """
    path = os.path.abspath(os.path.expanduser(path))
    name = "lite_assistant_tool_file_" + re.sub(r"\W", "_", os.path.splitext(os.path.basename(path))[0])
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or not os.path.isfile(path):
        raise ToolError(f"Tool file {path} isn't a Python file.")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses and pickling look a class's module up by name
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "tools", None)):
        raise ToolError(f"Tool file {path} has no tools(engines) function.")
    return list(module.tools(engines) or [])


def load_folders(folders: Sequence[str], engines, taken: Sequence[str] = ()) -> List[Callable]:
    """
    The tools in each folder's Python files, but those starting with _. A
    file that fails to load is skipped, and so is a tool whose name a
    built-in in taken or another file's tool has, saying which files.
    """
    found: Dict[str, List[Tuple[Callable, str]]] = {}
    for folder in folders:
        folder = os.path.abspath(os.path.expanduser(folder))
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".py") or name.startswith(("_", ".")):
                continue
            path = os.path.join(folder, name)
            try:
                functions = load(path, engines)
            except Exception as exc:
                print(f"[!] Skipped tool file {path}: {type(exc).__name__}: {exc}")
                continue
            for f in functions:
                found.setdefault(f.__name__, []).append((f, path))
    tools = []
    for name, each in found.items():
        if name in taken:
            print(f"[!] Skipped tool {name} from {each[0][1]}: a built-in tool has that name.")
        elif len(each) > 1:
            print(f"[!] Skipped tool {name}: {' and '.join(path for _, path in each)} both have one by that name.")
        else:
            print(f"Built-in tool {name} from {each[0][1]}")
            tools.append(each[0][0])
    return tools


def run(function: Callable, call: ToolCall, conversation: Optional[List[Dict[str, Any]]] = None,
        user=None) -> Tuple[str, List[str]]:
    """
    Runs a built-in for a call, with its output, or what went wrong, as JSON,
    and the images it made to show the user, which the model doesn't see. A
    built-in that takes the conversation or the user gets it.
    """
    images: List[str] = []
    try:
        arguments = json.loads(call.arguments or "{}")
        # Models sometimes add arguments a function doesn't take; leave them out
        parameters = inspect.signature(function).parameters
        arguments = {name: value for name, value in arguments.items()
                     if name in parameters and name not in (CONVERSATION, USER)}
        if CONVERSATION in parameters:
            arguments[CONVERSATION] = conversation or []
        if USER in parameters:
            arguments[USER] = user
        output = function(**arguments)
        if isinstance(output, dict):
            images = output.pop(IMAGES, None) or []
    except Exception as exc:
        output = {"error": f"{type(exc).__name__}: {exc}"}
    output = output if isinstance(output, str) else json.dumps(output)
    print(f"[Tool] {call.name}({call.arguments}) -> {output}" + (f" and {len(images)} image(s)" if images else ""))
    return output, images


def reply_with(output: str) -> Optional[str]:
    """ The reply a built-in's output says to give in place of the model's, if it says one. """
    try:
        found = json.loads(output)
    except ValueError:
        return None
    return found.get(REPLY_WITH) if isinstance(found, dict) else None


def stand_in(call: ToolCall, functions: Dict[str, Callable]) -> Tuple[ToolCall, Optional[Callable]]:
    """
    A built-in that answers a call to a tool that isn't one, such as
    load_skill for a call to a skill by its name, with the call it makes
    instead; or the call, and None.
    """
    for name, function in functions.items():
        arguments = getattr(function, "answers", lambda _: None)(call.name)
        if arguments:
            print(f"[Tool] {call.name} isn't a tool; calling {name} for it")
            return ToolCall(name, json.dumps(arguments), call.id), function
    return call, None


def not_offered(call: ToolCall, offered: List[Dict[str, Any]]) -> str:
    """ The error a call to a tool the model wasn't offered gets back, so it can try again. """
    names = ", ".join(tool["function"]["name"] for tool in offered) or "none"
    output = json.dumps({"error": f"There's no tool named {call.name!r}. The tools are: {names}. "
                                  "If none of them can do what was asked, say so."})
    print(f"[Tool] {call.name}({call.arguments}) wasn't offered -> {output}")
    return output


class BuiltinLLM:
    """
    An LLM engine that runs built-in tools itself. When the reply calls one,
    it runs it and generates again with the result, until the reply is
    written. Text and calls to the client's function tools pass through. A
    call to a tool that wasn't offered gets an error back in the same way,
    rather than reach a client that doesn't have it. A built-in with a
    for_user, such as load_skill, is the one for the user asking, if any.
    """
    def __init__(self, llm, functions: Dict[str, Callable], report: bool = False, user=None):
        self.inner, self.report, self.user = llm, report, user
        self.functions = {}
        for name, function in functions.items():
            function = function.for_user(user) if hasattr(function, "for_user") else function
            if function:
                self.functions[name] = function

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def generate(self, messages: List[Dict[str, Any]], max_tokens: Optional[int] = None,
                 usage: Optional[Dict[str, int]] = None, tools: Optional[List[Dict[str, Any]]] = None,
                 **options) -> Iterator[Union[str, ToolCall, BuiltinCall]]:
        # A reply held to a response schema can't call tools: the client's
        # are an error, and the built-ins are left out
        if options.get("response_schema") is not None and tools:
            raise ValueError("response_format can't be used with tools: a reply held to a schema can't call them.")
        # A client's function of the same name takes the place of a built-in
        clients = {tool["function"]["name"] for tool in tools or []}
        functions = {name: f for name, f in self.functions.items() if name not in clients
                     and options.get("response_schema") is None}
        offered = (tools or []) + [definition(f) for f in functions.values()]
        messages = list(messages)
        totals = {"prompt_tokens": 0, "completion_tokens": 0}
        written = ""
        for round in range(MAX_ROUNDS):
            # The last round can't call a built-in, so it answers
            last = round == MAX_ROUNDS - 1
            round_tools = tools if last else offered
            left = max_tokens - totals["completion_tokens"] if max_tokens else None
            round_usage, text, server_calls, client_called = {}, "", [], False
            chunks = self.inner.generate(messages, max_tokens=left, usage=round_usage, **options,
                                         **({"tools": round_tools} if round_tools else {}))
            try:
                for chunk in chunks:
                    # A built-in, or a tool that wasn't offered, is answered here
                    if isinstance(chunk, ToolCall) and chunk.name not in clients:
                        server_calls.append(chunk)
                        continue
                    if isinstance(chunk, ToolCall):
                        client_called = True
                    elif isinstance(chunk, str) and chunk:
                        # Keep the text before and after a call from running together
                        if not text and written and not written[-1].isspace() and not chunk[0].isspace():
                            chunk = " " + chunk
                        text += chunk
                        written += chunk
                    yield chunk
            finally:
                chunks.close()
                for key in totals:
                    totals[key] += round_usage.get(key, 0)
                if usage is not None:
                    usage.update(totals)
                    # Each round resends the whole conversation, so the last one says how full it got
                    if "context_tokens" in round_usage:
                        usage["context_tokens"] = round_usage["context_tokens"]
            if server_calls and (client_called or last):
                print(f"[Tool] Skipped {', '.join(c.name for c in server_calls)}: " +
                      ("the reply also called the client's tools" if client_called else "out of rounds"))
            if not server_calls or client_called or last or (max_tokens and totals["completion_tokens"] >= max_tokens):
                return
            messages.append({"role": "assistant", "content": text, "tool_calls": [
                {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
                for call in server_calls]})
            replies = []
            for call in server_calls:
                builtin = functions.get(call.name)
                if not builtin:
                    call, builtin = stand_in(call, functions)
                output, images = run(builtin, call, messages, self.user) if builtin else (not_offered(call, offered), [])
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
                replies.append(reply_with(output))
                if self.report and builtin:
                    yield BuiltinCall(call.name, call.arguments, output, call.id, images)
            # Built-ins that all say what to reply end the reply with it, rather than trust the model to
            if all(replies):
                reply = replies[0]
                yield " " + reply if written and not written[-1].isspace() else reply
                return
