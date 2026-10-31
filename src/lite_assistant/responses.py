"""
The Responses API (/v1/responses): turning its input items into the chat
messages the LLM engines take, and building response objects and the typed
event stream from a reply.
"""
import json
import time
import uuid
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from pydantic import BaseModel, ConfigDict

from .engines import Thought, ToolCall
from .tools import BuiltinCall

# The server_label of the mcp_call items that report built-in tool calls
BUILTIN_LABEL = "lite-assistant"


class InputError(ValueError):
    """ Input the server can't use, answered with a 400. """


class ResponseRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    input: Union[str, List[Dict[str, Any]]]
    model: Optional[str] = None
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    metadata: Optional[Dict[str, str]] = None
    # Stored responses can be fetched, listed and chained onto by id
    store: bool = True
    previous_response_id: Optional[str] = None
    # A stored conversation, by id or as {"id": ...}, to continue and add this turn to
    conversation: Optional[Union[str, Dict[str, Any]]] = None
    # Generate on a worker thread and answer right away; poll or cancel by id
    background: bool = False
    # Function tools, which the client runs: a call comes back as a
    # function_call output item, and its result goes in as function_call_output
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    # Beyond OpenAI's API: the server's own tools to offer, by name, or false
    # for none; they come back as mcp_call items
    builtin_tools: Optional[Union[bool, List[str]]] = None
    # {"effort": ...} turns the model's thinking on or off (see thinking_option);
    # its thoughts come back as reasoning items
    reasoning: Optional[Dict[str, Any]] = None
    # {"format": ...}: a json_schema or json_object format constrains the reply (see schema_option)
    text: Optional[Dict[str, Any]] = None


# Token budgets for thinking at each reasoning effort; None is no limit, and
# "none" and "minimal" turn it off
EFFORT_BUDGETS = {"low": 512, "medium": 2048, "high": None, "xhigh": None, "max": None}


def thinking_option(effort: Optional[str]) -> Dict[str, Any]:
    """
    The engine's thinking option for an OpenAI reasoning effort, as keyword
    arguments: none if the effort isn't set, so the model does as it does.
    """
    if effort is None:
        return {}
    if effort in ("none", "minimal"):
        return {"thinking": False}
    if effort not in EFFORT_BUDGETS:
        raise InputError(f"reasoning effort {effort!r} isn't one of none, minimal, {', '.join(EFFORT_BUDGETS)}")
    return {"thinking": EFFORT_BUDGETS[effort] or True}


def schema_option(format: Optional[Dict[str, Any]], flat: bool = False) -> Dict[str, Any]:
    """
    The engine's response_schema option for a response format, as keyword
    arguments: chat completions' response_format, or with flat, the format
    in Responses' text (whose json_schema has no "json_schema" wrapper).
    json_object asks for any object; text, or no format, leaves the reply free.
    """
    kind = (format or {}).get("type", "text")
    if kind == "text":
        return {}
    if kind == "json_object":
        return {"response_schema": {"type": "object"}}
    if kind != "json_schema":
        raise InputError(f"response format type {kind!r} isn't one of text, json_object, json_schema")
    schema = (format if flat else format.get("json_schema") or {}).get("schema")
    if schema is not None and not isinstance(schema, dict):
        raise InputError("a json_schema format's schema must be a JSON Schema object")
    return {"response_schema": schema or {"type": "object"}}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def input_items(input: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """ The input as a list of items, each a message with an id, as the API stores and lists them. """
    if isinstance(input, str):
        input = [{"role": "user", "content": input}]
    items = []
    for item in input:
        kind = item.get("type", "message")
        if kind == "function_call":
            if not (item.get("call_id") and item.get("name")):
                raise InputError("a function_call item needs a call_id and a name")
            items.append({"id": item.get("id") or new_id("fc"), "type": kind, "status": "completed",
                          "call_id": item["call_id"], "name": item["name"],
                          "arguments": item.get("arguments") or "{}"})
            continue
        if kind == "function_call_output":
            if not item.get("call_id"):
                raise InputError("a function_call_output item needs a call_id")
            output = item.get("output", "")
            if isinstance(output, list):
                output = "".join(part.get("text", "") for part in output if isinstance(part, dict))
            items.append({"id": item.get("id") or new_id("fco"), "type": kind, "status": "completed",
                          "call_id": item["call_id"], "output": output})
            continue
        if kind == "mcp_call":
            # A call a server ran, such as a built-in tool's, given back as context
            if not item.get("name"):
                raise InputError("an mcp_call item needs a name")
            items.append({"id": item.get("id") or new_id("mcp"), "type": kind, "status": "completed",
                          "server_label": item.get("server_label", BUILTIN_LABEL), "name": item["name"],
                          "arguments": item.get("arguments") or "{}", "output": item.get("output") or "",
                          "error": item.get("error")})
            continue
        if kind == "image_generation_call":
            # An image a built-in showed the user; models don't see it again
            items.append({"id": item.get("id") or new_id("ig"), "type": kind, "status": "completed",
                          "result": item.get("result") or ""})
            continue
        if kind == "reasoning":
            # Earlier thoughts are kept in the record, but models don't reread them
            text = "".join(part.get("text", "") for part in item.get("content") or [] if isinstance(part, dict))
            items.append({"id": item.get("id") or new_id("rs"), "type": kind, "status": "completed",
                          "summary": item.get("summary") or [],
                          "content": [{"type": "reasoning_text", "text": text}]})
            continue
        if kind != "message":
            raise InputError(f"input items of type {kind!r} aren't supported")
        content = item.get("content")
        if isinstance(content, str):
            text_type = "output_text" if item.get("role") == "assistant" else "input_text"
            content = [{"type": text_type, "text": content}]
        elif not isinstance(content, list):
            raise InputError("a message's content must be a string or a list of parts")
        items.append({"id": item.get("id") or new_id("msg"), "type": "message",
                      "role": item.get("role", "user"), "status": item.get("status", "completed"),
                      "content": content})
    return items


def to_chat_part(part: Dict[str, Any]) -> Dict[str, Any]:
    """ A Responses content part as the chat completions part the engines take. """
    kind = part.get("type")
    if kind in ("input_text", "output_text", "text"):
        return {"type": "text", "text": part.get("text", "")}
    if kind == "refusal":
        return {"type": "text", "text": part.get("refusal", "")}
    if kind == "input_image":
        if not part.get("image_url"):
            raise InputError("input_image needs an image_url; file_id isn't supported")
        return {"type": "image_url", "image_url": {"url": part["image_url"]}}
    if kind == "input_audio":
        return {"type": "input_audio", "input_audio": part["input_audio"]}
    if kind == "input_file":
        if not part.get("file_data"):
            raise InputError("input_file needs file_data; file_id and file_url aren't supported")
        return {"type": "file", "file": {"filename": part.get("filename") or "file", "file_data": part["file_data"]}}
    raise InputError(f"content parts of type {kind!r} aren't supported")


def to_messages(instructions: Optional[str], items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """ Chat messages for the engine: the instructions as a system message, then each input message. """
    messages = [{"role": "system", "content": instructions}] if instructions else []
    for item in items:
        if item["type"] in ("reasoning", "image_generation_call"):
            continue
        if item["type"] in ("function_call", "mcp_call"):
            call_id = item["call_id"] if item["type"] == "function_call" else item["id"]
            call = {"id": call_id, "type": "function",
                    "function": {"name": item["name"], "arguments": item["arguments"]}}
            # Calls made together belong to one assistant message, with any text before them
            if messages and messages[-1]["role"] == "assistant":
                messages[-1].setdefault("tool_calls", []).append(call)
            else:
                messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
            if item["type"] == "mcp_call":
                messages.append({"role": "tool", "tool_call_id": call_id,
                                 "content": item.get("output") or json.dumps({"error": item.get("error")})})
            continue
        if item["type"] == "function_call_output":
            messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": item["output"]})
            continue
        role = {"developer": "system"}.get(item["role"], item["role"])
        parts = [to_chat_part(p) for p in item["content"]]
        if all(p["type"] == "text" for p in parts):
            messages.append({"role": role, "content": "".join(p["text"] for p in parts)})
        else:
            messages.append({"role": role, "content": parts})
    if not any(m["role"] != "system" for m in messages):
        raise InputError("input must contain at least one message")
    return messages


def sse(event: Dict[str, Any]) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode("utf-8")


class ResponseBuilder:
    """ Builds one response's object, as it stands at any point while its reply is written. """
    def __init__(self, body: ResponseRequest, model_id: str):
        self.body = body
        self.id = new_id("resp")
        self.created_at = int(time.time())
        self.model_id = model_id
        self.status = "in_progress"
        # The output items in order. Messages and reasoning are {"type", "id",
        # "text"} here, written to until something else follows them; calls
        # are their items.
        self.order: List[Dict[str, Any]] = []
        self.writing: Optional[Dict[str, Any]] = None
        self.usage = None
        # Each thought chunk is about a token, so they count the reasoning tokens
        self.thoughts = 0

    def add(self, chunk: Union[str, Thought, ToolCall, BuiltinCall]) -> Optional[Dict[str, Any]]:
        """ Adds a chunk of the reply; for a call, returns its output item. """
        if isinstance(chunk, (str, Thought)):
            kind = "reasoning" if isinstance(chunk, Thought) else "message"
            if not self.writing or self.writing["type"] != kind:
                self.writing = {"type": kind, "id": new_id("rs" if kind == "reasoning" else "msg"), "text": ""}
                self.order.append(self.writing)
            if isinstance(chunk, Thought):
                self.thoughts += 1
                self.writing["text"] += chunk.text
            else:
                self.writing["text"] += chunk
            return None
        if isinstance(chunk, ToolCall):
            item = {"id": new_id("fc"), "type": "function_call", "status": "completed", "call_id": chunk.id,
                    "name": chunk.name, "arguments": chunk.arguments}
        else:
            item = {"id": new_id("mcp"), "type": "mcp_call", "status": "completed", "server_label": BUILTIN_LABEL,
                    "name": chunk.name, "arguments": chunk.arguments, "output": chunk.output, "error": None}
        # Text after a call goes in a new message
        self.writing = None
        self.order.append(item)
        # Images a built-in made follow it, as OpenAI's image generation tool gives them
        for image in getattr(chunk, "images", None) or []:
            self.order.append({"id": new_id("ig"), "type": "image_generation_call", "status": "completed",
                               "result": image})
        return item

    def output(self) -> List[Dict[str, Any]]:
        if not self.order:
            # A reply with no text and no calls is still an (empty) message
            self.add("")
        return [self.entry(item) for item in self.order]

    def entry(self, item: Dict[str, Any], status: Optional[str] = None) -> Dict[str, Any]:
        """ An output item as the API gives it. """
        return {"message": self.message, "reasoning": self.reasoning}.get(item["type"], lambda i, s: i)(item, status)

    def finish(self, usage: Dict[str, int]):
        """ Records the finished reply's token counts, and whether it was cut off. """
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        self.usage = {"input_tokens": input_tokens, "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens": output_tokens,
                      "output_tokens_details": {"reasoning_tokens": min(self.thoughts, output_tokens)},
                      "total_tokens": input_tokens + output_tokens}
        # Beyond OpenAI's API: how much of the model's context the reply filled
        if "context_tokens" in usage:
            self.usage["context_tokens"] = usage["context_tokens"]
        limit = self.body.max_output_tokens
        self.status = "incomplete" if limit and output_tokens >= limit else "completed"

    def status_of(self, entry: Dict[str, Any]) -> str:
        # A message or reasoning is only ever in progress, completed or
        # incomplete; only the one being written follows the response's status
        if entry is not self.writing:
            return "completed"
        return {"completed": "completed", "queued": "in_progress", "in_progress": "in_progress"}.get(
            self.status, "incomplete")

    def message(self, entry: Dict[str, Any], status: Optional[str] = None) -> Dict[str, Any]:
        return {"id": entry["id"], "type": "message", "role": "assistant", "status": status or self.status_of(entry),
                "content": [self.text_part(entry["text"])]}

    def reasoning(self, entry: Dict[str, Any], status: Optional[str] = None) -> Dict[str, Any]:
        return {"id": entry["id"], "type": "reasoning", "status": status or self.status_of(entry), "summary": [],
                "content": [self.text_part(entry["text"], "reasoning")]}

    @staticmethod
    def text_part(text: str, kind: str = "message") -> Dict[str, Any]:
        if kind == "reasoning":
            return {"type": "reasoning_text", "text": text}
        return {"type": "output_text", "text": text, "annotations": [], "logprobs": []}

    def conversation_id(self) -> Optional[str]:
        conversation = self.body.conversation
        return conversation.get("id") if isinstance(conversation, dict) else conversation

    def object(self, with_output: bool = True) -> Dict[str, Any]:
        b = self.body
        conversation_id = self.conversation_id()
        return {
            "id": self.id, "object": "response", "created_at": self.created_at, "status": self.status,
            "error": None,
            "incomplete_details": {"reason": "max_output_tokens"} if self.status == "incomplete" else None,
            "instructions": b.instructions, "max_output_tokens": b.max_output_tokens, "model": self.model_id,
            "output": self.output() if with_output else [],
            "parallel_tool_calls": True, "previous_response_id": b.previous_response_id,
            "temperature": b.temperature, "top_p": b.top_p, "tool_choice": b.tool_choice or "auto",
            "tools": b.tools or [], "text": {"format": {"type": "text"}}, "truncation": "disabled",
            "reasoning": {"effort": (b.reasoning or {}).get("effort"), "summary": None},
            "usage": self.usage, "metadata": b.metadata or {}, "store": b.store, "background": b.background,
            "conversation": {"id": conversation_id} if conversation_id else None,
        }

    def events(self, chunks: Iterator[Union[str, ToolCall, BuiltinCall]], usage: Dict[str, int],
               on_done: Optional[Callable[[Dict[str, Any]], None]] = None) -> Iterator[bytes]:
        """ The typed server-sent events of a streamed response, as the reply is written. """
        sequence = 0
        # Where the message or reasoning being written is, while one is open
        where = {}

        def event(kind: str, **fields) -> bytes:
            nonlocal sequence
            sequence += 1
            return sse({"type": kind, "sequence_number": sequence - 1, **fields})

        def open_message() -> Iterator[bytes]:
            entry = self.order[-1]
            where.update(item_id=entry["id"], output_index=len(self.order) - 1, content_index=0)
            yield event("response.output_item.added", output_index=where["output_index"],
                        item={**self.entry(entry, "in_progress"), "content": []})
            yield event("response.content_part.added", **where, part=self.text_part("", entry["type"]))

        def close_message(status: Optional[str] = None) -> Iterator[bytes]:
            entry = self.order[where["output_index"]]
            if entry["type"] == "reasoning":
                yield event("response.reasoning_text.done", **where, text=entry["text"])
            else:
                yield event("response.output_text.done", **where, text=entry["text"], logprobs=[])
            yield event("response.content_part.done", **where, part=self.text_part(entry["text"], entry["type"]))
            yield event("response.output_item.done", output_index=where["output_index"],
                        item=self.entry(entry, status))
            where.clear()

        def call_events(item: Dict[str, Any], index: int) -> Iterator[bytes]:
            ids = {"item_id": item["id"], "output_index": index}
            if item["type"] == "image_generation_call":
                yield event("response.output_item.added", output_index=index,
                            item={**item, "status": "in_progress", "result": None})
                yield event("response.image_generation_call.completed", **ids)
            elif item["type"] == "function_call":
                yield event("response.output_item.added", output_index=index,
                            item={**item, "status": "in_progress", "arguments": ""})
                # A call arrives whole, so its arguments are one delta
                yield event("response.function_call_arguments.delta", **ids, delta=item["arguments"])
                yield event("response.function_call_arguments.done", **ids, name=item["name"],
                            arguments=item["arguments"])
            else:
                # A built-in has already run by the time it's reported
                yield event("response.output_item.added", output_index=index,
                            item={**item, "status": "in_progress", "output": None})
                yield event("response.mcp_call_arguments.done", **ids, arguments=item["arguments"])
                yield event("response.mcp_call.completed", **ids)
            yield event("response.output_item.done", output_index=index, item=item)

        yield event("response.created", response=self.object(with_output=False))
        yield event("response.in_progress", response=self.object(with_output=False))
        for chunk in chunks:
            if not isinstance(chunk, (str, Thought)):
                if where:
                    yield from close_message("completed")
                before = len(self.order)
                self.add(chunk)
                for index in range(before, len(self.order)):
                    yield from call_events(self.order[index], index)
                continue
            kind = "reasoning" if isinstance(chunk, Thought) else "message"
            opening = not self.writing or self.writing["type"] != kind
            if opening and where:
                # Thinking done, the reply starts
                yield from close_message("completed")
            self.add(chunk)
            if opening:
                yield from open_message()
            if kind == "reasoning":
                yield event("response.reasoning_text.delta", **where, delta=chunk.text)
            else:
                yield event("response.output_text.delta", **where, delta=chunk, logprobs=[])
        self.finish(usage)
        if not self.order:
            self.add("")
            yield from open_message()
        if where:
            yield from close_message()
        final = self.object()
        if on_done:
            on_done(final)
        yield event(f"response.{self.status}", response=final)
