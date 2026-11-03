#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Brad Barnett
#
# SPDX-License-Identifier: MIT

"""
Check that a server answers the calls a portable client makes.

Runs the same small requests against this server or a provider, through
examples/board_client.py, and says which ones pass. A provider charges for them,
a fraction of a cent for the lot on small models. The key comes from an
environment variable, so it isn't on the command line.

    python scripts/check_provider.py --provider local
    python scripts/check_provider.py --provider deepinfra --key-env DEEPINFRA_API_KEY
"""
import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "examples"))

import board_client  # noqa: E402
import providers  # noqa: E402

PHOTO = os.path.join(ROOT, "tests", "test_image.jpg")
PROVIDERS = {"local": providers.LOCAL, "deepinfra": providers.DEEPINFRA, "openrouter": providers.OPENROUTER}
SAID = "The quick brown fox jumps over the lazy dog."
COMMAND = {"name": "command", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["action"],
    "properties": {"action": {"type": "string", "enum": ["stop", "go"]}}}}


def matches(value: Any, schema: Dict[str, Any]) -> bool:
    """ Whether a value fits the parts of JSON Schema the published schemas use. """
    kinds = schema.get("type", [])
    kinds = [kinds] if isinstance(kinds, str) else kinds
    python = {"object": dict, "array": list, "string": str, "number": (int, float), "null": type(None)}
    if kinds and not any(isinstance(value, python[kind]) for kind in kinds):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        return set(schema.get("required", [])) <= set(value) and set(value) <= set(properties) \
            and all(matches(item, properties[name]) for name, item in value.items())
    if isinstance(value, list):
        return all(matches(item, schema.get("items", {})) for item in value)
    return True


def expect(condition: bool, what: Any) -> None:
    if not condition:
        raise AssertionError(what)


def check_models(client, photo: bytes) -> str:
    listed = client.get("/v1/models")["data"]
    expect(listed and "id" in listed[0], listed[:1])
    return f"{len(listed)} models"


def check_chat(client, photo: bytes) -> str:
    text = client.chat("Reply with one word: ready.", max_tokens=200)
    expect(text, client.reply)
    expect(client.reply["choices"][0]["finish_reason"], client.reply)
    return repr(text[:40])


def check_stream(client, photo: bytes) -> str:
    text = "".join(client.chat_stream("Reply with one word: ready.", max_tokens=200))
    expect(text, "no text")
    return f"first piece in {client.first_ms} ms"


def check_no_thinking(client, photo: bytes) -> str:
    text = client.chat("Reply with one word: ready.", max_tokens=200, reasoning={"enabled": False})
    expect(text, client.reply)
    return repr(text[:40])


def check_json(client, photo: bytes) -> str:
    text = client.chat("The road is blocked. Reply with the action.", max_tokens=200,
                       response_format={"type": "json_schema", "json_schema": COMMAND})
    expect(matches(json.loads(text), COMMAND["schema"]), text)
    return text


def check_objects(client, photo: bytes) -> str:
    found = client.detect_objects(photo)
    expect(matches(found, board_client.OBJECT_DETECTION["schema"]), found)
    return ", ".join(item["label"] for item in found["objects"]) or "nothing found"


def check_text(client, photo: bytes) -> str:
    found = client.read_text(photo)
    expect(matches(found, board_client.OCR_EXTRACTION["schema"]), found)
    return repr(found["text"][:40])


def check_speech_and_back(client, photo: bytes) -> str:
    request = {"input": SAID, "response_format": "wav"}
    request.update((name, client.models[key]) for name, key in (("model", "speech"), ("voice", "voice"))
                   if key in client.models)
    audio = client.post_json("/v1/audio/speech", request)
    expect(audio[:4] == b"RIFF" and audio[8:12] == b"WAVE", audio[:12])
    heard = client.post_files("/v1/audio/transcriptions", [("file", "speech.wav", "audio/wav", audio)],
                              model=client.models.get("transcribe"))
    expect(heard.get("text"), heard)
    return f"{len(audio)} bytes, heard {heard['text'][:40]!r}"


CHECKS: List[Tuple[str, Callable]] = [
    ("models", check_models), ("chat", check_chat), ("stream", check_stream), ("no-thinking", check_no_thinking),
    ("json", check_json), ("objects", check_objects), ("text", check_text), ("speech", check_speech_and_back)]


def run(client, photo: bytes, only=None, out=print) -> List[str]:
    """ Runs the checks and returns the names of those that failed. """
    failed = []
    for name, check in CHECKS:
        if only and name not in only:
            continue
        start = time.monotonic()
        try:
            out(f"PASS {name}: {check(client, photo)} ({time.monotonic() - start:.1f} s)")
        except Exception as exc:
            failed.append(name)
            out(f"FAIL {name}: {type(exc).__name__}: {str(exc)[:300]}")
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--provider", choices=PROVIDERS, default="local")
    parser.add_argument("--base-url", help="use this instead of the provider's")
    parser.add_argument("--key-env", help="the environment variable that holds the key")
    parser.add_argument("--model", action="append", default=[], metavar="KIND=ID",
                        help="use this model for a kind of request, such as chat=google/gemma-4-31b-it")
    parser.add_argument("--timeout", type=int, default=120, help="seconds to wait for a reply, default 120")
    parser.add_argument("--only", action="append", choices=[name for name, _ in CHECKS], help="run only this check")
    args = parser.parse_args()
    settings = PROVIDERS[args.provider]
    models = dict(settings["models"], **dict(item.split("=", 1) for item in args.model))
    key = os.environ.get(args.key_env) if args.key_env else None
    if args.key_env and not key:
        parser.error(f"{args.key_env} isn't set")
    client = board_client.Client(base_url=args.base_url or settings["base_url"], key=key, models=models, timeout=args.timeout)
    with open(PHOTO, "rb") as file:
        failed = run(client, file.read(), args.only)
    print(f"{len(failed)} failed" if failed else "All passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
