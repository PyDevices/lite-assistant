# SPDX-FileCopyrightText: 2026 Brad Barnett
#
# SPDX-License-Identifier: MIT

"""
Ask for the objects or the text in a photo the way you ask any vision model.

A client sends an ordinary chat completion: a photo in an image_url part, and
one of the schemas below as its response_format. A vision model at a provider
or on this server answers by writing the JSON. Set model to a detection or OCR
engine here, such as yolox or ppocr, and the engine answers instead, in the same
shape, so the client changes only its base_url and model.
"""
import json
from typing import Any, Callable, Dict, Iterator, List, Optional

from . import detection
from .engines import as_text
from .images import brighten_if_dark
from .media import latest_image
from .responses import InputError

_BOX = {"type": "object", "additionalProperties": False,
        "required": ["left", "top", "right", "bottom"],
        "properties": {side: {"type": "number"} for side in ("left", "top", "right", "bottom")}}

# Strict mode at the providers wants every property required and no others, so
# a value that may be missing is nullable.
OBJECT_DETECTION = {
    "name": "object_detection", "strict": True,
    "schema": {
        "type": "object", "additionalProperties": False, "required": ["objects"],
        "properties": {"objects": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["label", "score", "box", "distance"],
            "properties": {"label": {"type": "string"}, "score": {"type": "number"}, "box": _BOX,
                           "distance": {"type": ["number", "null"]}}}}}}}

OCR_EXTRACTION = {
    "name": "ocr_extraction", "strict": True,
    "schema": {
        "type": "object", "additionalProperties": False, "required": ["text", "lines"],
        "properties": {"text": {"type": "string"}, "lines": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["text", "box"],
            "properties": {"text": {"type": "string"}, "box": _BOX}}}}}}


class VisionTask:
    """ A detection or OCR engine standing in for a chat model on one schema. """

    def __init__(self, engine: Any, answer: Callable[[bytes], Dict[str, Any]]):
        self.engine = engine
        self.answer = answer

    def get_model_id(self) -> str:
        return self.engine.get_model_id()

    def generate(self, messages: List[Dict[str, Any]], tools=None, **options) -> Iterator[str]:
        """ The whole reply as one chunk. Raises ValueError for a request the engine can't take. """
        if tools:
            raise ValueError(f"{self.get_model_id()} can't call tools.")
        photo = latest_image(messages)
        if photo is None:
            raise ValueError("The request needs a photo in an image_url part.")
        yield json.dumps(self.answer(photo))


def _objects(engines, engine) -> Callable[[bytes], Dict[str, Any]]:
    def answer(photo: bytes) -> Dict[str, Any]:
        # The same photo /v1/detect would look at, and the same distances
        photo, _ = brighten_if_dark(photo, None)
        found = engine.detect(photo, None)
        depth_map = next(iter(engines.depth.values())).estimate(photo) if engines.depth and found else None
        return {"objects": [{"label": item["label"], "score": item["score"], "box": item["box"],
                             "distance": item.get("distance")} for item in detection.objects(found, depth_map)]}
    return answer


def _text(engine) -> Callable[[bytes], Dict[str, Any]]:
    def answer(photo: bytes) -> Dict[str, Any]:
        lines = engine.read(photo)
        return {"text": as_text(lines),
                "lines": [{"text": line.text, "box": dict(zip(("left", "top", "right", "bottom"), line.box))}
                          for line in lines]}
    return answer


def vision_task(engines, model: Optional[str], response_format: Optional[Dict[str, Any]]) -> Optional[VisionTask]:
    """
    The task for a chat completion that names a detection or OCR engine as its
    model, or None for any other model. Only the model chooses: a schema's
    name is the client's, and a chat model can answer a schema of that name.

    Raises:
        InputError: The request doesn't ask for the engine's own schema.
    """
    if model in engines.detect:
        engine, wanted = engines.detect[model], OBJECT_DETECTION
        answer = _objects(engines, engine)
    elif model in engines.ocr:
        engine, wanted = engines.ocr[model], OCR_EXTRACTION
        answer = _text(engine)
    else:
        return None
    asked = (response_format or {}).get("json_schema") or {}
    if (response_format or {}).get("type") != "json_schema" or asked.get("name") != wanted["name"] \
            or asked.get("schema") != wanted["schema"]:
        raise InputError(f"Model {model!r} answers only the published {wanted['name']} schema, "
                         "as a json_schema response_format.")
    return VisionTask(engine, answer)
