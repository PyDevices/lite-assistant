"""
The endpoints that aren't chat: what the engines see in a photo and hear in
audio, and what text is about. None of them are OpenAI endpoints.
"""
from typing import Any, Dict, List, Optional, Union

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import Response
from pydantic import BaseModel

from . import depth, detection, lines, people, segmentation, sounds
from .api_common import APIError, read_photo, run_engine
from .engines import as_text
from .audio import read_audio
from .images import brighten_if_dark

router = APIRouter()


class ClassifyRequest(BaseModel):
    # One text, or several to sort into the same labels
    input: Union[str, List[str]]
    labels: List[str]
    # What the labels are about, read just before the text, e.g. "The intent of this command: "
    prompt: Optional[str] = None
    # Off: the labels compete and one wins. On: each label fits or not on its own
    multi_label: bool = False
    threshold: float = 0.5
    model: Optional[str] = None


@router.post("/v1/depth")
async def estimate_depth(request: Request):
    """
    How far things in a photo are from the camera, in meters: not an
    OpenAI endpoint. The photo and options come as read_photo takes
    them. Options: columns and rows of the grid, and response_format
    json, or png for the depth image.
    """
    e = request.app.state.engines
    if not e.depth:
        raise APIError(500, "No depth engines loaded.", "server_error")
    photo, options = await read_photo(request)

    try:
        columns, rows = int(options.get("columns", 8)), int(options.get("rows", 8))
    except (TypeError, ValueError):
        raise APIError(400, "columns and rows must be whole numbers.")
    if not (1 <= columns <= depth.MAX_GRID and 1 <= rows <= depth.MAX_GRID):
        raise APIError(400, f"columns and rows must be from 1 to {depth.MAX_GRID}.")
    response_format = options.get("response_format", "json")
    if response_format not in ("json", "png"):
        raise APIError(400, "response_format must be json or png")

    engine = e.depth.get(options.get("model")) or next(iter(e.depth.values()))
    found = await run_engine(engine.estimate, photo, "Depth")
    if response_format == "png":
        png, near, far = depth.image(found)
        # What the brightest and darkest grays stand for, in meters
        return Response(png, media_type="image/png",
                        headers={"X-Depth-Near": str(near), "X-Depth-Far": str(far)})
    return {"object": "depth", "model": engine.get_model_id(), "units": "meters",
            **depth.summary(found, columns, rows)}

@router.post("/v1/detect")
async def detect_objects(request: Request):
    """
    The objects in a photo: not an OpenAI endpoint. The photo and options
    come as read_photo takes them. Options: model, yolox for objects (the
    default) or yunet for faces; threshold, how sure the model must be
    (0 to 1, by default the engine's own); distances, false to skip
    measuring how far each object is, which a depth engine does by default;
    brighten, auto (the default) to brighten a dark photo first, or true
    or false; describe, false to skip what the face engines tell about each
    face, such as its expression and gaze; pose, true for each person's
    joints and raised hands; and hands, true for each person's hands'
    joints and the fingers they hold up too.
    """
    e = request.app.state.engines
    if not e.detect:
        raise APIError(500, "No detection engines loaded.", "server_error")
    photo, options = await read_photo(request)
    # A dark photo is brightened first, so the detector sees more of it
    photo, brightened = await run_engine(lambda p: brighten_if_dark(p, options.get("brighten")), photo, "Brighten")
    try:
        threshold = None if options.get("threshold") is None else float(options["threshold"])
    except (TypeError, ValueError):
        raise APIError(400, "threshold must be a number.")
    if threshold is not None and not 0 <= threshold <= 1:
        raise APIError(400, "threshold must be from 0 to 1.")
    flag = lambda name, default: str(options.get(name, default)).lower() not in ("false", "0", "no")
    measure = flag("distances", "true")
    hands = flag("hands", "false")
    pose_engines = {getattr(engine, "TASK", None): engine for engine in e.pose.values()}
    body = pose_engines.get("body") if flag("pose", "false") or hands else None
    hand = pose_engines.get("hand") if hands else None
    faces = list(e.face.values()) if flag("describe", "true") else []
    model = options.get("model")
    if model is not None and model not in e.detect:
        raise APIError(404, f"No detection model {model!r}; the server has {', '.join(e.detect)}.",
                       code="model_not_found")
    engine = e.detect[model or e.default_detect]
    found = await run_engine(lambda p: engine.detect(p, threshold), photo, "Detection")
    result = {"object": "detection", "model": engine.get_model_id()}
    depth_map = None
    if measure and e.depth and found:
        depth_engine = next(iter(e.depth.values()))
        depth_map = await run_engine(depth_engine.estimate, photo, "Depth")
        result.update({"depth_model": depth_engine.get_model_id(), "units": "meters"})
    if brightened:
        result["brightened"] = True
    result["objects"] = detection.objects(found, depth_map)
    labels = {d.label for d in found}
    if (faces and "face" in labels) or (body and "person" in labels):
        from .engines.tflite import open_photo
        extras = await run_engine(lambda p: people.describe(found, open_photo(p), faces, body, hand), photo, "People")
        for listed, extra in zip(result["objects"], extras):
            listed.update(extra)
    return result

@router.post("/v1/segment")
async def find_things(request: Request):
    """
    Where things named in a prompt are in a photo, such as "the cup": not
    an OpenAI endpoint. The photo and options come as read_photo takes
    them. prompt names one thing, or several separated by commas, or in
    JSON a list. Options: threshold, how sure the engine must be that a
    point is the thing (0 to 1, default 0.4); distances, false to skip
    measuring how far each is; and response_format json, or png for the
    first thing's mask, white where it is.
    """
    e = request.app.state.engines
    if not e.segment:
        raise APIError(500, "No segmentation engines loaded.", "server_error")
    photo, options = await read_photo(request)
    prompts = options.get("prompt")
    prompts = [p.strip() for p in (prompts.split(",") if isinstance(prompts, str) else prompts or [])
               if isinstance(p, str) and p.strip()]
    if not prompts:
        raise APIError(400, "prompt must name what to find, such as \"the cup\".")
    try:
        threshold = float(options.get("threshold", segmentation.THRESHOLD))
    except (TypeError, ValueError):
        raise APIError(400, "threshold must be a number.")
    if not 0 < threshold <= 1:
        raise APIError(400, "threshold must be above 0, and at most 1.")
    response_format = options.get("response_format", "json")
    if response_format not in ("json", "png"):
        raise APIError(400, "response_format must be json or png")
    engine = e.segment.get(options.get("model")) or next(iter(e.segment.values()))
    maps = await run_engine(lambda p: engine.segment(p, prompts[:1] if response_format == "png" else prompts),
                            photo, "Segmentation")
    if response_format == "png":
        from .engines.tflite import open_photo
        size = (await run_engine(open_photo, photo, "Segmentation")).size
        return Response(segmentation.mask(maps[0], size), media_type="image/png")
    found = [segmentation.region(m, threshold) for m in maps]
    result = {"object": "segmentation", "model": engine.get_model_id()}
    measure = str(options.get("distances", "true")).lower() not in ("false", "0", "no")
    depth_map = None
    if measure and e.depth and any(found):
        depth_engine = next(iter(e.depth.values()))
        depth_map = await run_engine(depth_engine.estimate, photo, "Depth")
        result.update({"depth_model": depth_engine.get_model_id(), "units": "meters"})
    result["data"] = [{"prompt": prompt, "found": bool(where), **(where or {}),
                       **({"distance": segmentation.distance(depth_map, m, threshold)} if where and depth_map else {})}
                      for prompt, m, where in zip(prompts, maps, found)]
    return result

@router.post("/v1/lines")
async def follow_lines(request: Request):
    """
    Lines to follow in a photo from a camera looking ahead: not an OpenAI
    endpoint. The photo and options come as read_photo takes them.
    Options: model, line (the default, tape or paint on a floor) or lanes
    (a road's lane markings), or an engine's id; and response_format
    json, or jpeg for the photo with what was found drawn on it.
    """
    e = request.app.state.engines
    if not e.line:
        raise APIError(500, "No line engines loaded.", "server_error")
    photo, options = await read_photo(request)
    model = options.get("model") or "line"
    engine = e.line.get(model) or next((engine for engine in e.line.values() if model in (
        getattr(engine, "TASK", None), getattr(engine, "MODEL_ID", None))), None)
    if engine is None:
        raise APIError(404, f"No line model {model!r}; the server has {', '.join(e.line)}.", code="model_not_found")
    response_format = options.get("response_format", "json")
    if response_format not in ("json", "jpeg"):
        raise APIError(400, "response_format must be json or jpeg")

    def follow(data: bytes):
        from .engines.tflite import open_photo
        opened = open_photo(data)
        found = engine.find(opened)
        return opened, found, engine.path(found)
    opened, found, path = await run_engine(follow, photo, "Lines")
    if response_format == "jpeg":
        return Response(lines.drawn(opened, found, path), media_type="image/jpeg")
    return {"object": "lines", "model": engine.get_model_id(), "found": bool(path),
            "path": lines.measure(path, opened.size), "lines": [[list(p) for p in line] for line in found]}


@router.post("/v1/images/classifications")
async def classify_image(request: Request):
    """
    What a photo shows, from an image classification engine's list, such
    as a plant's species: not an OpenAI endpoint. The photo and options
    come as read_photo takes them. Options: model, an engine's id or what
    it names, such as plant (the default is the first); and top, how many
    labels to give (1 to 20, default 5).
    """
    e = request.app.state.engines
    if not e.recognize:
        raise APIError(500, "No image classification engines loaded.", "server_error")
    photo, options = await read_photo(request)
    model = options.get("model")
    engine = e.recognize.get(model) or next((engine for engine in e.recognize.values() if model in (
        getattr(engine, "TASK", None), getattr(engine, "MODEL_ID", None))), None)
    if engine is None and model:
        raise APIError(404, f"No image classification model {model!r}; the server has {', '.join(e.recognize)}.",
                       code="model_not_found")
    engine = engine or next(iter(e.recognize.values()))
    try:
        top = int(options.get("top", 5))
    except (TypeError, ValueError):
        raise APIError(400, "top must be a whole number.")
    if not 1 <= top <= 20:
        raise APIError(400, "top must be from 1 to 20.")
    found = await run_engine(lambda p: engine.classify(p, top), photo, "Image classification")
    return {"object": "list", "model": engine.get_model_id(),
            "data": [{"label": label, "score": round(score, 4)} for label, score in found]}

@router.post("/v1/audio/classifications")
async def classify_sounds(request: Request):
    """
    The sounds in audio, such as speech, music, a doorbell or a dog: not
    an OpenAI endpoint. The audio comes as a multipart file, as
    transcriptions takes it, or as the request body itself (e.g.
    Content-Type: audio/wav). Options, as form fields or query
    parameters: threshold, how sure the model must be (0 to 1, default
    0.1), and top, the most sounds to list (1 to 50, default 5).
    """
    e = request.app.state.engines
    if not e.sound:
        raise APIError(500, "No sound engines loaded.", "server_error")
    audio, options = await read_audio(request)
    try:
        threshold, top = float(options.get("threshold", sounds.THRESHOLD)), int(options.get("top", sounds.TOP))
    except (TypeError, ValueError):
        raise APIError(400, "threshold and top must be numbers.")
    if not 0 <= threshold <= 1 or not 1 <= top <= 50:
        raise APIError(400, "threshold must be from 0 to 1, and top from 1 to 50.")
    engine = e.sound.get(options.get("model")) or next(iter(e.sound.values()))
    found = await run_engine(engine.classify, audio, "Sound", "audio")
    return {"object": "audio.classification", "model": engine.get_model_id(), "duration": found.duration,
            "sounds": sounds.heard(found, threshold, top), "segments": sounds.segments(found, threshold, top)}

@router.post("/v1/classify")
def classify(body: ClassifyRequest, request: Request):
    """
    Sorts text into labels the request names, such as a voice command's
    intent, in milliseconds: not an OpenAI endpoint. Each result lists
    every label, best first. By default the labels compete, scores add
    up to 1, and label is the winner; with multi_label, each score says
    on its own whether the label fits, and chosen lists those at or
    above threshold.
    """
    e = request.app.state.engines
    if not e.classify:
        raise APIError(500, "No classification engines loaded.", "server_error")
    engine = e.classify.get(body.model) or next(iter(e.classify.values()))
    texts = body.input if isinstance(body.input, list) else [body.input]
    if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise APIError(400, "input must be a text or a list of texts, none empty.")
    if any(not label.strip() for label in body.labels):
        raise APIError(400, "labels must not be empty strings.")
    if not 0 <= body.threshold <= 1:
        raise APIError(400, "threshold must be from 0 to 1.")
    data = []
    for index, text in enumerate(texts):
        try:
            logits = np.array(engine.logits(text, body.labels, body.prompt), np.float64)
        except ValueError as exc:
            raise APIError(400, f"input[{index}]: {exc}")
        if body.multi_label:
            scores = 1 / (1 + np.exp(-logits))
        else:
            scores = np.exp(logits - logits.max())
            scores /= scores.sum()
        # A stable sort, so a tie goes to the earlier label
        ranked = [{"label": body.labels[i], "score": round(float(scores[i]), 4)}
                  for i in np.argsort(-scores, kind="stable")]
        result = {"object": "classification", "index": index}
        if body.multi_label:
            result["chosen"] = [r["label"] for r in ranked if r["score"] >= body.threshold]
        else:
            result.update(ranked[0])
        data.append({**result, "labels": ranked})
    return {"object": "list", "model": engine.get_model_id(), "data": data}


@router.post("/v1/ocr")
async def read_text(request: Request):
    """
    The text in a photo, such as a sign, a label or a page: not an OpenAI
    endpoint. The photo and options come as read_photo takes them; the only
    option is model. Gives the text, with lines side by side in a row
    joined by spaces, and each line with how sure the reader is of it and
    its box, as fractions of the photo (left, top, right, bottom).
    """
    e = request.app.state.engines
    if not e.ocr:
        raise APIError(500, "No OCR engines loaded.", "server_error")
    photo, options = await read_photo(request)
    engine = e.ocr.get(options.get("model")) or next(iter(e.ocr.values()))
    lines = await run_engine(engine.read, photo, "OCR")
    return {"object": "ocr", "model": engine.get_model_id(), "text": as_text(lines),
            "lines": [{"text": line.text, "score": round(line.score, 3), "box": list(line.box)} for line in lines]}
