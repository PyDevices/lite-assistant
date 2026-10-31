"""
The image endpoints: editing a photo, as OpenAI's /v1/images/edits does, and
scoring how good photos are. Also brightening a dark photo before the
detector sees it.
"""
import base64
import io
import time
from typing import Any, Dict, Optional

import anyio
from fastapi import APIRouter, Request
from fastapi.responses import Response

from .api_common import APIError, read_photo, run_engine
from .media import MediaError, decode_data_url, resolve_image_url

router = APIRouter()

# A photo darker than this, as its mean brightness from 0 to 1, is stretched
# before detection sees it
DARK = 0.2
# The stretch makes this percentile of a photo's values white
STRETCH_PERCENTILE = 99


def edit_photo(engine, data: bytes, prompt: Optional[str] = None, mask=None) -> "Image.Image":
    """
    A photo's bytes edited by an editing engine, which takes RGB. A photo
    with transparency, such as one with its background removed, keeps it,
    stretched to the edit's size, unless the edit makes its own.
    """
    from PIL import Image, ImageOps

    from .engines.tflite import open_photo
    edited = engine.edit(open_photo(data), prompt, mask)
    try:
        source = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    except Exception:
        return edited
    if edited.mode != "RGBA" and (source.mode in ("RGBA", "LA", "PA") or "transparency" in source.info):
        edited = edited.convert("RGB")
        edited.putalpha(source.convert("RGBA").getchannel("A").resize(edited.size, Image.LANCZOS))
    return edited


def edit_engine(e, model: Optional[str]):
    """ The editing engine a request's model names, by its id or what it does ("brighten"). """
    if model in e.edit:
        return e.edit[model]
    for engine_id, engine in e.edit.items():
        if model and model in (getattr(engine, "TASK", None), getattr(engine, "MODEL_ID", None)):
            return engine
    offered = ", ".join(f"{engine_id} ({getattr(engine, 'TASK', '')})" for engine_id, engine in e.edit.items())
    if not model:
        raise APIError(400, f"model must say which edit to make: {offered}.")
    raise APIError(404, f"No image editing model {model!r}; the server has {offered}.", code="model_not_found")


def png(image) -> bytes:
    out = io.BytesIO()
    image.save(out, "PNG")
    return out.getvalue()


def brightness(photo) -> float:
    """ A photo's mean brightness, from 0 to 1. """
    import numpy as np
    return float(np.asarray(photo.convert("L").resize((64, 64)), np.float32).mean() / 255)


def stretch(photo):
    """ The photo with its contrast stretched, so its brightest values are white. """
    import numpy as np
    from PIL import Image
    values = np.asarray(photo, np.float32)
    top = max(float(np.percentile(values, STRETCH_PERCENTILE)), 1.0)
    return Image.fromarray(np.clip(values * (255 / top), 0, 255).astype(np.uint8))


def brighten_if_dark(photo: bytes, option: Any) -> tuple:
    """
    The photo, brightened for a detector if the option says to: "auto" (the
    default) when it's dark, true always, false never. Returns it, as bytes,
    and whether it was brightened. It's a plain contrast stretch: on COCO
    photos darkened to 7% and given sensor noise, YOLOX found 41 of their 61
    objects after it, 31 after CPGA-Net and 24 with neither, and CPGA-Net
    cost it 17 objects on photos that weren't dark, where the stretch cost 1.
    """
    from .engines.tflite import open_photo
    choice = str(option if option is not None else "auto").lower()
    if choice in ("false", "0", "no"):
        return photo, False
    image = open_photo(photo)
    if choice == "auto" and brightness(image) >= DARK:
        return photo, False
    return png(stretch(image)), True


async def mask_of(segmenter, photo: bytes, prompt: str):
    """ A mask, white where a segmentation engine finds what a prompt names, such as "the dog". """
    from .detection import subject
    from .segmentation import found_mask
    probability = await run_engine(lambda data: segmenter.segment(data, [subject(prompt)])[0], photo, "Segmentation")
    found = found_mask(probability)
    if found is None:
        raise APIError(400, f"Couldn't find {prompt!r} in the photo.")
    return found


async def read_mask(request: Request, options: Dict[str, Any]):
    """ The mask a request sends, as a multipart file or a data or http(s) URL, or None. """
    from .engines.tflite import open_photo
    data = None
    if request.headers.get("content-type", "").startswith("multipart/form-data"):
        upload = (await request.form()).get("mask")
        if upload is not None and not isinstance(upload, str):
            data = await upload.read()
    if data is None and isinstance(options.get("mask"), str) and options["mask"]:
        try:
            data = decode_data_url(await anyio.to_thread.run_sync(resolve_image_url, options["mask"]))[1]
        except MediaError as exc:
            raise APIError(400, f"mask: {exc}")
    if not data:
        return None
    try:
        return open_photo(data).convert("L")
    except ValueError as exc:
        raise APIError(400, f"Can't read the mask: {exc}")


@router.post("/v1/images/edits")
async def edit_image(request: Request):
    """
    Edits a photo, as OpenAI's endpoint does, but with the model naming the
    edit: an editing engine's id, or what it does, such as "brighten". The
    photo comes as read_photo takes it, in an image or file field; prompt
    says what to change, for the edits that need it, and mask, white where
    to change, can say it instead. response_format is b64_json (the
    default), as OpenAI's, or png for the image itself.
    """
    e = request.app.state.engines
    if not e.edit:
        raise APIError(500, "No image editing engines loaded.", "server_error")
    photo, options = await read_photo(request)
    engine = edit_engine(e, options.get("model"))
    response_format = options.get("response_format", "b64_json")
    if response_format not in ("b64_json", "png"):
        raise APIError(400, "response_format must be b64_json or png; this server doesn't host image URLs.")
    mask = await read_mask(request, options)
    prompt = options.get("prompt") or None
    if getattr(engine, "MASKED", False) and mask is None and not (prompt and e.segment):
        raise APIError(400, "This edit needs a mask, or a prompt saying what to change"
                       + (" and a segmentation engine to find it." if prompt else "."))
    if getattr(engine, "MASKED", False) and mask is None:
        mask = await mask_of(next(iter(e.segment.values())), photo, prompt)

    def work(data: bytes) -> bytes:
        return png(edit_photo(engine, data, prompt, mask))
    edited = await run_engine(work, photo, "Image edit")
    if response_format == "png":
        return Response(edited, media_type="image/png")
    return {"created": int(time.time()), "model": engine.get_model_id(),
            "data": [{"b64_json": base64.b64encode(edited).decode("ascii")}]}


@router.post("/v1/images/quality")
async def score_images(request: Request):
    """
    How good one or more photos are, from 1 to 10: technical (sharpness,
    noise, exposure) and aesthetic. Not an OpenAI endpoint. The photos come
    as multipart file fields, several of them for a burst, or as one photo
    as read_photo takes it; best is the index of the technically best, the
    one a board would keep.
    """
    e = request.app.state.engines
    if not e.quality:
        raise APIError(500, "No image quality engines loaded.", "server_error")
    photo, options = await read_photo(request)
    photos = [photo]
    if request.headers.get("content-type", "").startswith("multipart/form-data"):
        uploads = [u for u in (await request.form()).getlist("file") if not isinstance(u, str)]
        if len(uploads) > 1:
            photos = []
            for upload in uploads:
                # read_photo has read one of them already
                await upload.seek(0)
                photos.append(await upload.read())
    engine = e.quality.get(options.get("model")) or next(iter(e.quality.values()))
    data = []
    for index, data_bytes in enumerate(photos):
        scores = await run_engine(engine.score, data_bytes, "Image quality", f"photo {index}")
        data.append({"index": index, **scores})
    best = max(data, key=lambda d: (d["technical"], d["aesthetic"]))["index"]
    return {"object": "list", "model": engine.get_model_id(), "data": data, "best": best}
