"""
What the server's HTTP endpoints share: OpenAI-style errors, and reading the
photo or audio a request sends to an engine.
"""
from typing import Any, Dict, Optional, Tuple

import anyio
from fastapi import Request

from .media import MediaError, decode_data_url, resolve_image_url


class APIError(Exception):
    """ Ends a request with an OpenAI-style error body. """
    def __init__(self, status: int, message: str, kind: str = "invalid_request_error",
                 code: Optional[str] = None):
        super().__init__(message)
        self.status, self.message, self.kind, self.code = status, message, kind, code


async def read_photo(request: Request) -> Tuple[bytes, Dict[str, Any]]:
    """
    The photo a request sends, and its options. The photo comes as the
    request body itself (e.g. Content-Type: image/jpeg), as a multipart
    file, or in JSON as an image URL; options come as query parameters,
    form fields or JSON fields.
    """
    options: Dict[str, Any] = dict(request.query_params)
    kind = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if kind.startswith("image/") or kind == "application/octet-stream":
        photo = await request.body()
    elif kind == "multipart/form-data":
        form = await request.form()
        options.update({k: v for k, v in form.items() if isinstance(v, str)})
        # OpenAI's image endpoints call the field image
        upload = form.get("file") or form.get("image")
        if upload is None or isinstance(upload, str):
            raise APIError(400, "The form needs the photo as a file field.")
        photo = await upload.read()
    elif kind == "application/json":
        try:
            fields = await request.json()
        except ValueError:
            raise APIError(400, "The body isn't valid JSON.")
        if not isinstance(fields, dict):
            raise APIError(400, "The body must be a JSON object.")
        options.update({k: v for k, v in fields.items() if k != "image"})
        image = fields.get("image")
        if isinstance(image, dict):
            image = image.get("url")
        if not isinstance(image, str) or not image:
            raise APIError(400, "image must be a data: or http(s) URL.")
        try:
            photo = decode_data_url(await anyio.to_thread.run_sync(resolve_image_url, image))[1]
        except MediaError as exc:
            raise APIError(400, str(exc))
    else:
        raise APIError(400, "Send the photo as an image body, a multipart file, or JSON with an image URL.")
    if not photo:
        raise APIError(400, "The photo is empty.")
    return photo, options


async def run_engine(work, media: bytes, what: str, kind: str = "photo"):
    """ An engine's work on a photo or audio, off the event loop, with a 400 for one it can't read. """
    try:
        return await anyio.to_thread.run_sync(work, media)
    except ValueError as exc:
        raise APIError(400, f"Can't read the {kind}: {exc}")
    except Exception as exc:
        print(f"{what} error: {exc}")
        raise APIError(500, str(exc), "server_error")
