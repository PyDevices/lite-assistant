"""
Images and files in OpenAI content parts. An image_url part may name an
http(s) URL, which is fetched here, so engines only ever see the image's
bytes as a data: URL. A file part, a document attached to a message, is
read here into a text part.
"""
import base64
import hashlib
import re
from collections import OrderedDict
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

MAX_IMAGE_BYTES = 20 * 1024 * 1024
FETCH_TIMEOUT = 10


class MediaError(ValueError):
    """ An image that can't be used: a malformed data: URL, or a fetch that failed. """


def image_url(part: Dict[str, Any]) -> str:
    """ The URL of an image_url part, which OpenAI clients send as {"url": ...} or, loosely, a bare string. """
    value = part.get("image_url")
    url = value.get("url") if isinstance(value, dict) else value
    if not isinstance(url, str) or not url:
        raise MediaError("image_url part has no url")
    return url


def decode_data_url(url: str) -> Tuple[str, bytes]:
    """ The media type and bytes of a data: URL. """
    try:
        header, data = url[len("data:"):].split(",", 1)
    except ValueError:
        raise MediaError("malformed data: URL")
    media_type, *params = header.split(";")
    try:
        payload = base64.b64decode(data, validate=True) if "base64" in params else urllib.parse.unquote_to_bytes(data)
    except ValueError:
        raise MediaError("data: URL has invalid base64")
    return media_type or "text/plain", payload


def fetch_image(url: str) -> str:
    """ Downloads an http(s) image and returns it as a data: URL. """
    request = urllib.request.Request(url, headers={"User-Agent": "lite-assistant"})
    try:
        with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT) as response:
            media_type = response.headers.get_content_type()
            if not media_type.startswith("image/"):
                raise MediaError(f"{url} is {media_type}, not an image")
            data = response.read(MAX_IMAGE_BYTES + 1)
    except (urllib.error.URLError, OSError) as e:
        raise MediaError(f"couldn't fetch {url}: {getattr(e, 'reason', e)}")
    if len(data) > MAX_IMAGE_BYTES:
        raise MediaError(f"{url} is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
    return f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}"


def resolve_image_url(url: str) -> str:
    """ A data: URL for any image URL a client may send. """
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme == "data":
        decode_data_url(url)  # reject a malformed one here, as a bad request
        return url
    if scheme in ("http", "https"):
        return fetch_image(url)
    raise MediaError(f"unsupported image URL scheme {scheme or url[:20]!r}")


def is_image_part(part: Any) -> bool:
    return isinstance(part, dict) and part.get("type") == "image_url"


def resolve_image_part(part: Dict[str, Any]) -> Dict[str, Any]:
    url = resolve_image_url(image_url(part))
    options = part["image_url"] if isinstance(part["image_url"], dict) else {}
    return {**part, "image_url": {**options, "url": url}}


def resolve_images(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """ The messages with every image_url part's URL replaced by a data: URL. """
    resolved = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(is_image_part(p) for p in content):
            content = [resolve_image_part(p) if is_image_part(p) else p for p in content]
            message = {**message, "content": content}
        resolved.append(message)
    return resolved


def latest_audio(messages: List[Dict[str, Any]]) -> Optional[bytes]:
    """ The last audio clip the user sent in the conversation, as an input_audio part. """
    for message in reversed(messages):
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            continue
        for part in reversed(content):
            if isinstance(part, dict) and part.get("type") == "input_audio":
                try:
                    return base64.b64decode(part["input_audio"]["data"], validate=True)
                except (KeyError, TypeError, ValueError):
                    continue
    return None


def latest_image(messages: List[Dict[str, Any]]) -> Optional[bytes]:
    """
    The last image the user sent in the conversation: the bytes of its
    data: URL, in chat (image_url) or Responses (input_image) form.
    """
    for message in reversed(messages):
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            continue
        for part in reversed(content):
            if not isinstance(part, dict) or part.get("type") not in ("image_url", "input_image"):
                continue
            try:
                url = image_url(part) if part["type"] == "image_url" else part.get("image_url")
                if isinstance(url, str) and url.startswith("data:"):
                    return decode_data_url(url)[1]
            except MediaError:
                continue
    return None


# Attached files' text, by a hash of the file, so a conversation's files aren't read again each turn
_read_files: "OrderedDict[str, str]" = OrderedDict()
READ_FILES_KEPT = 16


def file_text(part: Dict[str, Any], ocr=None) -> str:
    """ The text of a file part, {"type": "file", "file": {"filename", "file_data"}}, for the model. """
    from .documents import attached_text
    file = part.get("file") if isinstance(part.get("file"), dict) else {}
    if not file.get("file_data"):
        raise MediaError("a file part needs file_data; file_id and file_url aren't supported")
    filename = file.get("filename") or "file"
    data = file["file_data"]
    if data.startswith("data:"):
        data = decode_data_url(data)[1]
    else:
        try:
            data = base64.b64decode(data, validate=True)
        except ValueError:
            raise MediaError(f"{filename}'s file_data isn't a data: URL or base64")
    key = hashlib.sha256(filename.encode() + b"\0" + data).hexdigest()
    if key in _read_files:
        _read_files.move_to_end(key)
        return _read_files[key]
    try:
        text = attached_text(filename, data, ocr)
    except ValueError as e:
        raise MediaError(str(e))
    _read_files[key] = text
    while len(_read_files) > READ_FILES_KEPT:
        _read_files.popitem(last=False)
    return text


def resolve_files(messages: List[Dict[str, Any]], ocr=None) -> List[Dict[str, Any]]:
    """ The messages with every file part replaced by a text part of the file's text. """
    resolved = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(isinstance(p, dict) and p.get("type") == "file" for p in content):
            content = [{"type": "text", "text": file_text(p, ocr)} if isinstance(p, dict) and p.get("type") == "file"
                       else p for p in content]
            message = {**message, "content": content}
        resolved.append(message)
    return resolved


def attached_files(messages: List[Dict[str, Any]]) -> List[str]:
    """ The names of the files the user attached in the conversation, whose text is already in it. """
    names = []
    for message in messages:
        content = message.get("content")
        texts = [content] if isinstance(content, str) else \
            [p.get("text", "") for p in content or [] if isinstance(p, dict) and p.get("type") == "text"]
        if message.get("role") == "user":
            names += [m for text in texts for m in re.findall(r"^The attached file (.+?):\n", text, re.M)]
    return names
