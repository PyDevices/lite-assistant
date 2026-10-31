"""
A user's files, as OpenAI's Files API gives them. A file of any purpose
but "vision" goes in the user's notes folder, which search_notes searches,
so uploading a document is how to add it to their notes, and notes put in
that folder directly are listed too. Files of purpose "vision" go in their
media folder, which it doesn't search: the photos, voice clips and
documents of their saved chats are kept there, one folder a conversation,
in place of base64 in the conversation's items.

A file's id comes from its folder and its path in it, so a file keeps its
id until it's moved or renamed.
"""
import base64
import binascii
import hashlib
import mimetypes
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

from .users import User

# The purpose of the files kept in the media folder; any other goes in the notes folder
MEDIA = "vision"
NOTES = "user_data"
# The largest file an upload may be, in bytes
MAX_UPLOAD = 50 * 1024 * 1024


class Files:
    def __init__(self, user: User):
        self.user = user
        self.folders = {NOTES: user.folder("notes"), MEDIA: user.folder("media")}

    @staticmethod
    def file_id(purpose: str, path: str) -> str:
        return "file-" + hashlib.sha256(f"{purpose}/{path}".encode()).hexdigest()[:24]

    def object(self, purpose: str, path: str) -> Dict[str, Any]:
        """ A file as the API gives it; its filename is its path in its folder. """
        full = os.path.join(self.folders[purpose], path)
        stat = os.stat(full)
        return {"id": self.file_id(purpose, path), "object": "file", "bytes": stat.st_size,
                "created_at": int(stat.st_mtime), "filename": path, "purpose": purpose, "status": "processed"}

    def paths(self, purpose: Optional[str] = None) -> List[Tuple[str, str]]:
        """ Each file's purpose and path in its folder, of a purpose, or of every purpose. """
        found = []
        for kind, folder in self.folders.items():
            if purpose not in (None, kind):
                continue
            for root, _, names in os.walk(folder):
                for name in names:
                    if not name.startswith("."):
                        found.append((kind, os.path.relpath(os.path.join(root, name), folder).replace(os.sep, "/")))
        return sorted(found)

    def list(self, purpose: Optional[str] = None) -> List[Dict[str, Any]]:
        """ The files, oldest first. """
        return sorted((self.object(kind, path) for kind, path in self.paths(purpose)),
                      key=lambda f: (f["created_at"], f["filename"]))

    def find(self, file_id: str) -> Optional[Tuple[Dict[str, Any], str]]:
        """ A file, and where it is, by its id. """
        for kind, path in self.paths():
            if self.file_id(kind, path) == file_id:
                return self.object(kind, path), os.path.join(self.folders[kind], path)
        return None

    def read(self, file_id: str) -> Optional[Tuple[Dict[str, Any], bytes]]:
        found = self.find(file_id)
        if not found:
            return None
        with open(found[1], "rb") as f:
            return found[0], f.read()

    def add(self, filename: str, data: bytes, purpose: str = NOTES, folder: str = "") -> Dict[str, Any]:
        """
        Saves a file, in a folder within the purpose's if given, named as
        filename is, but for characters a file name can't have; a file of
        that name already there isn't replaced: the new one gets a number.
        """
        purpose = MEDIA if purpose == MEDIA else NOTES
        name = re.sub(r'[\x00-\x1f<>:"/\\|?*]', "_", os.path.basename(filename or "")).strip(" .") or "file"
        base = os.path.join(self.folders[purpose], folder)
        os.makedirs(base, exist_ok=True)
        stem, ext = os.path.splitext(name)
        number = 1
        while os.path.exists(os.path.join(base, name)):
            number += 1
            name = f"{stem} ({number}){ext}"
        with open(os.path.join(base, name), "wb") as f:
            f.write(data)
        return self.object(purpose, "/".join(filter(None, (folder, name))))

    def delete(self, file_id: str) -> bool:
        found = self.find(file_id)
        if found:
            os.remove(found[1])
        return bool(found)

    def delete_folder(self, folder: str) -> None:
        """ Deletes a folder of media, such as a conversation's. """
        path = os.path.join(self.folders[MEDIA], folder)
        if folder and os.path.isdir(path):
            shutil.rmtree(path)

    def media_type(self, path: str) -> str:
        return mimetypes.guess_type(path)[0] or "application/octet-stream"


# --- Chat media as files ------------------------------------------------------

DATA_URL = re.compile(r"data:([\w.+-]+/[\w.+-]+)?(;[^,]*)?,(.*)", re.S)


def data_url_bytes(url: str) -> Optional[Tuple[str, bytes]]:
    """ A base64 data: URL's media type and bytes, or None for anything else. """
    found = DATA_URL.fullmatch(url or "")
    if not found or "base64" not in (found.group(2) or ""):
        return None
    try:
        return found.group(1) or "application/octet-stream", base64.b64decode(found.group(3), validate=False)
    except (binascii.Error, ValueError):
        return None


def extension(media_type: str) -> str:
    return {"image/jpeg": ".jpg", "audio/wav": ".wav", "audio/x-wav": ".wav"}.get(
        media_type, mimetypes.guess_extension(media_type) or ".bin")


def to_files(items: List[Dict[str, Any]], files: Files, folder: str) -> List[Dict[str, Any]]:
    """
    The items with the photos, voice clips and documents in them saved as
    media files in folder, and each part naming its file by file_id instead:
    input_image and input_file as OpenAI's take one, and input_audio and an
    image_generation_call's result likewise.
    """
    saved = []
    for item in items:
        item = dict(item)
        if item.get("type") == "image_generation_call" and item.get("result"):
            data = base64.b64decode(item["result"])
            name = "image" + (".jpg" if item["result"].startswith("/9j/") else ".png")
            item["file_id"] = files.add(name, data, MEDIA, folder)["id"]
            item["result"] = ""
        elif item.get("type") == "message" and isinstance(item.get("content"), list):
            item["content"] = [to_file(part, files, folder) for part in item["content"]]
        saved.append(item)
    return saved


def to_file(part: Dict[str, Any], files: Files, folder: str) -> Dict[str, Any]:
    kind = part.get("type")
    if kind == "input_image" and (found := data_url_bytes(part.get("image_url", ""))):
        rest = {k: v for k, v in part.items() if k != "image_url"}
        return {**rest, "file_id": files.add("image" + extension(found[0]), found[1], MEDIA, folder)["id"]}
    if kind == "input_file" and (found := data_url_bytes(part.get("file_data", ""))):
        rest = {k: v for k, v in part.items() if k != "file_data"}
        return {**rest, "file_id": files.add(part.get("filename") or "file", found[1], MEDIA, folder)["id"]}
    if kind == "input_audio" and (part.get("input_audio") or {}).get("data"):
        audio = part["input_audio"]
        audio_format = audio.get("format") or "wav"
        added = files.add(f"clip.{audio_format}", base64.b64decode(audio["data"]), MEDIA, folder)
        return {**part, "input_audio": {"file_id": added["id"], "format": audio_format}}
    return part


def from_files(items: List[Dict[str, Any]], files: Files, missing: str = "say") -> List[Dict[str, Any]]:
    """
    The items with each file_id in their parts read back as the base64 the
    engines take. A file that's gone the model is told of, or with missing
    "raise", raises FileNotFoundError, as a request naming one should.
    """
    def read(file_id: str) -> Tuple[str, str]:
        found = files.read(file_id)
        if not found:
            raise FileNotFoundError(file_id)
        return files.media_type(found[0]["filename"]), base64.b64encode(found[1]).decode()

    restored = []
    for item in items:
        if item.get("type") != "message" or not isinstance(item.get("content"), list):
            restored.append(item)
            continue
        content = []
        for part in item["content"]:
            try:
                if part.get("type") == "input_image" and part.get("file_id"):
                    media_type, data = read(part["file_id"])
                    part = {**part, "image_url": f"data:{media_type};base64,{data}"}
                elif part.get("type") == "input_file" and part.get("file_id"):
                    media_type, data = read(part["file_id"])
                    part = {**part, "file_data": f"data:{media_type};base64,{data}"}
                elif part.get("type") == "input_audio" and (part.get("input_audio") or {}).get("file_id"):
                    _, data = read(part["input_audio"]["file_id"])
                    part = {**part, "input_audio": {"data": data, "format": part["input_audio"].get("format", "wav")}}
            except FileNotFoundError as gone:
                if missing == "raise":
                    raise
                part = {"type": "input_text", "text": f"(A file that was here, {gone}, has been deleted.)"}
            content.append(part)
        restored.append({**item, "content": content})
    return restored
