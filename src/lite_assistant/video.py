"""
The video endpoint: what people are doing across a video's frames, from a
video file or from photos a board sends as it takes them.
"""
import io
import os
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, Request

from .api_common import APIError, run_engine

router = APIRouter()
# The longest video taken, in seconds, and the most frames
MAX_SECONDS, MAX_FRAMES = 60, 600
# A session's memory is dropped once it's gone unused this long, in seconds, and the
# oldest is dropped beyond this many
SESSION_IDLE, MAX_SESSIONS = 120, 32


class Sessions:
    """
    Each session's memory of the frames sent so far, with how many, by its
    id, so a board can send them one at a time.
    """
    def __init__(self):
        self.memories: Dict[str, Tuple[Any, float]] = {}
        self.lock = threading.Lock()

    def take(self, session: str) -> Optional[Any]:
        """ A session's memory, if it has one; it's put back with keep once the frames are watched. """
        with self.lock:
            now = time.monotonic()
            for stale in [s for s, (_, used) in self.memories.items() if now - used > SESSION_IDLE]:
                del self.memories[stale]
            found = self.memories.pop(session, None)
            return found[0] if found else None

    def keep(self, session: str, memory: Any):
        with self.lock:
            self.memories[session] = (memory, time.monotonic())
            while len(self.memories) > MAX_SESSIONS:
                del self.memories[min(self.memories, key=lambda s: self.memories[s][1])]


def video_frames(video: bytes, fps: float) -> List[Any]:
    """ A video's frames, at fps, as PIL images, decoded by ffmpeg. Raises ValueError if it can't be. """
    from PIL import Image
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("decoding video needs ffmpeg installed; send the frames as photos instead")
    # A file, not a pipe, since an MP4 may keep its index at the end
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "video")
        with open(path, "wb") as f:
            f.write(video)
        made = subprocess.run([ffmpeg, "-v", "error", "-t", str(MAX_SECONDS), "-i", path, "-vf",
                               f"fps={fps},scale=-2:240", "-frames:v", str(MAX_FRAMES), "-f", "image2pipe",
                               "-vcodec", "bmp", "-"], capture_output=True, timeout=120)
    if made.returncode or not made.stdout:
        raise ValueError(made.stderr.decode(errors="replace").strip()[-200:] or "no frames")
    frames, data = [], made.stdout
    while data:
        # Each BMP says its own size, after its "BM"
        size = int.from_bytes(data[2:6], "little")
        frames.append(Image.open(io.BytesIO(data[:size])).convert("RGB"))
        data = data[size:]
    return frames


def media_frames(media: List[Tuple[bytes, str]], fps: float) -> List[Any]:
    """ The frames in what was sent: each photo is a frame, and each video's frames at fps. """
    from .engines.tflite import open_photo
    frames = []
    for data, kind in media:
        if kind.startswith("image/"):
            frames.append(open_photo(data))
        elif kind.startswith(("video/", "audio/ogg", "application/ogg")):
            frames += video_frames(data, fps)
        else:
            # Without a type, it's a photo if Pillow reads it
            try:
                frames.append(open_photo(data))
            except ValueError:
                frames += video_frames(data, fps)
    if not frames:
        raise ValueError("no frames")
    return frames[:MAX_FRAMES]


@router.post("/v1/video/actions")
async def recognize_actions(request: Request):
    """
    What people are doing across a video's frames, such as jumping jacks
    or clapping: not an OpenAI endpoint. Takes a video file (with ffmpeg
    installed), or frames as photos, in order: multipart file fields, or
    one photo as the body. Options, as query parameters or form fields:
    session, any id, to go on from the frames sent before under it; top,
    how many actions to give (1 to 20, default 5); and fps, the rate a
    video is sampled at (the engine's own by default, 5).
    """
    e = request.app.state.engines
    if not e.action:
        raise APIError(500, "No action engines loaded.", "server_error")
    options: Dict[str, Any] = dict(request.query_params)
    kind = request.headers.get("content-type", "").split(";")[0].strip().lower()
    media: List[Tuple[bytes, str]] = []
    if kind == "multipart/form-data":
        form = await request.form()
        options.update({k: v for k, v in form.items() if isinstance(v, str)})
        for upload in form.getlist("file"):
            if not isinstance(upload, str):
                media.append((await upload.read(), (upload.content_type or "").lower()))
    elif kind.startswith(("image/", "video/")) or kind == "application/octet-stream":
        media.append((await request.body(), kind))
    if not any(data for data, _ in media):
        raise APIError(400, "Send a video or photos, as multipart file fields or as the body.")
    engine = e.action.get(options.get("model")) or next(iter(e.action.values()))
    try:
        top = int(options.get("top", 5))
        fps = float(options.get("fps", engine.FPS))
    except (TypeError, ValueError):
        raise APIError(400, "top and fps must be numbers.")
    if not 1 <= top <= 20 or not 0 < fps <= 30:
        raise APIError(400, "top must be from 1 to 20, and fps above 0 and at most 30.")
    session = options.get("session")
    sessions = request.app.state.__dict__.setdefault("action_sessions", Sessions())

    def watch(_):
        # The engine's memory, and how many frames it has seen
        memory, seen = (sessions.take(session) if session else None) or (engine.start(), 0)
        try:
            frames = media_frames(media, fps)
            for frame in frames:
                chances = engine.watch(memory, frame)
            seen += len(frames)
        finally:
            if session:
                sessions.keep(session, (memory, seen))
        return chances, seen
    chances, frames = await run_engine(watch, b"", "Actions", "video")
    result = {"object": "video.actions", "model": engine.get_model_id(), "frames": frames,
              "actions": [{"label": engine.labels[i], "score": round(float(chances[i]), 4)}
                          for i in np.argsort(-chances)[:top]]}
    if session:
        result["session"] = session
    return result
