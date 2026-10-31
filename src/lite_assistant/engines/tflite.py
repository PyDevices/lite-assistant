"""
What the engines that run .tflite graphs from the Hub share: finding and
downloading their files, loading a graph, and decoding a photo or audio.
"""
import io
import os
import threading
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import numpy as np

from ..paths import cached_hf_file

if TYPE_CHECKING:
    from PIL import Image


def open_photo(image: bytes) -> "Image.Image":
    """ A photo's bytes as an upright RGB image. Raises ValueError if it isn't one. """
    from PIL import Image, ImageOps
    try:
        return ImageOps.exif_transpose(Image.open(io.BytesIO(image))).convert("RGB")
    except Exception as e:
        raise ValueError(f"not an image Pillow can read: {e}")


def open_audio(audio: bytes, rate: int = 16000) -> np.ndarray:
    """
    Audio's bytes, WAV, FLAC, MP3, OGG or another format libsndfile reads,
    as mono float samples at a rate, resampled by soxr as librosa does.
    Raises ValueError if it isn't audio.
    """
    import soundfile
    import soxr
    try:
        samples, found = soundfile.read(io.BytesIO(audio), dtype="float32", always_2d=True)
    except Exception as e:
        raise ValueError(f"not audio libsndfile can read: {e}")
    samples = samples.mean(axis=1)
    if not len(samples):
        raise ValueError("the audio is empty")
    if found != rate:
        samples = soxr.resample(samples, found, rate, quality="HQ")
    return np.ascontiguousarray(samples, np.float32)


def wav(samples: np.ndarray, rate: int = 16000) -> bytes:
    """ Float samples from -1 to 1 as a 16-bit mono WAV. """
    import wave
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return out.getvalue()


def files(tag: str, repo: str, names: Sequence[str]) -> Optional[List[str]]:
    """ The paths of a repo's files in the Hugging Face cache, or None, with a note, if any isn't downloaded. """
    paths = [cached_hf_file(repo, name) for name in names]
    if not all(paths):
        missing = [name for name, path in zip(names, paths) if not path]
        print(f"[{tag}] {', '.join(missing)} isn't downloaded; run 'lite-assistant setup'.")
        return None
    return paths


def download(tag: str, repo: str, names: Sequence[str]) -> None:
    """ Downloads the files of a repo that aren't in the Hugging Face cache yet. """
    from huggingface_hub import hf_hub_download
    for name in names:
        if not cached_hf_file(repo, name):
            print(f"[{tag}] Downloading {name} from {repo}...")
            hf_hub_download(repo, name)


class Graph:
    """
    One .tflite graph on LiteRT's CPU interpreter. Calling it runs the graph
    on its inputs, in order, and returns copies of its outputs, in order;
    one call runs at a time, since the interpreter's tensors are shared.
    """
    def __init__(self, path: str, threads: Optional[int] = None, delegates: bool = True):
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
        options = {} if delegates else {
            "experimental_op_resolver_type": OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES}
        self.interpreter = Interpreter(model_path=path, num_threads=threads or os.cpu_count(), **options)
        self.interpreter.allocate_tensors()
        self.inputs = self.interpreter.get_input_details()
        self.outputs = self.interpreter.get_output_details()
        self._lock = threading.Lock()

    def shape(self, index: int = 0) -> List[int]:
        """ An input's shape. """
        return [int(n) for n in self.inputs[index]["shape"]]

    def __call__(self, *inputs: np.ndarray) -> List[np.ndarray]:
        with self._lock:
            for detail, value in zip(self.inputs, inputs):
                self.interpreter.set_tensor(detail["index"], np.asarray(value, dtype=detail["dtype"]))
            self.interpreter.invoke()
            return [self.interpreter.get_tensor(detail["index"]).copy() for detail in self.outputs]


def load_graphs(tag: str, paths: Dict[str, str], **options) -> Optional[Dict[str, Graph]]:
    """ Loads graphs by name from their paths, timing it; None, with a note, if one fails. """
    try:
        import ai_edge_litert  # noqa: F401
    except ImportError:
        print(f"[{tag}] ai-edge-litert package is not installed.")
        return None
    print(f"[{tag}] Loading {', '.join(os.path.basename(p) for p in paths.values())}...")
    start = time.time()
    try:
        graphs = {name: Graph(path, **options) for name, path in paths.items()}
    except Exception as e:
        print(f"[{tag}] Failed to load: {e}")
        return None
    print(f"[{tag}] Loaded in {time.time() - start:.1f}s.")
    return graphs


def crop(photo: "Image.Image", box: Sequence[float], scale: float = 1.0, aspect: Optional[float] = None):
    """
    The part of a photo in a box (left, top, right and bottom, as fractions
    of its width and height), grown about its center by scale, and widened
    or heightened to an aspect ratio (width over height) if one is given;
    beyond the photo's edges it's black. Returns the crop and its box in
    the photo's pixels, (left, top, width, height).
    """
    left, top, right, bottom = box[0] * photo.width, box[1] * photo.height, box[2] * photo.width, box[3] * photo.height
    x, y = (left + right) / 2, (top + bottom) / 2
    width, height = max(right - left, 1.0) * scale, max(bottom - top, 1.0) * scale
    if aspect:
        width, height = max(width, height * aspect), max(height, width / aspect)
    corner = (x - width / 2, y - height / 2)
    region = tuple(round(v) for v in (corner[0], corner[1], corner[0] + width, corner[1] + height))
    return photo.crop(region), (corner[0], corner[1], width, height)
