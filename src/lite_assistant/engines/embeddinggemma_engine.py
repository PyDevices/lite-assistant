"""
EmbeddingGemma 2 embedding engine, on LiteRT-LM.
"""
import base64
import os
import threading
import time
from typing import Any, Dict, List, Optional, Union
from .embedding_base import EmbeddingEngine
from ..media import decode_data_url, image_url
from ..paths import cache_dir, cached_hf_file

# The 740M model embeds text, images and audio into one vector space
REPO = "litert-community/embeddinggemma-2-740m-litert-lm"
FILE = "embeddinggemma-2-740m.litertlm"
# Matryoshka sizes: any prefix of these lengths is itself a usable embedding
DIMENSIONS = [128, 256, 512, 768]
# Shared with the Gemma engine's compiled caches
CACHE_DIR = cache_dir("litertlm")


class EmbeddingGemmaEngine(EmbeddingEngine):
    """
    Google's EmbeddingGemma 2, run by LiteRT-LM on the CPU, where it is fast
    enough and leaves the GPU to Gemma.
    """
    MODEL_ID = FILE[:-len(".litertlm")]

    def __init__(self):
        self.engine = None
        # LiteRT-LM's embedding engine isn't documented as thread-safe
        self._lock = threading.Lock()
        try:
            import litert_lm
        except ImportError:
            print("[EmbeddingGemma] litert-lm package is not installed. Skipping EmbeddingGemmaEngine.")
            return
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
        path = cached_hf_file(REPO, FILE)
        if not path:
            print(f"[EmbeddingGemma] {FILE} isn't downloaded; run 'lite-assistant setup'.")
            return
        print(f"[EmbeddingGemma] Loading {FILE}...")
        start = time.time()
        # LiteRT-LM refuses a cache directory that doesn't exist yet
        os.makedirs(CACHE_DIR, exist_ok=True)
        try:
            self.engine = litert_lm.EmbeddingEngine(
                path, backend=litert_lm.Backend.CPU(), cache_dir=CACHE_DIR,
                # The image and audio encoders load the first time an input needs them
                vision_backend=litert_lm.Backend.CPU(), audio_backend=litert_lm.Backend.CPU())
        except Exception as e:
            print(f"[EmbeddingGemma] Failed to load: {e}")
            return
        print(f"[EmbeddingGemma] Loaded in {time.time() - start:.1f}s.")

    def get_model_id(self) -> str:
        return self.MODEL_ID

    def get_dimensions(self) -> List[int]:
        return DIMENSIONS

    @staticmethod
    def _to_content(item: Union[str, Dict[str, Any]]):
        import litert_lm
        if isinstance(item, str):
            return item
        kind = item.get("type")
        if kind == "text":
            return item.get("text", "")
        if kind == "image_url":
            _, image = decode_data_url(image_url(item))
            return litert_lm.Content.ImageBytes(image)
        if kind == "input_audio":
            return litert_lm.Content.AudioBytes(base64.b64decode(item["input_audio"]["data"]))
        raise ValueError(f"can't embed a {kind!r} part")

    def embed(self, item: Union[str, Dict[str, Any]], dimensions: Optional[int] = None) -> List[float]:
        if not self.engine:
            raise RuntimeError("EmbeddingGemma engine is not available.")
        import litert_lm
        options = litert_lm.EmbeddingOptions(normalize=True, output_size=dimensions)
        content = self._to_content(item)
        with self._lock:
            return self.engine.compute_embedding(content, options).embedding

    @classmethod
    def setup(cls):
        if not cached_hf_file(REPO, FILE):
            from huggingface_hub import hf_hub_download
            print(f"[EmbeddingGemma] Downloading {FILE} from {REPO}...")
            hf_hub_download(REPO, FILE)
