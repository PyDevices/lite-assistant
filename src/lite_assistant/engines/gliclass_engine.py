"""
GLiClass zero-shot text classification engine, on LiteRT.
"""
import os
import threading
import time
from typing import List, Optional, Sequence

import numpy as np
from .classifier_base import ClassifierEngine
from ..paths import cached_hf_file

# Knowledgator's GLiClass-Edge v3.0, a 32.7M ModernBERT, converted by litert-community; Apache-2.0
REPO = "litert-community/GLiClass-Edge-v3.0-LiteRT"
# The graphs for up to 128 and 256 tokens of labels, prompt and text
GRAPHS = {128: "gliclass_edge_v3_s128_fp32.tflite", 256: "gliclass_edge_v3_s256_fp32.tflite"}
TABLE = "host_assets/tok_embeddings_fp16.bin"
TOKENIZER = "host_assets/tokenizer.json"
LABEL_SLOTS, HIDDEN, VOCAB = 25, 384, 50370
CLS_ID, SEP_ID, PAD_ID, LABEL_ID = 50281, 50282, 50283, 50368
LABEL_TOKEN, TEXT_SEP_TOKEN = "<<LABEL>>", "<<SEP>>"
# More threads were slower on a 12-thread PC: 22 ms with 4, 27 ms with 8
THREADS = min(4, os.cpu_count() or 1)


class GLiClassEngine(ClassifierEngine):
    """
    GLiClass-Edge, which scores up to 25 labels against a text in one pass,
    in about 20 ms on the CPU: fast enough to route a voice command before
    an LLM would have read it. English only.
    """
    MODEL_ID = "gliclass-edge-v3.0"

    def __init__(self):
        self.runners = {}
        # One request at a time: the interpreters' tensors are shared
        self._lock = threading.Lock()
        try:
            from ai_edge_litert.interpreter import Interpreter
            from tokenizers import Tokenizer
        except ImportError as e:
            print(f"[GLiClass] {e.name} is not installed. Skipping GLiClassEngine.")
            return
        paths = {name: cached_hf_file(REPO, name) for name in (*GRAPHS.values(), TABLE, TOKENIZER)}
        if not all(paths.values()):
            print(f"[GLiClass] {REPO} isn't downloaded; run 'lite-assistant setup'.")
            return
        print("[GLiClass] Loading GLiClass-Edge v3.0...")
        start = time.time()
        try:
            for window, name in GRAPHS.items():
                interpreter = Interpreter(model_path=paths[name], num_threads=THREADS)
                self.runners[window] = interpreter.get_signature_runner("serving_default")
            # Each token's embedding row, looked up here rather than in the graph
            self.table = np.memmap(paths[TABLE], dtype="<f2", mode="r", shape=(VOCAB, HIDDEN))
            self.tokenizer = Tokenizer.from_file(paths[TOKENIZER])
            self.tokenizer.no_truncation()
            self.tokenizer.no_padding()
        except Exception as e:
            print(f"[GLiClass] Failed to load: {e}")
            self.runners = {}
            return
        print(f"[GLiClass] Loaded in {time.time() - start:.1f}s.")

    def get_model_id(self) -> str:
        return self.MODEL_ID

    def logits(self, text: str, labels: Sequence[str], prompt: Optional[str] = None) -> List[float]:
        if not self.runners:
            raise RuntimeError("GLiClass classification engine is not available.")
        labels = list(labels)
        if not labels:
            raise ValueError("at least one label is needed")
        if len(labels) > LABEL_SLOTS:
            raise ValueError(f"{len(labels)} labels; the model takes at most {LABEL_SLOTS}")
        # As the gliclass pipeline builds it: labels, then the prompt and text with nothing between
        request = "".join(LABEL_TOKEN + label for label in labels) + TEXT_SEP_TOKEN + (prompt or "") + text
        ids = self.tokenizer.encode(request).ids
        positions = [i for i, token in enumerate(ids) if token == LABEL_ID]
        if len(positions) != len(labels):
            raise ValueError(f"the text, prompt or a label contains {LABEL_TOKEN}")
        window = next((w for w in GRAPHS if len(ids) <= w), None)
        if window is None:
            raise ValueError(f"the labels, prompt and text are {len(ids)} tokens; the model takes at most "
                             f"{max(GRAPHS)}")

        padded = np.full(window, PAD_ID, dtype=np.int64)
        padded[:len(ids)] = ids
        embeds = np.ascontiguousarray(self.table[padded].astype(np.float32)[None])
        mask = np.zeros((1, window), np.float32)
        mask[0, :len(ids)] = 1
        # Row k marks where the k-th label's token is
        routing = np.zeros((1, LABEL_SLOTS, window), np.float32)
        routing[0, np.arange(len(positions)), positions] = 1
        with self._lock:
            out = self.runners[window](inputs_embeds=embeds, attention_mask=mask, label_routing=routing)
        return [float(v) for v in np.asarray(out["logits"]).ravel()[:len(labels)]]

    @classmethod
    def setup(cls):
        from huggingface_hub import hf_hub_download
        for name in (*GRAPHS.values(), TABLE, TOKENIZER):
            if not cached_hf_file(REPO, name):
                print(f"[GLiClass] Downloading {name} from {REPO}...")
                hf_hub_download(REPO, name)
