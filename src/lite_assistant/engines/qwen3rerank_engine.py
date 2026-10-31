"""
Qwen3-Reranker-0.6B reranking engine, on LiteRT.
"""
import os
import threading
import time
from typing import List, Optional

import numpy as np

from . import tflite
from .rerank_base import RerankEngine
from ..paths import cached_hf_file

# Qwen's Qwen3-Reranker-0.6B, converted by litert-community with its token
# embeddings looked up outside the graph and its yes/no head built in; Apache-2.0
REPO = "litert-community/Qwen3-Reranker-0.6B-LiteRT"
GRAPH, TABLE = "qwen3rerank_gpu_fp16.tflite", "embeddings_fp16.bin"
TOKENIZER_REPO, TOKENIZER = "Qwen/Qwen3-Reranker-0.6B", "tokenizer.json"
TOKENS, HIDDEN, VOCAB = 256, 1024, 151669
# Qwen's prompt for the reranker: it answers yes or no to whether the document meets the query
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
          "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


class Qwen3RerankEngine(RerankEngine):
    """
    Qwen3-Reranker-0.6B: reads the query and a document together, which
    judges relevance better than comparing their embeddings, at a cost of
    one pass of a 0.6B model per document, up to 256 tokens of both.
    """
    MODEL_ID = "qwen3-reranker-0.6b"

    def __init__(self):
        self.runner = None
        paths = tflite.files("Qwen3-Reranker", REPO, [GRAPH, TABLE])
        tokenizer = cached_hf_file(TOKENIZER_REPO, TOKENIZER)
        if not paths or not tokenizer:
            if paths:
                print(f"[Qwen3-Reranker] {TOKENIZER} isn't downloaded; run 'lite-assistant setup'.")
            return
        try:
            from ai_edge_litert.interpreter import Interpreter
            from tokenizers import Tokenizer
        except ImportError as e:
            print(f"[Qwen3-Reranker] {e.name} isn't installed. Skipping.")
            return
        print(f"[Qwen3-Reranker] Loading {GRAPH}...")
        start = time.time()
        interpreter = Interpreter(model_path=paths[0], num_threads=os.cpu_count())
        self.runner = interpreter.get_signature_runner()
        self.input = next(iter(self.runner.get_input_details()))
        # Each token's embedding row, looked up here rather than in the graph
        self.table = np.memmap(paths[1], dtype="<f2", mode="r", shape=(VOCAB, HIDDEN))
        self.tokenizer = Tokenizer.from_file(tokenizer)
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.prefix = self.tokenizer.encode(PREFIX, add_special_tokens=False).ids
        self.suffix = self.tokenizer.encode(SUFFIX, add_special_tokens=False).ids
        self.lock = threading.Lock()
        print(f"[Qwen3-Reranker] Loaded in {time.time() - start:.1f}s.")

    def ids(self, query: str, document: str, instruction: str) -> List[int]:
        """ The prompt's tokens, the document cut short to fit the graph's 256. """
        asked = self.tokenizer.encode(f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: ",
                                      add_special_tokens=False).ids
        room = TOKENS - len(self.prefix) - len(asked) - len(self.suffix)
        if room < 1:
            raise ValueError("the query is too long to rerank by")
        said = self.tokenizer.encode(document, add_special_tokens=False).ids[:room]
        return self.prefix + asked + said + self.suffix

    def rank(self, query: str, documents: List[str], instruction: Optional[str] = None) -> List[float]:
        if not self.runner:
            raise RuntimeError("Qwen3-Reranker engine is not available.")
        scores = []
        with self.lock:
            for document in documents:
                ids = self.ids(query, document, instruction or INSTRUCTION)
                # Padded on the right: the graph is causal, so the last real token never sees the padding
                embeds = np.zeros((1, TOKENS, HIDDEN), np.float32)
                embeds[0, :len(ids)] = self.table[ids]
                no, yes = next(iter(self.runner(**{self.input: embeds}).values()))[0, len(ids) - 1]
                scores.append(float(1 / (1 + np.exp(no - yes))))
        return scores

    @classmethod
    def setup(cls):
        tflite.download("Qwen3-Reranker", REPO, [GRAPH, TABLE])
        tflite.download("Qwen3-Reranker", TOKENIZER_REPO, [TOKENIZER])
