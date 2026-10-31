"""
CLIPSeg segmentation engine, on LiteRT.
"""
import json
from typing import List

import numpy as np

from . import tflite
from .segment_base import SegmentEngine

# CIDAS's CLIPSeg rd64 refined, converted by litert-community; Apache-2.0, 277 MB
REPO = "litert-community/CLIPSeg-rd64-LiteRT"
VISION, TEXT, DECODER = "clipseg_vision_fp16.tflite", "clipseg_text_fp16.tflite", "clipseg_decoder.tflite"
EMBEDDING, PROJECTION = "token_embedding_f16.bin", "text_projection_f16.bin"
VOCAB, MERGES = "vocab.json", "merges.txt"
FILES = [VISION, TEXT, DECODER, EMBEDDING, PROJECTION, VOCAB, MERGES]
SIZE = 352
# CLIP's text: up to 77 tokens, from start to end of text, 512 numbers each
CONTEXT, WIDTH = 77, 512
START, END = 49406, 49407
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
# CLIP's pre-tokenizer: contractions, runs of letters, single digits, and runs of anything else
WORDS = r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""


def clip_tokenizer(vocab: str, merges: str):
    """ CLIP's byte-level BPE tokenizer, from its vocab.json and merges.txt. """
    from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers
    from tokenizers.models import BPE
    with open(vocab, encoding="utf-8") as f:
        ids = json.load(f)
    with open(merges, encoding="utf-8") as f:
        pairs = [tuple(line.split()) for line in f.read().split("\n")[1:] if len(line.split()) == 2]
    tokenizer = Tokenizer(BPE(ids, pairs, end_of_word_suffix="</w>", unk_token="<|endoftext|>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFC(), normalizers.Replace(Regex(r"\s+"), " "),
                                                 normalizers.Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(WORDS), behavior="removed", invert=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False)])
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


class CLIPSegEngine(SegmentEngine):
    """
    CLIPSeg: finds anything named, such as "the cup" or "the sky", as a map
    of where in the photo it is. CLIP's vision encoder sees the photo once,
    at 352x352, in about half a second; its text encoder and CLIPSeg's
    decoder then take about 50 ms a prompt.
    """
    MODEL_ID = "clipseg-rd64-refined"

    def __init__(self):
        paths = tflite.files("CLIPSeg", REPO, FILES)
        self.graphs = None
        if paths:
            found = dict(zip(FILES, paths))
            self.embedding = np.fromfile(found[EMBEDDING], np.float16).reshape(-1, WIDTH)
            self.projection = np.fromfile(found[PROJECTION], np.float16).astype(np.float32).reshape(WIDTH, WIDTH)
            self.tokenizer = clip_tokenizer(found[VOCAB], found[MERGES])
            self.graphs = tflite.load_graphs("CLIPSeg", {"vision": found[VISION], "text": found[TEXT],
                                                         "decoder": found[DECODER]})

    def condition(self, prompt: str) -> np.ndarray:
        """ CLIP's embedding of a prompt, which tells the decoder what to find. """
        ids = [START] + self.tokenizer.encode(prompt, add_special_tokens=False).ids[:CONTEXT - 2] + [END]
        tokens = np.zeros((1, CONTEXT, WIDTH), np.float32)
        tokens[0, :len(ids)] = self.embedding[ids]
        tokens[0, len(ids):] = self.embedding[END]
        hidden = self.graphs["text"](tokens)[0][0]
        # The end of text token's state, projected, as CLIP's text model gives it
        return (hidden[len(ids) - 1] @ self.projection)[None]

    def segment(self, image: bytes, prompts: List[str]) -> List[np.ndarray]:
        if not self.graphs:
            raise RuntimeError("CLIPSeg is not available.")
        photo = tflite.open_photo(image).resize((SIZE, SIZE), resample=2)
        x = ((np.asarray(photo, np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
        layers = self.graphs["vision"](x)
        maps = []
        for prompt in prompts:
            logits = self.graphs["decoder"](*layers, self.condition(prompt))[0].reshape(SIZE, SIZE)
            maps.append(1 / (1 + np.exp(-logits)))
        return maps

    @classmethod
    def setup(cls):
        tflite.download("CLIPSeg", REPO, FILES)
