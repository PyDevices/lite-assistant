"""
Moonshine STT engine, on ONNX Runtime. Its decoding loop is Useful Sensors'
from useful-moonshine-onnx (MIT), here so as not to need that package, which
needs librosa and numba.
"""
import os
import threading
from typing import Optional

import numpy as np

from . import tflite
from .stt_base import STTEngine
from ..paths import cached_hf_file

REPO = "UsefulSensors/moonshine"
WEIGHTS = ("encoder_model.onnx", "decoder_model_merged.onnx", "tokenizer.json")
RATE = 16000
# Each model's decoder: its layers, key/value heads and head size
SHAPES = {"tiny": (6, 8, 36), "base": (8, 8, 52)}
START, END, MAX_TOKENS = 1, 2, 192
# Moonshine says nothing at all for a clip that starts with more than about
# 1.5 s of room noise, as a board's recording often does, so the clip is cut
# to the speech Silero VAD hears in it, with this much either side
MARGIN = RATE * 3 // 10


def _weights_folder(model_name: str) -> str:
    return f"onnx/merged/{model_name.split('/')[-1]}/float"


class MoonshineEngine(STTEngine):
    """
    Moonshine STT Engine Implementation.
    High performance, privacy-focused Automatic Speech Recognition (ASR), on ONNX Runtime.
    """
    def __init__(self, model_name="moonshine/base"):
        self.model_name = model_name
        self.encoder = self.decoder = None
        self._tokenizer = None
        self.shape = SHAPES[model_name.split("/")[-1]]
        # The sessions run one clip at a time
        self._lock = threading.Lock()
        self._vad = None  # Silero VAD, loaded with the first clip; False if it isn't downloaded
        self._initialize_moonshine()

    def _initialize_moonshine(self):
        try:
            import onnxruntime
            import tokenizers
        except ImportError:
            print("[Moonshine] onnxruntime or tokenizers isn't installed. Skipping.")
            return
        folder = _weights_folder(self.model_name)
        paths = [cached_hf_file(REPO, f"{folder}/{name}") for name in WEIGHTS]
        if not all(paths):
            print(f"[Moonshine] {self.model_name} isn't downloaded; run 'lite-assistant setup'. Skipping.")
            return
        print(f"[Moonshine] Loading STT model {self.model_name}...")
        self.encoder = onnxruntime.InferenceSession(paths[0])
        self.decoder = onnxruntime.InferenceSession(paths[1])
        self._tokenizer = tokenizers.Tokenizer.from_file(paths[2])
        print("[Moonshine] Model loaded successfully.")

    def generate(self, audio: np.ndarray) -> list:
        """ The tokens Moonshine hears in [1, samples] of audio at 16 kHz, greedily, with its KV cache. """
        mask = np.ones_like(audio, dtype=np.int64)
        encoder_inputs = {"input_values": audio}
        if "attention_mask" in [x.name for x in self.encoder.get_inputs()]:
            encoder_inputs["attention_mask"] = mask
        hidden = self.encoder.run(None, encoder_inputs)[0]
        layers, heads, size = self.shape
        cache = {f"past_key_values.{i}.{a}.{b}": np.zeros((0, heads, 1, size), np.float32)
                 for i in range(layers) for a in ("decoder", "encoder") for b in ("key", "value")}
        masked = "encoder_attention_mask" in [x.name for x in self.decoder.get_inputs()]
        tokens = [START]
        for i in range(MAX_TOKENS):
            inputs = {"input_ids": [[tokens[-1]]], "encoder_hidden_states": hidden, "use_cache_branch": [i > 0],
                      **cache}
            if masked:
                inputs["encoder_attention_mask"] = mask
            logits, *present = self.decoder.run(None, inputs)
            tokens.append(int(logits[0, -1].argmax()))
            if tokens[-1] == END:
                break
            # The encoder's keys and values are computed once, on the first step
            for name, value in zip(cache, present):
                if i == 0 or "decoder" in name:
                    cache[name] = value
        return tokens

    def transcribe(self, audio_bytes: bytes, language: Optional[str] = None) -> str:
        if not self.decoder:
            raise RuntimeError("Moonshine STT engine is not available.")
        audio = tflite.open_audio(audio_bytes, RATE)
        seconds = len(audio) / RATE
        if not 0.1 < seconds < 64:
            raise ValueError(f"Moonshine transcribes clips of 0.1 to 64 seconds; this one is {seconds:.1f}s")
        with self._lock:
            audio = self.speech(audio)
            tokens = self.generate(audio[None])
        return self._tokenizer.decode(tokens).strip()

    def speech(self, audio: np.ndarray) -> np.ndarray:
        """ The audio cut to where Silero VAD hears speech, with a margin; all of it if it hears none. """
        if self._vad is None:
            from ..vad import SileroVAD
            self._vad = SileroVAD.load() or False
        if not self._vad:
            return audio
        from ..vad import speech_span
        span = speech_span(self._vad.probability, audio)
        if span is None:
            return audio
        return audio[max(0, span[0] - MARGIN):span[1] + MARGIN]

    @classmethod
    def setup(cls):
        from huggingface_hub import hf_hub_download
        print("[Moonshine] Ensuring model 'moonshine/base' is downloaded...")
        for name in WEIGHTS:
            hf_hub_download(REPO, name, subfolder=_weights_folder("moonshine/base"))
