"""
Whisper large-v3-turbo STT engine, on LiteRT.
"""
import functools
import os
import threading
from typing import List, Optional

import numpy as np

from . import tflite
from .stt_base import STTEngine
from ..paths import cached_hf_file

# OpenAI's Whisper large-v3-turbo, int8 weights, converted by litert-community; MIT, 1.1 GB. The
# 721 MB int4 file was 2.4 times slower to encode and 3.7 times slower a token here on XNNPACK
REPO = "litert-community/whisper-large-v3-turbo"
FILE = "whisper_large_v3_turbo_30s_i8.tflite"
TOKENIZER_REPO, TOKENIZER = "openai/whisper-large-v3-turbo", "tokenizer.json"
RATE = 16000
# The graph hears 30 s at a time, as 3000 frames of 128 mel bins, and decodes up to 128 tokens
WINDOW, FFT, HOP, MELS, TOKENS = 30 * RATE, 400, 160, 128, 128
# Its special tokens, in the 51866 token vocabulary of large-v3
START, END, TRANSCRIBE, NO_TIMESTAMPS = 50258, 50257, 50360, 50364
LANGUAGES = range(50259, 50359)
# Longer audio is heard 30 s at a time, which is slow, so it's limited
MAX_SECONDS = 120


@functools.lru_cache(maxsize=1)
def mel_filters(rate: int = RATE, fft: int = FFT, mels: int = MELS) -> np.ndarray:
    """ Slaney's mel filterbank, [mels, fft // 2 + 1], as librosa.filters.mel makes it and Whisper uses. """
    def to_mel(hz):
        hz = np.asarray(hz, np.float64)
        return np.where(hz < 1000, hz * 3 / 200, 15 + 27 * np.log(np.maximum(hz, 1000) / 1000) / np.log(6.4))

    def to_hz(mel):
        return np.where(mel < 15, mel * 200 / 3, 1000 * np.exp(np.log(6.4) / 27 * (mel - 15)))

    bins = np.fft.rfftfreq(fft, 1 / rate)
    edges = to_hz(np.linspace(to_mel(0), to_mel(rate / 2), mels + 2))
    below, above = (bins[None] - edges[:-2, None]) / np.diff(edges)[:-1, None], \
        (edges[2:, None] - bins[None]) / np.diff(edges)[1:, None]
    weights = np.maximum(0, np.minimum(below, above))
    return (weights * (2 / (edges[2:] - edges[:-2]))[:, None]).astype(np.float32)


def log_mel(samples: np.ndarray) -> np.ndarray:
    """ Whisper's input: 30 s of audio, padded with silence, as a [128, 3000] log-mel spectrogram. """
    padded = np.zeros(WINDOW, np.float32)
    padded[:min(len(samples), WINDOW)] = samples[:WINDOW]
    # Centered with reflection, a periodic Hann window, and the last frame dropped, as Whisper does
    padded = np.pad(padded, FFT // 2, mode="reflect")
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FFT) / FFT)
    frames = np.arange(FFT)[None] + HOP * np.arange(WINDOW // HOP)[:, None]
    power = np.abs(np.fft.rfft(padded[frames] * window)) ** 2
    mel = np.log10(np.maximum(mel_filters() @ power.T, 1e-10))
    return ((np.maximum(mel, mel.max() - 8) + 4) / 4).astype(np.float32)


class WhisperEngine(STTEngine):
    """
    Whisper large-v3-turbo: slower than Moonshine but more accurate, and it
    hears about 100 languages, finding which one if it isn't told. The graph
    has no KV cache, so each token reruns its whole decoder. It's big, so
    it loads on first use and unloads when idle, unlike the other STT
    engines.
    """
    MODEL_ID = "whisper-large-v3-turbo"
    LAZY = True

    def __init__(self):
        self.encode = self.decode = None
        paths = tflite.files("Whisper", REPO, [FILE])
        tokenizer = cached_hf_file(TOKENIZER_REPO, TOKENIZER)
        if not paths or not tokenizer:
            if paths:
                print(f"[Whisper] {TOKENIZER} isn't downloaded; run 'lite-assistant setup'.")
            return
        try:
            from ai_edge_litert.interpreter import Interpreter
            from tokenizers import Tokenizer
        except ImportError as e:
            print(f"[Whisper] {e.name} isn't installed. Skipping.")
            return
        print(f"[Whisper] Loading {FILE}...")
        interpreter = Interpreter(model_path=paths[0], num_threads=os.cpu_count())
        self.encode = interpreter.get_signature_runner("encode")
        self.decode = interpreter.get_signature_runner("decode")
        self.tokenizer = Tokenizer.from_file(tokenizer)
        self.languages = {self.tokenizer.id_to_token(t)[2:-2]: t for t in LANGUAGES}
        # Each token sees itself and those before it
        self.mask = np.triu(np.full((TOKENS, TOKENS), -1e9, np.float32), 1)[None, None]
        self.lock = threading.Lock()

    def get_model_id(self) -> str:
        return self.MODEL_ID

    def logits(self, states: np.ndarray, tokens: List[int]) -> np.ndarray:
        """ The scores of the token after these. """
        ids = np.zeros((1, TOKENS), np.int32)
        ids[0, :len(tokens)] = tokens
        return self.decode(args_0=states, args_1=ids, args_2=self.mask)["output_0"][0, len(tokens) - 1]

    def window(self, samples: np.ndarray, language: Optional[str]) -> str:
        """ The text of up to 30 s of speech, greedily decoded. """
        states = self.encode(args_0=log_mel(samples)[None])["output_0"]
        if language not in self.languages:
            scores = self.logits(states, [START])
            language = max(self.languages, key=lambda code: scores[self.languages[code]])
        tokens = [START, self.languages[language], TRANSCRIBE, NO_TIMESTAMPS]
        prompt = len(tokens)
        while len(tokens) < TOKENS:
            token = int(self.logits(states, tokens).argmax())
            if token == END:
                break
            tokens.append(token)
        return self.tokenizer.decode(tokens[prompt:]).strip()

    def transcribe(self, audio_bytes: bytes, language: Optional[str] = None) -> str:
        if not self.decode:
            raise RuntimeError("Whisper STT engine is not available.")
        samples = tflite.open_audio(audio_bytes, RATE)
        seconds = len(samples) / RATE
        if not 0.1 < seconds <= MAX_SECONDS:
            raise ValueError(f"Whisper transcribes clips of 0.1 to {MAX_SECONDS} seconds; this one is {seconds:.1f}s")
        with self.lock:
            return " ".join(text for text in (self.window(samples[k:k + WINDOW], language)
                                              for k in range(0, len(samples), WINDOW)) if text)

    @classmethod
    def setup(cls):
        tflite.download("Whisper", REPO, [FILE])
        tflite.download("Whisper", TOKENIZER_REPO, [TOKENIZER])
