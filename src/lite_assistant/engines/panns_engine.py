"""
PANNs sound engine: CNN14 tagging AudioSet's 527 kinds of sound, on LiteRT.
"""
import os
import threading
import time

import numpy as np
from . import tflite
from .sound_base import SoundEngine, Sounds
from ..paths import cached_hf_file

# Kong et al.'s CNN14 (mAP 0.431), converted by litert-community; weights CC-BY-4.0
REPO = "litert-community/PANNs-CNN14-AudioSet-LiteRT"
FILE = "cnn14_audioset_fp16.tflite"
MEL_BASIS = "mel_basis.bin"
LABELS = "audioset_labels.txt"
# The model hears 10 s at 32 kHz, as a log-mel spectrogram the host computes
RATE = 32000
WINDOW = 10 * RATE
FFT_SIZE, HOP, MELS = 1024, 320, 64


def log_mel(audio: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """
    The spectrogram CNN14 takes, [frames, 64], computed as torchlibrosa does:
    centered with reflection, a periodic Hann window, power and log mel.
    """
    padded = np.pad(audio, FFT_SIZE // 2, mode="reflect")
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FFT_SIZE) / FFT_SIZE)
    frames = np.arange(FFT_SIZE)[None] + HOP * np.arange(1 + len(audio) // HOP)[:, None]
    power = np.abs(np.fft.rfft(padded[frames] * window)) ** 2
    return (10 * np.log10(np.maximum(power @ basis.T, 1e-10))).astype(np.float32)


def windows(audio: np.ndarray) -> list:
    """
    The audio in 10 s windows, as (start, end, samples). A clip shorter than
    a window is repeated to fill it, which the model hears far better than
    silence after it; the last window of a longer clip is its last 10 s.
    """
    if len(audio) <= WINDOW:
        return [(0, len(audio), np.resize(audio, WINDOW))]
    starts = list(range(0, len(audio) - WINDOW, WINDOW)) + [len(audio) - WINDOW]
    return [(start, start + WINDOW, audio[start:start + WINDOW]) for start in starts]


class PANNsEngine(SoundEngine):
    """
    PANNs CNN14, which names the sounds in audio, from speech and music to a
    doorbell, a smoke alarm or breaking glass, in about 0.1 s per 10 s on the CPU.
    """
    MODEL_ID = "panns-cnn14"

    def __init__(self):
        self.interpreter = None
        # One clip at a time: the interpreter's tensors are shared
        self._lock = threading.Lock()
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            print("[PANNs] ai-edge-litert package is not installed. Skipping PANNsEngine.")
            return
        paths = [cached_hf_file(REPO, name) for name in (FILE, MEL_BASIS, LABELS)]
        if not all(paths):
            print(f"[PANNs] {FILE} isn't downloaded; run 'lite-assistant setup'.")
            return
        print(f"[PANNs] Loading {FILE}...")
        start = time.time()
        try:
            self.interpreter = Interpreter(model_path=paths[0], num_threads=os.cpu_count())
            self.interpreter.allocate_tensors()
        except Exception as e:
            print(f"[PANNs] Failed to load: {e}")
            self.interpreter = None
            return
        self.basis = np.fromfile(paths[1], np.float32).reshape(MELS, FFT_SIZE // 2 + 1)
        with open(paths[2], encoding="utf-8") as f:
            self.labels = f.read().splitlines()
        print(f"[PANNs] Loaded in {time.time() - start:.1f}s.")

    def get_model_id(self) -> str:
        return self.MODEL_ID

    def classify(self, audio: bytes) -> Sounds:
        if not self.interpreter:
            raise RuntimeError("PANNs sound engine is not available.")
        samples = tflite.open_audio(audio, RATE)

        scores, spans = [], []
        for start, end, window in windows(samples):
            features = log_mel(window, self.basis)[None, None]
            with self._lock:
                interpreter = self.interpreter
                interpreter.set_tensor(interpreter.get_input_details()[0]["index"], features)
                interpreter.invoke()
                scores.append(interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0].copy())
            spans.append((round(start / RATE, 2), round(end / RATE, 2)))
        return Sounds(self.labels, np.stack(scores), spans, round(len(samples) / RATE, 2))

    @classmethod
    def setup(cls):
        from huggingface_hub import hf_hub_download
        for name in (FILE, MEL_BASIS, LABELS):
            if not cached_hf_file(REPO, name):
                print(f"[PANNs] Downloading {name} from {REPO}...")
                hf_hub_download(REPO, name)
