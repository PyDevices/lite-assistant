"""
Speaker engine on pyannote 3.1's recipe: WeSpeaker embeddings on LiteRT and
pyannote's segmentation on onnxruntime.
"""
from functools import lru_cache

import numpy as np

from . import tflite
from .speaker_base import SpeakerEngine

# Converted by litert-community: WeSpeaker's ResNet34 (CC-BY-4.0 weights, 13 MB)
# and pyannote's segmentation-3.0 (MIT, 6 MB)
REPO = "litert-community/Speaker-Diarization-LiteRT"
EMBEDDER, SEGMENTER = "wespeaker_emb_fp16.tflite", "pyannote_seg30.onnx"
# The embedder takes 500 frames of 80 Kaldi filter bank features: 5.015 s
FRAMES, BINS = 500, 80
FRAME_LENGTH, FRAME_SHIFT, FFT = 400, 160, 512
EMBED_SAMPLES = (FRAMES - 1) * FRAME_SHIFT + FRAME_LENGTH
# The segmenter's classes, as the voices speaking: none, one of three, or two at once
POWERSET = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]


@lru_cache(maxsize=1)
def mel_banks() -> np.ndarray:
    """ Kaldi's triangular mel filters, 20 Hz to 8 kHz, over the FFT's bins below Nyquist. """
    mel = lambda f: 1127 * np.log(1 + f / 700)
    low, high = mel(20.0), mel(8000.0)
    step = (high - low) / (BINS + 1)
    bins = mel(np.arange(FFT // 2) * 16000 / FFT)
    banks = np.zeros((BINS, FFT // 2), np.float32)
    for b in range(BINS):
        left, center, right = low + b * step, low + (b + 1) * step, low + (b + 2) * step
        up, down = (bins - left) / (center - left), (right - bins) / (right - center)
        banks[b] = np.maximum(0, np.minimum(up, down))
    return banks


def fbank(samples: np.ndarray) -> np.ndarray:
    """ Kaldi's log mel filter bank features, as WeSpeaker was trained on: 25 ms frames every 10 ms. """
    x = samples.astype(np.float64) * 32768
    count = 1 + (len(x) - FRAME_LENGTH) // FRAME_SHIFT
    frames = np.lib.stride_tricks.sliding_window_view(x, FRAME_LENGTH)[::FRAME_SHIFT][:count].copy()
    frames -= frames.mean(axis=1, keepdims=True)
    # Pre-emphasis, with the first sample emphasized against itself
    frames = np.concatenate([frames[:, :1] * (1 - 0.97), frames[:, 1:] - 0.97 * frames[:, :-1]], axis=1)
    frames *= np.hamming(FRAME_LENGTH)
    power = np.abs(np.fft.rfft(frames, n=FFT)) ** 2
    energies = power[:, :FFT // 2] @ mel_banks().T
    return np.log(np.maximum(energies, np.finfo(np.float32).eps)).astype(np.float32)


class PyannoteEngine(SpeakerEngine):
    """
    WeSpeaker's ResNet34, which embeds 5 s of a voice, and pyannote's
    segmentation, which finds up to three voices in 10 s windows, two at
    once. Together they're pyannote 3.1's diarization; the clustering is
    in speakers.py.
    """
    MODEL_ID = "pyannote-3.1"
    # pyannote 3.1's clustering threshold
    SAME = 0.7046

    def __init__(self):
        self.embedder, self.segmenter = None, None
        paths = tflite.files("pyannote", REPO, [EMBEDDER, SEGMENTER])
        if not paths:
            return
        loaded = tflite.load_graphs("pyannote", {"embedder": paths[0]})
        self.embedder = loaded["embedder"] if loaded else None
        try:
            import onnxruntime
            self.segmenter = onnxruntime.InferenceSession(paths[1], providers=["CPUExecutionProvider"])
            self.FRAME = self.WINDOW / self.segmenter.get_outputs()[0].shape[1]
        except Exception as e:
            print(f"[pyannote] Segmentation didn't load: {e}")

    def embed(self, samples: np.ndarray) -> np.ndarray:
        if not self.embedder:
            raise RuntimeError("WeSpeaker is not available.")
        # Speech shorter than the window is repeated to fill it; longer, the windows' embeddings are averaged
        if len(samples) < EMBED_SAMPLES:
            samples = np.resize(samples, EMBED_SAMPLES)
        total = np.zeros(256, np.float32)
        last = len(samples) - EMBED_SAMPLES
        for start in sorted({*range(0, last + 1, EMBED_SAMPLES), last}):
            features = fbank(samples[start:start + EMBED_SAMPLES])
            vector = self.embedder((features - features.mean(axis=0))[None])[0][0]
            total += vector / (np.linalg.norm(vector) + 1e-9)
        return total / (np.linalg.norm(total) + 1e-9)

    def activity(self, window: np.ndarray) -> np.ndarray:
        if not self.segmenter:
            raise RuntimeError("pyannote's segmentation is not available.")
        size = int(self.WINDOW * self.RATE)
        padded = np.zeros(size, np.float32)
        padded[:min(len(window), size)] = window[:size]
        classes = self.segmenter.run(None, {"waveform": padded[None, None]})[0][0].argmax(axis=1)
        active = np.zeros((len(classes), 3), bool)
        for frame, c in enumerate(classes):
            active[frame, list(POWERSET[c])] = True
        return active

    @classmethod
    def setup(cls):
        tflite.download("pyannote", REPO, [EMBEDDER, SEGMENTER])
