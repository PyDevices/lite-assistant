"""
CMGAN noise removal engine, on LiteRT.
"""
import numpy as np

from . import tflite
from .denoise_base import DenoiseEngine

# Ruizhe Cao's CMGAN, trained on VoiceBank-DEMAND, converted by litert-community; MIT, 4 MB
REPO = "litert-community/CMGAN-LiteRT"
FILE = "cmgan_fp16.tflite"
# The graph takes 2 s chunks, padded by half an FFT each side, and gives their spectrograms
CHUNK, FFT, HOP = 32000, 400, 100
# Its spectrogram's magnitudes are compressed to this power
COMPRESS = 0.3
# The periodic Hamming window of its STFT
WINDOW = (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(FFT) / FFT)).astype(np.float32)


class CMGANEngine(DenoiseEngine):
    """
    CMGAN, a conformer that cleans speech 2 s at a time. The graph does the
    STFT and gives the cleaned, compressed spectrogram; here it's
    uncompressed and turned back into sound. XNNPACK declines its fp16
    graph, so it runs at about 1.25 times real time on a desktop CPU.
    """
    MODEL_ID = "cmgan"

    def __init__(self):
        paths = tflite.files("CMGAN", REPO, [FILE])
        loaded = tflite.load_graphs("CMGAN", {"net": paths[0]}) if paths else None
        self.graph = loaded["net"] if loaded else None

    def chunk(self, x: np.ndarray) -> np.ndarray:
        """ One 2 s chunk, cleaned. """
        # The graph wants the chunk at unit loudness
        gain = np.sqrt(CHUNK / (float(x @ x) + 1e-12))
        x = x * gain
        half = FFT // 2
        padded = np.concatenate([x[half:0:-1], x, x[-2:-half - 2:-1]])
        real, imag = (out[0, 0] for out in self.graph(padded[None].astype(np.float32)))
        scale = np.maximum(real * real + imag * imag, 1e-12) ** ((1 / COMPRESS - 1) / 2)
        frames = np.fft.irfft(real * scale + 1j * imag * scale, n=FFT, axis=1) * WINDOW
        # Overlap-add the frames, then undo the window's own overlap
        out = np.zeros((len(frames) - 1) * HOP + FFT)
        weight = np.zeros_like(out)
        for k, frame in enumerate(frames):
            out[k * HOP:k * HOP + FFT] += frame
            weight[k * HOP:k * HOP + FFT] += WINDOW ** 2
        return (out / np.maximum(weight, 1e-8))[half:half + CHUNK] / gain

    def denoise(self, samples: np.ndarray) -> np.ndarray:
        if not self.graph:
            raise RuntimeError("CMGAN is not available.")
        padded = np.zeros(-(-len(samples) // CHUNK) * CHUNK, np.float32)
        padded[:len(samples)] = samples
        return np.concatenate([self.chunk(padded[k:k + CHUNK]) for k in range(0, len(padded), CHUNK)])[
            :len(samples)].astype(np.float32)

    @classmethod
    def setup(cls):
        tflite.download("CMGAN", REPO, [FILE])
