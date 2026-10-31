"""
Voice activity detection for the Realtime API's server turn detection:
Silero VAD on ONNX Runtime finds where speech starts and stops in a stream
of 24 kHz PCM.
"""
from typing import List, Optional, Tuple

import numpy as np

from .paths import cached_hf_file

REPO = "onnx-community/silero-vad"
FILE = "onnx/model.onnx"
# Silero takes 512-sample windows at 16 kHz (32 ms), each with the 64
# samples before it as context
RATE = 16000
WINDOW = 512
CONTEXT = 64
WINDOW_MS = WINDOW * 1000 // RATE


class SileroVAD:
    """ The model, shared by every session; each stream keeps its own state. """
    def __init__(self, path: str):
        import onnxruntime
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = options.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(path, options, providers=["CPUExecutionProvider"])

    @classmethod
    def load(cls) -> Optional["SileroVAD"]:
        path = cached_hf_file(REPO, FILE)
        if not path:
            print("[VAD] Silero VAD isn't downloaded; run 'lite-assistant setup'. "
                  "Realtime turns must be committed by the client.")
            return None
        return cls(path)

    @classmethod
    def setup(cls):
        if not cached_hf_file(REPO, FILE):
            from huggingface_hub import hf_hub_download
            print(f"[VAD] Downloading {FILE} from {REPO}...")
            hf_hub_download(REPO, FILE)

    def stream(self) -> "VADStream":
        return VADStream(self.probability)

    def probability(self, window: np.ndarray, state: np.ndarray) -> Tuple[float, np.ndarray]:
        output, state = self.session.run(None, {"input": window[np.newaxis, :], "state": state,
                                                "sr": np.array(RATE, dtype=np.int64)})
        return float(output[0][0]), state


def speech_span(probability, samples: np.ndarray, threshold: float = 0.5) -> Optional[Tuple[int, int]]:
    """
    Where speech starts and stops in 16 kHz float samples, as sample
    indexes, from the first window Silero hears speech in to the end of the
    last; None if it hears none.
    """
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros(CONTEXT, dtype=np.float32)
    first = last = None
    for start in range(0, len(samples) - WINDOW + 1, WINDOW):
        window = np.asarray(samples[start:start + WINDOW], dtype=np.float32)
        p, state = probability(np.concatenate([context, window]), state)
        context = window[-CONTEXT:]
        if p >= threshold:
            first = start if first is None else first
            last = start + WINDOW
    return None if first is None else (first, last)


class VADStream:
    """ Speech probabilities, one per 32 ms window, for a stream of 24 kHz 16-bit PCM fed in pieces. """
    def __init__(self, probability):
        self.probability = probability
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros(CONTEXT, dtype=np.float32)
        self.pending = np.zeros(0, dtype=np.float32)
        self.leftover = b""

    def feed(self, pcm: bytes) -> List[float]:
        from scipy.signal import resample_poly
        pcm = self.leftover + pcm
        whole = len(pcm) - len(pcm) % 2
        self.leftover = pcm[whole:]
        samples = np.frombuffer(pcm[:whole], dtype="<i2").astype(np.float32) / 32768
        # 24 kHz to 16 kHz; resampling each piece on its own blurs only its
        # edges, which doesn't matter to a speech detector
        self.pending = np.concatenate([self.pending, resample_poly(samples, 2, 3).astype(np.float32)])
        probabilities = []
        while len(self.pending) >= WINDOW:
            window, self.pending = self.pending[:WINDOW], self.pending[WINDOW:]
            p, self.state = self.probability(np.concatenate([self.context, window]), self.state)
            self.context = window[-CONTEXT:]
            probabilities.append(p)
        return probabilities


class TurnDetector:
    """
    OpenAI's server_vad turn detection: speech starts when a window's
    probability passes the threshold, and stops after silence_duration_ms
    below it. Times are in ms of audio, counted from elapsed_ms.
    """
    def __init__(self, stream: VADStream, threshold: float = 0.5, silence_duration_ms: int = 500,
                 elapsed_ms: int = 0):
        self.stream = stream
        self.threshold, self.silence_duration_ms = threshold, silence_duration_ms
        self.speaking = False
        self.elapsed_ms = elapsed_ms
        self.silent_ms = 0

    def feed(self, pcm: bytes) -> List[Tuple[str, int]]:
        """ ("speech_started" or "speech_stopped", ms) for each change the audio brings. """
        events = []
        for p in self.stream.feed(pcm):
            self.elapsed_ms += WINDOW_MS
            if not self.speaking and p >= self.threshold:
                self.speaking, self.silent_ms = True, 0
                events.append(("speech_started", self.elapsed_ms - WINDOW_MS))
            elif self.speaking:
                # A little hysteresis, as Silero's own iterator uses, so speech
                # dipping just under the threshold doesn't count as silence
                self.silent_ms = self.silent_ms + WINDOW_MS if p < self.threshold - 0.15 else 0
                if self.silent_ms >= self.silence_duration_ms:
                    self.speaking = False
                    events.append(("speech_stopped", self.elapsed_ms - self.silent_ms))
        return events
