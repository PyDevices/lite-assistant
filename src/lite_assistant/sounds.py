"""
The sounds in audio, as /v1/audio/classifications reports them and the
identify_sounds tool tells the model.
"""
from typing import Any, Dict, List

import numpy as np

from .engines import Sounds

THRESHOLD = 0.1
TOP = 5


def ranked(labels: List[str], scores: np.ndarray, threshold: float = THRESHOLD, top: int = TOP) -> List[Dict[str, Any]]:
    """ The surest sounds, at least threshold sure, as label and score. """
    order = [i for i in np.argsort(-scores)[:top] if scores[i] >= threshold]
    return [{"label": labels[i], "score": round(float(scores[i]), 3)} for i in order]


def heard(sounds: Sounds, threshold: float = THRESHOLD, top: int = TOP) -> List[Dict[str, Any]]:
    """ The sounds anywhere in the audio: each kind's best score in any window. """
    return ranked(sounds.labels, sounds.scores.max(axis=0), threshold, top)


def segments(sounds: Sounds, threshold: float = THRESHOLD, top: int = TOP) -> List[Dict[str, Any]]:
    """ The sounds in each 10 s window, with its start and end in seconds. """
    return [{"start": start, "end": end, "sounds": ranked(sounds.labels, row, threshold, top)}
            for (start, end), row in zip(sounds.windows, sounds.scores)]


def likely(score: float) -> str:
    """ How sure a score is, in words a small model reads more reliably than numbers. """
    return "very likely" if score >= 0.5 else "likely" if score >= 0.25 else "possibly"
