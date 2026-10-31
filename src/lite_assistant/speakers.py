"""
Who is speaking: the voices a speaker engine finds in audio, when each
speaks, and which are voices the user has enrolled by name, such as the
people in a household.
"""
import json
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import paths

# Windows overlap by half, and each frame is judged by the window it's nearest the middle of
STEP = 0.5
# A voice needs this much speech alone in a window, in seconds, to be embedded
MIN_ALONE = 1.0
# A pause shorter than this, in seconds, doesn't end a turn, and a turn shorter than this is dropped
MIN_PAUSE, MIN_TURN = 0.5, 0.3


class Voices:
    """
    The enrolled voices: each name's embeddings, one per recording, and who
    enrolled it, kept in voices.json in the data directory. They're one
    household's: every voice is recognized for everyone, but only the user
    who enrolled one, or an admin, may add to it or remove it.
    """
    def __init__(self, path: str = ""):
        self.path = path or paths.data_dir("voices.json")
        self.lock = threading.Lock()

    def load(self) -> Dict[str, Dict[str, Any]]:
        """ Each name's owner and recordings; a name with only a list, as older files had, has no owner. """
        try:
            with open(self.path, encoding="utf-8") as f:
                found = json.load(f)
        except (FileNotFoundError, ValueError):
            return {}
        return {name: voice if isinstance(voice, dict) else {"owner": "", "recordings": voice}
                for name, voice in found.items()}

    def save(self, voices: Dict[str, Dict[str, Any]]):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(voices, f)
        os.replace(self.path + ".tmp", self.path)

    def owner(self, name: str) -> Optional[str]:
        """ Who enrolled a name, "" if no one knows, or None if it isn't enrolled. """
        voice = self.load().get(name)
        return None if voice is None else voice["owner"]

    def add(self, name: str, vector: np.ndarray, owner: str = "") -> int:
        """ Adds a recording of a name's voice; returns how many it has. A new name is owner's. """
        with self.lock:
            voices = self.load()
            voice = voices.setdefault(name, {"owner": owner, "recordings": []})
            voice["recordings"].append([round(float(v), 5) for v in vector])
            self.save(voices)
            return len(voice["recordings"])

    def remove(self, name: str) -> bool:
        with self.lock:
            voices = self.load()
            if name not in voices:
                return False
            del voices[name]
            self.save(voices)
            return True

    def names(self) -> Dict[str, Dict[str, Any]]:
        """ Each enrolled name, with who enrolled it and how many recordings it has. """
        return {name: {"owner": voice["owner"], "recordings": len(voice["recordings"])}
                for name, voice in sorted(self.load().items())}

    def match(self, vector: np.ndarray, same: float) -> Tuple[Optional[str], float]:
        """ The enrolled name whose voice is nearest, if it's nearer than same, and how near, as a unit distance. """
        best, nearest = None, 2.0
        for name, voice in self.load().items():
            mean = np.mean(np.asarray(voice["recordings"], np.float32), axis=0)
            distance = float(np.linalg.norm(vector - mean / (np.linalg.norm(mean) + 1e-9)))
            if distance < nearest:
                best, nearest = name, distance
        return (best if nearest < same else None), nearest


def cluster(vectors: np.ndarray, same: float) -> np.ndarray:
    """ Which voice each embedding is, as pyannote clusters them: centroid linkage, cut at the same distance. """
    if len(vectors) < 2:
        return np.zeros(len(vectors), int)
    from scipy.cluster.hierarchy import fcluster, linkage
    return fcluster(linkage(vectors, method="centroid", metric="euclidean"), same, criterion="distance") - 1


def turns(engine, samples: np.ndarray) -> Tuple[List[Tuple[float, float, int]], Dict[int, np.ndarray]]:
    """
    Who spoke when: each turn's start and end, in seconds, and its voice,
    numbered from 0, with each voice's embedding. Windows of the engine's
    length slide over the audio; in each, every voice with enough speech
    alone is embedded, and the embeddings are clustered into voices.
    """
    rate, size = engine.RATE, int(engine.WINDOW * engine.RATE)
    step = int(size * STEP)
    starts = list(range(0, max(len(samples) - size, 0) + 1, step))
    if starts[-1] + size < len(samples):
        starts.append(len(samples) - size)
    windows, vectors, owners = [], [], []
    for start in starts:
        active = engine.activity(samples[start:start + size])
        frame = engine.WINDOW / len(active)
        windows.append((start / rate, frame, active))
        for local in range(active.shape[1]):
            alone = active[:, local] & (active.sum(axis=1) == 1)
            if alone.sum() * frame < MIN_ALONE:
                continue
            # The voice's solo speech, frame by frame, joined up
            picked = np.repeat(alone, int(round(frame * rate)))[:min(size, len(samples) - start)]
            speech = samples[start:start + len(picked)][picked]
            vectors.append(engine.embed(speech))
            owners.append((len(windows) - 1, local))
    labels = cluster(np.asarray(vectors), engine.SAME) if vectors else np.array([], int)
    voice_of = {owner: int(label) for owner, label in zip(owners, labels)}
    embeddings = {}
    for label in set(labels.tolist()):
        mean = np.mean([v for v, l in zip(vectors, labels) if l == label], axis=0)
        embeddings[int(label)] = mean / (np.linalg.norm(mean) + 1e-9)
    # Each frame of the audio, from the window it's nearest the middle of: which voices speak
    resolution = windows[0][1]
    count = int(np.ceil(len(samples) / rate / resolution))
    speaking = np.zeros((count, max(embeddings, default=-1) + 1), bool)
    for t in range(count):
        time = (t + 0.5) * resolution
        w = min(range(len(windows)), key=lambda i: abs(windows[i][0] + engine.WINDOW / 2 - time))
        start, frame, active = windows[w]
        f = int((time - start) / frame)
        if 0 <= f < len(active):
            for local in np.flatnonzero(active[f]):
                if (w, local) in voice_of:
                    speaking[t, voice_of[(w, local)]] = True
    found = []
    for voice in range(speaking.shape[1]):
        found += [(start, end, voice) for start, end in spans(speaking[:, voice], resolution)]
    return sorted(found), embeddings


def spans(on: np.ndarray, frame: float) -> List[Tuple[float, float]]:
    """ The stretches a frame mask is on, in seconds, with short pauses bridged and short stretches dropped. """
    edges = np.diff(np.concatenate([[0], on.astype(int), [0]]))
    found: List[List[float]] = []
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if found and start * frame - found[-1][1] < MIN_PAUSE:
            found[-1][1] = end * frame
        else:
            found.append([start * frame, end * frame])
    return [(round(s, 2), round(e, 2)) for s, e in found if e - s >= MIN_TURN]


def diarize(engine, samples: np.ndarray, voices: Optional[Voices] = None) -> Dict[str, Any]:
    """
    Who spoke when, as the endpoint gives it: each voice is an enrolled
    name it matches, or "speaker 1", "speaker 2" and so on.
    """
    found, embeddings = turns(engine, samples)
    names: Dict[int, str] = {}
    # Unknown voices are numbered in the order they first speak
    for _, _, voice in found:
        if voice not in names:
            name = voices.match(embeddings[voice], engine.SAME)[0] if voices else None
            names[voice] = name or f"speaker {sum(n.startswith('speaker ') for n in names.values()) + 1}"
    segments = [{"start": s, "end": e, "speaker": names[v]} for s, e, v in found]
    talk: Dict[str, float] = {}
    for segment in segments:
        talk[segment["speaker"]] = round(talk.get(segment["speaker"], 0) + segment["end"] - segment["start"], 2)
    return {"duration": round(len(samples) / engine.RATE, 2), "speakers": talk, "segments": segments}
