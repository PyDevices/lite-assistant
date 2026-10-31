"""
Tuning: the note a pitch engine hears, how many cents off it is, and for
an instrument, which string it's nearest and how far off that string.
"""
import math
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Each instrument's strings in standard tuning, lowest first
TUNINGS = {
    "guitar": ["E2", "A2", "D3", "G3", "B3", "E4"],
    "bass": ["E1", "A1", "D2", "G2"],
    "ukulele": ["G4", "C4", "E4", "A4"],
    "violin": ["G3", "D4", "A4", "E5"],
    "viola": ["C3", "G3", "D4", "A4"],
    "cello": ["C2", "G2", "D3", "A3"],
    "mandolin": ["G3", "D4", "A4", "E5"],
    "banjo": ["G4", "D3", "G3", "B3", "D4"],
}
# Within this many cents is in tune: about what a trained ear notices
IN_TUNE = 5
# A pitch counts when the engine is at least this sure a note is playing
CONFIDENCE = 0.5


def midi(frequency: float, reference: float = 440.0) -> float:
    """ A frequency as a MIDI note number, with a fraction: 69 is A4 at the reference. """
    return 69 + 12 * math.log2(frequency / reference)


def name(note: int) -> str:
    return f"{NAMES[note % 12]}{note // 12 - 1}"


def number(note: str) -> int:
    """ A note's MIDI number from its name, such as A4. """
    pitch, octave = note[:-1], int(note[-1])
    return NAMES.index(pitch) + 12 * (octave + 1)


def frequency(note: int, reference: float = 440.0) -> float:
    return reference * 2 ** ((note - 69) / 12)


def tune(pitches: List[Tuple[float, float]], hop: float, reference: float = 440.0,
         instrument: Optional[str] = None, confidence: float = CONFIDENCE) -> Dict[str, Any]:
    """
    The note heard, from the frames the engine is sure of: its frequency
    (the median), the nearest note, how many cents off it is, and whether
    it's in tune; with an instrument, its nearest string and how far off
    that. Each frame is in track, as its time, frequency (None when no note
    is heard) and confidence.
    """
    sure = [hz for hz, sureness in pitches if sureness >= confidence]
    result: Dict[str, Any] = {"frequency": None, "note": None, "cents": None, "in_tune": None,
                              "confidence": round(median(s for _, s in pitches), 3) if pitches else 0.0}
    if sure:
        heard = median(sure)
        exact = midi(heard, reference)
        nearest = round(exact)
        cents = round((exact - nearest) * 100)
        result.update({"frequency": round(heard, 2), "note": name(nearest), "cents": cents,
                       "in_tune": abs(cents) <= IN_TUNE})
        if instrument:
            string = min(TUNINGS[instrument], key=lambda s: abs(exact - number(s)))
            off = round((exact - number(string)) * 100)
            result["string"] = {"note": string, "frequency": round(frequency(number(string), reference), 2),
                                "cents": off, "in_tune": abs(off) <= IN_TUNE}
    result["track"] = [[round(i * hop, 2), round(hz, 2) if sureness >= confidence else None, round(sureness, 3)]
                       for i, (hz, sureness) in enumerate(pitches)]
    return result


def advice(found: Dict[str, Any]) -> str:
    """ What to do, in words: for an instrument's string, tighten or loosen it; otherwise, sharp or flat. """
    target = found.get("string") or found
    if found["note"] is None:
        return "No note is heard; play one note at a time, near the microphone."
    if target["in_tune"]:
        return "In tune."
    off = f"{abs(target['cents'])} cents {'sharp' if target['cents'] > 0 else 'flat'}"
    if "string" not in found:
        return f"{off}."
    return f"{off}: {'loosen' if target['cents'] > 0 else 'tighten'} the string{' a little' if abs(target['cents']) < 20 else ''}."
