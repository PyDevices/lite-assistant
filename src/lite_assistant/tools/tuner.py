"""
tune_instrument: the note in the user's latest recording, and how far out
of tune it is, from the pitch engine.
"""
from typing import Any, Dict, List, Literal, Optional

from ..media import latest_audio
from ..pitch import TUNINGS, advice, tune

# A tuner needs only a few seconds of one note
SECONDS = 10


class Tuner:
    def __init__(self, engine):
        self.engine = engine

    def tune_instrument(self, instrument: Optional[Literal["guitar", "bass", "ukulele", "violin", "viola", "cello",
                                                           "mandolin", "banjo"]] = None,
                        conversation: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Hears the note in the user's latest recording, of an instrument or a voice: which note it is and
        how many cents sharp or flat. Name the instrument, if it's one of these, to get which string it's
        nearest and whether to tighten or loosen it. Use it when the user asks to tune, or what note
        they're playing or singing.
        """
        from ..engines.tflite import open_audio
        audio = latest_audio(conversation or [])
        if audio is None:
            return {"error": "The user hasn't sent a recording. Ask them to record one note being played."}
        samples = open_audio(audio, self.engine.RATE)[:SECONDS * self.engine.RATE]
        # Models name instruments the list doesn't have ("string"); the note is still found
        instrument = instrument.lower() if isinstance(instrument, str) and instrument.lower() in TUNINGS else None
        found = tune(self.engine.pitches(samples), self.engine.HOP, instrument=instrument)
        reply: Dict[str, Any] = {"advice": advice(found)}
        if found["note"]:
            reply.update({"note": found["note"], "frequency": f"{found['frequency']} Hz",
                          "cents_off_the_note": found["cents"]})
        if "string" in found:
            reply["string"] = found["string"]["note"]
            reply["cents_off_the_string"] = found["string"]["cents"]
        return reply
