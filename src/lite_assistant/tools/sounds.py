"""
identify_sounds: the sounds in the user's latest voice clip or recording,
from the sound engine, such as a doorbell, a dog or an alarm.
"""
from typing import Any, Dict, List, Optional

from ..media import latest_audio
from ..sounds import heard, likely, segments


class SoundTagger:
    def __init__(self, engine):
        self.engine = engine

    def identify_sounds(self, conversation: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Names the sounds in the user's latest audio recording, such as a doorbell, a dog barking, music or
        an alarm. Use it only when the user asks what a sound is or what else can be heard. Not for what
        the user says: you hear their words yourself.
        """
        audio = latest_audio(conversation or [])
        if audio is None:
            return {"error": "The user hasn't sent a recording. Ask them to record the sound."}
        sounds = self.engine.classify(audio)
        ranked = heard(sounds)
        if not ranked:
            return {"heard": "nothing recognizable"}
        # A small model hedges between near-equal scores unless told which is best
        found: Dict[str, Any] = {"best_guess": ranked[0]["label"],
                                 "heard": [f"{s['label']} ({likely(s['score'])})" for s in ranked]}
        if len(sounds.windows) > 1:
            found["over_time"] = [f"{s['start']:.0f} to {s['end']:.0f} seconds: "
                                  + (", ".join(t["label"] for t in s["sounds"]) or "nothing recognizable")
                                  for s in segments(sounds)]
        return found
