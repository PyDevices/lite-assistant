"""
identify_speakers: who is speaking in the user's latest recording, and when,
naming the voices the user has enrolled, from the speaker engine.
"""
from typing import Any, Dict, List, Optional

from ..media import latest_audio
from ..speakers import Voices, diarize


class SpeakerFinder:
    def __init__(self, engine, voices: Optional[Voices] = None):
        self.engine, self.voices = engine, voices or Voices()

    def identify_speakers(self, conversation: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Tells who is speaking in the user's latest audio recording, and when: how many voices, which
        are people whose voices are enrolled, by name, and each turn. Use it only when the user asks who
        is speaking, how many people are talking, or who said something.
        """
        from ..engines.tflite import open_audio
        audio = latest_audio(conversation or [])
        if audio is None:
            return {"error": "The user hasn't sent a recording. Ask them to record the conversation."}
        found = diarize(self.engine, open_audio(audio, self.engine.RATE), self.voices)
        if not found["segments"]:
            return {"speakers": "no one is heard speaking"}
        known = [name for name in found["speakers"] if not name.startswith("speaker ")]
        reply: Dict[str, Any] = {
            "voices": len(found["speakers"]),
            "speaking_time": [f"{name}: {seconds:.0f} seconds" for name, seconds in found["speakers"].items()],
            "turns": [f"{s['start']:.1f} to {s['end']:.1f} s: {s['speaker']}" for s in found["segments"]]}
        if len(found["speakers"]) > len(known):
            reply["note"] = ("Voices called speaker 1, speaker 2 and so on aren't enrolled. "
                             + ("The enrolled voices are " + ", ".join(self.voices.names()) + "." if self.voices.names()
                                else "No voices are enrolled; the user can enroll theirs in the chat page's settings."))
        return reply
