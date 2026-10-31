"""
read_text: the text in the user's latest photo, such as a sign, a label or
a page, from the OCR engine.
"""
from typing import Any, Dict, List, Optional

from ..engines import as_text
from ..media import attached_files, latest_image

# A line read with less certainty than this is marked as a guess
UNSURE = 0.8


class TextReader:
    def __init__(self, engine):
        self.engine = engine

    def read_text(self, conversation: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Reads the text in the user's latest photo, such as a sign, a label, a screen or a page, word for
        word. Use it when the user asks what something in a photo says, or to read, copy or translate its
        text. An attached file's text is already in the user's message, so it needs no reading.
        """
        photo = latest_image(conversation or [])
        if photo is None:
            files = attached_files(conversation or [])
            if files:
                return {"error": f"The user sent no photo, but attached {', '.join(files)}, whose text is "
                                 "already in their message, after \"The attached file\". Answer from that text."}
            return {"error": "The user hasn't sent a photo. Ask them to send one."}
        lines = self.engine.read(photo)
        if not lines:
            return {"text": "", "note": "No text was found in the photo."}
        found: Dict[str, Any] = {"text": as_text(lines)}
        unsure = [line.text for line in lines if line.score < UNSURE]
        if unsure:
            found["unsure"] = unsure
        return found
