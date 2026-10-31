"""
edit_photo: edits the user's latest photo, or the last edit of it when they
ask for more without sending another, with the
image editing engines, and shows the result in the chat: its background
removed, something in it erased, upscaled or brightened.
"""
import base64
import hashlib
import io
from typing import Any, Dict, List, Literal, Optional

from ..media import latest_image

# The JPEG quality of edited photos without transparency
JPEG_QUALITY = 90
# The output key whose PNGs or JPEGs are shown to the user rather than given to the model
IMAGES = "images"


class PhotoEditor:
    def __init__(self, engines: Dict[str, Any], segmenter=None):
        """ engines are the image editing engines, by their TASK, which is the edit's name; segmenter finds what to erase. """
        self.engines, self.segmenter = engines, segmenter
        # The last edit made of each photo, by its hash, for edits of an edit
        self.last: Dict[str, bytes] = {}

    def edit_photo(self, edit: Literal["remove_background", "erase", "upscale", "brighten"], what: str = "",
                   conversation: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Edits the user's latest photo and shows them the result: removes its background, erases
        something from it, upscales it to 4 times the size, or brightens it if it's dark. Use it only
        when the user asks to change their photo.

        Args:
            edit: remove_background, erase, upscale or brighten.
            what: For erase, what to erase, such as "the person on the left" or "the power lines".
        """
        from PIL import Image

        from ..detection import subject
        from ..images import edit_photo
        from ..segmentation import found_mask
        sent = latest_image(conversation or [])
        if sent is None:
            return {"error": "The user hasn't sent a photo. Ask them to send one."}
        engine = self.engines.get(edit)
        if engine is None:
            return {"error": f"Can't {edit}; the edits are {', '.join(self.engines)}."}
        key = hashlib.sha1(sent).hexdigest()
        # A photo sent with this message is edited afresh; asked for more without one, the edits add up.
        # There's no argument for it, since small models set every flag they're given.
        newest = next((m for m in reversed(conversation or []) if m.get("role") == "user"), {})
        photo = sent if latest_image([newest]) == sent else self.last.get(key, sent)
        mask = None
        if getattr(engine, "MASKED", False):
            if not what.strip():
                return {"error": "Say what to erase, in the what argument."}
            if self.segmenter is None:
                return {"error": "There's no segmentation engine to find what to erase."}
            mask = found_mask(self.segmenter.segment(photo, [subject(what)])[0])
            if mask is None:
                return {"error": f"Couldn't find {what!r} in the photo. Ask the user to describe it another way."}
        edited = edit_photo(engine, photo, what or None, mask)
        out = io.BytesIO()
        # A photo with transparency stays a PNG; the rest are JPEGs, which keep a 4x upscale to a few MB
        if edited.mode == "RGBA":
            edited.save(out, "PNG")
        else:
            edited.convert("RGB").save(out, "JPEG", quality=JPEG_QUALITY)
        self.last = {key: out.getvalue()}
        done = {"remove_background": "with its background removed", "erase": f"with {what} erased",
                "upscale": f"upscaled to {edited.width}x{edited.height}", "brighten": "brightened"}[edit]
        # Small models repeat what a tool says, so it says what they should
        return {"done": f"The photo is below your reply, {done}. Say only something like \"Here's your photo, "
                        f"{done}.\" You can't see it, so don't describe it.",
                IMAGES: [base64.b64encode(out.getvalue()).decode("ascii")]}
