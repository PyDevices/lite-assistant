"""
Editing photos: the background removal, erasing and upscaling engines with
fake graphs, erasing what a prompt names through /v1/images/edits, and the
edit_photo tool, whose photos reach the chat as image_generation_call items.
"""
import base64
import io
import json
import unittest

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from lite_assistant.app import Engines, create_app
from lite_assistant.engines import ImageEditEngine, ToolCall
from lite_assistant.engines.esrgan_engine import ESRGANEngine
from lite_assistant.engines.migan_engine import MIGANEngine
from lite_assistant.engines.ormbg_engine import OrmbgEngine
from lite_assistant.responses import ResponseBuilder, ResponseRequest, input_items, to_messages
from lite_assistant.tools import BuiltinCall, definition, run
from lite_assistant.tools.editing import PhotoEditor

from test_segment import FakeSegmenter


def photo(size=(60, 40)) -> Image.Image:
    """ A photo whose every pixel differs, so a misplaced tile shows. """
    width, height = size
    ys, xs = np.mgrid[0:height, 0:width]
    return Image.fromarray(np.stack([xs * 255 // width, ys * 255 // height, (xs + ys) % 256], -1).astype(np.uint8))


def jpeg(size=(60, 40)) -> bytes:
    out = io.BytesIO()
    photo(size).save(out, "JPEG")
    return out.getvalue()


def engine(cls, graph):
    """ An engine running a fake graph, without loading its model. """
    made = cls.__new__(cls)
    made.graph = graph
    return made


def nearest_x4(x):
    """ A fake Real-ESRGAN: each pixel of an NHWC tile becomes 4x4, NCHW. """
    return [np.repeat(np.repeat(x, 4, 1), 4, 2).transpose(0, 3, 1, 2)]


class FakeMIGAN(MIGANEngine):
    """ Fills what it erases with gray, and remembers what it was given. """
    def __init__(self):
        self.inputs = []
        self.graph = lambda x: self.inputs.append(x) or [np.zeros((1, 3, 512, 512), np.float32)]


class TestEngines(unittest.TestCase):
    def test_upscaling_tiles_meet_without_seams(self):
        for size in ((300, 200), (100, 40)):
            upscaled = engine(ESRGANEngine, nearest_x4).edit(photo(size))
            expected = photo(size).resize((size[0] * 4, size[1] * 4), Image.NEAREST)
            self.assertEqual(upscaled.size, expected.size)
            self.assertTrue(np.array_equal(np.asarray(upscaled), np.asarray(expected)), size)

    def test_upscaling_has_a_limit(self):
        with self.assertRaises(ValueError):
            engine(ESRGANEngine, nearest_x4).edit(Image.new("RGB", (1200, 1000)))

    def test_background_removal_makes_the_matte_alpha(self):
        def matte(x):
            alpha = np.zeros((1, 1, 1024, 1024), np.float32)
            alpha[..., 512:] = 0.7
            return [alpha]
        removed = engine(OrmbgEngine, matte).edit(photo())
        self.assertEqual(removed.mode, "RGBA")
        self.assertEqual(removed.getpixel((5, 20))[3], 0)
        self.assertEqual(removed.getpixel((55, 20)), photo().getpixel((55, 20)) + (255,))

    def test_erasing_changes_only_the_mask_and_what_it_encloses(self):
        eraser = FakeMIGAN()
        mask = Image.new("L", (60, 40))
        # An outline, like a pair of glasses' frame
        mask.paste(255, (20, 10, 40, 30))
        mask.paste(0, (25, 15, 35, 25))
        erased = np.asarray(eraser.edit(photo(), None, mask)).astype(int)
        original = np.asarray(photo()).astype(int)
        self.assertTrue(np.all(abs(erased[20, 30] - 128) <= 1), "the enclosed middle is erased")
        self.assertTrue(np.array_equal(erased[:5], original[:5]))
        # The graph saw the photo, with the erased part blanked and the mask flagged in its first channel
        given = eraser.inputs[0][0]
        self.assertEqual(given.shape, (4, 512, 512))
        self.assertEqual(given[0].min(), -0.5)
        self.assertEqual(given[0].max(), 0.5)

    def test_erasing_needs_a_mask(self):
        with self.assertRaises(ValueError):
            FakeMIGAN().edit(photo())


class TestEndpoint(unittest.TestCase):
    def setUp(self):
        self.eraser = FakeMIGAN()
        engines = Engines({}, {}, {}, edit={"migan": self.eraser}, segment={"clipseg": FakeSegmenter()})
        self.http = TestClient(create_app(engines, None, "none"), client=("127.0.0.1", 50000))

    def erase(self, prompt):
        return self.http.post("/v1/images/edits", files={"image": ("p.jpg", jpeg())},
                              data={"model": "erase", "prompt": prompt, "response_format": "png"})

    def test_erases_what_the_prompt_names(self):
        found = self.erase("the tree on the right")
        self.assertEqual(found.status_code, 200, found.text)
        erased = np.asarray(Image.open(io.BytesIO(found.content))).astype(int)
        # The fake segmenter's tree is the bottom right quarter
        self.assertTrue(np.all(abs(erased[35, 55] - 128) <= 2))
        self.assertLess(abs(erased[5, 5] - np.asarray(Image.open(io.BytesIO(jpeg())))[5, 5]).max(), 10)

    def test_nothing_found_or_nothing_said(self):
        self.assertIn("Couldn't find", self.erase("a cat").json()["error"]["message"])
        self.assertEqual(self.http.post("/v1/images/edits", files={"image": ("p.jpg", jpeg())},
                                        data={"model": "erase"}).status_code, 400)


class FakeUpscale(ImageEditEngine):
    TASK = "upscale"

    def edit(self, photo, prompt=None, mask=None):
        return photo.resize((photo.width * 4, photo.height * 4))


class TestTool(unittest.TestCase):
    @staticmethod
    def sent(size=(60, 40)):
        return [{"role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(jpeg(size)).decode()}]}]

    def test_edits_are_shown_not_given_to_the_model(self):
        editor = PhotoEditor({"upscale": FakeUpscale()})
        output, images = run(editor.edit_photo, ToolCall("edit_photo", '{"edit": "upscale"}'), self.sent())
        self.assertIn("240x160", json.loads(output)["done"])
        self.assertEqual(Image.open(io.BytesIO(base64.b64decode(images[0]))).size, (240, 160))
        # Asked again without a new photo, it edits its last edit
        later = self.sent() + [{"role": "assistant", "content": "Here it is."}, {"role": "user", "content": "Again"}]
        output, images = run(editor.edit_photo, ToolCall("edit_photo", '{"edit": "upscale"}'), later)
        self.assertIn("960x640", json.loads(output)["done"])
        # A photo sent with the message is edited afresh
        output, images = run(editor.edit_photo, ToolCall("edit_photo", '{"edit": "upscale"}'), self.sent())
        self.assertIn("240x160", json.loads(output)["done"])

    def test_a_cut_out_stays_cut_out(self):
        cut = Image.new("RGBA", (60, 40), (200, 0, 0, 0))
        cut.paste((200, 0, 0, 255), (20, 10, 40, 30))
        data = io.BytesIO()
        cut.save(data, "PNG")
        sent = [{"role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()}]}]
        found = PhotoEditor({"upscale": FakeUpscale()}).edit_photo("upscale", conversation=sent)
        edited = Image.open(io.BytesIO(base64.b64decode(found["images"][0])))
        self.assertEqual((edited.size, edited.mode), ((240, 160), "RGBA"))
        self.assertEqual((edited.getpixel((5, 5))[3], edited.getpixel((120, 80))[3]), (0, 255))

    def test_erase_finds_what_to_erase(self):
        editor = PhotoEditor({"erase": FakeMIGAN()}, FakeSegmenter())
        self.assertIn("images", editor.edit_photo("erase", "the tree", conversation=self.sent()))
        self.assertIn("Couldn't find", editor.edit_photo("erase", "the cat", conversation=self.sent())["error"])
        self.assertIn("Say what", editor.edit_photo("erase", conversation=self.sent())["error"])

    def test_errors(self):
        editor = PhotoEditor({"upscale": FakeUpscale()})
        self.assertIn("hasn't sent", editor.edit_photo("upscale", conversation=[])["error"])
        self.assertIn("upscale", editor.edit_photo("brighten", conversation=self.sent())["error"])

    def test_edits_are_an_enum(self):
        parameters = definition(PhotoEditor({}).edit_photo)["function"]["parameters"]
        self.assertEqual(parameters["properties"]["edit"]["enum"], ["remove_background", "erase", "upscale", "brighten"])
        self.assertEqual(set(parameters["properties"]), {"edit", "what"})
        self.assertEqual(parameters["required"], ["edit"])


class TestResponses(unittest.TestCase):
    def test_images_follow_the_call(self):
        builder = ResponseBuilder(ResponseRequest(input="Upscale it"), "fake")
        call = BuiltinCall("edit_photo", "{}", '{"done": "upscaled"}', "call_1", ["aW1hZ2U="])
        events = [json.loads(line[6:]) for chunk in builder.events(iter([call, "Here it is."]), {})
                  for line in chunk.decode().splitlines() if line.startswith("data: ")]
        output = events[-1]["response"]["output"]
        self.assertEqual([item["type"] for item in output], ["mcp_call", "image_generation_call", "message"])
        self.assertEqual(output[1]["result"], "aW1hZ2U=")
        added = [(e["output_index"], e["item"]["type"]) for e in events if e["type"] == "response.output_item.added"]
        self.assertEqual(added, [(0, "mcp_call"), (1, "image_generation_call"), (2, "message")])
        # Given back, the image is kept but the model doesn't see it
        items = input_items([{"role": "user", "content": "Upscale it"}] + output)
        self.assertEqual(items[2]["type"], "image_generation_call")
        self.assertEqual([m["role"] for m in to_messages(None, items)], ["user", "assistant", "tool", "assistant"])


if __name__ == "__main__":
    unittest.main()
