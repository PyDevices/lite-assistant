"""
Files: each user's, through OpenAI's Files API, their notes folder for most
purposes and their media folder for "vision"; and a saved chat's photos,
clips and documents kept as media files, read back for the model, and
deleted with the chat.
"""
import base64
import io
import os
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient
from PIL import Image

from lite_assistant.app import Engines, create_app
from lite_assistant.users import Users

from test_app import FakeLLM, FakeSTT, FakeTTS

ELSEWHERE = ("192.168.1.20", 50000)


def png() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), (200, 40, 40)).save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class TestFiles(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.home = folder.name
        patch = mock.patch.dict(os.environ, {"LITE_ASSISTANT_HOME": self.home})
        patch.start()
        self.addCleanup(patch.stop)
        self.users = Users(os.path.join(self.home, "users.json"))
        self.llm = FakeLLM()
        engines = Engines({"kokoro": FakeTTS()}, {"moonshine": FakeSTT()}, {"litertlm": self.llm})
        app = create_app(engines, None, "none", users=self.users, database=os.path.join(self.home, "chats.db"))
        self.alice = TestClient(app, client=ELSEWHERE, headers={"Authorization": f"Bearer {self.users.add('alice')}"})
        self.bob = TestClient(app, client=ELSEWHERE, headers={"Authorization": f"Bearer {self.users.add('bob')}"})

    def test_uploads_are_the_users_notes(self):
        added = self.alice.post("/v1/files", files={"file": ("dentist.txt", b"The dentist is on Tuesday.")},
                                data={"purpose": "user_data"}).json()
        self.assertEqual((added["object"], added["filename"], added["bytes"], added["purpose"]),
                         ("file", "dentist.txt", 26, "user_data"))
        notes = os.path.join(self.home, "users", "alice", "notes")
        self.assertEqual(os.listdir(notes), ["dentist.txt"])
        # A note put in the folder directly is a file too
        with open(os.path.join(notes, "shopping.md"), "w") as f:
            f.write("Milk")
        listed = self.alice.get("/v1/files").json()["data"]
        self.assertEqual(sorted(f["filename"] for f in listed), ["dentist.txt", "shopping.md"])
        self.assertEqual(self.alice.get(f"/v1/files/{added['id']}").json(), added)
        content = self.alice.get(f"/v1/files/{added['id']}/content")
        self.assertEqual((content.content, content.headers["content-type"].split(";")[0]),
                         (b"The dentist is on Tuesday.", "text/plain"))
        # The same name again gets a number, rather than replace the first
        again = self.alice.post("/v1/files", files={"file": ("dentist.txt", b"Moved to Wednesday.")}).json()
        self.assertEqual(again["filename"], "dentist (2).txt")
        # Bob sees none of them
        self.assertEqual(self.bob.get("/v1/files").json()["data"], [])
        self.assertEqual(self.bob.get(f"/v1/files/{added['id']}/content").status_code, 404)
        self.assertEqual(self.bob.delete(f"/v1/files/{added['id']}").status_code, 404)
        self.assertTrue(self.alice.delete(f"/v1/files/{added['id']}").json()["deleted"])
        self.assertEqual(self.alice.get(f"/v1/files/{added['id']}").status_code, 404)

    def test_a_name_cant_leave_the_folder(self):
        added = self.alice.post("/v1/files", files={"file": ("../../evil.txt", b"x")}).json()
        self.assertEqual(added["filename"], "evil.txt")
        self.assertTrue(os.path.isfile(os.path.join(self.home, "users", "alice", "notes", "evil.txt")))

    def test_a_chats_media_is_kept_as_files(self):
        image = png()
        conversation = self.alice.post("/v1/conversations", json={}).json()["id"]
        photo = {"role": "user", "content": [{"type": "input_text", "text": "What color is it?"},
                                             {"type": "input_image", "image_url": f"data:image/png;base64,{image}"}]}
        self.alice.post("/v1/responses", json={"input": [photo], "conversation": conversation})
        items = self.alice.get(f"/v1/conversations/{conversation}/items", params={"order": "asc"}).json()["data"]
        part = items[0]["content"][1]
        self.assertNotIn("image_url", part)
        self.assertEqual(self.alice.get(f"/v1/files/{part['file_id']}/content").content, base64.b64decode(image))
        media = os.path.join(self.home, "users", "alice", "media", conversation)
        self.assertEqual(os.listdir(media), ["image.png"])
        # It's a media file, not a note
        self.assertEqual(self.alice.get("/v1/files", params={"purpose": "user_data"}).json()["data"], [])
        # The next turn's model still sees the photo
        self.alice.post("/v1/responses", json={"input": "And now?", "conversation": conversation})
        first = self.llm.requests[-1]["messages"][0]
        self.assertTrue(any(p.get("type") != "text" for p in first["content"]), first)
        # Deleting the chat deletes its media
        self.alice.delete(f"/v1/conversations/{conversation}")
        self.assertFalse(os.path.exists(media))

    def test_a_deleted_file_is_said_to_be_gone(self):
        conversation = self.alice.post("/v1/conversations", json={"items": [
            {"role": "user", "content": [{"type": "input_image", "image_url": f"data:image/png;base64,{png()}"}]}]}).json()["id"]
        item = self.alice.get(f"/v1/conversations/{conversation}/items").json()["data"][0]
        self.alice.delete(f"/v1/files/{item['content'][0]['file_id']}")
        self.alice.post("/v1/responses", json={"input": "What was it?", "conversation": conversation})
        self.assertIn("has been deleted", str(self.llm.requests[-1]["messages"][0]["content"]))


if __name__ == "__main__":
    unittest.main()
