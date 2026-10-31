"""
The conversation store: conversations kept in SQLite outlast a restart,
are listed by user, the last changed first, and old ones are forgotten,
but pinned ones.
"""
import os
import tempfile
import time
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from lite_assistant.app import Engines, create_app
from lite_assistant.store import Conversation, Store

from test_app import FakeLLM, FakeSTT, FakeTTS


def message(item_id, text):
    return {"id": item_id, "type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


class TestStore(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = os.path.join(folder.name, "conversations.db")

    def test_conversations_outlast_a_restart(self):
        store = Store(self.path)
        conversation = Conversation("conv_1", {"title": "Hi"}, "alice")
        conversation.items = [message("msg_1", "Hi")]
        store.put_conversation(conversation)
        store.add_items(conversation, [message("msg_2", "Again"), message("msg_3", "And again")])
        store.delete_item(conversation, "msg_2")
        store.set_metadata(conversation, {"title": "Renamed"})
        again = Store(self.path).get_conversation("conv_1")
        self.assertEqual((again.user, again.object["metadata"]), ("alice", {"title": "Renamed"}))
        self.assertEqual([item["id"] for item in again.items], ["msg_1", "msg_3"])
        self.assertEqual(again.items, conversation.items)
        self.assertTrue(Store(self.path).delete_conversation("conv_1"))
        self.assertIsNone(Store(self.path).get_conversation("conv_1"))
        self.assertFalse(Store(self.path).delete_conversation("conv_1"))

    def test_each_users_last_changed_first(self):
        store = Store()
        with mock.patch("time.time", return_value=1000):
            for name, user in (("conv_a", "alice"), ("conv_b", "alice"), ("conv_c", "bob")):
                store.put_conversation(Conversation(name, {}, user))
        with mock.patch("time.time", return_value=2000):
            store.add_items(store.get_conversation("conv_a"), [message("msg_1", "Hi")])
        self.assertEqual([c.id for c in store.list_conversations("alice")], ["conv_a", "conv_b"])
        self.assertEqual(store.list_conversations("alice")[0].updated_at, 2000)
        self.assertEqual([c.id for c in store.list_conversations("bob")], ["conv_c"])

    def test_old_conversations_are_forgotten_but_pinned_ones(self):
        store = Store()
        day = 86400
        with mock.patch("time.time", return_value=time.time() - 40 * day):
            for name, metadata in (("conv_old", {}), ("conv_pinned", {"pinned": "true"}),
                                   ("conv_unpinned", {"pinned": "false"})):
                store.put_conversation(Conversation(name, metadata, "alice"))
        store.put_conversation(Conversation("conv_new", {}, "alice"))
        self.assertEqual(store.forget(30), 2)
        self.assertEqual(sorted(c.id for c in store.list_conversations("alice")), ["conv_new", "conv_pinned"])


    def test_a_chat_outlasts_a_restart_of_the_server(self):
        def server():
            engines = Engines({"kokoro": FakeTTS()}, {"moonshine": FakeSTT()}, {"litertlm": FakeLLM()})
            return TestClient(create_app(engines, None, "none", database=self.path), client=("127.0.0.1", 50000))
        conversation = server().post("/v1/conversations", json={}).json()["id"]
        server().post("/v1/responses", json={"input": "Hi", "conversation": conversation})
        items = server().get(f"/v1/conversations/{conversation}/items", params={"order": "asc"}).json()["data"]
        self.assertEqual([item["role"] for item in items], ["user", "assistant"])
        # The chat list: the last changed first, a page at a time
        http = server()
        newer = http.post("/v1/conversations", json={"metadata": {"title": "Newer"}}).json()["id"]
        listed = http.get("/v1/conversations", params={"limit": 1}).json()
        self.assertEqual(([c["id"] for c in listed["data"]], listed["has_more"]), ([newer], True))
        self.assertEqual(listed["data"][0]["metadata"], {"title": "Newer"})
        self.assertIn("updated_at", listed["data"][0])
        rest = http.get("/v1/conversations", params={"after": newer}).json()
        self.assertEqual(([c["id"] for c in rest["data"]], rest["has_more"]), ([conversation], False))


if __name__ == "__main__":
    unittest.main()
