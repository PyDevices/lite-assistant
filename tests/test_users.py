"""
Users and their keys: adding and removing them, and which requests the
server takes, from this PC, from elsewhere, and behind a proxy.
"""
import os
import tempfile
import unittest

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from lite_assistant.app import Engines, create_app
from lite_assistant.users import FAILURES, Users

from test_app import FakeLLM, FakeSTT, FakeTTS

LOCAL, ELSEWHERE, PUBLIC = ("127.0.0.1", 50000), ("192.168.1.20", 50000), ("9.9.9.9", 50000)


class TestUsers(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = os.path.join(folder.name, "users.json")
        self.users = Users(self.path)

    def test_the_owner_needs_no_file(self):
        owner = self.users.owner()
        self.assertTrue(owner.admin)
        self.assertEqual([u["name"] for u in self.users.list()], [owner.name])
        self.assertFalse(os.path.exists(self.path))

    def test_keys_find_their_users_and_are_not_kept(self):
        key = self.users.add("alice")
        self.assertEqual(self.users.by_key(key).name, "alice")
        self.assertFalse(self.users.by_key(key).admin)
        self.assertNotIn(key, open(self.path).read())
        self.assertIsNone(self.users.by_key(key + "x"))
        # A new key replaces the old
        newer = self.users.new_key("alice")
        self.assertIsNone(self.users.by_key(key))
        self.assertEqual(self.users.by_key(newer).name, "alice")
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)

    def test_names_and_the_owner_are_guarded(self):
        for name in ("Alice", "a b", "../x", "", "x" * 33):
            with self.assertRaises(ValueError):
                self.users.add(name)
        self.users.add("alice")
        with self.assertRaises(ValueError):
            self.users.add("alice")
        owner = self.users.owner().name
        with self.assertRaises(ValueError):
            self.users.remove(owner)
        with self.assertRaises(ValueError):
            self.users.set_admin(owner, False)

    def test_changes_from_the_command_line_are_seen(self):
        # The server's copy, and the command line's
        key = Users(self.path).add("bob", admin=True)
        self.assertTrue(self.users.by_key(key).admin)
        Users(self.path).remove("bob")
        self.assertIsNone(self.users.by_key(key))


class TestRequests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.users = Users(os.path.join(folder.name, "users.json"))
        self.alice = self.users.add("alice")
        self.bob = self.users.add("bob")

    def client(self, where=ELSEWHERE, key=None, behind_proxy=False, allow_public=False):
        engines = Engines({"kokoro": FakeTTS()}, {"moonshine": FakeSTT()}, {"litertlm": FakeLLM()})
        http = TestClient(create_app(engines, None, "none", users=self.users, behind_proxy=behind_proxy,
                                     allow_public=allow_public), client=where)
        if key:
            http.headers["Authorization"] = f"Bearer {key}"
        return http

    def who(self, http):
        found = http.get("/me")
        return found.json()["name"] if found.status_code == 200 else found.status_code

    def test_this_pc_is_the_owner(self):
        owner = self.users.owner().name
        self.assertEqual(self.who(self.client(LOCAL)), owner)
        # The placeholder key the openai client needs is still the owner
        self.assertEqual(self.who(self.client(LOCAL, "unused")), owner)
        self.assertEqual(self.who(self.client(LOCAL, self.alice)), "alice")

    def test_elsewhere_needs_a_key(self):
        refused = self.client().get("/v1/models")
        self.assertEqual((refused.status_code, refused.json()["error"]["code"]), (401, "missing_api_key"))
        self.assertEqual(refused.headers["www-authenticate"], "Bearer")
        wrong = self.client(key="la-wrong").get("/v1/models")
        self.assertEqual((wrong.status_code, wrong.json()["error"]["code"]), (401, "invalid_api_key"))
        self.assertEqual(self.who(self.client(key=self.alice)), "alice")
        # The page itself asks for the key, so anyone may load it
        self.assertEqual(self.client().get("/").status_code, 200)
        self.assertEqual(self.client().get("/docs").status_code, 401)
        # So are the pages that say how to call the server
        for path in ("/llms.txt", "/v1", "/v1/"):
            self.assertEqual(self.client().get(path).status_code, 200)

    def test_behind_a_proxy_this_pc_needs_a_key_too(self):
        self.assertEqual(self.who(self.client(LOCAL, behind_proxy=True)), 401)
        self.assertEqual(self.who(self.client(LOCAL, "unused", behind_proxy=True)), 401)
        self.assertEqual(self.who(self.client(LOCAL, self.bob, behind_proxy=True)), "bob")

    def test_only_private_addresses_unless_allowed(self):
        refused = self.client(PUBLIC, self.alice).get("/me")
        self.assertEqual((refused.status_code, refused.json()["error"]["code"]), (403, "public_address"))
        self.assertEqual(self.client(PUBLIC).get("/").status_code, 403)
        for host in ("10.1.2.3", "172.20.0.5", "100.101.102.103", "fd7a:115c:a1e0::1", "fe80::1", "::1",
                     "::ffff:192.168.1.5"):
            self.assertEqual(self.who(self.client((host, 50000), self.alice)), "alice", host)
        for host in ("8.8.8.8", "2001:4860:4860::8888", "::ffff:8.8.8.8", "testclient"):
            self.assertEqual(self.who(self.client((host, 50000), self.alice)), 403, host)
        self.assertEqual(self.who(self.client(PUBLIC, self.alice, allow_public=True)), "alice")
        self.assertEqual(self.who(self.client(PUBLIC, allow_public=True)), 401)

    def test_guessing_keys_is_slowed(self):
        http = self.client(key="la-guess")
        for _ in range(FAILURES):
            self.assertEqual(http.get("/me").status_code, 401)
        self.assertEqual(http.get("/me").status_code, 429)
        # Even the right key waits
        http.headers["Authorization"] = f"Bearer {self.alice}"
        self.assertEqual(http.get("/me").status_code, 429)

    def test_only_admins_manage_models(self):
        refused = self.client(key=self.alice).delete("/v1/models/some-model")
        self.assertEqual((refused.status_code, refused.json()["error"]["code"]), (403, "admin_only"))
        self.users.set_admin("alice", True)
        self.assertEqual(self.client(key=self.alice).delete("/v1/models/some-model").status_code, 404)

    def test_each_user_sees_only_their_own(self):
        alice, bob = self.client(key=self.alice), self.client(key=self.bob)
        conversation = alice.post("/v1/conversations", json={"items": []}).json()["id"]
        self.assertEqual(alice.get(f"/v1/conversations/{conversation}").status_code, 200)
        self.assertEqual(bob.get(f"/v1/conversations/{conversation}").status_code, 404)
        self.assertEqual(bob.delete(f"/v1/conversations/{conversation}").status_code, 404)
        response = alice.post("/v1/responses", json={"input": "Hi"}).json()["id"]
        self.assertEqual(alice.get(f"/v1/responses/{response}").status_code, 200)
        self.assertEqual(bob.get(f"/v1/responses/{response}").status_code, 404)
        chained = bob.post("/v1/responses", json={"input": "And?", "previous_response_id": response})
        self.assertEqual(chained.status_code, 404)
        self.assertEqual([c["id"] for c in alice.get("/v1/conversations").json()["data"]], [conversation])
        self.assertEqual(bob.get("/v1/conversations").json()["data"], [])

    def test_tools_know_who_is_asking(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        with open(os.path.join(folder.name, "who.py"), "w") as f:
            f.write("def tools(engines):\n"
                    "    def who_is_asking(user=None) -> str:\n"
                    "        \"\"\" Says who is asking. \"\"\"\n"
                    "        return user.name\n"
                    "    return [who_is_asking]\n")
        engines = Engines({"kokoro": FakeTTS()}, {"moonshine": FakeSTT()}, {"litertlm": FakeLLM()})
        app = create_app(engines, None, "none", tool_dirs=[folder.name], users=self.users)
        http = TestClient(app, client=ELSEWHERE, headers={"Authorization": f"Bearer {self.bob}"})
        output = http.post("/v1/responses", json={"input": "Who?", "builtin_tools": ["who_is_asking"]}).json()["output"]
        self.assertEqual([item["output"] for item in output if item["type"] == "mcp_call"], ["bob"])

    def test_a_browser_sends_its_key_as_a_subprotocol(self):
        with self.assertRaises(WebSocketDisconnect):
            with self.client().websocket_connect("/v1/realtime"):
                pass
        protocols = ["realtime", f"openai-insecure-api-key.{self.alice}"]
        with self.client().websocket_connect("/v1/realtime", subprotocols=protocols) as ws:
            self.assertEqual(ws.accepted_subprotocol, "realtime")
            self.assertEqual(ws.receive_json()["type"], "session.created")


if __name__ == "__main__":
    unittest.main()
