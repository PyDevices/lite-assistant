"""
Tests for examples/board_client.py that need no server: reading replies and
WebSocket frames from a socket that hands over bytes in uneven pieces.
"""
import importlib.util
import json
import os
import unittest

PATH = os.path.join(os.path.dirname(__file__), "..", "examples", "board_client.py")
spec = importlib.util.spec_from_file_location("board_client", PATH)
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)


class FakeSocket:
    """Hands back ``data`` in pieces of at most ``step`` bytes."""

    def __init__(self, data, step=1 << 20):
        self.data = data
        self.step = step
        self.sent = b""

    def recv(self, n):
        piece, self.data = self.data[:min(n, self.step)], self.data[min(n, self.step):]
        return piece

    def recv_into(self, view):
        piece = self.recv(len(view))
        view[:len(piece)] = piece
        return len(piece)

    def send(self, data):
        self.sent += bytes(data)
        return len(data)


def frame(event):
    data = json.dumps(event).encode()
    assert len(data) < 126
    return bytes((0x81, len(data))) + data


class TestReply(unittest.TestCase):
    def test_read_takes_only_what_was_asked_from_bytes_read_with_the_head(self):
        # The head and what follows it arrive in one recv
        reply = bc._Reply(FakeSocket(b"HTTP/1.1 200 OK\r\n\r\nabcdef"))
        reply.head()
        self.assertEqual(reply.read(2), b"ab")
        self.assertEqual(reply.read(4), b"cdef")

    def test_readline(self):
        reply = bc._Reply(FakeSocket(b"HTTP/1.0 200 OK\r\n\r\ndata: 1\n\ndata: 2\n", step=3))
        reply.head()
        self.assertEqual([reply.readline() for _ in range(4)], [b"data: 1\n", b"\n", b"data: 2\n", b""])


class TestRealtimeFrames(unittest.TestCase):
    def session(self, data, step=1 << 20):
        sock = FakeSocket(b"HTTP/1.1 101 Switching Protocols\r\n\r\n" + data, step)
        reply = bc._Reply(sock)
        reply.head()
        return bc.Realtime(sock, reply), sock

    def test_frames_that_arrive_with_the_handshake(self):
        rt, _ = self.session(frame({"type": "session.created"}) + frame({"type": "session.updated"}))
        self.assertEqual(rt.recv()["type"], "session.created")
        self.assertEqual(rt.recv()["type"], "session.updated")
        self.assertIsNone(rt.recv())

    def test_frames_a_byte_at_a_time(self):
        rt, _ = self.session(frame({"type": "a"}) + frame({"type": "b"}), step=1)
        self.assertEqual([rt.recv()["type"], rt.recv()["type"]], ["a", "b"])

    def test_ping_is_answered_and_skipped(self):
        rt, sock = self.session(b"\x89\x02hi" + frame({"type": "a"}))
        self.assertEqual(rt.recv()["type"], "a")
        self.assertEqual(sock.sent, b"\x8a\x82\0\0\0\0hi")

    def test_sent_events_are_masked_text_frames(self):
        rt, sock = self.session(b"")
        rt.send({"type": "x" * 200})
        self.assertEqual(sock.sent[:2], b"\x81\xfe")
        n = (sock.sent[2] << 8) | sock.sent[3]
        self.assertEqual(json.loads(sock.sent[8:8 + n])["type"], "x" * 200)

    def test_reply_yields_audio_and_keeps_the_transcript(self):
        import base64
        events = [{"type": "response.output_audio.delta", "delta": base64.b64encode(b"\1\2\3\4").decode()},
                  {"type": "response.output_audio_transcript.delta", "delta": "Hi."},
                  {"type": "response.done", "response": {"status": "completed"}}]
        rt, sock = self.session(b"".join(frame(e) for e in events))
        self.assertEqual(b"".join(rt.reply()), b"\1\2\3\4")
        self.assertEqual(rt.transcript, "Hi.")
        self.assertEqual(json.loads(sock.sent[6:])["type"], "response.create")


if __name__ == "__main__":
    unittest.main()
