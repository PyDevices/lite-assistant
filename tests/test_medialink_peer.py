"""
Tests for examples/medialink_peer.py that need no server: resampling, what
each mode answers, the paced reply, and, where medialink is importable, a
turn answered over a real link on localhost.
"""
import importlib.util
import math
import os
import struct
import sys
import threading
import time
import unittest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "examples"))
PATH = os.path.join(HERE, "..", "examples", "medialink_peer.py")
spec = importlib.util.spec_from_file_location("medialink_peer", PATH)
mp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mp)

try:
    import medialink
except ImportError:
    medialink = None


def tone(rate, seconds, freq=440, level=8000):
    count = int(rate * seconds)
    return struct.pack("<%dh" % count, *(int(level * math.sin(2 * math.pi * freq * n / rate)) for n in range(count)))


def peak_hz(pcm, rate):
    """The strongest frequency, by counting rising zero crossings."""
    samples = struct.unpack("<%dh" % (len(pcm) // 2), pcm)
    rises = sum(1 for a, b in zip(samples, samples[1:]) if a < 0 <= b)
    return rises * rate / len(samples)


class FakeClient:
    """Answers as the server would, handing speech back in a reused buffer."""

    def __init__(self, heard="hello there"):
        self.said = None
        self.heard_text = heard
        self.heard = None

    def transcribe(self, pcm, rate):
        return {"text": self.heard_text}

    def _stream(self, pcm):
        buf = bytearray(1024)
        for at in range(0, len(pcm), 1024):
            piece = pcm[at:at + 1024]
            buf[:len(piece)] = piece
            yield memoryview(buf)[:len(piece)]

    def speech(self, text, voice=None):
        self.said = text
        return self._stream(tone(24000, 0.5))

    def voice_stream(self, pcm, rate, **fields):
        self.heard = self.heard_text
        return self._stream(tone(24000, 0.25, freq=300))


class Resample(unittest.TestCase):
    def test_24k_to_16k_keeps_the_pitch_and_the_length(self):
        out = mp.resample(tone(24000, 1.0), 24000, 16000)
        self.assertEqual(len(out), 16000 * 2)
        self.assertAlmostEqual(peak_hz(out, 16000), 440, delta=5)

    def test_same_rate_and_empty_are_unchanged(self):
        self.assertEqual(mp.resample(b"\x01\x00\x02\x00", 16000, 16000), b"\x01\x00\x02\x00")
        self.assertEqual(mp.resample(b"", 24000, 16000), b"")


class Answer(unittest.TestCase):
    def test_echo_sends_the_turn_back(self):
        self.assertEqual(mp.answer(FakeClient(), "echo", b"abcd"), (None, b"abcd"))

    def test_transcribe_says_what_it_heard(self):
        client = FakeClient("turn the lights on")
        heard, reply = mp.answer(client, "transcribe", tone(16000, 1))
        self.assertEqual(heard, "turn the lights on")
        self.assertEqual(client.said, "turn the lights on")
        # Every piece was copied before the client reused its buffer.
        self.assertEqual(reply, mp.resample(tone(24000, 0.5), 24000, 16000))

    def test_transcribe_of_nothing_says_so(self):
        client = FakeClient("")
        mp.answer(client, "transcribe", tone(16000, 1))
        self.assertEqual(client.said, "I didn't catch that.")

    def test_chat_answers_through_the_voice_endpoint(self):
        heard, reply = mp.answer(FakeClient("what time is it"), "chat", tone(16000, 1))
        self.assertEqual(heard, "what time is it")
        self.assertEqual(len(reply), 4000 * 2)


class Reply(unittest.TestCase):
    def test_the_reply_comes_at_the_pace_it_plays(self):
        mic = mp.ReplyMic()
        mic.load(bytes(16000))  # half a second
        mic.open()
        buf = bytearray(640)
        got = 0
        start = time.monotonic()
        while not mic.done:
            got += mic.readinto(buf)
            time.sleep(0.002)
        self.assertEqual(got, 16000)
        self.assertGreater(time.monotonic() - start, 0.45)


@unittest.skipIf(medialink is None, "needs medialink on the path")
class OverALink(unittest.TestCase):
    def test_a_turn_is_answered_over_the_link(self):
        peer_link = medialink.Link(medialink.UDPTransport(0), retries=2)
        side_link = medialink.Link(medialink.UDPTransport(0), retries=2)
        recorder = mp.TurnRecorder()
        reply_mic = mp.ReplyMic()
        peer = medialink.Intercom(peer_link, reply_mic, recorder, prebuffer=1)
        heard_back = mp.TurnRecorder()
        talk = mp.ReplyMic()
        side = medialink.Intercom(side_link, talk, heard_back, prebuffer=1)
        stop = [False]

        def run_peer():
            done = 0
            while not stop[0]:
                peer.service()
                if peer.stats["spurts"] != done and peer_link.floor == medialink.IDLE:
                    done = peer.stats["spurts"]
                    _heard, reply = mp.answer(FakeClient(), "echo", recorder.take())
                    mp.send_reply(peer, reply_mic, reply)
                time.sleep(0.002)

        thread = threading.Thread(target=run_peer, daemon=True)
        thread.start()
        try:
            side_link.connect(("127.0.0.1", peer_link.transport.port))
            said = tone(16000, 0.4)
            self.assertTrue(mp.send_reply(side, talk, said))
            deadline = time.monotonic() + 5
            while len(heard_back.pcm) < len(said) and time.monotonic() < deadline:
                side.service()
                time.sleep(0.002)
            self.assertGreaterEqual(len(heard_back.pcm), len(said) * 9 // 10)
            self.assertAlmostEqual(peak_hz(bytes(heard_back.pcm), 16000), 440, delta=10)
        finally:
            stop[0] = True
            thread.join()
            side.close()
            peer.close()
            side_link.close()
            peer_link.close()


if __name__ == "__main__":
    unittest.main()
