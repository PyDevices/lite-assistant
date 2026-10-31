"""
medialink peer for lite-assistant
Joins a medialink link as one more peer, so a board running medialink's
walkie-talkie or hands-free example talks to the server the way it talks to
another board. Each turn the board sends is answered over the same link:

    echo        sends the turn straight back, to test the link
    transcribe  says back what it heard, by Moonshine and the server's voice
    chat        answers it through /v1/voice (needs the server's chat model)

It is a client of the server's HTTP API, like board_client.py, which it
uses, so it runs beside the server or on any machine that can reach it.
medialink comes with PyDevices (pip install pydevices, or PYTHONPATH set to
a pydevices checkout's lib/, and its utils/ too for finding peers by name).

    python examples/medialink_peer.py --mode transcribe
    python examples/medialink_peer.py --mode chat --name assistant

On the board, set PEER_ADDRESS to this machine's address and --port, or give
the board's example PEER = "assistant" to find it by name (that needs
python-zeroconf here).
"""
import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board_client import SPEECH_RATE, Client  # noqa: E402

RATE = 16000  # medialink's intercom examples send 16 kHz mono


def resample(pcm, src_rate, dst_rate):
    """16-bit mono PCM from ``src_rate`` to ``dst_rate`` by linear interpolation."""
    count = len(pcm) // 2
    if not count or src_rate == dst_rate:
        return bytes(pcm)
    samples = memoryview(bytes(pcm)).cast("h")
    out_count = count * dst_rate // src_rate
    out = bytearray(out_count * 2)
    view = memoryview(out).cast("h")
    step = src_rate / dst_rate
    last = count - 1
    for j in range(out_count):
        pos = j * step
        i = int(pos)
        if i >= last:
            view[j] = samples[last]
            continue
        frac = pos - i
        view[j] = int(samples[i] + (samples[i + 1] - samples[i]) * frac)
    return bytes(out)


def answer(client, mode, pcm, voice=None):
    """What to say back to a turn: ``(heard, reply)``, with ``reply`` as
    16-bit mono PCM at 16 kHz and ``heard`` the transcript, if any."""
    if mode == "echo":
        return None, bytes(pcm)
    if mode == "transcribe":
        heard = client.transcribe(pcm, RATE).get("text", "").strip()
        text = heard if heard else "I didn't catch that."
        spoken = b"".join(bytes(piece) for piece in client.speech(text, voice=voice))
        return heard, resample(spoken, SPEECH_RATE, RATE)
    if mode == "chat":
        fields = {"stt_model": "moonshine"}
        if voice:
            fields["voice"] = voice
        spoken = b"".join(bytes(piece) for piece in client.voice_stream(pcm, RATE, **fields))
        return client.heard, resample(spoken, SPEECH_RATE, RATE)
    raise ValueError("unknown mode %r" % mode)


class Format:
    def __init__(self, rate, channels):
        self.rate = rate
        self.channels = channels


class TurnRecorder:
    """Stands in for a speaker: keeps what the peer says."""

    def __init__(self, rate=RATE):
        self.format = Format(rate, 1)
        self.pcm = bytearray()

    def try_write(self, data):
        self.pcm += data
        return len(data)

    def queued_size(self):
        return 0

    def take(self):
        pcm, self.pcm = bytes(self.pcm), bytearray()
        return pcm

    def close(self):
        pass


class ReplyMic:
    """Stands in for a microphone: gives ``pcm`` at the pace it would play."""

    def __init__(self, rate=RATE):
        self.format = Format(rate, 1)
        self.pcm = b""
        self.pos = 0
        self.start = None

    def load(self, pcm):
        self.pcm = pcm
        self.pos = 0
        self.start = None

    @property
    def done(self):
        return self.pos >= len(self.pcm)

    def open(self):
        if self.start is None:
            self.start = time.monotonic()

    def readinto(self, buf):
        if self.start is None:
            self.open()
        due = int((time.monotonic() - self.start) * self.format.rate) * 2
        count = min(len(buf), due - self.pos, len(self.pcm) - self.pos)
        if count <= 0:
            return 0
        buf[:count] = self.pcm[self.pos:self.pos + count]
        self.pos += count
        return count

    def close(self):
        pass


def send_reply(intercom, mic, pcm, pace=0.002):
    """Takes the floor and sends ``pcm`` in real time. False if refused."""
    mic.load(pcm)
    if not intercom.press():
        return False
    while not mic.done and intercom.talking:
        intercom.service()
        time.sleep(pace)
    intercom.release()
    return True


def answer_while_serving(intercom, client, mode, pcm, voice=None):
    """answer(), on a thread, while the link keeps answering pings: a chat
    reply can take longer than the peer waits before it calls us gone."""
    result = {}

    def work():
        try:
            result["value"] = answer(client, mode, pcm, voice)
        except Exception as e:
            result["error"] = e

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    while worker.is_alive():
        intercom.service()
        time.sleep(0.005)
    if "error" in result:
        raise result["error"]
    return result["value"]


def serve(args, client):
    import medialink

    link = medialink.Link(medialink.UDPTransport(args.port), retries=2)
    speaker = TurnRecorder()
    mic = ReplyMic()
    intercom = medialink.Intercom(link, mic, speaker, prebuffer=1)
    finder = None
    try:
        finder = medialink.MDNS()
        finder.advertise(args.name, args.port)
    except Exception as e:  # no python-zeroconf: reach it by address instead
        print("not advertising %r (%s); peers need this machine's address" % (args.name, e))
    print("medialink peer %r on UDP port %d, mode %s" % (args.name, args.port, args.mode))
    heard = 0
    try:
        while True:
            intercom.service()
            if intercom.stats["spurts"] != heard and link.floor == medialink.IDLE:
                heard = intercom.stats["spurts"]
                turn = speaker.take()
                if len(turn) < RATE // 5:  # under 100 ms: nothing to answer
                    continue
                started = time.monotonic()
                try:
                    text, reply = answer_while_serving(intercom, client, args.mode, turn, args.voice)
                except Exception as e:
                    print("couldn't answer: %s" % e)
                    continue
                took = time.monotonic() - started
                print("heard %.1f s%s; answering with %.1f s after %.1f s" % (
                    len(turn) / 2 / RATE, (": %r" % text) if text else "", len(reply) / 2 / RATE, took))
                try:
                    if not send_reply(intercom, mic, reply):
                        print("the peer took the floor first")
                except medialink.PeerUnreachable:
                    print("the peer went away")
                speaker.take()  # nothing it said while we answered counts
            time.sleep(0.002)
    finally:
        intercom.close()
        link.close()
        if finder:
            finder.deinit()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--mode", choices=("echo", "transcribe", "chat"), default="transcribe")
    parser.add_argument("--name", default="assistant", help="this peer's name on the network")
    parser.add_argument("--port", type=int, default=47840, help="UDP port to listen on")
    parser.add_argument("--server", default="127.0.0.1", help="the lite-assistant server's host")
    parser.add_argument("--server-port", type=int, default=5000)
    parser.add_argument("--voice", default=None, help="a voice id from /voices")
    args = parser.parse_args(argv)
    serve(args, Client(args.server, args.server_port))


if __name__ == "__main__":
    main()
