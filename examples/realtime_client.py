"""
Realtime voice chat with lite-assistant from this computer's microphone and
speakers, through the openai SDK's Realtime client. The server's VAD finds
where each turn ends, so just talk; speaking over a reply interrupts it.

Use headphones: from speakers, the mic hears the reply and interrupts it.

    pip install "openai[realtime]" sounddevice
    python examples/realtime_client.py [--url http://localhost:5000/v1] [--voice af_heart]
"""
import argparse
import base64
import os
import threading

import sounddevice
from openai import OpenAI

RATE = 24000
BYTES_PER_MS = RATE * 2 // 1000


class Speaker:
    """ Plays reply audio as it arrives, and knows how much of each reply has been heard. """
    def __init__(self):
        self.lock = threading.Lock()
        self.queued = bytearray()
        self.item_id = None
        self.played = 0
        self.stream = sounddevice.RawOutputStream(samplerate=RATE, channels=1, dtype="int16",
                                                  callback=self.fill)
        self.stream.start()

    def fill(self, out, frames, time, status):
        with self.lock:
            n = min(len(out), len(self.queued))
            out[:n] = self.queued[:n]
            del self.queued[:n]
            self.played += n
        out[n:] = b"\0" * (len(out) - n)

    def play(self, item_id, pcm):
        with self.lock:
            if item_id != self.item_id:
                self.item_id, self.played = item_id, 0
            self.queued += pcm

    def stop(self):
        """ Stops playing, returning the item that was playing and how many ms of it were heard. """
        with self.lock:
            if not self.queued:
                return None, 0
            self.queued.clear()
            return self.item_id, self.played // BYTES_PER_MS


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--url", default="http://localhost:5000/v1")
    parser.add_argument("--voice", help="a voice id from GET /voices")
    args = parser.parse_args()

    # Any key works from the server's PC; from elsewhere, set OPENAI_API_KEY to a user's key
    client = OpenAI(base_url=args.url, api_key=os.environ.get("OPENAI_API_KEY", "unused"))
    with client.realtime.connect(model="gpt-realtime") as connection:
        session = {"type": "realtime", "audio": {"input": {"transcription": {"model": "whisper-1"}}}}
        if args.voice:
            session["audio"]["output"] = {"voice": args.voice}
        connection.session.update(session=session)

        speaker = Speaker()

        def send_mic(data, frames, time, status):
            connection.input_audio_buffer.append(audio=base64.b64encode(bytes(data)).decode("ascii"))

        with sounddevice.RawInputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=RATE // 10,
                                        callback=send_mic):
            print("Listening; press Ctrl+C to stop.")
            for event in connection:
                if event.type == "input_audio_buffer.speech_started":
                    item_id, heard_ms = speaker.stop()
                    if item_id:
                        # Tell the server how much of the reply was heard, so its transcript ends there
                        connection.conversation.item.truncate(item_id=item_id, content_index=0,
                                                              audio_end_ms=heard_ms)
                elif event.type == "conversation.item.input_audio_transcription.completed":
                    print(f"\nYou: {event.transcript}")
                elif event.type == "response.created":
                    print("Gemma: ", end="", flush=True)
                elif event.type == "response.output_audio_transcript.delta":
                    print(event.delta, end="", flush=True)
                elif event.type == "response.output_audio.delta":
                    speaker.play(event.item_id, base64.b64decode(event.delta))
                elif event.type == "response.done":
                    print(" [interrupted]" if event.response.status == "cancelled" else "")
                elif event.type == "error":
                    print(f"\n[error] {event.error.message}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
