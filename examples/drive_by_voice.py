# SPDX-FileCopyrightText: 2026 Brad Barnett
#
# SPDX-License-Identifier: MIT

"""Turn a spoken command into a drive command, without a chat model.

Moonshine transcribes a short recording and the server's fast classifier
sorts the words into one of a few commands: "left", "faster", "stop". Both
run in milliseconds on the server, so a push-to-talk button on a phone or a
board can drive a robot car with no language model behind it.

Needs: a lite-assistant server with the Moonshine and classifier engines,
reachable at HOST, and a 16-bit mono WAV of someone speaking a command.

Run it:

    python examples/drive_by_voice.py

It prints what was heard and the command, such as "heard 'go left a bit'"
and "command: left".

Tested on: Python 3.12 on Linux, against a lite-assistant server with the
Moonshine and gliclass engines.
"""
import os
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board_client import Client  # noqa: E402

# Change these to match your setup.
HOST = "localhost"  # the machine running lite-assistant
PORT = 5000
RECORDING = "command.wav"  # 16-bit mono speech
COMMANDS = ["forward", "back", "left", "right", "faster", "slower", "stop"]
MIN_SCORE = 0.5  # below this the car stops instead of guessing (0 to 1)


def choose(label, score):
    """The command to act on: ``label``, or "stop" when the server isn't sure."""
    return label if score >= MIN_SCORE else "stop"


def main():
    # Read the recording's samples and its rate.
    with wave.open(RECORDING, "rb") as file:
        rate = file.getframerate()
        pcm = file.readframes(file.getnframes())

    # Transcribe it.
    la = Client(HOST, PORT)
    text = la.transcribe(pcm, rate)["text"].strip()
    print("heard %r" % text)
    if not text:
        print("command: stop")
        return "stop"

    # Sort the words into one command. The labels compete, so exactly one wins.
    best = la.post_json("/v1/classify", {"input": text, "labels": COMMANDS})["data"][0]
    command = choose(best["label"], best["score"])
    print("command: %s (%.2f)" % (command, best["score"]))
    return command


if __name__ == "__main__":
    main()
