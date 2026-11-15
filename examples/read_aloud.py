# SPDX-FileCopyrightText: 2026 Brad Barnett
#
# SPDX-License-Identifier: MIT

"""Read the text in a photo aloud and save the speech as a WAV file.

Sends a photo of a page, a sign or a label to the server's /v1/ocr, then
sends the text it read to /v1/audio/speech. Hold a page up to a camera and
run this on the frame, and the speech is what a desktop speaker would say.

Needs: a lite-assistant server with the OCR and Kokoro engines, reachable at
HOST, and a JPEG or PNG with printed text.

Run it:

    python examples/read_aloud.py
    aplay read_aloud.wav

It prints the text it read and the length of the speech, and writes
read_aloud.wav.

Tested on: Python 3.12 on Linux, against a lite-assistant server with the
ppocr and kokoro engines.
"""
import os
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board_client import SPEECH_RATE, Client  # noqa: E402

# Change these to match your setup.
HOST = "localhost"  # the machine running lite-assistant
PORT = 5000
PHOTO = "page.jpg"  # a JPEG or PNG with printed text
OUTPUT = "read_aloud.wav"  # where the speech is saved
VOICE = None  # a voice name from /voices; None for the server's default
MAX_CHARACTERS = 600  # read no more than this much, so a page doesn't take minutes


def main():
    # Read the photo.
    with open(PHOTO, "rb") as file:
        photo = file.read()

    # Have the server read the text in it.
    la = Client(HOST, PORT)
    content_type = "image/png" if PHOTO.endswith(".png") else "image/jpeg"
    text = la.post_body("/v1/ocr", photo, content_type)["text"].strip()
    if not text:
        print("no text found")
        return None
    text = text[:MAX_CHARACTERS]
    print("read: %s" % text)

    # Speak it. The server streams 16-bit mono PCM, which goes into a WAV.
    pcm = b"".join(bytes(piece) for piece in la.speech(text, voice=VOICE))
    with wave.open(OUTPUT, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(SPEECH_RATE)
        file.writeframes(pcm)
    print("%s: %.1f seconds of speech" % (OUTPUT, len(pcm) / 2 / SPEECH_RATE))
    return OUTPUT


if __name__ == "__main__":
    main()
