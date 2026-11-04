# SPDX-FileCopyrightText: 2026 Brad Barnett
#
# SPDX-License-Identifier: MIT

"""
Point a board or desktop client at any server with one line.

Each entry is a base URL and the model to use for each kind of request, so a
program takes the same calls to this server, to DeepInfra or to OpenRouter. A
provider also wants its key, which you keep in a file of your own.

Needs: board_client.py beside it, and Python, MicroPython or CircuitPython
with sockets. https needs a firmware with the ssl module.

Run it:

    python examples/providers.py

Tested on: CPython 3.12 and MicroPython's Unix port.
"""
from board_client import Client

LOCAL = {
    "base_url": "http://localhost:5000/v1",
    "models": {"chat": "litertlm", "speech": "kokoro", "voice": "af_heart", "transcribe": "moonshine",
               "detect": "yolox", "read": "ppocr"},
}

DEEPINFRA = {
    "base_url": "https://api.deepinfra.com/v1/openai",
    "models": {"chat": "google/gemma-4-26B-A4B-it", "speech": "hexgrad/Kokoro-82M", "voice": "af_heart",
               "transcribe": "openai/whisper-large-v3",
               "detect": "google/gemma-4-26B-A4B-it", "read": "google/gemma-4-26B-A4B-it"},
}

OPENROUTER = {
    "base_url": "https://openrouter.ai/api/v1",
    "models": {"chat": "google/gemma-4-26b-a4b-it", "speech": "mistralai/voxtral-mini-tts-3-0",
               "transcribe": "mistralai/voxtral-mini-transcribe-3-0",
               "detect": "google/gemma-4-26b-a4b-it", "read": "google/gemma-4-26b-a4b-it"},
}


if __name__ == "__main__":
    # Change LOCAL to a provider, and pass its key=, and nothing else changes
    client = Client(**LOCAL)
    print(client.chat("Say hello in three words.", max_tokens=20))
