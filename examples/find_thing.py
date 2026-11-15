# SPDX-FileCopyrightText: 2026 Brad Barnett
#
# SPDX-License-Identifier: MIT

"""Find a thing in a photo by name and say where it is and how far.

Sends a photo and a name, such as "the red cup", to the server's /v1/segment.
The server finds the thing by its name and, when it has a depth engine,
measures how far away it is. The example turns that into a sentence a robot
could say or act on: which side the thing is on and its distance in metres.

Needs: a lite-assistant server with the segmentation engine (and the depth
engine, for distances), reachable at HOST, and a JPEG or PNG to search.

Run it:

    python examples/find_thing.py

It prints one line, such as "the red cup: right of centre, 0.8 m away".

Tested on: Python 3.12 on Linux, against a lite-assistant server with the
clipseg and moge engines.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board_client import Client  # noqa: E402

# Change these to match your setup.
HOST = "localhost"  # the machine running lite-assistant
PORT = 5000
PHOTO = "desk.jpg"  # the photo to search
THING = "the red cup"  # what to look for, in a few words
THRESHOLD = 0.4  # how sure the engine must be, from 0 to 1


def describe(thing, result):
    """One sentence for a ``/v1/segment`` answer's entry about ``thing``."""
    if not result["found"]:
        return "%s: not found" % thing
    # The box's sides are fractions of the photo, so 0.5 is the middle.
    middle = (result["box"][0] + result["box"][2]) / 2
    side = "left of" if middle < 0.4 else "right of" if middle > 0.6 else "in the"
    sentence = "%s: %s %s" % (thing, side, "centre" if side != "in the" else "middle")
    if "distance" in result and result["distance"] is not None:
        sentence += ", %.1f m away" % result["distance"]
    return sentence


def main():
    # Read the photo to search.
    with open(PHOTO, "rb") as file:
        photo = file.read()

    # Ask the server to find the thing by name.
    content_type = "image/png" if PHOTO.endswith(".png") else "image/jpeg"
    answer = Client(HOST, PORT).post_body("/v1/segment", photo, content_type, prompt=THING, threshold=THRESHOLD)

    # Say where it is.
    sentence = describe(THING, answer["data"][0])
    print(sentence)
    return sentence


if __name__ == "__main__":
    main()
