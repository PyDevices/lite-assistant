# SPDX-FileCopyrightText: 2026 Brad Barnett
#
# SPDX-License-Identifier: MIT

"""Steer along a line on the floor from a photo of it.

Sends a photo from a camera looking ahead to the server's /v1/lines, then
turns where the line is into a steering command a robot car can act on:
left, right or straight, with how hard to turn. Run it on each frame a car
streams and the result is the drive loop's steering input.

Needs: a lite-assistant server with its line engine, reachable at HOST, and
a JPEG of tape or paint on a floor, taken from low down, looking ahead.

Run it:

    python examples/follow_line.py

It prints what the server found and then the command, such as
"steer right 0.52".

Tested on: Python 3.12 on Linux, against a lite-assistant server with the
line-finder engine.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board_client import Client  # noqa: E402

# Change these to match your setup.
HOST = "localhost"  # the machine running lite-assistant
PORT = 5000
PHOTO = "line.jpg"  # a JPEG looking along the line
DEAD_BAND = 0.08  # offsets smaller than this count as on the line (0 to 1)


def steering(path):
    """The command for a path from /v1/lines: ``(direction, amount)``.

    The line is judged by its nearest point, how far it sits from the
    middle of the photo, plus how far it leans, so a line that is centred
    but curving away still turns the car early. ``amount`` runs from 0 to 1.
    """
    if not path:
        return "stop", 0.0
    # Offset is -1 at the left edge and 1 at the right; angle is degrees
    # right of straight up the photo. Blend them into one number.
    want = path["offset"] + path["angle"] / 90 + path["ahead"] * 0.25
    want = max(-1.0, min(1.0, want))
    if abs(want) < DEAD_BAND:
        return "straight", 0.0
    return ("right" if want > 0 else "left"), round(abs(want), 2)


def main():
    # Read the photo the camera took.
    with open(PHOTO, "rb") as file:
        jpeg = file.read()

    # Ask the server where the line is. With no line in sight, found is false.
    found = Client(HOST, PORT).post_body("/v1/lines", jpeg, "image/jpeg")
    print("line found:", found["found"], "by", found["model"])

    # Turn that into a command for the motors.
    direction, amount = steering(found["path"])
    if direction in ("straight", "stop"):
        print(direction)
    else:
        print("steer %s %.2f" % (direction, amount))
    return direction, amount


if __name__ == "__main__":
    main()
