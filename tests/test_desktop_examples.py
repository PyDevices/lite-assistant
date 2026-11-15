# SPDX-FileCopyrightText: 2026 Brad Barnett
#
# SPDX-License-Identifier: MIT

"""
Tests for the desktop examples (follow_line, drive_by_voice, find_thing and
read_aloud). Each runs end to end against a stub server on localhost that
answers the way the real endpoints do, so no models are needed.
"""
import importlib.util
import json
import os
import struct
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")
sys.path.insert(0, EXAMPLES)


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(EXAMPLES, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stub(BaseHTTPRequestHandler):
    """The endpoints the examples call. ``answers`` maps a path to its JSON,
    and ``seen`` records each request as (path, body)."""

    answers = {}
    seen = []
    speech = struct.pack("<4h", 100, -100, 100, -100) * 600

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        path = self.path.split("?")[0]
        self.seen.append((self.path, body))
        if path == "/v1/audio/speech":
            data, kind = self.speech, "audio/pcm"
        else:
            data, kind = json.dumps(self.answers[path]).encode(), "application/json"
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def server():
    Stub.answers, Stub.seen = {}, []
    httpd = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield Stub, httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def lines_answer(offset, angle, ahead, found=True):
    path = {"offset": offset, "angle": angle, "ahead": ahead, "points": []} if found else None
    return {"object": "lines", "model": "line-finder", "found": found, "path": path, "lines": []}


@pytest.mark.parametrize("path, expected", [
    (None, ("stop", 0.0)),
    ({"offset": 0.0, "angle": 0.0, "ahead": 0.0}, ("straight", 0.0)),
    ({"offset": 0.5, "angle": 0.0, "ahead": 0.0}, ("right", 0.5)),
    ({"offset": -0.5, "angle": 0.0, "ahead": 0.0}, ("left", 0.5)),
    ({"offset": 0.0, "angle": -45.0, "ahead": 0.0}, ("left", 0.5)),
    ({"offset": 1.0, "angle": 90.0, "ahead": 1.0}, ("right", 1.0)),
])
def test_steering(path, expected):
    assert load("follow_line").steering(path) == expected


def test_follow_line_sends_the_photo_and_steers(server, tmp_path, capsys):
    stub, port = server
    stub.answers["/v1/lines"] = lines_answer(0.4, 0.0, 0.0)
    photo = tmp_path / "line.jpg"
    photo.write_bytes(b"\xff\xd8jpeg\xff\xd9")
    example = load("follow_line")
    example.HOST, example.PORT, example.PHOTO = "127.0.0.1", port, str(photo)
    assert example.main() == ("right", 0.4)
    assert stub.seen == [("/v1/lines", photo.read_bytes())]
    assert "steer right 0.40" in capsys.readouterr().out


def test_follow_line_stops_with_no_line(server, tmp_path):
    stub, port = server
    stub.answers["/v1/lines"] = lines_answer(0, 0, 0, found=False)
    photo = tmp_path / "line.jpg"
    photo.write_bytes(b"x")
    example = load("follow_line")
    example.HOST, example.PORT, example.PHOTO = "127.0.0.1", port, str(photo)
    assert example.main() == ("stop", 0.0)


def write_wav(path, rate=16000):
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(struct.pack("<8h", *range(8)))


def test_choose_stops_when_unsure():
    example = load("drive_by_voice")
    assert example.choose("left", 0.9) == "left"
    assert example.choose("left", 0.2) == "stop"


@pytest.mark.parametrize("heard, label, score, expected", [
    ("go left a bit", "left", 0.8, "left"),
    ("mumble", "faster", 0.3, "stop"),
    ("", None, 0, "stop"),
])
def test_drive_by_voice(server, tmp_path, heard, label, score, expected):
    stub, port = server
    stub.answers["/v1/audio/transcriptions"] = {"text": heard}
    stub.answers["/v1/classify"] = {"data": [{"label": label, "score": score}]}
    recording = tmp_path / "command.wav"
    write_wav(recording)
    example = load("drive_by_voice")
    example.HOST, example.PORT, example.RECORDING = "127.0.0.1", port, str(recording)
    assert example.main() == expected
    asked = [path for path, _ in stub.seen]
    # With nothing heard, the classifier isn't asked at all.
    assert ("/v1/classify" in asked) == bool(heard)
    if heard:
        request = json.loads(stub.seen[-1][1])
        assert request["input"] == heard
        assert request["labels"] == example.COMMANDS


def segment_answer(found, box=None, distance=None):
    item = {"prompt": "the cup", "found": found}
    if found:
        item.update(box=box, center=[0, 0])
    if distance is not None:
        item["distance"] = distance
    return {"object": "segmentation", "data": [item]}


@pytest.mark.parametrize("answer, expected", [
    (segment_answer(False), "the cup: not found"),
    (segment_answer(True, [0.7, 0.2, 0.9, 0.6], 0.84), "the cup: right of centre, 0.8 m away"),
    (segment_answer(True, [0.1, 0.2, 0.3, 0.6]), "the cup: left of centre"),
    (segment_answer(True, [0.4, 0.2, 0.6, 0.6], 2.0), "the cup: in the middle, 2.0 m away"),
])
def test_find_thing(server, tmp_path, answer, expected):
    stub, port = server
    stub.answers["/v1/segment"] = answer
    photo = tmp_path / "desk.png"
    photo.write_bytes(b"png")
    example = load("find_thing")
    example.HOST, example.PORT, example.PHOTO, example.THING = "127.0.0.1", port, str(photo), "the cup"
    assert example.main() == expected
    path, body = stub.seen[0]
    assert "prompt=the%20cup" in path and body == b"png"


def test_read_aloud_saves_the_speech(server, tmp_path):
    stub, port = server
    stub.answers["/v1/ocr"] = {"text": "EXIT ONLY\nOpen 9am", "lines": []}
    photo = tmp_path / "page.jpg"
    photo.write_bytes(b"jpeg")
    example = load("read_aloud")
    example.HOST, example.PORT, example.PHOTO = "127.0.0.1", port, str(photo)
    example.OUTPUT = str(tmp_path / "out.wav")
    assert example.main() == example.OUTPUT
    spoken = json.loads(stub.seen[-1][1])
    assert spoken["input"] == "EXIT ONLY\nOpen 9am"
    with wave.open(example.OUTPUT, "rb") as file:
        assert (file.getnchannels(), file.getsampwidth(), file.getframerate()) == (1, 2, 24000)
        assert file.readframes(file.getnframes()) == Stub.speech


def test_read_aloud_says_when_there_is_no_text(server, tmp_path):
    stub, port = server
    stub.answers["/v1/ocr"] = {"text": "  ", "lines": []}
    photo = tmp_path / "page.jpg"
    photo.write_bytes(b"jpeg")
    example = load("read_aloud")
    example.HOST, example.PORT, example.PHOTO = "127.0.0.1", port, str(photo)
    example.OUTPUT = str(tmp_path / "out.wav")
    assert example.main() is None
    assert not os.path.exists(example.OUTPUT)
