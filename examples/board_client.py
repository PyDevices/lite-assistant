"""
Board client for lite-assistant
A sockets-only client for every kind of request the server takes: GETs, JSON
POSTs, a photo or a recording sent as the body, multipart uploads, speech
streamed back as raw PCM, chat (streamed or not), a whole spoken turn, and the
Realtime API's WebSocket. It runs unchanged on MicroPython (an ESP32-P4 board
was the first) and on CPython.

From another machine the server wants a user's key. Make one for the board on
the server (`lite-assistant users add kitchen-board`) and put it on the board
in `lite_assistant_key.py`, as `KEY = "..."`. `Client()` imports it from there
when you don't pass `key`; from the server's own machine no key is needed.

    from board_client import Client
    la = Client("192.168.1.10")
    la.say("Hey Jarvis", voice="am_michael")        # speaks on the board's speaker
    print(la.post_json("/v1/classify", {"input": "lights on", "labels": ["lights", "music"]}))
    print(la.post_body("/v1/detect", jpeg, "image/jpeg"))
    print(la.transcribe(pcm, 16000)["text"])
    print(la.chat("Why is the sky blue?"))          # needs the server's chat model
    la.voice(pcm, 16000)                            # a spoken question in, the reply spoken
"""
import json
import socket

try:
    from time import ticks_diff, ticks_ms
except ImportError:  # CPython
    from time import monotonic

    def ticks_ms():
        return int(monotonic() * 1000)

    def ticks_diff(a, b):
        return a - b


BOUNDARY = "----lite-assistant-board-7f3a9c"
SPEECH_RATE = 24000  # /v1/audio/speech streams 16-bit mono PCM at 24 kHz


class ServerError(Exception):
    """The server answered with an error: ``status`` and its ``message``."""

    def __init__(self, status, message):
        super().__init__("HTTP %d: %s" % (status, message))
        self.status = status
        self.message = message


def wav_header(data_bytes, rate, channels=1, bits=16):
    """The 44-byte header of a PCM WAV holding ``data_bytes`` of samples."""
    import struct

    align = channels * bits // 8
    return (b"RIFF" + struct.pack("<I", 36 + data_bytes) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, channels, rate, rate * align, align, bits)
            + b"data" + struct.pack("<I", data_bytes))


def _query(params):
    if not params:
        return ""
    parts = []
    for k, v in params.items():
        if v is None:
            continue
        if v is True or v is False:
            v = "true" if v else "false"
        v = str(v).replace("%", "%25").replace(" ", "%20").replace("&", "%26").replace(",", "%2C")
        parts.append("%s=%s" % (k, v))
    return "?" + "&".join(parts) if parts else ""


def _messages(messages):
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return messages


def _unquote(text):
    """Undoes the percent-encoding of a header, such as X-Transcript."""
    out = bytearray()
    raw = text.encode()
    i = 0
    while i < len(raw):
        if raw[i] == 37 and i + 3 <= len(raw):  # "%"
            try:
                out.append(int(raw[i + 1:i + 3], 16))
                i += 3
                continue
            except ValueError:
                pass
        out.append(raw[i])
        i += 1
    return out.decode()


def _send(sock, data):
    """Send all of ``data``: a board's socket.send() may take only part."""
    view = memoryview(data)
    while view:
        n = sock.send(view)
        view = view[n:]


def _timed_out(e):
    # CPython raises socket.timeout; MicroPython OSError ETIMEDOUT or EAGAIN.
    return isinstance(e, getattr(socket, "timeout", ())) or (e.args and e.args[0] in (11, 110, 116))


class _Reply:
    """A reply read straight from the socket, so a timeout can't lose bytes
    the way it can inside CPython's buffered makefile()."""

    def __init__(self, sock):
        self.sock = sock
        self.extra = b""
        self._into = getattr(sock, "readinto", None) or sock.recv_into

    def head(self):
        data = b""
        while b"\r\n\r\n" not in data:
            piece = self.sock.recv(512)
            if not piece:
                if not data:
                    raise OSError("the server closed the connection without answering")
                break
            data += piece
        head, _, self.extra = data.partition(b"\r\n\r\n")
        return head

    def readinto(self, view):
        if self.extra:
            n = min(len(view), len(self.extra))
            view[:n] = self.extra[:n]
            self.extra = self.extra[n:]
            return n
        return self._into(view)

    def readline(self):
        """One line, with its newline, or b"" at the end of the reply."""
        while b"\n" not in self.extra:
            piece = self.sock.recv(512)
            if not piece:
                line, self.extra = self.extra, b""
                return line
            self.extra += piece
        line, _, self.extra = self.extra.partition(b"\n")
        return line + b"\n"

    def read(self, n=None):
        """``n`` bytes, or everything to the end when ``n`` is None."""
        if n is not None and len(self.extra) >= n:
            data, self.extra = self.extra[:n], self.extra[n:]
            return data
        parts = [self.extra]
        have = len(self.extra)
        self.extra = b""
        while n is None or have < n:
            piece = self.sock.recv(4096 if n is None else min(4096, n - have))
            if not piece:
                break
            parts.append(piece)
            have += len(piece)
        return b"".join(parts)


def _sleep_ms(ms):
    import time

    if hasattr(time, "sleep_ms"):
        time.sleep_ms(ms)
    else:
        time.sleep(ms / 1000)


def _feed(out, queue, space):
    """Writes queued PCM to ``out``: all of it, or only what ``space()``
    says fits. Returns the bytes written."""
    wrote = 0
    while queue:
        piece = queue[0]
        n = len(piece)
        if space is not None:
            n = min(n, space())
            if n <= 0:
                break
        out.write(piece[:n] if n < len(piece) else piece)
        wrote += n
        if n < len(piece):
            queue[0] = memoryview(piece)[n:]
        else:
            queue.pop(0)
    return wrote


class Client:
    """One lite-assistant server. Each request is its own HTTP/1.0 connection,
    so nothing is held open between them."""

    def __init__(self, host, port=5000, key=None, timeout=60):
        if key is None:
            try:
                from lite_assistant_key import KEY as key
            except ImportError:
                key = None  # fine from the server's own machine
        self.host = host
        self.port = port
        self.timeout = timeout
        self._auth = "Authorization: Bearer %s\r\n" % key if key else ""
        self._addr = None
        self.last_ms = 0  # how long the last request took, sent to last byte

    # -- the HTTP underneath -------------------------------------------------

    def _open(self, method, path, content_type=None, length=0):
        if self._addr is None:
            self._addr = socket.getaddrinfo(self.host, self.port)[0][-1]
        sock = socket.socket()
        sock.settimeout(self.timeout)
        try:
            sock.connect(self._addr)
        except OSError as e:
            sock.close()
            raise OSError("can't reach lite-assistant at %s:%d: %s" % (self.host, self.port, e))
        head = "%s %s HTTP/1.0\r\nHost: %s\r\n%s" % (method, path, self.host, self._auth)
        if content_type:
            head += "Content-Type: %s\r\nContent-Length: %d\r\n" % (content_type, length)
        _send(sock, (head + "\r\n").encode())
        return sock

    def _response(self, sock):
        """Reads the status line and headers; returns (status, headers, reply)."""
        reply = _Reply(sock)
        head = reply.head()
        lines = head.split(b"\r\n")
        try:
            status = int(lines[0].split(None, 2)[1])
        except (IndexError, ValueError):
            raise OSError("not an HTTP reply: %r" % lines[0][:40])
        headers = {}
        for line in lines[1:]:
            k, _, v = line.decode().partition(":")
            headers[k.strip().lower()] = v.strip()
        return status, headers, reply

    def _finish(self, sock, start):
        """Reads the whole reply and returns it as JSON, or as bytes when it
        isn't JSON (a WAV, a PNG)."""
        try:
            status, headers, reply = self._response(sock)
            n = headers.get("content-length")
            body = reply.read(int(n) if n is not None else None)
        finally:
            sock.close()
        self.last_ms = ticks_diff(ticks_ms(), start)
        is_json = "json" in headers.get("content-type", "")
        if status >= 400:
            message = body.decode()
            if is_json:
                try:
                    err = json.loads(body)
                    err = err.get("error", err.get("detail", err))
                    message = err.get("message", err) if isinstance(err, dict) else err
                except ValueError:
                    pass
            raise ServerError(status, message)
        return json.loads(body) if is_json else body

    # -- requests -----------------------------------------------------------

    def get(self, path, **params):
        """GET a path, such as ``/v1/models`` or ``/voices``."""
        start = ticks_ms()
        return self._finish(self._open("GET", path + _query(params)), start)

    def post_json(self, path, obj, **params):
        """POST a JSON body, such as ``/v1/classify`` or ``/v1/rerank``."""
        start = ticks_ms()
        body = json.dumps(obj).encode()
        sock = self._open("POST", path + _query(params), "application/json", len(body))
        try:
            _send(sock, body)
        except Exception:
            sock.close()
            raise
        return self._finish(sock, start)

    def post_body(self, path, data, content_type, **params):
        """POST bytes as the body itself: a camera's JPEG to ``/v1/detect``,
        or a WAV to ``/v1/audio/classifications``. Options go in the query."""
        start = ticks_ms()
        sock = self._open("POST", path + _query(params), content_type, len(data))
        try:
            _send(sock, data)
        except Exception:
            sock.close()
            raise
        return self._finish(sock, start)

    def post_files(self, path, files, **fields):
        """POST multipart/form-data. ``files`` is a list of
        ``(field, filename, content_type, data)``, where ``data`` is bytes or
        a list of pieces (a WAV header and the samples, say) so a recording in
        memory is never copied; ``fields`` are plain form fields."""
        start = ticks_ms()
        return self._finish(self._send_files(path, files, fields), start)

    def _send_files(self, path, files, fields):
        """Sends a multipart request and returns its socket, to be read."""
        parts = []
        for name, value in fields.items():
            if value is None:
                continue
            if value is True or value is False:
                value = "true" if value else "false"
            parts.append(('--%s\r\nContent-Disposition: form-data; name="%s"\r\n\r\n%s\r\n'
                          % (BOUNDARY, name, value)).encode())
        for name, filename, ctype, data in files:
            parts.append(('--%s\r\nContent-Disposition: form-data; name="%s"; filename="%s"\r\n'
                          'Content-Type: %s\r\n\r\n' % (BOUNDARY, name, filename, ctype)).encode())
            parts.extend(data if isinstance(data, (list, tuple)) else (data,))
            parts.append(b"\r\n")
        parts.append(("--%s--\r\n" % BOUNDARY).encode())
        length = sum(len(p) for p in parts)
        sock = self._open("POST", path, "multipart/form-data; boundary=" + BOUNDARY, length)
        try:
            for p in parts:
                _send(sock, p)
        except Exception:
            sock.close()
            raise
        return sock

    # -- chat ---------------------------------------------------------------

    def chat(self, messages, **opts):
        """Chat completions: ``messages`` is a list of ``{"role", "content"}``,
        or a string for one user message. Returns the reply's text, and keeps
        the server's whole JSON in ``self.reply``. ``opts`` go in the request,
        such as ``max_tokens`` or ``model``."""
        self.reply = self.post_json("/v1/chat/completions", dict(opts, messages=_messages(messages)))
        return self.reply["choices"][0]["message"]["content"]

    def chat_stream(self, messages, **opts):
        """Chat completions, streamed: yields the reply's text a piece at a
        time as the model writes it. ``self.first_ms`` is how long the first
        piece took."""
        for event in self._events("/v1/chat/completions", dict(opts, messages=_messages(messages), stream=True)):
            for choice in event.get("choices", ()):
                text = choice.get("delta", {}).get("content")
                if text:
                    yield text

    def respond(self, input, stream=False, **opts):
        """OpenAI's Responses API: ``input`` is a string or a list of items.
        Returns the reply's text, and keeps the whole response in
        ``self.reply``; with ``stream=True``, yields the text as it comes."""
        req = dict(opts, input=input)
        if stream:
            return self._response_stream(req)
        self.reply = self.post_json("/v1/responses", req)
        return "".join(part.get("text", "") for item in self.reply.get("output", ())
                       if item.get("type") == "message" for part in item.get("content", ()))

    def _response_stream(self, req):
        req["stream"] = True
        for event in self._events("/v1/responses", req):
            if event.get("type") == "response.output_text.delta":
                yield event.get("delta", "")
            elif event.get("type") == "response.completed":
                self.reply = event.get("response")

    def _events(self, path, req):
        """POSTs ``req`` and yields each server-sent event's JSON."""
        start = ticks_ms()
        self.first_ms = None
        body = json.dumps(req).encode()
        sock = self._open("POST", path, "application/json", len(body))
        try:
            _send(sock, body)
            status, headers, reply = self._response(sock)
            if status >= 400:
                raise ServerError(status, reply.read().decode())
            while True:
                line = reply.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                if self.first_ms is None:
                    self.first_ms = ticks_diff(ticks_ms(), start)
                yield json.loads(data)
        finally:
            sock.close()
            self.last_ms = ticks_diff(ticks_ms(), start)

    # -- speech -------------------------------------------------------------

    def speech(self, text, voice=None, model=None, chunk=2048, poll_ms=None):
        """Yields the spoken ``text`` as raw 16-bit mono PCM at 24 kHz, as it
        streams in, in pieces of whole samples. ``self.first_ms`` is how long
        the first piece took. With ``poll_ms``, a read that waits that long
        yields an empty piece, so the caller can do other work meanwhile."""
        start = ticks_ms()
        req = {"input": text, "response_format": "pcm"}
        if voice:
            req["voice"] = voice
        if model:
            req["model"] = model
        body = json.dumps(req).encode()
        sock = self._open("POST", "/v1/audio/speech", "application/json", len(body))
        try:
            _send(sock, body)
        except Exception:
            sock.close()
            raise
        return self._pcm(sock, start, chunk, poll_ms)

    def _pcm(self, sock, start, chunk=2048, poll_ms=None):
        """Reads a reply of raw PCM from ``sock`` and yields it in pieces of
        whole samples, as speech() does."""
        try:
            status, headers, reply = self._response(sock)
            if status >= 400:
                raise ServerError(status, reply.read().decode())
            self.headers = headers
            if poll_ms:
                sock.settimeout(poll_ms / 1000)
            buf = bytearray(chunk)
            view = memoryview(buf)
            have = 0  # a byte left over from an odd read
            self.first_ms = None
            while True:
                try:
                    n = reply.readinto(view[have:])
                except OSError as e:
                    if not (poll_ms and _timed_out(e)):
                        raise
                    yield view[:0]
                    continue
                if n is None:  # MicroPython's timed-out read
                    if not poll_ms:
                        raise OSError("timed out reading the speech")
                    yield view[:0]
                    continue
                if not n:
                    break
                if self.first_ms is None:
                    self.first_ms = ticks_diff(ticks_ms(), start)
                have += n
                whole = have & ~1
                if whole:
                    yield view[:whole]
                    if have > whole:
                        buf[0] = buf[whole]
                    have -= whole
        finally:
            sock.close()
            self.last_ms = ticks_diff(ticks_ms(), start)

    def say(self, text, voice=None, model=None, out=None, prebuffer_ms=None):
        """Speaks ``text`` on the board's speaker and returns the bytes played.

        ``out`` is anything with ``write()``; by default the board's own
        ``board_peripherals.pcm_out()`` at 24 kHz mono, opened and closed
        here. By default the whole reply arrives before it plays, so a slow
        or uneven Wi-Fi link can't break it up; ``self.sound_ms`` says how
        long that took. ``prebuffer_ms`` starts sooner, once that much has
        arrived, and plays the rest as it comes, which is smooth only while
        the link delivers faster than the speaker plays. An ``out`` with
        ``space()``, as a board's is, is fed only what it can take, so
        nothing is dropped: a board's ``pcm_out()`` drops what it can't hold."""
        return self.play(lambda poll_ms: self.speech(text, voice=voice, model=model, chunk=4096, poll_ms=poll_ms),
                         out, prebuffer_ms)

    def voice(self, pcm, rate, out=None, prebuffer_ms=None, **fields):
        """A whole spoken turn: sends 16-bit mono PCM of a question, recorded
        at ``rate``, to ``/v1/voice`` and plays the spoken reply as say()
        does. Returns the bytes played. ``fields`` are the endpoint's form
        fields, such as ``voice``, ``max_tokens`` or ``stt_model``; with an
        ``stt_model`` such as ``"moonshine"``, ``self.heard`` is what it
        heard."""
        return self.play(lambda poll_ms: self.voice_stream(pcm, rate, poll_ms=poll_ms, **fields), out, prebuffer_ms)

    def voice_stream(self, pcm, rate, chunk=4096, poll_ms=None, **fields):
        """The spoken reply to ``pcm`` as raw 24 kHz PCM, yielded as it
        streams in, as speech() yields it."""
        start = ticks_ms()
        self.heard = None
        sock = self._send_files("/v1/voice", [("file", "turn.wav", "audio/wav", [wav_header(len(pcm), rate), pcm])],
                                fields)
        for piece in self._pcm(sock, start, chunk, poll_ms):
            if self.heard is None:
                self.heard = _unquote(self.headers.get("x-transcript", ""))
            yield piece

    def play(self, source, out=None, prebuffer_ms=None):
        """Plays the PCM that ``source(poll_ms)`` yields, as say() does."""
        self._said = ticks_ms()
        own = out is None
        if own:
            import board_peripherals

            out = board_peripherals.pcm_out()
        space = getattr(out, "space", None)
        need = None if prebuffer_ms is None else prebuffer_ms * SPEECH_RATE * 2 // 1000
        queue = []
        played = 0
        started = False
        try:
            for pcm in source(10 if space and need is not None else None):
                if pcm:
                    queue.append(bytes(pcm))
                if not started and need is not None and sum(len(q) for q in queue) >= need:
                    started = True
                    self.sound_ms = ticks_diff(ticks_ms(), self._said)
                if started:
                    played += _feed(out, queue, space)
            if not started:
                self.sound_ms = ticks_diff(ticks_ms(), self._said)
            while queue:
                n = _feed(out, queue, space)
                played += n
                if not n:
                    _sleep_ms(5)
            if hasattr(out, "drain"):
                out.drain()
            if space is not None and hasattr(out, "queued_size"):
                # Wait for the speaker to play what it holds; space() also
                # flushes the board's stash into I2S. Bounded, in case the
                # device's count never reaches zero.
                for _ in range(300):
                    space()
                    if out.queued_size() <= 0:
                        break
                    _sleep_ms(10)
        finally:
            if own:
                out.close()
        return played

    def realtime(self, model=None, **session):
        """Opens a ``/v1/realtime`` session, OpenAI's Realtime API over a
        WebSocket, and returns it as a ``Realtime``. ``session`` is sent as a
        ``session.update``, such as ``audio={"input": {"turn_detection": None}}``
        to commit each turn yourself."""
        import binascii
        import os

        key = binascii.b2a_base64(os.urandom(16)).strip().decode()
        if self._addr is None:
            self._addr = socket.getaddrinfo(self.host, self.port)[0][-1]
        sock = socket.socket()
        sock.settimeout(self.timeout)
        try:
            sock.connect(self._addr)
            path = "/v1/realtime" + _query({"model": model})
            _send(sock, ("GET %s HTTP/1.1\r\nHost: %s\r\n%sUpgrade: websocket\r\nConnection: Upgrade\r\n"
                         "Sec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\n\r\n"
                         % (path, self.host, self._auth, key)).encode())
            status, headers, reply = self._response(sock)
            if status != 101:
                raise ServerError(status, reply.read().decode())
        except Exception:
            sock.close()
            raise
        rt = Realtime(sock, reply)
        rt.session = rt.recv().get("session")  # session.created
        if session:
            rt.send({"type": "session.update", "session": session})
            while rt.recv().get("type") != "session.updated":
                pass
        return rt

    def transcribe(self, pcm, rate, model=None, language=None):
        """Transcribes 16-bit mono PCM (bytes, bytearray or memoryview) by
        uploading it as a WAV; returns the server's JSON, with ``text``."""
        wav = [wav_header(len(pcm), rate), pcm]
        return self.post_files("/v1/audio/transcriptions", [("file", "speech.wav", "audio/wav", wav)],
                               model=model, language=language)


class Realtime:
    """A Realtime API session over a WebSocket, from ``Client.realtime()``.
    Its audio is 16-bit mono PCM at 24 kHz both ways."""

    def __init__(self, sock, reply):
        self.sock = sock
        self._reply = reply
        self.transcript = ""

    def send(self, event):
        """Sends one client event, a dict with its ``type``."""
        data = json.dumps(event).encode()
        n = len(data)
        if n < 126:
            head = bytes((0x81, 0x80 | n))
        elif n < 65536:
            head = bytes((0x81, 0xFE, n >> 8, n & 0xFF))
        else:
            head = bytes((0x81, 0xFF, 0, 0, 0, 0, n >> 24, (n >> 16) & 0xFF, (n >> 8) & 0xFF, n & 0xFF))
        # A client must mask what it sends; a mask of zeros leaves the data
        # as it is, which spares a board XORing every byte.
        _send(self.sock, head + b"\0\0\0\0")
        _send(self.sock, data)

    def recv(self):
        """The next server event, as a dict; None when the server closed."""
        message = b""
        while True:
            head = self._reply.read(2)
            if len(head) < 2:
                return None
            op = head[0] & 0x0F
            n = head[1] & 0x7F
            if n == 126:
                b = self._reply.read(2)
                n = (b[0] << 8) | b[1]
            elif n == 127:
                n = 0
                for c in self._reply.read(8):
                    n = (n << 8) | c
            mask = self._reply.read(4) if head[1] & 0x80 else None
            data = self._reply.read(n)
            if mask:
                data = bytes(c ^ mask[i & 3] for i, c in enumerate(data))
            if op == 8:  # close
                return None
            if op == 9:  # ping: answer with a pong
                _send(self.sock, bytes((0x8A, 0x80 | len(data))) + b"\0\0\0\0" + data)
                continue
            if op == 10:
                continue
            message += data
            if head[0] & 0x80:
                return json.loads(message)

    def append(self, pcm, piece=24000):
        """Sends 24 kHz PCM to the input buffer, ``piece`` bytes an event."""
        import binascii

        view = memoryview(pcm)
        for i in range(0, len(view), piece):
            audio = binascii.b2a_base64(view[i:i + piece]).strip().decode()
            self.send({"type": "input_audio_buffer.append", "audio": audio})

    def reply(self, poll_ms=None, create=True):
        """Asks for a response (unless ``create`` is False, as when the
        server's turn detection asks for it) and yields its audio as it
        streams in, until it's done. ``self.transcript`` is what it said,
        ``self.done`` the ``response.done`` event."""
        import binascii

        if create:
            self.send({"type": "response.create"})
        self.transcript = ""
        self.done = None
        while True:
            event = self.recv()
            if event is None:
                return
            kind = event.get("type")
            if kind == "response.output_audio.delta":
                yield binascii.a2b_base64(event["delta"])
            elif kind == "response.output_audio_transcript.delta":
                self.transcript += event.get("delta", "")
            elif kind == "error":
                raise ServerError(400, event["error"].get("message"))
            elif kind == "response.done":
                self.done = event
                return

    def close(self):
        try:
            _send(self.sock, b"\x88\x80\0\0\0\0")
        except OSError:
            pass
        self.sock.close()


if __name__ == "__main__":
    import sys

    la = Client(sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1")
    print("models:", [m["id"] for m in la.get("/v1/models")["data"]], "in %d ms" % la.last_ms)
    total = sum(len(p) for p in la.speech("Hello from a board.", voice="af_heart"))
    print("speech: %d bytes, first after %d ms, all after %d ms" % (total, la.first_ms, la.last_ms))
    print("classify:", la.post_json("/v1/classify", {"input": "turn on the light", "labels": ["lights", "music"]})["data"][0]["label"])
