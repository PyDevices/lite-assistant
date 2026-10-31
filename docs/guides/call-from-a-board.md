# Call the server from a board

[`examples/board_client.py`](../../examples/board_client.py) calls the server from MicroPython with nothing but sockets, so it needs no library on the board. [Results](../how-it-works/board-results.md) shows what an ESP32-P4 board got from each endpoint and how long each took.

Copy `board_client.py` to the board, and a `lite_assistant_key.py` holding `KEY = "..."`, the key `lite-assistant users add <board>` printed. Then:

```python
from board_client import Client
la = Client("192.168.1.10")                       # the server's address
la.say("Hey Jarvis", voice="am_michael")          # spoken on the board's speaker
print(la.transcribe(pcm, 16000)["text"])          # 16-bit mono PCM in memory, sent as a WAV
print(la.post_body("/v1/detect", jpeg, "image/jpeg"))
print(la.post_files("/v1/audio/pitch", [("file", "a.wav", "audio/wav", wav)], instrument="guitar"))
print(la.post_json("/v1/classify", {"input": "lights on", "labels": ["lights", "music"]}))

# With the chat model loaded (not --llm none):
print(la.chat("Why is the sky blue?", max_tokens=80))
for text in la.chat_stream("Why is the sky blue?"):  # the reply as it's written
    print(text, end="")
print(la.respond("Why is the sky blue?"))         # the Responses API
la.voice(pcm, 24000)                              # a spoken question in, the spoken reply played
rt = la.realtime(audio={"input": {"turn_detection": None}})
rt.append(pcm)                                    # 24 kHz only, for the Realtime API
rt.send({"type": "input_audio_buffer.commit"})
la.play(lambda poll_ms: rt.reply())               # its spoken reply; rt.transcript is the text
rt.close()
```

- **`say()`** plays on `board_peripherals.pcm_out()` by default, or on any `out` with `write()`. It waits for the whole reply before playing, since a weak Wi-Fi link delivers speech unevenly; `prebuffer_ms=1000` starts sooner, and plays smoothly only while the link keeps ahead of the speaker. A board's `pcm_out()` drops what it can't hold, so `say()` writes only what `space()` says fits.
- **`speech()`** yields the raw 24 kHz PCM as it arrives, for a caller that does its own playing.
- **`post_files()`** takes a file's data as a list of pieces, such as a WAV header and a recording in memory, so a recording is sent without being copied.
- **`last_ms`** is how long the last request took, from sending to the last byte.
- **`chat()`** and **`respond()`** return the reply's text and keep the server's JSON in `reply`; **`chat_stream()`**, and `respond(..., stream=True)`, yield the text as the model writes it, with `first_ms` for the first piece.
- **`voice()`** sends a recording to `/v1/voice` and plays the spoken reply as `say()` does; `voice_stream()` yields it instead. With `stt_model="whisper"` or `"moonshine"`, `heard` is what the server heard; by default the chat model hears the audio itself.
- **`realtime()`** opens the Realtime API's WebSocket, with no library: `send()` an event, `recv()` the next one, `append()` PCM to the input buffer, and `reply()` to ask for a response and yield its audio.
