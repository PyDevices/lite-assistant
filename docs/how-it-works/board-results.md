# What boards got from the server

Every endpoint below was called from a microcontroller, a Waveshare ESP32-P4-WIFI6-DEV-KIT on MicroPython 1.30, with [`examples/board_client.py`](../../examples/board_client.py), over Wi-Fi with a user's key. The board's input was its own: speech and tones from its speaker, recorded by a Logitech C920e webcam's microphone on its USB host, and a frame from the same webcam.

## Results

These were measured with the Piper voice that was the default then, which lite-assistant no longer ships, so the speech and spoken-reply figures below describe Piper, not Kokoro. The server's own time for the speech row is re-measured with Kokoro below the first table. Everything else here does not depend on the voice.

The latency is from the board, sending to the last byte of the reply, the best of two or three calls (one for face detection and transcription). The board's Wi-Fi was weak, at -67 dBm, with pings from 100 ms to 2 s, so each request spent 0.5 to 3 s on the link, and sometimes longer. The last column is the same request from the server's own PC, which is what the server took.

| Endpoint | Input from the board | Result | Board | Server's PC |
|---|---|---|---|---|
| `GET /voices` | | 76 voices | 1.8 s | 8 ms |
| `GET /v1/models` | | 29 engines | 97 ms | 3 ms |
| `/v1/audio/speech` | "The quick brown fox jumps over the lazy dog." | 3.5 s of speech, played on the speaker; Whisper and Moonshine heard every word of it from a PC's microphone | first byte 0.3 s, all 0.65 s | 0.18 s |
| `/v1/audio/transcriptions` | That speech from the speaker, recorded by the webcam's mic | Moonshine: "The quick brown fox jumps over the lazy dog" (after the fix below); Whisper: the same | 1.1 s | 0.13 s |
| `/v1/detect` | A 640x480 webcam frame of the bench at night | A cell phone at 1.58 m, a TV; brightened first | 5.7 s | 1.1 s |
| `/v1/detect`, `model=yunet` | The same frame | No faces, rightly | 5.6 s | 0.14 s |
| `/v1/ocr` | The same frame | One misread line from a small screen, score 0.26 | 1.4 s | 0.11 s |
| `/v1/depth` | The same frame | 1.54 m at the center, 4x3 grid | 2.3 s | 1.0 s |
| `/v1/segment` | The same frame, "a person, a chair, a keyboard" | "a person" found (a coat on a chair), the others not | 4.3 s | 1.6 s |
| `/v1/images/classifications` | The same frame | PlantNet's guess, 48%: there's no plant | 0.64 s | 32 ms |
| `/v1/images/quality` | Two frames | technical 5.29 and 5.28, best 0 | 1.6 s | 24 ms |
| `/v1/lines` | The same frame | No line, rightly | 2.0 s | 9 ms |
| `/v1/audio/classifications` | The recorded speech | Speech 0.79, Telephone 0.47 | 2.2 s | 0.16 s |
| `/v1/audio/pitch` | A 440 Hz tone from the speaker, recorded by the webcam's mic | A4, 439.75 Hz, -1 cent, in tune | 1.8 s | |
| `/v1/audio/denoise` | The recorded speech | A 16 kHz WAV; Moonshine then heard "Turn on the kitchen blind" | 5.5 s | 2.4 s |
| `/v1/audio/speakers/identify` | The recorded speech | No enrolled voice matched (`speaker: null`) | 1.7 s | 0.12 s |
| `/v1/classify` | "Please turn on the kitchen light", as Moonshine heard it | `lights`, 0.90 | 0.23 s | 20 ms |
| `/v1/embeddings` | That text and two more | 3 vectors of 768 | 2.0 s | 62 ms |
| `/v1/rerank` | That text and three documents | The kitchen light document first, 0.99 | 2.6 s | 2.1 s |

With Kokoro, the same sentence in `af_heart` is 2.9 s of speech, and the server took a median of 1.5 s (1.1 to 2.0 s, five runs, 2026-10-30) to send all of it, with the first byte after about 5 ms. Other work was running on the server's PC at the time. The board's side of that request was not re-measured: it needs a board.

The first call to an engine that loads on first use takes several seconds more: 8.8 s for `/v1/detect`, 9.1 s for `/v1/embeddings`.

## What broke on the board, and the fixes

- **Speech came out broken up.** A board's `pcm_out()` keeps 80 ms of what it's given and drops the rest, so writing speech as fast as it arrives played about a tenth of it: a PC's microphone heard 1.9 s of a 3.5 s sentence, which Moonshine made into "The outfox was overly". `say()` now writes only what `space()` says fits, and waits for the whole reply first.
- **Moonshine heard nothing in the board's recording.** It returns an empty transcript for a clip that starts with more than about 1.5 s of room noise, as the webcam's recordings did. The server now cuts a clip to the speech Silero VAD hears in it before Moonshine sees it.
- **MicroPython's timed-out read returns `None`, where CPython's raises.** The client handles both.

## The chat endpoints

These were called from a second board, a Waveshare ESP32-P4-WIFI6-Touch-LCD-4B panel on the same MicroPython, with the server running Gemma 4 E2B on its PC's CPU (`serve --llm e2b`). The spoken questions came from the PC's speakers and were recorded by the panel's own microphone, an ES7210 at gain 100, at 24 kHz; the replies played on the panel's speaker, and a USB microphone across the room recorded them, which Whisper then transcribed. Its Wi-Fi was steadier than the DEV-KIT's.

| Endpoint | Input from the board | Result | Board | Server's PC |
|---|---|---|---|---|
| `/v1/chat/completions` | "In one sentence, why is the sky blue?" | A sentence on Rayleigh scattering | 7.4 s | 6.0 s |
| `/v1/chat/completions`, streamed | The same | The same, a piece at a time | first text 6.0 s, all 7.4 s | first text 5.0 s, all 6.6 s |
| `/v1/responses` | The same | The same | 7.4 s | 6.7 s |
| `/v1/responses`, streamed | The same | The same | first text 9.2 s, all 11.1 s (one call) | first text 5.4 s, all 7.1 s |
| `/v1/voice` | "What color is the sky on a clear day?", spoken by the PC, recorded by the panel | 7.5 s of speech on the panel's speaker; the room's microphone heard "The sky on a clear day is typically blue. This is because of how sunlight interacts with the Earth's..." | first byte 7.2 s, all 8.3 s | first byte 6.6 s, all 7.2 s |
| `/v1/voice`, `stt_model=whisper` | The same | Whisper heard "what color is the sky on a clear day"; 6.8 s of speech, heard as "The sky on a clear day is typically blue..." | first byte 14.0 s, all 15.5 s | first byte 9.6 s, all 9.9 s |
| `WS /v1/realtime` | The same 8 s recording, appended and committed | "The sky on a clear day is typically blue. This is because of how sunlight interacts with the Earth's atmosphere.", 7.3 s of speech, heard by the room's microphone word for word | connect 1.1 s, sending the audio 2.7 s, then first audio 9.7 s, all 13.8 s | first audio 6.7 s, all 7.3 s |
| `/v1/audio/translations` | "¿Dónde está la estación de tren?", spoken by the PC, recorded by the panel | "Where is the train station?" | 4.7 s | 1.2 s |

Most of each chat's wait is the model reading its prompt before it writes a word: about 1,700 tokens, mostly the descriptions of the built-in tools it's offered. A reply of a sentence or two then takes another second or so.

How well the model hears matters more than the endpoint. From a first recording of "What is the capital of France?", quieter, Whisper and Moonshine heard "Frank" for "France", and Gemma, hearing the audio itself, answered a question about the caves of France.

## What broke with the chat endpoints, and the fixes

- **The Realtime session failed on its first event,** with "syntax error in JSON", on the board but not on the PC. When the server's first WebSocket frame arrived in the same read as the handshake's headers, the client's `read(n)` returned all of it rather than `n` bytes. It now returns exactly what was asked.
