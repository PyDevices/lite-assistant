# Tune an instrument

`POST /v1/audio/pitch` is a tuner: it hears the note in a recording of one note, played or sung, taking the audio as `/v1/audio/classifications` does, up to 10 seconds. `instrument` (`guitar`, `bass`, `ukulele`, `violin`, `viola`, `cello`, `mandolin` or `banjo`) adds the nearest string in standard tuning and how far off it is:

```bash
curl -F file=@guitar_a2.wav -F instrument=guitar http://localhost:5000/v1/audio/pitch
```

```json
{"object": "audio.pitch", "model": "crepe-full", "advice": "In tune.",
 "frequency": 110.0, "note": "A2", "cents": 0, "in_tune": true, "confidence": 0.71,
 "string": {"note": "A2", "frequency": 110.0, "cents": 0, "in_tune": true},
 "track": [[0.0, null, 0.322], [0.05, 109.81, 0.711], [0.1, 109.95, 0.805], ...]}
```

- **`frequency`** is the median of the pitches heard, in Hz, and **`note`** the nearest note. **`cents`** says how far off it, a hundredth of a semitone: positive is sharp, negative flat. **`in_tune`** is within 5 cents.
- **`string`** gives the same for the instrument's nearest string, and **`advice`** says which way to turn it.
- **`track`** has each 50 ms: its time, the frequency (null where no note is heard) and how sure CREPE is a note is playing. A board can follow a note as it's tuned.
- **`reference`** is A4's frequency, 440 Hz by default, and **`confidence`** how sure CREPE must be (default 0.5).

On a recording of a guitar being tuned, it heard D2, A2, G3, B3 and E4 in turn, each within 4 cents of in tune, and missed the D3. A violin's open D string was 3 cents sharp. On made tones of known pitch, it was within 2 to 4 cents. Two seconds take about 1.3 seconds. In the chat, the `tune_instrument` tool does the same for the user's latest recording.
