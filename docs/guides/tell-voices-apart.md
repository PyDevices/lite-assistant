# Tell voices apart

`POST /v1/audio/diarization` says who spoke when in a recording, taking it as `/v1/audio/classifications` does. For a 21 s conversation between two of Kokoro's male voices, it found both, with every turn right:

```json
{"object": "audio.diarization", "model": "pyannote-3.1", "duration": 20.97,
 "speakers": {"Adam": 6.65, "Michael": 9.09},
 "segments": [{"start": 0.0, "end": 2.53, "speaker": "Adam"}, {"start": 3.4, "end": 7.08, "speaker": "Michael"},
              {"start": 7.93, "end": 10.32, "speaker": "Adam"}, ...]}
```

A voice no one has enrolled is `speaker 1`, `speaker 2` and so on, in the order they first speak. To enroll one, send a few seconds of it speaking alone with a `name`; more recordings, in other rooms or moods, recognize it better. `GET /v1/audio/speakers` lists the voices, with who enrolled each, and `DELETE /v1/audio/speakers/{name}` forgets one. The voices are the household's: everyone's are recognized for everyone, but only the user who enrolled a voice, or an admin, may add to it or remove it. They're kept in `voices.json` in the [data directory](../reference/where-files-go.md); a voice enrolled before there were users has no owner, so only an admin may change it. The chat page's settings enroll your voice from the microphone.

```bash
curl -F name=Alex -F file=@me.wav http://localhost:5000/v1/audio/speakers
curl -F file=@command.wav http://localhost:5000/v1/audio/speakers/identify
```

`identify` says which enrolled voice one voice is, such as a board's spoken command: `speaker` is null for none, and `distance` how near the nearest was, under 0.70 for a match. A 2 s command matched the right voice at 0.58 and 0.64, and an unenrolled voice was 1.31 from the nearest.

Qwen3-TTS, which speaks in a voice copied from a short sample, was tried and not added: on this PC's CPU, each 80 ms of speech took about 340 ms to make, and copying a voice needs PyTorch and the full original model.
