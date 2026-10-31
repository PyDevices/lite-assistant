# Classify sounds

`POST /v1/audio/classifications` takes audio as a multipart `file`, as transcriptions does, or as the request body itself (`Content-Type: audio/wav` or another audio type), in any format libsndfile reads, such as WAV, FLAC, MP3 or OGG:

```bash
curl http://localhost:5000/v1/audio/classifications -F file="@recording.wav"
```

For 25 seconds of speech, then quiet, then a beeping alarm, it returns:

```json
{"object": "audio.classification", "model": "panns-cnn14", "duration": 24.71,
 "sounds": [{"label": "Speech", "score": 0.825}, {"label": "Smoke detector, smoke alarm", "score": 0.518},
            {"label": "Buzzer", "score": 0.217}, {"label": "Fire alarm", "score": 0.216}, {"label": "Beep, bleep", "score": 0.209}],
 "segments": [
  {"start": 0.0, "end": 10.0, "sounds": [{"label": "Speech", "score": 0.825}, {"label": "Speech synthesizer", "score": 0.145}]},
  {"start": 10.0, "end": 20.0, "sounds": [{"label": "Smoke detector, smoke alarm", "score": 0.518}, ...]},
  {"start": 14.71, "end": 24.71, "sounds": [{"label": "Smoke detector, smoke alarm", "score": 0.447}, ...]}]}
```

- **`sounds`** are AudioSet's labels, with how sure the model is from 0 to 1, best first. Several can be high at once, such as speech over music: they aren't shares of one whole.
- **`segments`** are the windows the model heard, 10 seconds each. A longer recording's last window is its last 10 seconds, so it may overlap the one before. A shorter one is repeated to fill the window, which the model hears far better than silence after it. `sounds` takes each label's best score in any window.
- **`threshold`** (0 to 1, default 0.1) is how sure the model must be, and **`top`** (1 to 50, default 5) the most sounds listed, overall and in each segment. They go in form fields or the query string.
