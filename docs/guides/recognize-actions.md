# Recognize actions in video

`POST /v1/video/actions` says what people are doing across a video's frames, from Kinetics-600's 600 actions: exercises such as jumping jacks and push-ups, waving, clapping, dancing, cooking, playing instruments and sports. Send a video file (any format ffmpeg reads, up to 60 s; it needs `ffmpeg` installed), or the frames as photos, in order, as multipart `file` fields:

```bash
curl -F file=@jumping_jacks.ogg http://localhost:5000/v1/video/actions
```

```json
{"object": "video.actions", "model": "movinet-a0-stream", "frames": 40,
 "actions": [{"label": "jumping jacks", "score": 0.3221}, {"label": "making balloon shapes", "score": 0.2956},
             {"label": "tiptoeing", "score": 0.0451}, ...]}
```

A board can send a photo at a time as it takes them, as the request body, with a `session` of its choosing: the model remembers the frames sent before under it, and each reply says what it makes of them all so far. The model expects about 5 frames a second. A session is forgotten after 2 minutes unused.

```bash
curl --data-binary @frame.jpg -H "Content-Type: image/jpeg" "http://localhost:5000/v1/video/actions?session=car1"
```

- **`score`s** are shares of one whole, from 0 to 1, best first; **`top`** (1 to 20, default 5) says how many to give.
- **`fps`** sets the rate a video is sampled at; leave it at 5, the rate the model learned at.
- **`frames`** is how many frames it has seen, in the session if there is one.

Sent one at a time with a session, a 3D figure doing jumping jacks was "jumping jacks" from the 20th frame, at 62 to 77%, in about 33 ms a request. A video of girls playing a hand-clapping game was 77% "playing hand clapping games". It knows actions, not gestures meant for it, so it's better at noticing someone exercising or dancing than at taking commands.
