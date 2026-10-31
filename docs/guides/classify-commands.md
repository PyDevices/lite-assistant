# Classify commands fast

A board can act on "turn on the kitchen light" without waiting for the LLM: transcribe the command, sort it into the intents it handles with `POST /v1/classify`, and send only what doesn't fit to chat. The labels are whatever the request names, so each board can have its own:

```bash
curl http://localhost:5000/v1/classify -H "Content-Type: application/json" -d '{
  "input": ["Turn on the kitchen light", "Who wrote Hamlet?"],
  "labels": ["lights", "timer", "weather", "music", "something else"]}'
```

```json
{"object": "list", "model": "gliclass-edge-v3.0", "data": [
  {"object": "classification", "index": 0, "label": "lights", "score": 0.9051,
   "labels": [{"label": "lights", "score": 0.9051}, {"label": "weather", "score": 0.0589}, ...]},
  {"object": "classification", "index": 1, "label": "something else", "score": 0.5874,
   "labels": [{"label": "something else", "score": 0.5874}, {"label": "music", "score": 0.3082}, ...]}]}
```

- **`input`** is one text or a list; **`labels`** are up to 25, and the labels, prompt and text together up to 256 tokens.
- **By default the labels compete:** scores add up to 1, and `label` is the winner. A label such as "something else" catches what isn't a command, and a low winning score is a reason to ask the LLM instead.
- **`multi_label: true`** scores each label on its own, from 0 to 1, and `chosen` lists those at or above `threshold` (default 0.5).
- **`prompt`** says what the labels are about, and goes just before the text: `"The room this command is about: "`.
- **The details of a command** come from asking again with other labels: the rooms the board knows, for example. This works when the command names the room ("the kitchen light", 0.77); it doesn't work out a room from a description ("where I watch TV").
