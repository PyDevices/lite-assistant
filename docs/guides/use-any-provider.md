# Use this server or a provider, with the same client

Write your client against three settings, the base URL, the key and each task's model, and it can talk to this server, to DeepInfra or to OpenRouter without any other change.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5000/v1", api_key="your-key")
# client = OpenAI(base_url="https://api.deepinfra.com/v1/openai", api_key=deepinfra_key)
# client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)
```

On a board, [`examples/board_client.py`](../../examples/board_client.py) takes the same settings, and [`examples/providers.py`](../../examples/providers.py) holds them for all three:

```python
from board_client import Client
from providers import DEEPINFRA

client = Client(key=KEY, **DEEPINFRA)
print(client.chat("Why is the sky blue?"))
for text in client.chat_stream("Why is the sky blue?"):
    print(text, end="")
pcm = b"".join(bytes(piece) for piece in client.speech("Hello!"))
print(client.transcribe(recording, 16000)["text"])
```

A model id stays yours to choose: `hexgrad/Kokoro-82M` at DeepInfra, `kokoro` here. This server also takes a provider's id, so one settings file can serve both: `openai/whisper-large-v3` gets its Whisper engine, and an id it doesn't know gets its default engine for that task.

## Find objects and read text

Ask for objects the way you'd ask any vision model: a chat completion with a photo and the `object_detection` schema, or the `ocr_extraction` schema for text. The [schemas](../reference/provider-api.md#the-vision-schemas) are the same everywhere, so the reply is too.

```python
found = client.detect_objects(jpeg)
for thing in found["objects"]:
    print(thing["label"], thing["box"]["left"], thing["distance"])
print(client.read_text(jpeg)["text"])
```

The `detect` and `read` models say who answers. Here, `yolox` or `yunet` and `ppocr` are engines: they answer in a second or two where Gemma E2B on a PC's CPU took 25 to 65 seconds, with no chat model loaded, and at no cost per photo. Name a vision model instead, here or at a provider, and it writes the same JSON.

The two don't know the same things. An engine's `score` is measured, and its `distance` comes from the depth engine, in meters. A chat model's `score` is its own guess, its boxes can be loose, and its `distance` should be `null`.

## What's different on purpose

- **Speech:** always send `response_format`. Asked for nothing, this server sends PCM and most providers send something else. This server makes `pcm` and `wav`. PCM's sample rate depends on the voice model, 24 kHz for Kokoro; a WAV says what it is.
- **A whole spoken turn:** `/v1/voice` and `/v1/realtime` are this server's own. To work anywhere, your client makes the three calls itself: transcribe, stream the chat reply, and speak it a sentence at a time.
- **Everything else beyond OpenAI's API,** such as `/v1/detect` with its faces and poses, or `/v1/lines`, works only here. The [reference](../reference/provider-api.md) lists what each one has.

A provider costs money per call and goes away when the network does. If your client moves something, like a robot car, keep the decision to stop on the car or on a server beside it. Stop when a reply is late, whichever server you're using.

## Check a provider

[`scripts/check_provider.py`](../../scripts/check_provider.py) makes each of these calls and says which ones pass:

```bash
python scripts/check_provider.py --provider local
python scripts/check_provider.py --provider deepinfra --key-env DEEPINFRA_API_KEY
```
