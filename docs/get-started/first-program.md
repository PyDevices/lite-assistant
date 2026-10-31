# Your first program

With the server running ([Install](install.md)), ask it to speak:

```bash
curl http://localhost:5000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model": "kokoro", "input": "Hello world!", "voice": "af_heart"}' > hello.raw
```

`model` names the engine and `voice` one of its voices; `curl http://localhost:5000/voices` lists them. The reply streams as raw 16-bit, 24 kHz mono PCM, whatever `response_format` asks for. For "Hello world!" in `af_heart` that is about 56 KB, 1.2 seconds of speech. Play it with:

```bash
ffplay -f s16le -ar 24000 -ac 1 hello.raw
```

## From an OpenAI client

Any OpenAI client works. Point it at the server. From the server's own PC the API key can be anything; from elsewhere, it is a [user's key](../guides/manage-users-and-keys.md).

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5000/v1", api_key="unused")
reply = client.chat.completions.create(
    model="litertlm", messages=[{"role": "user", "content": "Write a haiku about the sea."}])
print(reply.choices[0].message.content)
```

This needs a chat model: a server started with `--llm none` answers with an error saying so. A model name the server doesn't know, such as `gpt-4o`, `tts-1` or `whisper-1`, goes to the default engine.

Next: [transcribe audio](../guides/transcribe-audio.md), [chat](../guides/chat-with-the-model.md), or open the server's own page, [the chat page](../guides/use-the-chat-page.md).
