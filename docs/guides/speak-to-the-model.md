# Speak to the model

POST an audio file to `/v1/voice` to get the spoken reply back as raw 16-bit 24kHz mono PCM. Each sentence goes to TTS as soon as the LLM finishes writing it, so audio starts before the reply is complete.

```bash
curl http://localhost:5000/v1/voice -F file="@tests/test_audio.wav" > reply.raw
```

Optional form fields:

- **`stt_model`:** `gemma` (the default) lets the LLM hear the audio. An STT engine id, such as `moonshine`, transcribes it first; the transcript then comes back percent-encoded in the `X-Transcript` header.
- **`voice`**, **`tts_model`** (default `kokoro`), **`system_prompt`** and **`max_tokens`** (default 256).
