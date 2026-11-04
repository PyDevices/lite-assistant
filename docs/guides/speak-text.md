# Speak text

```bash
curl http://localhost:5000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro", "input": "Hello world!", "voice": "af_heart"}' > output.raw
```

The reply is raw 16-bit mono PCM at 24 kHz. Set `response_format` to `wav` for the same samples with a WAV header, whose sizes are left open because the speech is still being made; any other format is a 400. `speed` makes it slower or faster: from 0.25 to 4 with Kokoro, where 1 is normal.
