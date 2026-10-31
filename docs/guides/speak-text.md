# Speak text

```bash
curl http://localhost:5000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro", "input": "Hello world!", "voice": "af_heart"}' > output.raw
```

The reply is raw PCM whatever `response_format` asks for.
