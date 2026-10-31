# Chat with the model

```bash
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "litertlm", "messages": [{"role": "user", "content": "Write a haiku about the sea."}], "stream": true}'
```
