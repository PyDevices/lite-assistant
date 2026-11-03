# The provider API

The calls a client can make against this server, DeepInfra or OpenRouter, and how to use each. [Use this server or a provider](../guides/use-any-provider.md) shows them in a program.

| | This server | DeepInfra | OpenRouter |
|---|---|---|---|
| Base URL | `http://host:5000/v1` | `https://api.deepinfra.com/v1/openai` | `https://openrouter.ai/api/v1` |
| Key | `Authorization: Bearer`, needed from another machine | The same, always | The same, always |
| `GET /models` | Yes, with each one's `kind` | Yes | Yes |
| `POST /chat/completions` | Yes | Yes | Yes |
| Streaming | Yes | Yes | Yes |
| Photos, as `image_url` parts | Yes, with a model that sees | Yes, with a model that sees | Yes, with a model that sees |
| `response_format` with `json_schema` | Yes | Yes | With a model that supports it |
| `reasoning` or `reasoning_effort` | Yes | Yes | Yes |
| `POST /audio/speech` | `pcm` and `wav` | `pcm`, `wav`, `mp3`, `opus`, `flac` | Depends on the model |
| `POST /audio/transcriptions` | Yes | Yes | Yes |
| `POST /embeddings` | Yes | Yes | Yes |
| `POST /rerank` | Yes | No: its own `/v1/inference/{model}` | Yes |
| `POST /responses` | Yes | Yes | Yes |
| `input_audio` parts in chat | Yes | No | With a model that hears |

A request's other fields follow OpenAI's API at all three. Thoughts come back as `reasoning` and `reasoning_content` here, so read whichever your provider sends.

These work only on this server: `/v1/voice`, `/v1/realtime`, `/v1/detect`, `/v1/depth`, `/v1/segment`, `/v1/lines`, `/v1/ocr`, `/v1/classify`, and the [other endpoints](endpoints.md) marked as not OpenAI's.

## The vision schemas

Send one of these as `response_format`, `{"type": "json_schema", "json_schema": ...}`, in a chat completion with a photo in an `image_url` part. Boxes are fractions of the photo, from 0 to 1. An engine on this server takes only its own schema, exactly as written here: any other is a 400.

`object_detection`, which `yolox`, `yunet` or a vision model answers:

```json
{
  "name": "object_detection",
  "strict": true,
  "schema": {
    "type": "object",
    "additionalProperties": false,
    "required": ["objects"],
    "properties": {
      "objects": {
        "type": "array",
        "items": {
          "type": "object",
          "additionalProperties": false,
          "required": ["label", "score", "box", "distance"],
          "properties": {
            "label": {"type": "string"},
            "score": {"type": "number"},
            "box": {
              "type": "object",
              "additionalProperties": false,
              "required": ["left", "top", "right", "bottom"],
              "properties": {
                "left": {"type": "number"},
                "top": {"type": "number"},
                "right": {"type": "number"},
                "bottom": {"type": "number"}
              }
            },
            "distance": {"type": ["number", "null"]}
          }
        }
      }
    }
  }
}
```

`ocr_extraction`, which `ppocr` or a vision model answers:

```json
{
  "name": "ocr_extraction",
  "strict": true,
  "schema": {
    "type": "object",
    "additionalProperties": false,
    "required": ["text", "lines"],
    "properties": {
      "text": {"type": "string"},
      "lines": {
        "type": "array",
        "items": {
          "type": "object",
          "additionalProperties": false,
          "required": ["text", "box"],
          "properties": {
            "text": {"type": "string"},
            "box": {
              "type": "object",
              "additionalProperties": false,
              "required": ["left", "top", "right", "bottom"],
              "properties": {
                "left": {"type": "number"},
                "top": {"type": "number"},
                "right": {"type": "number"},
                "bottom": {"type": "number"}
              }
            }
          }
        }
      }
    }
  }
}
```
