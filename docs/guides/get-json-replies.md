# Get JSON replies

`response_format` in chat completions, or `text: {"format": ...}` in Responses, holds the reply to JSON. LiteRT-LM's constrained decoding (llguidance) masks every token that would break it, so the reply always parses.

- **`json_schema`** holds it to the schema given: `{"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}` in chat completions, or `{"type": "json_schema", "name": ..., "schema": {...}}` in Responses.
- **`json_object`** asks for any JSON object, and **`text`** leaves the reply free.
- **Say what you want too.** The schema says what's allowed, not what to write. Describe the fields in the system message, and make fields that may not apply optional: Gemma fills in every field it's offered, with something.
- **No tools:** a reply held to a schema can't call one. Built-in tools are left out, and function `tools` with a `response_format` are a 400.
- **Thinking** still works: the thoughts are free, and the reply after them follows the schema.
- **Cost:** about 2 to 3 seconds more per reply on this PC's CPU.

```python
reply = client.chat.completions.create(model="e2b", messages=[
    {"role": "system", "content": "Reply with the action (led, servo or none) and its color or angle."},
    {"role": "user", "content": "Point the servo at 90 degrees."}],
    response_format={"type": "json_schema", "json_schema": {"name": "command", "schema": {
        "type": "object", "required": ["action"], "properties": {
            "action": {"enum": ["led", "servo", "none"]}, "angle": {"type": "integer"}}}}})
```
