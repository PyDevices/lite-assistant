# Let the model think

Some models can think before they reply. Qwen3 thinks unless told not to; Gemma 4 thinks only when asked. A request's effort turns it on or off: `reasoning_effort` in chat completions, or `reasoning: {"effort": ...}` in Responses.

- **`none`** or **`minimal`** turns thinking off.
- **`low`** and **`medium`** let the model think for up to 512 and 2048 tokens.
- **`high`**, **`xhigh`** and **`max`** let it think as long as it likes.
- **Leaving it out** gets the model's own default.

The thoughts come back apart from the reply: as `reasoning_content` on the chat message (or in its `delta` when streamed), the convention DeepSeek and vLLM use, and as a `reasoning` output item with `reasoning_text` content in Responses (streamed as `response.reasoning_text.delta` events). Usage counts them as `reasoning_tokens`. They aren't sent back to the model on the next turn. Spoken replies, Realtime and translations never think.

```python
reply = client.chat.completions.create(model="Qwen3-0.6B", reasoning_effort="high",
                                       messages=[{"role": "user", "content": "What is 17 * 23?"}])
print(reply.choices[0].message.reasoning_content)
print(reply.choices[0].message.content)
```
