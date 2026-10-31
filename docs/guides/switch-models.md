# Switch models

Only one LiteRT-LM model is in memory at a time, but a request can ask for any one that's downloaded: `/v1/models` lists them. Set `model` to its id, such as `gemma-4-E4B-it` (or just `e2b` or `e4b`) or `Qwen3-0.6B`, in chat completions, Responses, translations or `/v1/voice`, or in the `?model=` query of `/v1/realtime`. If another model is loaded, the server unloads it and loads this one on the same backend, so that reply waits a while: about half a minute for Gemma E4B.

The switch lasts for every client, until a request asks for another model or the server restarts. Any other `model`, such as `litertlm` or `gpt-4o`, gets whichever model is loaded now. Asking for a Gemma model that isn't downloaded gives a 404 `model_not_found`.

Models differ in what they take. Voice clips sent to a model that can't hear, such as Qwen3, are transcribed by Moonshine first, so it gets their words. An image sent to a model that can't see gives a 400.
