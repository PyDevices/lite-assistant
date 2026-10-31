# Let the model call your functions

Chat completions, Responses and Realtime take function `tools`, which the client runs, as OpenAI's do. The model decides when to call one. The call comes back whole, so its arguments arrive as one delta when streamed.

- **`tool_choice`:** `auto` and `none` work, and naming a function offers the model only that one. It can't be made to call a tool, so `required` acts as `auto`.
- **Other tool types,** such as `web_search`, are a 400.
- **Arguments:** `strict` is ignored, so check the arguments before using them. Holding calls to their schemas with LiteRT-LM's constrained decoding was tried: Gemma 4 E2B never broke an enum without it, and it cost about 2 seconds a reply.
- **Tools that weren't offered:** a call to one never reaches the client. The model gets an error back that lists the tools it has, and tries again or says it can't.
- **Sampling:** without a `temperature` or `top_p`, replies decode greedily, which suits tool calls best. Setting either samples.
- **Cut-off calls:** a call that `max_tokens` cuts off is dropped, rather than left in the reply as text, and `finish_reason` is `length`.
- **Silence after a result:** Gemma sometimes ends its turn without a word when given a tool's result. The server then asks it once more to answer with the results.
