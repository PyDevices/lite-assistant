# Add models

Any LiteRT-LM chat model on Hugging Face can be added, from the command line or [the chat page](use-the-chat-page.md):

```bash
lite-assistant models search qwen3
lite-assistant models add litert-community/Qwen3-1.7B
lite-assistant models
lite-assistant models remove Qwen3_1.7B
```

- **`search [words]`** lists LiteRT-LM chat models on Hugging Face, most downloaded first (`--sort likes`, `trending` or `recent` for another order), with the file `add` would take, its size, and whether it's likely to fit in this machine's memory beside the rest of the server. Search leaves out base models (Qwen's `-Base`, Gemma's `-pt`), which only continue text rather than follow instructions, as well as speech and embedding models. Repos often hold several files: builds for one kind of NPU, the browser or the GPU only, and other quantizations. `add` takes the file named after the repo, or else the smallest that runs anywhere.
- **`add REPO [FILE]`** downloads a model into the Hugging Face cache; FILE picks another of the repo's `.litertlm` files. REPO may be the repo's huggingface.co address, or a file's, which picks that file. The running server offers it right away, without a restart.
- **`add PATH`** copies a `.litertlm` file you have into the `models` folder of the data directory (see [Where models go](../reference/where-files-go.md)), where any chat model file you put is offered too.
- **`remove MODEL`** deletes one, and the server stops offering it.

Some models on the Hub, such as Google's Gemma 3 and 3n, are gated: you accept their terms on the model's page, logged in, and log this machine in with `hf auth login` (or set `HF_TOKEN`) so the download can use your token. Search marks them `gated`, and adding one that isn't unlocked, or a repo that's private or missing, says which and what to do.

On the chat page, the Models dialog's search also takes a repo's address, has the same orders, and shows a download's speed and time left.
