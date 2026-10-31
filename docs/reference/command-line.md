# The command line

`lite-assistant` has four commands. `lite-assistant COMMAND --help` prints each one's options.

## `serve`

Runs the server.

| Option | Meaning |
|---|---|
| `--host HOST`, `--port PORT` | Address to listen on, `0.0.0.0` and `5000` by default |
| `--llm e2b\|e4b\|none` | The chat model to load. `none` serves voice and the rest without one, which leaves its memory to other work. Without it, the answer `setup --llm` kept, or else the backend's own |
| `--tools-dir FOLDER` | A folder of [tool files](../guides/write-a-tool-file.md), besides `~/.config/lite-assistant/tools`; repeatable |
| `--certfile PEM`, `--keyfile PEM` | Serve https with this certificate and its private key. See [Serve over https](../guides/serve-over-https.md) |
| `--behind-proxy` | Require a key from every request, even from this PC, as when an https proxy on this PC forwards requests from elsewhere |
| `--allow-public` | Take requests with a key from public addresses too, not only this PC, the local network and Tailscale, as for a server in the cloud |
| `--forget-chats-after DAYS` | Delete chats unchanged for this many days, but pinned ones. By default they are kept |

## `setup`

Downloads the models every engine needs, and Silero VAD. `--llm e2b|e4b|none` picks the chat model, or changes the kept answer. See [Install](../get-started/install.md).

## `models`

Lists, finds, adds and removes LiteRT-LM chat models: `list` (the default), `search`, `add` and `remove`. See [Add models](../guides/add-models.md).

## `users`

Lists, adds and removes users, and gives them keys: `list` (the default), `add`, `key`, `admin` and `remove`. See [Manage users and keys](../guides/manage-users-and-keys.md).

## Environment variables

| Variable | Meaning |
|---|---|
| `LITERTLM_BACKEND` | `gpu` or `cpu` forces a backend for the chat model |
| `LITERTLM_MODEL` | `e2b` or `e4b` chooses the chat model |
| `LITERTLM_THREADS` | The CPU thread count |
| `LITERTLM_MAX_TOKENS` | The context length, 4096 by default |
| `LITE_ASSISTANT_IDLE_UNLOAD` | Seconds before an unused engine unloads, 600 by default; `0` keeps them loaded |
| `LITE_ASSISTANT_TOOLS` | Which built-in tools are on by default |
| `LITE_ASSISTANT_HOME` | A folder that holds the data, cache and config |
| `HF_HOME` | Moves the Hugging Face cache |
| `SEARXNG_URL` | Searches with your own SearXNG server instead of DuckDuckGo |
| `YOLOX_MODEL` | `nano` picks the smaller object detector |
