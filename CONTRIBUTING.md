# Contributing

## Run the tests

```bash
pip install pytest httpx openai
pytest tests
```

The engine tests load real models, so on a machine with little memory, `scripts/test-each.sh` runs each file in a process of its own instead, which frees each file's models before the next.

- **`tests/test_app.py`** checks the API against fake engines, through the official `openai` client. It needs no models.
- **`tests/test_litert_models.py`** checks finding, adding and removing models, and the `models` command, with the Hugging Face cache and Hub faked.
- **`tests/test_helpers.py`** checks tool folders, and the example tool file, with the web and Claude Code faked.
- **`tests/test_live_server.py`** checks a running server with real models (`LITE_ASSISTANT_URL`, default `http://localhost:5000`), and skips if there's none.
- **The engine tests** load the real models, and skip what isn't downloaded.

CI runs the files that use fake engines and need no models; `.github/workflows/tests.yml` lists them.

A server may already be running on port 5000. Don't restart it to try a change: run a second one with `serve --port`, or use the fake-engine tests.

## Add an engine

An engine is a module in `src/lite_assistant/engines/` ending in `_engine.py`; the server discovers it at startup. [Add an engine](docs/guides/add-an-engine.md) has the steps, and [Find LiteRT models](docs/guides/find-litert-models.md) covers checking a model before writing code. Models are never committed. They download to the Hugging Face cache, so check a model's licence before adding an engine.

## Record the showcase

`showcase/` records two videos of a running server at 1080p, with their sound and a still of each scene, in dark mode. The waits are cut short, with a badge saying how long each one was.

```bash
pip install playwright && playwright install chromium   # and ffmpeg
python showcase/chat.py     # the chat page, with Gemma 4 E4B, replies spoken
python showcase/boards.py   # the endpoints a board calls, request and reply on screen
```

Name scenes to record just those (`python showcase/chat.py plant documents`). They use the photos and clips in `test-media/`, which is git-ignored, so bring your own, and write to `showcase/output/`.

## Benchmarks

`scripts/benchmark.py` runs against a live server over HTTP and records what [Benchmarks](docs/how-it-works/benchmarks.md) reports:

```bash
python scripts/benchmark.py --url http://localhost:5000
```
