# AGENTS.md

- Tests with fake engines need no models: `pytest tests/test_app.py` and the files CI runs (see `.github/workflows/tests.yml`). The engine tests load real models and skip what isn't downloaded; `scripts/test-each.sh` runs them one file at a time on a small machine.
- A server may already be running on port 5000. Don't restart it to try a change: run a second one with `serve --port`, or use the fake-engine tests.
- An engine is a module in `src/lite_assistant/engines/` ending in `_engine.py`; the server discovers it at startup. Steps are in `docs/guides/add-an-engine.md`.
- `test-media/` is git-ignored: the showcase scripts and some engine tests expect your own photos and clips there.
- Models are never committed. They download to the Hugging Face cache; check a model's licence before adding an engine (`docs/guides/find-litert-models.md`).
