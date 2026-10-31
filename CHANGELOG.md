# Changelog

## Unreleased

- Kokoro is the speech engine and the default voice; the Piper engine and its `piper-tts` dependency are gone. A request for `model: piper` now goes to the default engine like any model name the server doesn't know.
- The docs are reorganised into Get started, Guides, Reference and How it works.
- `tests/test_audio.wav` is regenerated with Kokoro by `scripts/generate_test_audio.py`.
