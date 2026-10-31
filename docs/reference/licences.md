# Licences

The code in this repository is [MIT](../../LICENSE). The test photo and its source are listed in [tests/README.md](../../tests/README.md).

## Dependencies

Installing lite-assistant also installs packages under other licences, and some of them are GPL. If you redistribute a bundle that includes them, such as a container image or an installer, you take on their terms. The ones to know about:

- `phonemizer` (installed by `kokoro-onnx`): GPL-3.0 or later.
- espeak-ng, bundled by `espeakng-loader` (installed by `kokoro-onnx`): GPL-3.0 or later.
- `soxr`: LGPL-2.1 or later.

The rest are MIT, BSD, Apache-2.0 or PSF: fastapi, uvicorn, websockets, python-multipart, platformdirs, huggingface_hub, kokoro-onnx, onnxruntime, soundfile, litert-lm, scipy, ai-edge-litert, pillow, tokenizers, ddgs, trafilatura, pypdf and audioop-lts.

Models are never part of the repository. Each downloads from its own source under its own licence, which [Engines](engines.md) names.
