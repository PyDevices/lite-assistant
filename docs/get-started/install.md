# Install

lite-assistant needs Python 3.10 or later. It installs from a clone:

```bash
git clone https://github.com/bdbarnett/lite-assistant
cd lite-assistant
pip install -e .
lite-assistant setup
lite-assistant serve
```

`run.sh` (or `run.cmd` on Windows) does all three in a `.venv`. To run the server on Windows from WSL, see [Run the server on Windows from WSL](../guides/run-on-windows.md).

## What each command does

- **`lite-assistant setup`** downloads the models: about 8.5 GB with Kokoro's voices, Gemma E2B, EmbeddingGemma 2, MoGe-2, YOLOX, YuNet, PANNs, pyannote, WeSpeaker, CMGAN, GLiClass, CPGA-Net, ormbg, MI-GAN, Real-ESRGAN, NIMA, PP-OCRv6, CLIPSeg, HSEmotion, L2CS-Net, RTMPose, PlantNet and Silero VAD. That is the size on disk of everything the engines fetch, counting the E2B chat model and not E4B, which setup adds only if a GPU works. [Where files go](../reference/where-files-go.md) says where they land.
- **`lite-assistant serve`** listens on `0.0.0.0:5000`. Change it with `--host` and `--port`. The [command line](../reference/command-line.md) lists the rest of its options.

## The chat model

In a terminal, setup asks once which chat model to download: Gemma 4 E2B (2.6 GB), E4B (3.7 GB, smarter, but half as fast without a GPU), or none, for a server only boards use, which still sees, hears and speaks for them. It recommends E4B on a machine with 12 GB of memory or more, since a server on E4B holds about 5 GB, and E2B, which holds 1.8 GB, on a smaller one.

`setup --llm e2b|e4b|none` answers without asking, or changes the answer. It is kept in `setup.json` in the config folder, so `run.sh` asks only the first time. From a script, with no answer, setup downloads E2B, and E4B too if a GPU works on this machine.

`serve --llm e2b|e4b|none` picks the chat model to load; without it, the answer `setup --llm` kept. With `none` no chat model loads, speech, transcription and the rest work, and chat answers with an error saying so.

Add more models later with `lite-assistant models`, or **Add or remove models...** in the chat page's model menu; see [Add models](../guides/add-models.md). With none, add one with `setup --llm e2b` or `lite-assistant models add`, and restart the server.

Next: [make it speak](first-program.md).
