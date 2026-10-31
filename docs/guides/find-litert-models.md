# Finding models lite-assistant can run

[litert-community](https://huggingface.co/litert-community) on Hugging Face holds over 400 models converted for LiteRT, Google's on-device runtime. This explains how to find ones lite-assistant can use, check them before writing any code, and add one as an engine. [MoGe-2](../../src/lite_assistant/engines/moge_engine.py) is the worked example throughout.

## Two kinds of model file

| File | Runtime | What it holds | Examples |
|---|---|---|---|
| `.litertlm` | `litert-lm` (`litert_lm` in Python) | A whole language model: weights, tokenizer, prompt template, and sometimes vision and audio encoders | Gemma 4, Qwen3, EmbeddingGemma 2 |
| `.tflite` | `ai-edge-litert` (`ai_edge_litert.interpreter.Interpreter`) | One graph: tensors in, tensors out. Everything before and after it is yours to write | MoGe-2, YOLOX, YuNet, Whisper |

- **`.litertlm` chat models** need no code: `lite-assistant models search` finds them, and `lite-assistant models add` downloads one. See [Add models](add-models.md).
- **`.tflite` models** each need an engine: code that prepares the input, runs the graph and reads the output.
- **Other files:** a repo may also hold `.task` files (MediaPipe bundles), ONNX or PyTorch files. lite-assistant doesn't run those.

## Listing the models

The Hub API lists them all, without a login:

```bash
curl -s "https://huggingface.co/api/models?author=litert-community&limit=500" \
  | python3 -c "import json,sys; [print(m['id'], m.get('pipeline_tag'), m.get('downloads')) for m in json.load(sys.stdin)]"
```

- **By task:** add `&pipeline_tag=object-detection`. Other tags include `depth-estimation`, `image-classification`, `image-segmentation`, `keypoint-detection`, `automatic-speech-recognition`, `audio-classification`, `text-to-speech`, `token-classification`, `zero-shot-classification` and `feature-extraction`. Some repos have no tag, so also scan the full list by name.
- **By name:** add `&search=yolo`.
- **Files, sizes and license:** `https://huggingface.co/api/models/litert-community/<name>?blobs=true` lists each file with its size, and the `license:` tag. For example:

```bash
curl -s "https://huggingface.co/api/models/litert-community/yolox-nano-litert?blobs=true" | python3 -c "
import json,sys; d=json.load(sys.stdin)
print([t for t in d['tags'] if t.startswith('license:')])
for s in d['siblings']: print(s['rfilename'], s.get('size'))"
```

## Checking a model before writing code

### 1. Read the model card

Look for:
- **The input:** its size, channel order (RGB or BGR), layout (NCHW `[1,3,H,W]` or NHWC `[1,H,W,3]`), value range, and whether normalization happens inside the graph. MoGe-2 takes NCHW RGB in [0,1], and does the ImageNet normalization itself.
- **The outputs:** what each one means. MoGe-2's card names four, and says their order isn't fixed, so they're told apart by shape.
- **Pre- and post-processing:** what the graph leaves out, such as letterboxing, a mel spectrogram, a tokenizer, anchor decoding, non-maximum suppression or CTC decoding.
- **Variants:** fp32, fp16 and int8 files of the same graph, and which one suits the CPU.
- **Sample code:** many cards link to [google-ai-edge/litert-samples](https://github.com/google-ai-edge/litert-samples). Its web demos (`samples/web_demos/src/<name>/main.js`) show the full pipeline. MoGe's showed letterboxing with gray padding.

When the card is thin, the original model's repo (the card's `base_model`) has the reference pre- and post-processing.

### 2. Check the license

- The repo's `license:` tag covers the conversion. The `base_model` may carry its own terms, and its training data may too.
- MIT, Apache-2.0 and BSD are easy. CC-BY needs attribution. CC-BY-NC rules out commercial use. Treat "other" as unknown until you've read it.

### 3. Load it and look

This script downloads a file, prints its inputs and outputs, and times it on random input:

```python
import os, resource, sys, time
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from huggingface_hub import hf_hub_download

path = hf_hub_download(sys.argv[1], sys.argv[2])
start = time.time()
interpreter = Interpreter(model_path=path, num_threads=os.cpu_count())
interpreter.allocate_tensors()
print(f"loaded in {time.time() - start:.1f}s")
print("signatures:", interpreter.get_signature_list())
for d in interpreter.get_input_details():
    print("in ", d["name"], d["shape"], d["dtype"].__name__, d["quantization"])
for d in interpreter.get_output_details():
    print("out", d["name"], d["shape"], d["dtype"].__name__)
# Random input, just to time it and see the outputs' ranges
for d in interpreter.get_input_details():
    interpreter.set_tensor(d["index"], np.random.rand(*d["shape"]).astype(d["dtype"]))
for _ in range(3):
    start = time.time()
    interpreter.invoke()
    print(f"invoke {time.time() - start:.2f}s")
for d in interpreter.get_output_details():
    t = interpreter.get_tensor(d["index"])
    print("out", d["name"], f"range {t.min():.3g} .. {t.max():.3g}")
print(f"peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB")
```

```bash
.venv/bin/python inspect_tflite.py litert-community/yolox-nano-litert <file>.tflite
```

For yolox-nano, it shows one NHWC input of `[1, 416, 416, 3]` and one raw output of `[1, 3549, 85]`. That's 3549 candidate boxes, each with 4 box numbers, an objectness score and 80 class scores, still to be decoded and thinned out with non-maximum suppression. It runs in 10 ms.

What can go wrong:
- **It won't load:** "Select TF ops" or an unknown custom op means the graph needs ops the Python interpreter doesn't have. Pass it by.
- **It's very slow on the CPU:** an fp16 graph can be declined by XNNPACK, LiteRT's fast CPU kernels, and fall back to reference kernels. MoGe-2's fp16 file is about 31× slower than its fp32 one. On the CPU, use fp32 or int8, and keep fp16 for the GPU.
- **Several files or signatures:** a model may be split into stages. OCR has detection and recognition, SAM an image encoder and a mask decoder, and Parakeet comes as 47 files. Each stage is a step to wire up.
- **Inputs of a fixed size:** most graphs take one size, so photos must be resized or letterboxed to it, and the output mapped back to the photo.

### 4. Check it on real input

Random input proves only that it runs. Feed it something whose answer you know, and compare against the model card, or the original model if you can run it. For MoGe-2, photos of a room, a street and a table gave sensible distances. A tight close-up of a cat, with nothing around it for scale, gave 6 m. Note what it gets wrong, for the docs.

## Adding it to lite-assistant

Follow the MoGe-2 engine:

1. **A base class** for the kind, if there isn't one, in `src/lite_assistant/engines/<kind>_base.py`. Add it to `BASES` in `engines/__init__.py`, and give `Engines` in `app.py` a registry for it. See `depth_base.py`: `get_model_id()`, the method that does the work (`estimate(image)`), and `setup()`.
2. **The engine** in `engines/<name>_engine.py`:
   - It finds the file with `paths.cached_hf_file(REPO, FILE)`, without touching the network. `setup()` downloads it with `hf_hub_download`, and `lite-assistant setup` calls `setup()`.
   - It loads in `__init__`, and prints why if it can't: the package is missing, or the file isn't downloaded.
   - It holds a lock around `set_tensor`/`invoke`/`get_tensor`, since the interpreter's tensors are shared between requests.
3. **An endpoint**, if a board should call it directly. Make it take the board's data as it comes: a JPEG as the request body, or a WAV. Return a small JSON answer a board can parse, and do the heavy work in `anyio.to_thread.run_sync`.
4. **A built-in tool**, if the LLM should use it in chat, in `src/lite_assistant/tools/`. A `conversation` parameter gets the chat so far, which is how `estimate_distance` finds the latest photo (`media.latest_image`). Return short strings that read well aloud.
5. **Tests:**
   - A fake engine in `tests/test_app.py`, for the endpoint and the tool.
   - A real-model test in `tests/test_engines.py`.
   - A live test in `tests/test_live_server.py`.
6. **The reference:** the engine, the endpoint, the tool, memory use and accuracy.
7. **Dependencies** in `pyproject.toml`. `ai-edge-litert` and `pillow` are already there; `scipy` and `numpy` are too.

## What suits a microcontroller

lite-assistant does the heavy work so a board doesn't have to. Good fits:
- **Small input:** take what a board already has, such as a JPEG from `cameraif`/`jpegio`, or PCM or WAV from `audioif`.
- **Small output:** a few numbers or words a board can act on, such as a distance, a label or a box, not a tensor.
- **Fast enough to wait for:** under about a second on the server's CPU.
- **Better than on the board:** the model is too big or too slow for the board, or it gets better by running next to the LLM and the other engines, such as detection combined with depth.

Models of a few MB in int8, such as `yolox-nano` or `YuNet`, may also run on an ESP32-P4 itself. The server is then a fallback for when it's out of reach.
