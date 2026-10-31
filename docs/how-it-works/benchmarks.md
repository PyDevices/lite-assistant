# Benchmarks

Latency of the Gemma chat engine (`engines/litertlm_engine.py`) and the `/v1/voice` endpoint.

## Method

`scripts/benchmark.py` runs against a live server over HTTP:

```bash
python scripts/benchmark.py --url http://localhost:5000
```

- **Chat:** two prompts, each sent with `stream: true` and `max_tokens: 200`. Each gets one warm-up request, then 5 timed runs.
  - The short prompt is "Explain how a refrigerator works." (15 tokens in).
  - The long prompt asks for a summary of 6,000 characters of `examples/pg3825.txt` (1,709 tokens in).
  - *First token* is the time from sending the request to the first content chunk. It includes prefill.
  - *Decode* is the tokens after the first, divided by the time after the first, using the server's own `completion_tokens` count.
- **Voice:** `tests/test_audio.wav` (a 6-second spoken sentence) goes to `/v1/voice`, with one warm-up, then 5 timed runs per STT option.
  - *First audio* is the time from sending the request to the first PCM byte.
  - *Whole reply* is the time to the last byte.
  - The default `max_tokens` is 256. The voice is Kokoro `af_heart`. The Piper rows in the voice table below were measured with `en_US-amy-medium`, which lite-assistant no longer ships.

All figures are the median of 5 runs, with min–max in brackets.

## WSL2 on Ubuntu (2026-10-24)

| | |
| --- | --- |
| CPU | Intel Core Ultra 7 155H, 12 threads visible to WSL |
| RAM | 7.9 GB visible to WSL |
| GPU | None usable. WebGPU only finds `llvmpipe`, a software renderer, and the GPU backend segfaults, so the engine runs on the CPU. |
| Model | `gemma-4-E2B-it.litertlm` (2.6 GB), CPU backend, 4,096-token context |
| Software | `litert-lm-api` 0.18.1, Python 3.12.3 |
| Server | `scripts/server.py` (the server before the FastAPI rewrite) under systemd, alongside Piper, Kokoro and Moonshine |

### Chat completions

| Prompt | First token | Decode |
| --- | --- | --- |
| Short (15 tokens in, 200 out) | 0.55 s (0.46–0.58) | 23.7 tok/s (21.8–25.0) |
| Long (1,709 tokens in, ~104 out) | 6.52 s (6.46–6.55) | 23.6 tok/s (22.6–29.5) |

The long prompt's first-token time is mostly prefill, at about 260 tokens/sec. Run-to-run variation is large on this laptop. An earlier spec-off run of the same prompts with a different client script measured 0.43 s / 27.2 tok/s (short) and 5.17 s / 26.9 tok/s (long).

Through curl, a non-streaming request for "In two sentences, why is the sky blue?" (19 tokens in, 58 out) took 2.86 s in total. The streaming version's first byte arrived at 0.39 s.

### Voice

| STT | First audio | Whole reply | Audio length |
| --- | --- | --- | --- |
| Moonshine, then Gemma, then Kokoro (2026-10-30) | 16.55 s (15.74–17.25) | 16.55 s (15.75–17.25) | 4.1 s |
| Gemma hears the audio directly, then Kokoro (2026-10-30) | 17.06 s (15.61–25.21) | 18.04 s (16.63–26.16) | 3.5 s |
| Moonshine, then Gemma, then Piper (2026-10-24) | 2.84 s (2.61–3.13) | 3.33 s (3.13–3.68) | 6.1 s |
| Gemma hears the audio directly, then Piper (2026-10-24) | 2.39 s (1.57–2.49) | 2.40 s (1.57–2.49) | 6.8 s |

The Kokoro rows are slower mostly for a different reason than the voice. By then the server offered the chat model its built-in tools, and it spent 13 to 17 s reading that prompt before its first sentence was done; the server's log puts Kokoro's own share, from first sentence to first audio, at about 1 to 2.5 s. They were run on 2026-10-30 against a second server (`--llm e2b`, CPU backend), on the same laptop, while other jobs ran on it (load average 3 to 5), with a new 6.5 s test clip, so they are not directly comparable to the Piper rows. The Piper rows predate the built-in tools. The reply audio is not the same reply either: the model answered the clip differently each way.

With `stt_model=gemma` on this clip, Gemma repeated what it heard rather than answering it. The clip is a statement, not a question.

### Speculative decoding on the CPU

The E2B file has a speculative decoding drafter (`ModelInfo.llm.has_speculative_decoding_support()` is true).

Using LiteRT-LM directly, with no server, on three prompts of 200 tokens out:

| Prompt | Off | On |
| --- | --- | --- |
| "Explain how a refrigerator works." | 27.0 tok/s | 30.9 tok/s |
| Rewrite a casual message in a formal tone | 27.3 tok/s | 47.5 tok/s |
| Write a palindrome check in Python | 27.3 tok/s | 37.8 tok/s |

Through the server with `scripts/benchmark.py`, the same switch didn't show a gain:

| | Short prompt decode | Long prompt decode |
| --- | --- | --- |
| Off | 23.7 tok/s | 23.6 tok/s |
| On | 26.5 tok/s (21.6–28.4) | 21.9 tok/s (18.0–28.4) |

The two disagree, and the run-to-run spread is as wide as the difference. So speculative decoding stays off for the CPU, as originally specified, until it's tested more carefully. It is on for the GPU.

### Gemma 4 E4B on the CPU

The larger model, timed with a standalone script with no server running (`LITERTLM_MODEL=e4b` selects it in the server). Each figure is the median of 3 replies of 128 tokens, plus one long-prompt prefill.

| Model | Short prompt first token | Decode | Long prompt first token (prefill) | Peak memory |
| --- | --- | --- | --- | --- |
| E4B | 1.45 s | 11.7 tok/s | 19.48 s (88 tok/s) | 3.9 GB |
| E4B, speculative decoding on | 1.54 s | 11.1 tok/s | 19.55 s (87 tok/s) | 4.0 GB |
| E2B, same script | 0.51 s | 25.1 tok/s | 8.72 s (196 tok/s) | |

E4B decodes at about half E2B's speed and reads prompts at under half. That's still faster than speech, so voice replies keep up, but long prompts wait much longer. The drafter doesn't help E4B on the CPU either.

### One-time costs

These aren't in the tables above because the warm-up requests absorb them.

- **Model load:** 3–15 s at server start, including a one-token warm-up generation.
- **First voice request after a restart:** 7.2 s of Moonshine STT and about 2.8 s more before the first audio. Piper loads a voice the first time it's used, and Moonshine's first call warms up torch. Warm, the whole voice pipeline reaches first audio in under 3 s.
- **Memory:** a process holding only Gemma 4 E2B peaks at 1.8 GB resident.

## Windows (2026-10-25)

The same laptop, with the server running on the Windows Python 3.14 (no venv) from a copy of the repo at `C:\piper`: `litert-lm-api` 0.18.1, `torch` 2.14.1. Setup is described in [Run the server on Windows from WSL](../guides/run-on-windows.md). The benchmark ran from WSL against `localhost:5000`.

Windows was much slower than WSL until two things were fixed:

1. **Power throttling.** Windows throttles processes it treats as background work (EcoQoS), and a server started with no visible window counts as one. That throttling cost about 3× on prompt reading and slowed decoding too. The server now opts out at startup on Windows.
2. **Thread count.** On Windows, LiteRT-LM's CPU backend decodes faster with fewer threads. `LITERTLM_THREADS=3` is the best balance.

### Chat and voice, CPU backend, E2B

| | WSL | Windows over `U:` | Windows from `C:` | Windows from `C:`, tuned |
| --- | --- | --- | --- | --- |
| Short prompt: first token | 0.55 s | 1.59 s | 1.48 s | **0.70 s** (0.60–0.74) |
| Short prompt: decode | 23.7 tok/s | 7.4 tok/s | 8.6 tok/s | **15.2 tok/s** (14.6–16.0) |
| Long prompt: first token | 6.52 s | 18.39 s | 16.78 s | **7.81 s** (6.66–8.61) |
| Long prompt: decode | 23.6 tok/s | 7.8 tok/s | 8.9 tok/s | **14.8 tok/s** (13.6–15.6) |
| Voice, Moonshine STT: first audio | 2.84 s | 8.28 s | 6.82 s | **4.44 s** (4.01–4.78) |
| Voice, Moonshine STT: whole reply | 3.33 s | 10.49 s | 8.78 s | **5.46 s** (5.03–5.82) |
| Voice, Gemma hears the audio: first audio | 2.39 s | 7.20 s | 6.48 s | **2.99 s** (2.39–3.07) |

- **Over `U:`:** the code, packages and models were read through the share into WSL, from a venv in the WSL checkout.
- **From `C:`:** a local copy, but still throttled, with the default thread count.
- **Tuned:** from `C:`, unthrottled, with `LITERTLM_THREADS=3`.

Running over the share cost only about 10–15%. Tuned, Windows matches WSL on prompt reading and comes within 0.2 s on the first token, but still decodes at about two-thirds of WSL's speed. The Moonshine pipeline is slower on Windows: its transcription runs on torch.

### Thread count, CPU backend

A standalone script loaded E2B with each `Backend.CPU(thread_count=...)`, with no server running. Each figure is the median of 3 replies of 128 tokens, plus one long-prompt prefill.

| Threads | Windows, throttled: decode | Windows, unthrottled: decode | Windows, unthrottled: long prompt first token (prefill) | WSL: decode | WSL: long prompt first token (prefill) |
| --- | --- | --- | --- | --- | --- |
| 1 | | 9.7 tok/s | 19.48 s (89 tok/s) | | |
| 2 | | **17.6 tok/s** | 10.03 s (172 tok/s) | 16.5 tok/s | 9.80 s (174 tok/s) |
| 3 | | 15.6 tok/s | 7.06 s (244 tok/s) | 22.6 tok/s | 7.60 s (225 tok/s) |
| 4 | 8.9 tok/s | | | 23.1 tok/s | 7.69 s (222 tok/s) |
| 6 | 7.0 tok/s | 8.3 tok/s | 6.96 s (248 tok/s) | 23.8 tok/s | 6.76 s (253 tok/s) |
| 8 | 5.4 tok/s | 5.6 tok/s | 7.34 s (235 tok/s) | | |
| 12 | 4.4 tok/s | | | | |
| 16 | 3.8 tok/s | | | | |
| Default | 9.0 tok/s | 11.4 tok/s | 6.63 s (260 tok/s) | **25.9 tok/s** | 6.72 s (254 tok/s) |

- **Windows:** decoding peaks at 2–3 threads and falls as more are added, while prefill keeps improving up to about 6.
- **WSL:** LiteRT-LM's default thread count is fastest, so `LITERTLM_THREADS` is left unset there.
- **Pinning:** pinning the Windows process to the 6 performance cores at high priority made no difference once throttling was off. It gave 11.4 tok/s at the default thread count, the same as unpinned.
- **YNNPACK:** `enable_ynnpack=True` fails on Windows: the engine won't load (`UNIMPLEMENTED`).
- **Power:** the laptop was on AC power, with Windows in its default balanced power mode.

### GPU backend (Intel Arc)

The GPU check passes on the Intel Arc integrated GPU (driver 32.0.101.8831), but the GPU is the slowest option on this laptop. Timed with a standalone script, unthrottled, on "Explain how a refrigerator works.":

| Model | First token | Decode |
| --- | --- | --- |
| E2B | 1.6–1.8 s | 3.6–4.0 tok/s |
| E4B | 4.7–5.0 s | 2.0–2.1 tok/s |
| E4B, speculative decoding on | 4.3–5.0 s | 2.7 tok/s |

Turning throttling off made no difference to the GPU.

Through the server, the 1,709-token prompt hung the GPU: `DXGI_ERROR_DEVICE_HUNG`. Windows resets a GPU that stays busy for more than about 2 seconds. That left the engine with a lost device, and every later request failed until a restart.

On this machine, set `LITERTLM_BACKEND=cpu` on Windows. In auto mode, the engine picks this GPU because it works, even though it's slower.
