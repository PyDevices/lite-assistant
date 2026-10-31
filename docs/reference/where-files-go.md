# Where files go

- **Hugging Face models** (Gemma, EmbeddingGemma, Moonshine, Whisper, MoGe-2, YOLOX, YuNet, PANNs, pyannote, WeSpeaker, CMGAN, GLiClass, CPGA-Net, ormbg, MI-GAN, Real-ESRGAN, NIMA, PP-OCRv6, CLIPSeg, HSEmotion, L2CS-Net, RTMPose, PlantNet, Ultra-Fast-Lane-Detection, MoViNet, CREPE, Qwen3-Reranker, Silero VAD) go to the Hugging Face cache, `~/.cache/huggingface/hub` by default (`HF_HOME` moves it). Other tools that use the same models share them.
- **Kokoro's files** come from GitHub and go to the user data directory: `~/.local/share/lite-assistant` on Linux, `%LOCALAPPDATA%\lite-assistant` on Windows.
- **LiteRT-LM's compiled caches** go to the user cache directory: `~/.cache/lite-assistant` on Linux. They can be deleted, and are rebuilt on the next load.
- **`LITE_ASSISTANT_HOME`**, if set, holds the data, the cache and the config (the tools folder, in `config/tools`) in place of those directories.
