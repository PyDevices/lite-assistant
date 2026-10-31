# What runs where

lite-assistant is a FastAPI app. It is one process that holds many small models and exposes them behind OpenAI's API shapes where OpenAI has one, and behind endpoints of its own where it doesn't (depth, detection, sounds, lines and the rest; [Endpoints](../reference/endpoints.md) lists them).

## Engines

Each model sits behind an engine, a class in `src/lite_assistant/engines/` whose module name ends in `_engine.py`. The server finds engines at startup by scanning that package, so adding a model is adding a file; [Add an engine](../guides/add-an-engine.md) has the steps. Engines of one kind share a base class (`TTSEngine`, `STTEngine`, `LLMEngine` and so on) and the server keeps them in one registry per kind, with a default for each.

Which engines load at startup and which on first use, and when an idle one unloads, is in [Engines](../reference/engines.md), with each one's size and licence.

## One chat model at a time

Only one LiteRT-LM chat model is in memory. A request that names another downloaded model makes the server unload the current one and load the new one, so that reply waits; the switch lasts for every client. [Switch models](../guides/switch-models.md) has the details.

## Tools the server runs itself

When the chat model calls a built-in tool, such as the clock or `estimate_distance`, the server runs it and gives the model the result, so a client sees only the reply the tool led to. The tools use the other engines: distance is MoGe-2 and YOLOX, sounds are PANNs, notes are searched with EmbeddingGemma and Qwen3-Reranker. [Built-in tools](../reference/built-in-tools.md) lists them.

## Who is asking

A request from the server's own PC is the owner. Any other request needs a user's key, and a request from a public address is refused. Each user has their own chats, responses, files, notes and skills. [Manage users and keys](../guides/manage-users-and-keys.md) covers the rules.

## Speed

[Benchmarks](benchmarks.md) measures the chat engine and `/v1/voice` on one laptop, and [What boards got from the server](board-results.md) measures the endpoints from two ESP32-P4 boards.
