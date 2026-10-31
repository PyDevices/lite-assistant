# Talk in real time

`/v1/realtime` detects turns with `server_vad` by default, as OpenAI's does. Silero VAD finds where speech starts and stops, and the server commits each turn and answers it. Speech that starts during a reply cancels the reply.

- **`server_vad` settings:** `threshold`, `prefix_padding_ms`, `silence_duration_ms`, `create_response` and `interrupt_response` work as OpenAI documents them.
- **`semantic_vad`** is accepted too, but Silero can't tell when a thought is finished. So its `eagerness` sets how much silence ends a turn: 1000 ms for `low`, 500 ms for `medium` and `auto`, 300 ms for `high`.
- **Manual turns:** set `turn_detection` to null to commit turns yourself. Without Silero VAD downloaded, turn detection is off.
- **Truncating:** after an interruption, send `conversation.item.truncate` with how much of the reply was played, so the model remembers only what the user heard. Whole sentences are kept.

[`examples/realtime_client.py`](../../examples/realtime_client.py) is a voice chat from the microphone, through the openai SDK (`pip install "openai[realtime]" sounddevice`). Use headphones, or the mic hears the reply and interrupts it.
