"""
Speaking a reply as it is written, for /v1/voice and the Realtime API: the
LLM writes sentences on a worker thread while the caller's thread voices
them, so the next sentence is ready when the current one finishes.
"""
import io
import queue
import threading
import wave
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .engines import ToolCall
from .voice_text import speakable, split_sentences

# The rate, width and channels every TTS engine produces, and Realtime's PCM format
SAMPLE_RATE = 24000


def wav(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    """ 16-bit mono PCM wrapped in a WAV header, as Gemma's audio encoder takes it. """
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return out.getvalue()


def spoken_reply(llm, tts, voice: str, messages: List[Dict[str, Any]], max_tokens: Optional[int] = None,
                 usage: Optional[Dict[str, int]] = None, cancel: Optional[threading.Event] = None,
                 tools: Optional[List[Dict[str, Any]]] = None) -> Iterator[Tuple[str, Any]]:
    """
    The LLM's reply to messages, spoken. Yields ("sentence", text) as each
    sentence is written, then ("audio", pcm) for each chunk of its audio,
    and ("tool_call", call) for each function the LLM calls from tools.
    Closing the generator, or setting cancel, stops the LLM within a token.
    """
    sentences = queue.Queue()
    stop = threading.Event()

    def until_stopped(chunks):
        for chunk in chunks:
            if stop.is_set() or (cancel and cancel.is_set()):
                return
            yield chunk

    calls = []

    def text_only(chunks):
        for chunk in chunks:
            if isinstance(chunk, str):
                yield chunk
            elif isinstance(chunk, ToolCall):
                calls.append(chunk)

    def write_sentences():
        # A spoken reply doesn't wait for the model to think
        chunks = llm.generate(messages, max_tokens=max_tokens, usage=usage, thinking=False,
                              **({"tools": tools} if tools else {}))
        try:
            for sentence in split_sentences(text_only(until_stopped(chunks))):
                sentences.put(("sentence", sentence))
            # Calls come after the text they follow, once it's all been spoken
            for call in calls:
                sentences.put(("tool_call", call))
        except Exception as exc:
            print(f"\n[!] Voice generation error: {exc}")
        finally:
            chunks.close()
            sentences.put(None)

    writer = threading.Thread(target=write_sentences, daemon=True)
    writer.start()
    try:
        while (next_up := sentences.get()) is not None:
            kind, sentence = next_up
            yield next_up
            if kind != "sentence":
                continue
            text = speakable(sentence)
            if text:
                for audio in tts.generate_pcm(text, voice):
                    yield "audio", audio
    finally:
        stop.set()
        # Waiting for the writer means the LLM is free, and usage filled in, once this returns
        writer.join()
