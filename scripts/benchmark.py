"""
Latency benchmark for a running server: chat completions (time to first token,
decode speed) and /v1/voice (time to first audio). Talks HTTP only, so it can
measure a server on any OS. Results are recorded in docs/how-it-works/benchmarks.md.
"""
import argparse
import json
import os
import statistics
import time
import urllib.request
import uuid

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def chat(base_url, prompt, max_tokens):
    """ Streams one completion; returns (seconds to first token, decode tokens/sec, usage). """
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                       "stream": True, "stream_options": {"include_usage": True}}).encode('utf-8')
    request = urllib.request.Request(base_url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    start = time.time()
    first = usage = None
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data: {"):
                continue
            event = json.loads(line[len(b"data: "):])
            if first is None and event["choices"] and event["choices"][0]["delta"].get("content"):
                first = time.time()
            usage = event.get("usage") or usage
    end = time.time()
    # Tokens after the first, over the time after the first
    return first - start, (usage["completion_tokens"] - 1) / (end - first), usage

def voice(base_url, audio, stt_model):
    """ Sends one spoken turn; returns (seconds to first audio byte, total seconds, seconds of audio). """
    boundary = uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="stt_model"\r\n\r\n{stt_model}\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="in.wav"\r\n'
            f'Content-Type: audio/wav\r\n\r\n').encode() + audio + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(base_url + "/v1/voice", body,
                                     {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    start = time.time()
    first = None
    size = 0
    with urllib.request.urlopen(request, timeout=600) as response:
        while (chunk := response.read1(65536)):
            first = first or time.time()
            size += len(chunk)
    # 16-bit 24kHz mono
    return first - start, time.time() - start, size / 48000

def summarize(values, unit, digits=2):
    return (f"median {statistics.median(values):.{digits}f}{unit} "
            f"(min {min(values):.{digits}f}, max {max(values):.{digits}f})")

def main():
    parser = argparse.ArgumentParser(description="Benchmark a running voice server")
    parser.add_argument("--url", default="http://localhost:5000")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    with open(os.path.join(REPO, "examples", "pg3825.txt"), errors="ignore") as f:
        passage = f.read()[20000:26000]
    prompts = [
        ("Short prompt", "Explain how a refrigerator works.", 200),
        ("Long prompt", "Summarize this passage in three sentences:\n\n" + passage, 200),
    ]
    for label, prompt, max_tokens in prompts:
        chat(args.url, prompt, 8)  # warm-up
        rows = [chat(args.url, prompt, max_tokens) for _ in range(args.runs)]
        usage = rows[0][2]
        print(f"{label} ({usage['prompt_tokens']} tokens in, {usage['completion_tokens']} out): "
              f"first token {summarize([r[0] for r in rows], 's')}, "
              f"decode {summarize([r[1] for r in rows], ' tok/s', 1)}")

    with open(os.path.join(REPO, "tests", "test_audio.wav"), "rb") as f:
        audio = f.read()
    for stt_model in ("moonshine", "gemma"):
        voice(args.url, audio, stt_model)  # warm-up
        rows = [voice(args.url, audio, stt_model) for _ in range(args.runs)]
        print(f"Voice, {stt_model} STT: first audio {summarize([r[0] for r in rows], 's')}, "
              f"whole reply {summarize([r[1] for r in rows], 's')}, "
              f"{statistics.median(r[2] for r in rows):.1f}s of audio")

if __name__ == "__main__":
    main()
