# SPDX-License-Identifier: MIT
"""Caption-bridge MVP: NatureLM-audio (CPU/4-bit) → GPT-OSS-20B (llama-server).

Pipeline:
    audio.wav
       │  (1) NatureLM-audio captions the audio
       ▼
    "Green Treefrog" / "Domestic cat meow" / etc.
       │  (2) GPT-OSS-20B reasons over the caption
       ▼
    "What might this cat be communicating? Is this a contented purr or a distress call? ..."

This is the simplest possible bridge — no GGUF conversion needed.
Once it works end-to-end we can upgrade to the mmproj path (real soft-prompt injection).

Run order:
    1. Start llama-server (separate terminal):
       /home/scott/llama.cpp/build-cuda/bin/llama-server \\
         --model /home/scott/models/gpt-oss-20b-gguf/gpt-oss-20b-Q4_K_M.gguf \\
         --port 9002 --n-gpu-layers 12 --ctx-size 4096
    2. Run this script:
       cd /home/scott/naturelm-cats && uv run python scripts/bridge_caption.py <audio_file>
"""
import argparse
import sys
import time
from pathlib import Path

import requests

LLAMA_SERVER = "http://127.0.0.1:9002"


def caption_with_naturelm(audio_path: Path, query: str) -> str:
    """Run NatureLM-audio to produce a text caption for audio_path.

    Forces CPU via CUDA_VISIBLE_DEVICES="" before importing torch — NatureLM's
    _DEVICE module global is set at import time. Slow (~30s/clip) but avoids
    VRAM contention with llama-server.
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from NatureLM.infer import Pipeline

    print(f"[caption] Loading NatureLM-audio (CPU)...", file=sys.stderr)
    t0 = time.time()
    pipeline = Pipeline()
    print(f"[caption] Loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    t1 = time.time()
    results = pipeline([str(audio_path)], [query], window_length_seconds=10.0, hop_length_seconds=10.0)
    print(f"[caption] Inference in {time.time() - t1:.1f}s", file=sys.stderr)

    return results[0] if isinstance(results, list) else str(results)


def reason_with_gpt_oss(caption: str, user_question: str) -> str:
    """Send caption + question to GPT-OSS-20B via llama-server harmony API."""
    # llama-server speaks OpenAI-compatible /v1/chat/completions
    messages = [
        {
            "role": "system",
            "content": (
                "You are a bioacoustics expert. The user gives you an audio caption "
                "and a question. Reason carefully about what the audio likely represents."
            ),
        },
        {
            "role": "user",
            "content": f"Audio caption from a specialized bioacoustic model: {caption!r}\n\nQuestion: {user_question}",
        },
    ]
    r = requests.post(
        f"{LLAMA_SERVER}/v1/chat/completions",
        json={"model": "gpt-oss-20b", "messages": messages, "max_tokens": 1200, "temperature": 0.3},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("audio", type=Path, help="Path to audio file (wav/mp3/flac/ogg)")
    p.add_argument(
        "--caption-query",
        default="Caption the audio. What animal vocalization is this, if any?",
        help="Query passed to NatureLM-audio for captioning",
    )
    p.add_argument(
        "--reason-question",
        default="What is this animal likely communicating? What is the behavioral context?",
        help="Follow-up question for GPT-OSS-20B reasoning",
    )
    args = p.parse_args()

    if not args.audio.exists():
        print(f"FAIL: {args.audio} not found")
        return 1

    # Health check llama-server
    try:
        r = requests.get(f"{LLAMA_SERVER}/health", timeout=3)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"FAIL: llama-server not reachable at {LLAMA_SERVER} — start it first ({e})")
        return 1

    caption = caption_with_naturelm(args.audio, args.caption_query)
    print(f"\n=== NatureLM-audio caption ===\n{caption}")

    answer = reason_with_gpt_oss(caption, args.reason_question)
    print(f"\n=== GPT-OSS-20B reasoning ===\n{answer}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
