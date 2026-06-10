# SPDX-License-Identifier: MIT
"""Smoke test: load GPT-OSS-20B with MoE expert offload, generate 20 tokens, measure VRAM.

Goal: validate that the 4070 can host the BF16 backbone while experts page from CPU RAM.
Run when /home/scott/models/gpt-oss-20b/model-*.safetensors are fully downloaded.

Exits non-zero on any failure. Prints peak VRAM and tokens/sec at the end.
"""
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = Path("/home/scott/models/gpt-oss-20b")
OFFLOAD_DIR = Path("/tmp/gpt-oss-offload")
OFFLOAD_DIR.mkdir(exist_ok=True)


def fmt_gb(b: int) -> str:
    return f"{b / 1024**3:.2f} GB"


def main() -> int:
    if not (MODEL_PATH / "model.safetensors.index.json").exists():
        print(f"FAIL: model not downloaded yet at {MODEL_PATH}")
        return 1

    sf_files = sorted(MODEL_PATH.glob("model-*.safetensors"))
    total = sum(f.stat().st_size for f in sf_files)
    print(f"Found {len(sf_files)} safetensors files, total {fmt_gb(total)}")
    if total < 10 * 1024**3:
        print(f"FAIL: expected ~13 GB, got {fmt_gb(total)} — download incomplete")
        return 1

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    print(f"  vocab_size={len(tok)}, pad_token={tok.pad_token} ({tok.pad_token_id})")

    print("Loading model (device_map='auto', BF16 backbone on GPU, experts paged from CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
        offload_folder=str(OFFLOAD_DIR),
        low_cpu_mem_usage=True,
    )
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.1f}s")
    print(f"  hidden_size={model.config.hidden_size}, layers={model.config.num_hidden_layers}")
    print(f"  Device map (first 5 entries): {dict(list(model.hf_device_map.items())[:5])}")

    print("\nGenerating 20 tokens, prompt='What sound does a cat make?'...")
    prompt = "What sound does a cat make?"
    inputs = tok(prompt, return_tensors="pt").to("cuda")

    t1 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    gen_time = time.time() - t1

    text = tok.decode(out[0], skip_special_tokens=True)
    n_new = out.shape[1] - inputs["input_ids"].shape[1]
    tps = n_new / gen_time if gen_time > 0 else float("inf")

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nOutput: {text!r}")
    print(f"\nGenerated {n_new} tokens in {gen_time:.2f}s = {tps:.2f} tok/s")
    print(f"Peak VRAM: {peak_gb:.2f} GB / 8.0 GB")

    if peak_gb > 7.5:
        print("WARN: peak VRAM > 7.5 GB — leaves no room for BEATs + Q-Former")
    elif peak_gb > 6.5:
        print("OK: peak VRAM under 7.5 GB — BEATs (180 MB) + Q-Former (150 MB) should fit")
    else:
        print(f"OK: comfortable headroom ({8.0 - peak_gb:.2f} GB free)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
