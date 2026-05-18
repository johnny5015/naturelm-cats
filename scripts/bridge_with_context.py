"""Bridge v2: NatureLM-audio caption + BEATs cat-context classifier → GPT-OSS-20B reasoning.

Pipeline (all CPU for audio side, GPU for LLM):
    audio.wav
       ├─► NatureLM-audio (CPU) ─► generic caption "Domestic cats vocalizing."
       └─► BEATs + context head (CPU) ─► context label + softmax probabilities
                                          {brushing: 0.4, isolation: 0.3, food: 0.3}
       │
       ▼ merge
    GPT-OSS-20B (4070 via llama-server :9002) ─► structured behavioral analysis

The context classifier was trained on CatMeows (3 classes: brushing/isolation/food).
For out-of-distribution clips (e.g., ESC-50) the softmax still tells us "this is most
similar to X but with low confidence." GPT-OSS-20B uses the uncertainty too.
"""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU-only audio side

import numpy as np
import requests
import resampy
import soundfile as sf
import torch
import torch.nn as nn
from safetensors.torch import load_file

sys.path.insert(0, "/home/scott/naturelm-cats")
from NatureLM.models.beats.BEATs import BEATs, BEATsConfig

LLAMA_SERVER = "http://127.0.0.1:9002"
CLASSIFIER_PATH = Path("/home/scott/datasets/cats/context_classifier.pt")
NLM_CKPT_PATH = Path("/home/scott/models/naturelm-audio/model.safetensors")


# Reuse BEATs builder from extract_beats_features (inlined for self-containment)
def build_beats() -> BEATs:
    cfg_dict = {
        "input_patch_size": 16, "embed_dim": 512, "conv_bias": False,
        "encoder_layers": 12, "encoder_embed_dim": 768, "encoder_ffn_embed_dim": 3072,
        "encoder_attention_heads": 12, "activation_fn": "gelu",
        "layer_wise_gradient_decay_ratio": 0.6, "layer_norm_first": False,
        "deep_norm": True, "dropout": 0.0, "attention_dropout": 0.0,
        "activation_dropout": 0.0, "encoder_layerdrop": 0.05, "dropout_input": 0.0,
        "conv_pos": 128, "conv_pos_groups": 16, "relative_position_embedding": True,
        "num_buckets": 320, "max_distance": 800, "gru_rel_pos": True,
        "finetuned_model": True, "predictor_dropout": 0.0, "predictor_class": 527,
    }
    beats = BEATs(cfg=BEATsConfig(cfg_dict))
    full = load_file(str(NLM_CKPT_PATH))
    beats_state = {k[len("beats."):]: v for k, v in full.items() if k.startswith("beats.")}
    beats.load_state_dict(beats_state, strict=False)
    beats.eval()
    return beats


class ContextHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_classes),
        )
    def forward(self, x): return self.net(x)


def classify_context(audio_path: Path, beats: BEATs, classifier: nn.Module,
                     mu: torch.Tensor, sd: torch.Tensor, label_names: dict[int, str]) -> dict:
    audio, sr = sf.read(str(audio_path))
    if audio.ndim == 2:
        audio = audio.mean(axis=int(np.argmin(audio.shape)))
    if sr != 16000:
        audio = resampy.resample(audio.astype(np.float32), sr, 16000)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

    with torch.inference_mode():
        wav = torch.from_numpy(audio).unsqueeze(0)
        feats, _ = beats(wav)             # [1, T, 768]
        pooled = feats.mean(dim=1)        # [1, 768]
        pooled = (pooled - mu) / sd       # z-score normalize using training stats
        logits = classifier(pooled)       # [1, 3]
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    ranked = sorted([(label_names[i], float(probs[i])) for i in range(len(probs))],
                    key=lambda x: -x[1])
    return {"probabilities": dict(ranked), "top_label": ranked[0][0], "top_prob": ranked[0][1]}


def caption_with_naturelm(audio_path: Path, query: str) -> str:
    from NatureLM.infer import Pipeline
    print("[caption] Loading NatureLM-audio (CPU)...", file=sys.stderr)
    t0 = time.time()
    pipeline = Pipeline()
    print(f"[caption] Loaded in {time.time() - t0:.1f}s", file=sys.stderr)
    t1 = time.time()
    results = pipeline([str(audio_path)], [query], window_length_seconds=10.0, hop_length_seconds=10.0)
    print(f"[caption] Inference in {time.time() - t1:.1f}s", file=sys.stderr)
    return results[0] if isinstance(results, list) else str(results)


def reason_with_gpt_oss(caption: str, context_result: dict, user_question: str) -> str:
    probs_pct = {k: f"{v:.1%}" for k, v in context_result["probabilities"].items()}
    context_summary = (
        f"Most likely: {context_result['top_label']} ({context_result['top_prob']:.1%}). "
        f"Full distribution: {probs_pct}. "
        f"Classifier was trained on CatMeows (3 contexts: brushing, isolation, waiting-for-food). "
        f"If top probability < 50%, treat as out-of-distribution."
    )
    messages = [
        {"role": "system", "content": (
            "You are a bioacoustics expert. The user gives you (1) a generic audio caption "
            "from a bioacoustic foundation model, (2) probabilities from a cat-context classifier "
            "trained on the CatMeows dataset, and (3) a question. Reason carefully — use both "
            "signals, and flag uncertainty when the classifier is not confident."
        )},
        {"role": "user", "content": (
            f"Generic caption: {caption!r}\n\n"
            f"Cat context classifier output: {context_summary}\n\n"
            f"Question: {user_question}"
        )},
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
    p.add_argument("audio", type=Path)
    p.add_argument("--caption-query", default="Caption the audio. What animal vocalization is this, if any?")
    p.add_argument("--reason-question", default="What is this cat likely communicating? What is the behavioral context?")
    args = p.parse_args()

    if not args.audio.exists():
        print(f"FAIL: {args.audio} not found")
        return 1

    try:
        requests.get(f"{LLAMA_SERVER}/health", timeout=3).raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"FAIL: llama-server not reachable at {LLAMA_SERVER} ({e})")
        return 1

    print("[ctx] Loading BEATs + context classifier...", file=sys.stderr)
    t0 = time.time()
    beats = build_beats()
    ckpt = torch.load(CLASSIFIER_PATH, weights_only=False)
    cfg = ckpt["config"]
    classifier = ContextHead(cfg["in_dim"], cfg["hidden"], cfg["n_classes"], cfg["dropout"])
    classifier.load_state_dict(ckpt["state_dict"])
    classifier.eval()
    mu = ckpt["feature_mu"]
    sd = ckpt["feature_sd"]
    label_names = ckpt["label_names"]
    print(f"[ctx] Loaded in {time.time() - t0:.1f}s. Train test acc was {ckpt['best_test_acc']:.3f}.", file=sys.stderr)

    print("[ctx] Running classifier...", file=sys.stderr)
    ctx = classify_context(args.audio, beats, classifier, mu, sd, label_names)
    print(f"\n=== Cat-context classifier ===\nTop: {ctx['top_label']} ({ctx['top_prob']:.1%})")
    for k, v in ctx["probabilities"].items():
        print(f"  {k}: {v:.1%}")

    caption = caption_with_naturelm(args.audio, args.caption_query)
    print(f"\n=== NatureLM-audio caption ===\n{caption}")

    answer = reason_with_gpt_oss(caption, ctx, args.reason_question)
    print(f"\n=== GPT-OSS-20B reasoning ===\n{answer}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
