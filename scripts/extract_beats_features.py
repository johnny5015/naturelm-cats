# SPDX-License-Identifier: MIT
"""Extract BEATs features for openfarm-catmeows train+test parquet rows.

BEATs encoder (loaded from NatureLM-audio's checkpoint, beats.* prefix subset).
For each row: decode audio bytes → resample 8→16 kHz → BEATs.forward → mean-pool over time.

Output:
    /home/scott/datasets/cats/features_train.npy  (201, 768)
    /home/scott/datasets/cats/labels_train.npy    (201,)
    /home/scott/datasets/cats/features_test.npy   (75, 768)
    /home/scott/datasets/cats/labels_test.npy     (75,)
    /home/scott/datasets/cats/label_names.json    {0: "brushing", 1: "isolation_...", 2: "waiting_for_food"}

Runs on CPU to avoid VRAM contention with llama-server.
"""
import io
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only

import numpy as np
import pyarrow.parquet as pq
import resampy
import soundfile as sf
import torch
from safetensors.torch import load_file

# Reuse NatureLM-audio's BEATs implementation
sys.path.insert(0, "/home/scott/naturelm-cats")
from NatureLM.models.beats.BEATs import BEATs, BEATsConfig

CKPT_PATH = Path("/home/scott/models/naturelm-audio/model.safetensors")
PARQUET_DIR = Path("/home/scott/datasets/cats/openfarm-catmeows/data")
OUT_DIR = Path("/home/scott/datasets/cats")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_beats() -> BEATs:
    """Construct BEATs and load weights from NatureLM-audio's safetensors."""
    cfg_dict = {
        "input_patch_size": 16,
        "embed_dim": 512,
        "conv_bias": False,
        "encoder_layers": 12,
        "encoder_embed_dim": 768,
        "encoder_ffn_embed_dim": 3072,
        "encoder_attention_heads": 12,
        "activation_fn": "gelu",
        "layer_wise_gradient_decay_ratio": 0.6,
        "layer_norm_first": False,
        "deep_norm": True,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "activation_dropout": 0.0,
        "encoder_layerdrop": 0.05,
        "dropout_input": 0.0,
        "conv_pos": 128,
        "conv_pos_groups": 16,
        "relative_position_embedding": True,
        "num_buckets": 320,
        "max_distance": 800,
        "gru_rel_pos": True,
        "finetuned_model": True,
        "predictor_dropout": 0.0,
        "predictor_class": 527,
    }
    beats = BEATs(cfg=BEATsConfig(cfg_dict))

    # Filter beats.* keys from the full checkpoint, strip the prefix
    full_state = load_file(str(CKPT_PATH))
    beats_state = {k[len("beats.") :]: v for k, v in full_state.items() if k.startswith("beats.")}
    print(f"Loading {len(beats_state)} BEATs tensors...")
    missing, unexpected = beats.load_state_dict(beats_state, strict=False)
    if missing:
        print(f"  missing (first 5): {missing[:5]}")
    if unexpected:
        print(f"  unexpected (first 5): {unexpected[:5]}")

    beats.eval()
    for p in beats.parameters():
        p.requires_grad = False
    return beats


def encode_split(beats: BEATs, parquet_path: Path, label_map: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    n = len(df)
    feats = np.zeros((n, 768), dtype=np.float32)
    labels = np.zeros((n,), dtype=np.int64)

    t0 = time.time()
    for i, row in df.iterrows():
        audio_bytes = row["audio"]["bytes"]
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        if audio.ndim == 2:
            audio = audio.mean(axis=int(np.argmin(audio.shape)))
        if sr != 16000:
            audio = resampy.resample(audio.astype(np.float32), sr, 16000)
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

        with torch.inference_mode():
            wav = torch.from_numpy(audio).unsqueeze(0)  # [1, T]
            x, _ = beats(wav)  # [1, T_patches, 768]
            pooled = x.mean(dim=1).squeeze(0).cpu().numpy()  # [768]

        feats[i] = pooled
        labels[i] = label_map[row["context"]]

        if (i + 1) % 25 == 0 or i == n - 1:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n} clips ({elapsed:.1f}s, {(i+1)/elapsed:.1f} clips/s)")

    return feats, labels


def main() -> int:
    print("Building BEATs encoder...")
    beats = build_beats()

    # Build label map from sorted unique contexts (deterministic)
    train_table = pq.read_table(PARQUET_DIR / "train-00000-of-00001.parquet")
    contexts = sorted(train_table.to_pandas()["context"].unique())
    label_map = {c: i for i, c in enumerate(contexts)}
    print(f"Label map: {label_map}")
    (OUT_DIR / "label_names.json").write_text(json.dumps({v: k for k, v in label_map.items()}, indent=2))

    for split in ["train", "test"]:
        print(f"\n=== {split} ===")
        feats, labels = encode_split(beats, PARQUET_DIR / f"{split}-00000-of-00001.parquet", label_map)
        np.save(OUT_DIR / f"features_{split}.npy", feats)
        np.save(OUT_DIR / f"labels_{split}.npy", labels)
        print(f"Saved {split}: features {feats.shape}, labels {labels.shape}")
        print(f"  label counts: {np.bincount(labels).tolist()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
