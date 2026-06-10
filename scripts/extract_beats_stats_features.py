# SPDX-License-Identifier: MIT
"""Extract BEATs features with statistical pooling (mean + std + max).

Same audio path as extract_beats_features.py, but instead of mean-only pooling
over time, compute three pooling statistics and concatenate:
    [mean_t(x), std_t(x), max_t(x)] -> 3 × 768 = 2304-dim feature per clip

Hypothesis: mean-pooling flattens temporal dynamics that distinguish cat
contexts (food-meow's sharp onset vs brushing-meow's gentler envelope).
Adding std + max preserves variability + peak information.

Outputs:
    /home/scott/datasets/cats/stats_features_train.npy  (201, 2304)
    /home/scott/datasets/cats/stats_labels_train.npy    (201,)
    /home/scott/datasets/cats/stats_features_test.npy   (75, 2304)
    /home/scott/datasets/cats/stats_labels_test.npy     (75,)
"""
import io
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pyarrow.parquet as pq
import resampy
import soundfile as sf
import torch
from safetensors.torch import load_file

sys.path.insert(0, "/home/scott/naturelm-cats")
from NatureLM.models.beats.BEATs import BEATs, BEATsConfig

CKPT_PATH = Path("/home/scott/models/naturelm-audio/model.safetensors")
PARQUET_DIR = Path("/home/scott/datasets/cats/openfarm-catmeows/data")
OUT_DIR = Path("/home/scott/datasets/cats")


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
    full = load_file(str(CKPT_PATH))
    beats_state = {k[len("beats."):]: v for k, v in full.items() if k.startswith("beats.")}
    beats.load_state_dict(beats_state, strict=False)
    beats.eval()
    return beats


def encode_split(beats: BEATs, parquet_path: Path, label_map: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    df = pq.read_table(parquet_path).to_pandas()
    n = len(df)
    feats = np.zeros((n, 768 * 3), dtype=np.float32)
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
            wav = torch.from_numpy(audio).unsqueeze(0)
            x, _ = beats(wav)              # [1, T, 768]
            x = x.squeeze(0)               # [T, 768]
            mean_p = x.mean(dim=0)         # [768]
            std_p = x.std(dim=0)           # [768]
            max_p = x.max(dim=0).values    # [768]
            pooled = torch.cat([mean_p, std_p, max_p], dim=0).cpu().numpy()  # [2304]

        feats[i] = pooled
        labels[i] = label_map[row["context"]]

        if (i + 1) % 25 == 0 or i == n - 1:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n} clips ({elapsed:.1f}s)")

    return feats, labels


def main() -> int:
    print("Building BEATs encoder...")
    beats = build_beats()

    label_names = json.loads((OUT_DIR / "label_names.json").read_text())
    label_map = {v: int(k) for k, v in label_names.items()}
    print(f"Label map: {label_map}")

    for split in ["train", "test"]:
        print(f"\n=== {split} ===")
        feats, labels = encode_split(beats, PARQUET_DIR / f"{split}-00000-of-00001.parquet", label_map)
        np.save(OUT_DIR / f"stats_features_{split}.npy", feats)
        np.save(OUT_DIR / f"stats_labels_{split}.npy", labels)
        print(f"Saved: features {feats.shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
