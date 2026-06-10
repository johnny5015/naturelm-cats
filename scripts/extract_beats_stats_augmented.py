# SPDX-License-Identifier: MIT
"""Extract BEATs statistical-pool features WITH AUGMENTATIONS.

Per source clip, produces 5 feature vectors:
    aug 0: original audio
    aug 1: time-stretch ∈ [0.9, 1.1] + light noise
    aug 2: pitch-shift ∈ [-2, +2] semitones
    aug 3: combined time-stretch + pitch-shift + gain
    aug 4: combined + light room-style noise

All five share the same context label and "group id" = source clip index.
Group ID is used in StratifiedGroupKFold to keep augmentations of a clip
in the same fold (no train/test leakage).

Outputs:
    aug_features_train.npy   (201 * 5, 2304)
    aug_labels_train.npy     (201 * 5,)
    aug_groups_train.npy     (201 * 5,)  — clip index for group-aware CV
    aug_features_test.npy    (75 * 5, 2304)
    aug_labels_test.npy      (75 * 5,)
    aug_groups_test.npy      (75 * 5,)

Test set is also augmented — useful for inspection/test-time-augmentation
but during 5-fold CV we'll combine train+test and re-split by group.
"""
import io
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import librosa
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

SAMPLE_RATE = 16000
TARGET_LEN = 10 * SAMPLE_RATE
N_AUG = 5  # 1 original + 4 augmented
SEED = 13


def build_beats():
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


def pad_or_truncate(audio: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
    if len(audio) < target_len:
        return np.pad(audio, (0, target_len - len(audio)))
    return audio[:target_len]


def augment(audio: np.ndarray, aug_id: int, rng: np.random.Generator) -> np.ndarray:
    """Augmentation aug_id ∈ {0..4}. 0 = passthrough."""
    sr = SAMPLE_RATE
    if aug_id == 0:
        return audio
    if aug_id == 1:
        rate = rng.uniform(0.9, 1.1)
        out = librosa.effects.time_stretch(audio, rate=rate)
        # Light white noise (SNR ~25 dB)
        rms = np.sqrt(np.mean(out ** 2))
        if rms > 0:
            noise = rng.standard_normal(len(out)).astype(np.float32) * (rms * 10 ** (-25 / 20))
            out = out + noise
        return out
    if aug_id == 2:
        n_steps = rng.uniform(-2, 2)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    if aug_id == 3:
        out = librosa.effects.time_stretch(audio, rate=rng.uniform(0.9, 1.1))
        out = librosa.effects.pitch_shift(out, sr=sr, n_steps=rng.uniform(-2, 2))
        gain_db = rng.uniform(-3, 3)
        out = out * (10 ** (gain_db / 20))
        return np.clip(out, -1.0, 1.0)
    if aug_id == 4:
        out = librosa.effects.time_stretch(audio, rate=rng.uniform(0.9, 1.1))
        out = librosa.effects.pitch_shift(out, sr=sr, n_steps=rng.uniform(-2, 2))
        # "Room" noise: pink-ish via 1/f white-noise filter
        n = len(out)
        white = rng.standard_normal(n).astype(np.float32)
        # Cheap pink approximation: cumulative running sum, then high-pass via diff
        pink = np.cumsum(white) / n
        pink = pink - np.mean(pink)
        pink_std = np.std(pink) or 1.0
        pink = pink / pink_std
        rms = np.sqrt(np.mean(out ** 2))
        if rms > 0:
            snr_db = rng.uniform(18, 28)
            out = out + pink * rms * 10 ** (-snr_db / 20)
        return np.clip(out, -1.0, 1.0)
    return audio


def beats_stats(audio: np.ndarray, beats: BEATs) -> np.ndarray:
    with torch.inference_mode():
        wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        x, _ = beats(wav)
        x = x.squeeze(0)
        pooled = torch.cat([x.mean(0), x.std(0), x.max(0).values], dim=0)
    return pooled.cpu().numpy().astype(np.float32)


def encode_split(beats: BEATs, parquet_path: Path, label_map: dict[str, int], split_offset: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pq.read_table(parquet_path).to_pandas()
    n = len(df)
    feats = np.zeros((n * N_AUG, 2304), dtype=np.float32)
    labels = np.zeros((n * N_AUG,), dtype=np.int64)
    groups = np.zeros((n * N_AUG,), dtype=np.int64)
    rng = np.random.default_rng(SEED + split_offset)
    target = TARGET_LEN

    t0 = time.time()
    for i, row in df.iterrows():
        audio_bytes = row["audio"]["bytes"]
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        if audio.ndim == 2:
            audio = audio.mean(axis=int(np.argmin(audio.shape)))
        if sr != SAMPLE_RATE:
            audio = resampy.resample(audio.astype(np.float32), sr, SAMPLE_RATE)
        audio = np.clip(audio.astype(np.float32), -1.0, 1.0)

        for k in range(N_AUG):
            aug_audio = augment(audio, k, rng)
            aug_audio = pad_or_truncate(aug_audio)
            feats[i * N_AUG + k] = beats_stats(aug_audio, beats)
            labels[i * N_AUG + k] = label_map[row["context"]]
            groups[i * N_AUG + k] = split_offset + i  # unique per source clip across splits

        if (i + 1) % 25 == 0 or i == n - 1:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n} clips × {N_AUG} augs ({elapsed:.1f}s, "
                  f"{(i+1) * N_AUG / elapsed:.1f} aug-clips/s)")

    return feats, labels, groups


def main() -> int:
    print(f"Building BEATs encoder ({N_AUG} augs/clip)...")
    beats = build_beats()

    label_names = json.loads((OUT_DIR / "label_names.json").read_text())
    label_map = {v: int(k) for k, v in label_names.items()}
    print(f"Label map: {label_map}")

    # Different group offsets so train + test clips have distinct group IDs
    n_train = pq.read_table(PARQUET_DIR / "train-00000-of-00001.parquet").num_rows
    for split, offset in [("train", 0), ("test", n_train)]:
        print(f"\n=== {split} (offset {offset}) ===")
        feats, labels, groups = encode_split(beats,
                                             PARQUET_DIR / f"{split}-00000-of-00001.parquet",
                                             label_map, offset)
        np.save(OUT_DIR / f"aug_features_{split}.npy", feats)
        np.save(OUT_DIR / f"aug_labels_{split}.npy", labels)
        np.save(OUT_DIR / f"aug_groups_{split}.npy", groups)
        print(f"Saved: features {feats.shape}, labels {labels.shape}, groups {groups.shape}")
        print(f"  unique groups in this split: {len(set(groups))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
