"""Extract classical acoustic features per CatMeows clip.

Per clip, computes (all via librosa) and pools mean + std over time:
    - MFCC (13 coefficients) + delta + delta-delta → 26+26+26 = 78 features
    - Zero crossing rate                            → 2 features
    - Spectral centroid                             → 2 features
    - Spectral bandwidth                            → 2 features
    - Spectral rolloff                              → 2 features
    - Spectral contrast (7 bands)                   → 14 features
    - RMS energy                                    → 2 features
    Total: 102 features per clip

Outputs:
    /home/scott/datasets/cats/classical_features_train.npy  (201, 102)
    /home/scott/datasets/cats/classical_features_test.npy   (75, 102)
    (labels reuse labels_{train,test}.npy from BEATs run)

Hypothesis: Ludovico 2020 achieved 86% with hand-crafted features; combining
with learned BEATs-stats should give better than either alone.
"""
import io
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pyarrow.parquet as pq
import resampy
import soundfile as sf

PARQUET_DIR = Path("/home/scott/datasets/cats/openfarm-catmeows/data")
OUT_DIR = Path("/home/scott/datasets/cats")
SAMPLE_RATE = 16000


def classical_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Returns (102,) feature vector."""
    n_fft = 1024
    hop = 256

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop)
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=n_fft, hop_length=hop)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop)
    bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=n_fft, hop_length=hop)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=n_fft, hop_length=hop)
    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop)

    # Pool each feature stream by mean + std over time
    def pool(x: np.ndarray) -> np.ndarray:
        return np.concatenate([x.mean(axis=-1), x.std(axis=-1)])

    feats = np.concatenate([
        pool(mfcc),       # 26
        pool(delta),      # 26
        pool(delta2),     # 26
        pool(zcr),        # 2
        pool(centroid),   # 2
        pool(bw),         # 2
        pool(rolloff),    # 2
        pool(contrast),   # 14 (7 bands × 2 stats)
        pool(rms),        # 2
    ])
    return feats.astype(np.float32)


def encode_split(parquet_path: Path) -> np.ndarray:
    df = pq.read_table(parquet_path).to_pandas()
    n = len(df)
    # First clip gives us the actual feature size
    sample_feat_size = None
    feats = None

    t0 = time.time()
    for i, row in df.iterrows():
        audio_bytes = row["audio"]["bytes"]
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        if audio.ndim == 2:
            audio = audio.mean(axis=int(np.argmin(audio.shape)))
        if sr != SAMPLE_RATE:
            audio = resampy.resample(audio.astype(np.float32), sr, SAMPLE_RATE)
        audio = np.clip(audio.astype(np.float32), -1.0, 1.0)

        f = classical_features(audio)
        if feats is None:
            sample_feat_size = len(f)
            feats = np.zeros((n, sample_feat_size), dtype=np.float32)
        feats[i] = f

        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n} clips ({elapsed:.1f}s, {(i+1)/elapsed:.1f} clips/s)")

    print(f"  Feature dim: {sample_feat_size}")
    return feats


def main() -> int:
    for split in ["train", "test"]:
        print(f"\n=== {split} ===")
        feats = encode_split(PARQUET_DIR / f"{split}-00000-of-00001.parquet")
        np.save(OUT_DIR / f"classical_features_{split}.npy", feats)
        print(f"Saved: {feats.shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
