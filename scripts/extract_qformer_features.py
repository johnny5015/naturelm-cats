"""Extract Q-Former features (post-cross-attention, pre-Llama-projection) for CatMeows.

Pipeline replicates NatureLM-audio's encode_audio() up to but not including audio_llama_proj:
    raw audio
        ↓ pad to 10s @ 16 kHz
    BEATs encoder → [1, T_patches=~500, 768]
        ↓ ln_audio LayerNorm
    audio_embeds
        ↓ unfold to overlapping windows (kernel=17, stride=17)
    [num_windows, 17, 768]
        ↓ Q-Former cross-attention with learnable query_tokens
    [num_windows, 1, 768]   (1 query token per window per inference.yml)
        ↓ reshape + mean-pool over num_windows
    [768]   ← final feature per clip

Outputs (parallel structure to extract_beats_features.py):
    /home/scott/datasets/cats/qformer_features_train.npy  (201, 768)
    /home/scott/datasets/cats/qformer_labels_train.npy    (201,)
    /home/scott/datasets/cats/qformer_features_test.npy   (75, 768)
    /home/scott/datasets/cats/qformer_labels_test.npy     (75,)
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
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import BertConfig

sys.path.insert(0, "/home/scott/naturelm-cats")
from NatureLM.models.beats.BEATs import BEATs, BEATsConfig
from NatureLM.models.Qformer import BertLMHeadModel

CKPT_PATH = Path("/home/scott/models/naturelm-audio/model.safetensors")
PARQUET_DIR = Path("/home/scott/datasets/cats/openfarm-catmeows/data")
OUT_DIR = Path("/home/scott/datasets/cats")

# From NatureLM-audio's inference.yml
SAMPLE_RATE = 16000
MAX_LENGTH_SECONDS = 10
NUM_AUDIO_QUERY_TOKEN = 1
SECOND_PER_WINDOW = 0.333333
SECOND_STRIDE = 0.333333


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
    return beats, full


def build_qformer(num_query_token: int, audio_width: int, num_hidden_layers: int = 2):
    """Mirrors NatureLM.init_audio_Qformer()."""
    encoder_config = BertConfig.from_pretrained("bert-base-uncased")
    encoder_config.num_hidden_layers = num_hidden_layers
    encoder_config.encoder_width = audio_width
    encoder_config.add_cross_attention = True
    encoder_config.cross_attention_freq = 1
    encoder_config.query_length = num_query_token
    qformer = BertLMHeadModel(config=encoder_config)
    query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
    query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
    return qformer, query_tokens, encoder_config


def load_qformer_weights(qformer: nn.Module, query_tokens: nn.Parameter,
                         ln_audio: nn.LayerNorm, full_state: dict) -> None:
    qformer_state = {k[len("audio_Qformer."):]: v for k, v in full_state.items()
                     if k.startswith("audio_Qformer.")}
    missing, unexpected = qformer.load_state_dict(qformer_state, strict=False)
    print(f"  Q-Former: loaded {len(qformer_state)} tensors "
          f"({len(missing)} missing, {len(unexpected)} unexpected)")

    if "audio_query_tokens" in full_state:
        query_tokens.data.copy_(full_state["audio_query_tokens"])
        print(f"  audio_query_tokens loaded: shape {query_tokens.shape}")
    else:
        print("  WARNING: audio_query_tokens not found in checkpoint, using random init")

    ln_state = {k[len("ln_audio."):]: v for k, v in full_state.items() if k.startswith("ln_audio.")}
    ln_audio.load_state_dict(ln_state)
    print(f"  ln_audio loaded: weight shape {ln_audio.weight.shape}")


def encode_clip_qformer(
    audio: np.ndarray,
    beats: BEATs,
    qformer: nn.Module,
    query_tokens: nn.Parameter,
    ln_audio: nn.LayerNorm,
) -> np.ndarray:
    """Returns mean-pooled Q-Former feature of shape (768,)."""
    with torch.inference_mode():
        wav = torch.from_numpy(audio).unsqueeze(0).float()  # [1, T]

        # BEATs → [1, T_patches, 768]
        audio_embeds, _ = beats(wav)

        # LayerNorm
        audio_embeds = ln_audio(audio_embeds)

        # Sliding-window unfold (replicating NatureLM.py L434-455)
        B, T, C = audio_embeds.shape
        kernel = round(1500 * SECOND_PER_WINDOW / 30.0)
        stride = round(1500 * SECOND_STRIDE / 30.0)
        kernel_2d = (1, kernel)
        stride_2d = (1, stride)

        audio_embeds_tr = audio_embeds.transpose(1, 2).unsqueeze(2)  # [B, C, 1, T]
        audio_embeds_overlap = F.unfold(audio_embeds_tr, kernel_size=kernel_2d, stride=stride_2d)
        _, _, L = audio_embeds_overlap.shape  # L = num_windows
        audio_embeds_overlap = audio_embeds_overlap.view(B, C, kernel, L)
        audio_embeds_overlap = audio_embeds_overlap.permute(0, 3, 2, 1)  # [B, L, kernel, C]
        audio_embeds_windows = audio_embeds_overlap.reshape(-1, kernel, C)  # [B*L, kernel, C]
        audio_atts = torch.ones(audio_embeds_windows.size()[:-1], dtype=torch.long)

        # Q-Former cross-attention
        n_windows = audio_embeds_windows.shape[0]
        query_in = query_tokens.expand(n_windows, -1, -1)  # [B*L, 1, 768]
        q_out = qformer.bert(
            query_embeds=query_in,
            encoder_hidden_states=audio_embeds_windows,
            encoder_attention_mask=audio_atts,
            return_dict=True,
        )
        # q_out.last_hidden_state: [B*L, num_query_tokens=1, 768]
        per_window_feat = q_out.last_hidden_state.squeeze(1)  # [B*L, 768]
        pooled = per_window_feat.mean(dim=0).cpu().numpy()    # [768]

    return pooled.astype(np.float32)


def main() -> int:
    print("Building BEATs + Q-Former + ln_audio + query_tokens...")
    beats, full_state = build_beats()
    qformer, query_tokens, qcfg = build_qformer(NUM_AUDIO_QUERY_TOKEN, audio_width=768)
    ln_audio = nn.LayerNorm(768)
    load_qformer_weights(qformer, query_tokens, ln_audio, full_state)
    qformer.eval()
    print(f"Q-Former hidden_size = {qcfg.hidden_size}")

    label_map_path = OUT_DIR / "label_names.json"
    if label_map_path.exists():
        label_names = json.loads(label_map_path.read_text())
        label_map = {v: int(k) for k, v in label_names.items()}
    else:
        # Build from train parquet (deterministic sort)
        train_ctx = sorted(pq.read_table(PARQUET_DIR / "train-00000-of-00001.parquet")
                          .to_pandas()["context"].unique())
        label_map = {c: i for i, c in enumerate(train_ctx)}
    print(f"Label map: {label_map}")

    for split in ["train", "test"]:
        print(f"\n=== {split} ===")
        df = pq.read_table(PARQUET_DIR / f"{split}-00000-of-00001.parquet").to_pandas()
        n = len(df)
        feats = np.zeros((n, qcfg.hidden_size), dtype=np.float32)
        labels = np.zeros((n,), dtype=np.int64)

        t0 = time.time()
        target_len = SAMPLE_RATE * MAX_LENGTH_SECONDS
        for i, row in df.iterrows():
            audio_bytes = row["audio"]["bytes"]
            audio, sr = sf.read(io.BytesIO(audio_bytes))
            if audio.ndim == 2:
                audio = audio.mean(axis=int(np.argmin(audio.shape)))
            if sr != SAMPLE_RATE:
                audio = resampy.resample(audio.astype(np.float32), sr, SAMPLE_RATE)
            audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
            # Pad/truncate to 10s
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            else:
                audio = audio[:target_len]

            feats[i] = encode_clip_qformer(audio, beats, qformer, query_tokens, ln_audio)
            labels[i] = label_map[row["context"]]

            if (i + 1) % 25 == 0 or i == n - 1:
                elapsed = time.time() - t0
                print(f"  {i+1}/{n} clips ({elapsed:.1f}s, {(i+1)/elapsed:.1f} clips/s)")

        np.save(OUT_DIR / f"qformer_features_{split}.npy", feats)
        np.save(OUT_DIR / f"qformer_labels_{split}.npy", labels)
        print(f"Saved: features {feats.shape}, labels {labels.shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
