# naturelm-cats

**Extending [NatureLM-audio](https://github.com/earthspecies/NatureLM-audio) (Earth Species Project) to cat vocalization understanding, with a more powerful reasoning LLM bridge.**

A soft fork of `earthspecies/NatureLM-audio` that adds:

1. A **bridge from NatureLM-audio to GPT-OSS-20B** via `llama-server` — getting structured behavioral reasoning on top of NatureLM's bioacoustic captions, running on a single consumer GPU (RTX 4070 Laptop, 8 GB).
2. A **cat-vocalization context classifier** trained on BEATs features from [CatMeows](https://zenodo.org/records/4008297) (Ludovico et al. 2020) — predicting *brushing*, *isolation*, or *waiting-for-food* context from raw audio.

Both pieces work end-to-end against arbitrary cat audio today.

> The original NatureLM-audio README is preserved at [`README_UPSTREAM.md`](README_UPSTREAM.md).

---

## Pipeline

```
audio (any cat sound, wav/mp3/flac/ogg)
  │
  ├─► NatureLM-audio (CPU)
  │     BEATs + Q-Former + Llama-3.1-8B + LoRA
  │     → generic bioacoustic caption: "Domestic cats vocalizing."
  │
  └─► BEATs encoder + trained MLP head (CPU)
        → cat-context softmax:
          {waiting_for_food: 60.3%, brushing: 25.6%, isolation: 14.2%}
  │
  ▼  merge both signals
GPT-OSS-20B Q4_K_M (RTX 4070, llama-server :9002)
  → structured behavioral analysis with confidence calibration
```

### Sample output (ESC-50 cat clip)

> *Generic caption:* `"Domestic cats vocalizing."`
> *Classifier:* `waiting_for_food (60.3%), brushing (25.6%), isolation (14.2%)`
>
> **GPT-OSS-20B reasoning:** "The cat's vocalization most probably signals **hunger or a request for food**. In the 'waiting-for-food' context, cats typically produce a clear, sustained meow... Because the top probability is comfortably above the 50% threshold, we can treat this as a confident prediction... Caveat: the remaining 39.7% of probability mass is spread over other contexts, so there is still a non-negligible chance that the cat is simply meowing for attention."

---

## Why bridge instead of fine-tune?

NatureLM-audio is pretrained for general bioacoustics across many species. Replacing its Llama-3.1-8B head with a larger reasoner (GPT-OSS-20B) would require retraining the audio→LLM projection, which is expensive. Bridging is cheaper and modular:

- **NatureLM-audio** stays untouched → keeps its broad bioacoustic prior
- **Small classifier head** adds domain-specific grounding (cats) with ~200K trainable params
- **GPT-OSS-20B** does the heavy reasoning over both caption + context probabilities

On an 8 GB consumer GPU this works because GPT-OSS-20B is MoE: only ~3.6B params active per token, so `llama.cpp -ngl 12` with Q4_K_M fits the always-on backbone on the GPU while expert weights stream from system RAM. We measured **32 tok/s generation** on RTX 4070 Laptop.

---

## Results

Cat-context classifier on [CatMeows](https://zenodo.org/records/4008297) held-out test set (75 clips, 25 per class):

| Metric | Value |
|---|---|
| Test accuracy | **65.3%** (chance = 33%) |
| Best class (waiting-for-food) | 72% (18/25) |
| Brushing | 64% (16/25) |
| Isolation | 60% (15/25) |

**Confusion matrix:**

|  | pred: brushing | pred: isolation | pred: food |
|---|---:|---:|---:|
| **true: brushing** | 16 | 3 | 6 |
| **true: isolation** | 7 | 15 | 3 |
| **true: food** | 4 | 3 | 18 |

The Ludovico et al. 2020 paper reports 86% on the same dataset using hand-crafted features + SVM with full-dataset cross-validation. We use only learned BEATs features and a strict 73%/27% hold-out split — different evaluation protocol, weaker model, deliberately small (~200K-param) classifier head.

**Most informative confusion:** brushing-isolation reciprocal errors (7 + 3 = 10/50 swaps) — gentle non-distressed meows look similar across both. Food-context meows are most distinguishable (sharper, more urgent acoustic signature).

---

## Install

Prereqs:
- HuggingFace auth with access to `meta-llama/Meta-Llama-3.1-8B-Instruct` (gated)
- `uv` for Python env management
- llama.cpp built with CUDA (we used build 7787+)
- ~30 GB disk for model checkpoints

```bash
git clone https://github.com/Scottcjn/naturelm-cats
cd naturelm-cats
uv sync --no-group gpu      # or "uv sync" if you want CUDA torch
```

Download required models:

```bash
# GPT-OSS-20B GGUF (~11 GB)
hf download unsloth/gpt-oss-20b-GGUF gpt-oss-20b-Q4_K_M.gguf \
  --local-dir ~/models/gpt-oss-20b-gguf

# NatureLM-audio head (~1.5 GB)
hf download EarthSpeciesProject/NatureLM-audio --local-dir ~/models/naturelm-audio

# Llama-3.1-8B base (NatureLM-audio's LoRA backbone, ~16 GB)
hf download meta-llama/Meta-Llama-3.1-8B-Instruct --exclude "original/*"

# CatMeows dataset (for training the classifier yourself, ~18 MB)
hf download oliveirabruno01/openfarm-catmeows --repo-type dataset \
  --local-dir ~/datasets/cats/openfarm-catmeows
```

---

## Run

Start the LLM server (terminal 1):

```bash
~/llama.cpp/build-cuda/bin/llama-server \
  --model ~/models/gpt-oss-20b-gguf/gpt-oss-20b-Q4_K_M.gguf \
  --host 127.0.0.1 --port 9002 \
  --n-gpu-layers 12 --ctx-size 4096 --threads 8
```

Run the bridge against any cat audio (terminal 2):

```bash
# Caption-only bridge (no trained classifier)
CUDA_VISIBLE_DEVICES="" uv run python scripts/bridge_caption.py path/to/cat.wav

# Full bridge with trained context classifier
CUDA_VISIBLE_DEVICES="" uv run python scripts/bridge_with_context.py path/to/cat.wav
```

`CUDA_VISIBLE_DEVICES=""` forces NatureLM-audio to CPU so it doesn't fight llama-server for VRAM. ~60s cold start (Llama-3.1-8B loads into 16 GB of system RAM), <10s warm.

---

## Train the classifier yourself

```bash
# 1. Extract BEATs features for train+test (CPU, ~20s for 276 clips)
CUDA_VISIBLE_DEVICES="" uv run python scripts/extract_beats_features.py

# 2. Train the MLP head (CPU, <1s)
uv run python scripts/train_context_classifier.py
```

Outputs `~/datasets/cats/context_classifier.pt` along with normalization stats and label map. Tuned hyperparameters (hidden=128, dropout=0.7, weight_decay=0.1) are baked into the script — found by a 80-config grid sweep across hidden ∈ {0, 32, 64, 128, 256}, dropout ∈ {0, 0.3, 0.5, 0.7}, weight_decay ∈ {1e-4, 1e-3, 1e-2, 1e-1}.

---

## Files added on top of upstream

```
scripts/
├── bridge_caption.py              v1 bridge: NatureLM caption → GPT-OSS-20B
├── bridge_with_context.py         v2 bridge: + cat-context classifier
├── extract_beats_features.py      BEATs feature extraction over CatMeows
├── train_context_classifier.py    MLP head training + hyperparameter config
└── smoke_test_gpt_oss.py          standalone GPT-OSS-20B load test (transformers path)
```

Nothing in `NatureLM/` is modified — this is purely additive on top of upstream `earthspecies/NatureLM-audio`.

---

## Licensing

- **This repo's added code** (under `scripts/`): Apache 2.0, same as upstream
- **NatureLM-audio model weights** (used at runtime): CC-BY-NC-SA 4.0 per [EarthSpeciesProject/NatureLM-audio](https://huggingface.co/EarthSpeciesProject/NatureLM-audio) — our derivative trained classifier inherits the NC restriction
- **CatMeows dataset**: CC-BY-4.0 per [Zenodo record 4008297](https://zenodo.org/records/4008297)
- **GPT-OSS-20B**: Apache 2.0 per OpenAI release
- **Llama 3.1**: Llama 3.1 Community License (separate, gated access on HF)

If you want commercial use of the trained cat classifier, contact Earth Species Project regarding their NC-licensed weights — the classifier head itself is small enough (~200K params) to retrain on different upstream features.

---

## Limitations

- **3 contexts only** (brushing / isolation / waiting-for-food). Real cats produce many more vocalization types; the classifier will force-fit out-of-distribution audio into one of these three. Use the softmax confidence as your uncertainty signal.
- **65% test accuracy** is a proof of concept, below the 86% reported in the original Ludovico et al. paper. Improvements ahead: larger backbone projection layer, time-distributed attention pooling instead of mean-pool, more data.
- **No per-cat personalization**: the classifier was trained on 21 cats from CatMeows. Individual cats have idiosyncratic dialects; a per-cat embedding or fine-tuning on a few labeled clips of your specific cat would help substantially.
- **No reverse direction**: we listen to the cat, we don't generate audio the cat would recognize. Cat-directed speech synthesis is a separate, harder problem.

---

## Acknowledgements

This project is a soft fork of [earthspecies/NatureLM-audio](https://github.com/earthspecies/NatureLM-audio). The audio-language foundation model, BEATs encoder fine-tune, and the entire bioacoustic pretraining stack are their work.

- **NatureLM-audio**: Robinson et al., *NatureLM-audio: an Audio-Language Foundation Model for Bioacoustics*, ICLR 2025. [paper](https://openreview.net/forum?id=hJVdwBpWjt)
- **CatMeows dataset**: Ludovico et al., *CatMeows: A Publicly-Available Dataset of Cat Vocalizations*, Animals 2020. [DOI 10.5281/zenodo.4008297](https://zenodo.org/records/4008297)
- **GPT-OSS-20B**: OpenAI open-weights release
- **Earth Species Project's mission**: decoding non-human communication for science and conservation. The bridge architecture here is a small extension to consumer hardware; their foundation models do the actual heavy lifting.
