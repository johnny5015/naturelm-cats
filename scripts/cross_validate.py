# SPDX-License-Identifier: MIT
"""5-fold stratified cross-validation across three feature variants.

Combines train + test (276 total samples), 5 stratified folds (~55 test per fold).
Reports mean ± std accuracy. Tighter estimate of real CatMeows ceiling at ~±3%
instead of single-split ±6%.

Each variant uses the best hyperparams from prior grid sweeps.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

DATA = Path("/home/scott/datasets/cats")

VARIANTS = {
    "BEATs-mean (768)": {
        "feat_file": "features", "label_file": "labels",
        "hidden": 128, "dropout": 0.7, "wd": 1e-1,
    },
    "Q-Former (768)": {
        "feat_file": "qformer_features", "label_file": "qformer_labels",
        "hidden": 256, "dropout": 0.0, "wd": 1e-2,
    },
    "BEATs-stats (2304)": {
        "feat_file": "stats_features", "label_file": "stats_labels",
        "hidden": 128, "dropout": 0.0, "wd": 1e-3,
    },
}

N_FOLDS = 5
MAX_EPOCHS = 100
PATIENCE = 20
LR = 1e-3
BATCH = 32
SEED = 13


class Head(nn.Module):
    def __init__(self, in_dim, hidden, dropout, n_classes=3):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden, n_classes),
            )
        else:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, n_classes))
    def forward(self, x): return self.net(x)


def load_all(feat_file: str, label_file: str):
    X = np.concatenate([np.load(DATA / f"{feat_file}_train.npy"),
                        np.load(DATA / f"{feat_file}_test.npy")], axis=0)
    y = np.concatenate([np.load(DATA / f"{label_file}_train.npy"),
                        np.load(DATA / f"{label_file}_test.npy")], axis=0)
    return X.astype(np.float32), y.astype(np.int64)


def train_one_fold(X_tr, y_tr, X_te, y_te, hidden, dropout, wd):
    torch.manual_seed(SEED); np.random.seed(SEED)
    mu = X_tr.mean(0, keepdim=True)
    sd = X_tr.std(0, keepdim=True).clamp(min=1e-6)
    X_tr_n = (X_tr - mu) / sd
    X_te_n = (X_te - mu) / sd

    model = Head(X_tr.shape[1], hidden, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=wd)
    loader = DataLoader(TensorDataset(X_tr_n, y_tr), batch_size=BATCH, shuffle=True)

    best_acc = 0.0
    no_improve = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.inference_mode():
            acc = (model(X_te_n).argmax(-1) == y_te).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break

    return best_acc


def main() -> int:
    print(f"5-fold stratified CV (n_folds={N_FOLDS}, max_epochs={MAX_EPOCHS}, patience={PATIENCE})\n")

    results = {}
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for name, cfg in VARIANTS.items():
        X, y = load_all(cfg["feat_file"], cfg["label_file"])
        print(f"=== {name}  X.shape={X.shape} ===")

        fold_accs = []
        for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
            X_tr = torch.from_numpy(X[tr_idx])
            y_tr = torch.from_numpy(y[tr_idx])
            X_te = torch.from_numpy(X[te_idx])
            y_te = torch.from_numpy(y[te_idx])
            acc = train_one_fold(X_tr, y_tr, X_te, y_te,
                                 cfg["hidden"], cfg["dropout"], cfg["wd"])
            fold_accs.append(acc)
            print(f"  fold {fold_idx+1}/{N_FOLDS}: {acc:.3f}  (train={len(tr_idx)}, test={len(te_idx)})")

        mean = float(np.mean(fold_accs))
        std = float(np.std(fold_accs))
        sem = std / np.sqrt(N_FOLDS)
        results[name] = {"folds": fold_accs, "mean": mean, "std": std, "sem": sem}
        print(f"  mean = {mean:.3f} ± {std:.3f} (std), sem={sem:.3f}\n")

    print("=" * 60)
    print("RANKING (5-fold CV):")
    print("=" * 60)
    for name in sorted(results, key=lambda n: -results[n]["mean"]):
        r = results[name]
        print(f"  {name:25}  {r['mean']:.3f} ± {r['sem']:.3f}  folds=[{', '.join(f'{a:.2f}' for a in r['folds'])}]")

    # Save numerical results
    out = DATA / "cv_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
