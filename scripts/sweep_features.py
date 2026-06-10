# SPDX-License-Identifier: MIT
"""5-fold CV hyperparam sweep across feature variants.

Sweeps:  hidden ∈ {0, 32, 64, 128, 256} × dropout ∈ {0, 0.3, 0.5, 0.7}
         × weight_decay ∈ {1e-4, 1e-3, 1e-2, 1e-1}    = 80 configs per variant

For each config: 5-fold CV mean accuracy. Reports best config per variant.

Usage:
    uv run python scripts/sweep_features.py
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

DATA = Path("/home/scott/datasets/cats")
N_FOLDS = 5
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


def load_combined(feat_train: str, feat_test: str, lbl_train: str = "labels_train.npy",
                  lbl_test: str = "labels_test.npy"):
    X = np.concatenate([np.load(DATA / feat_train), np.load(DATA / feat_test)], axis=0)
    y = np.concatenate([np.load(DATA / lbl_train), np.load(DATA / lbl_test)], axis=0)
    return X.astype(np.float32), y.astype(np.int64)


def cv_mean(X: np.ndarray, y: np.ndarray, hidden: int, dropout: float, wd: float) -> float:
    """5-fold mean CV accuracy at the given hyperparams."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_accs = []
    for tr_idx, te_idx in skf.split(X, y):
        torch.manual_seed(SEED); np.random.seed(SEED)
        Xt, yt = torch.from_numpy(X[tr_idx]), torch.from_numpy(y[tr_idx]).long()
        Xv, yv = torch.from_numpy(X[te_idx]), torch.from_numpy(y[te_idx]).long()
        mu, sd = Xt.mean(0, keepdim=True), Xt.std(0, keepdim=True).clamp(min=1e-6)
        Xt, Xv = (Xt - mu) / sd, (Xv - mu) / sd

        model = Head(Xt.shape[1], hidden, dropout)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=32, shuffle=True)
        best, stale = 0.0, 0
        for epoch in range(100):
            model.train()
            for xb, yb in loader:
                opt.zero_grad(); F.cross_entropy(model(xb), yb).backward(); opt.step()
            model.eval()
            with torch.inference_mode():
                acc = (model(Xv).argmax(-1) == yv).float().mean().item()
            if acc > best:
                best, stale = acc, 0
            else:
                stale += 1
            if stale >= 15:
                break
        fold_accs.append(best)
    return float(np.mean(fold_accs))


def sweep(name: str, X: np.ndarray, y: np.ndarray) -> dict:
    best = (0.0, None)
    for hidden in [0, 32, 64, 128, 256]:
        for dropout in [0.0, 0.3, 0.5, 0.7]:
            for wd in [1e-4, 1e-3, 1e-2, 1e-1]:
                m = cv_mean(X, y, hidden, dropout, wd)
                if m > best[0]:
                    best = (m, {"hidden": hidden, "dropout": dropout, "wd": wd})
                    print(f"  [{name}] NEW BEST: {m:.3f}  hidden={hidden} dropout={dropout} wd={wd}")
    return {"acc": best[0], **best[1]}


def main() -> int:
    variants = []

    # BEATs-mean
    bx, by = load_combined("features_train.npy", "features_test.npy")
    print(f"\n=== BEATs-mean (768) ===")
    variants.append(("BEATs-mean (768)", sweep("BEATs-mean", bx, by)))

    # BEATs-stats
    sx, sy = load_combined("stats_features_train.npy", "stats_features_test.npy",
                            "stats_labels_train.npy", "stats_labels_test.npy")
    print(f"\n=== BEATs-stats (2304) ===")
    variants.append(("BEATs-stats (2304)", sweep("BEATs-stats", sx, sy)))

    # Q-Former
    qx, qy = load_combined("qformer_features_train.npy", "qformer_features_test.npy",
                            "qformer_labels_train.npy", "qformer_labels_test.npy")
    print(f"\n=== Q-Former (768) ===")
    variants.append(("Q-Former (768)", sweep("Q-Former", qx, qy)))

    # Classical alone
    cx, cy = load_combined("classical_features_train.npy", "classical_features_test.npy")
    print(f"\n=== Classical (102) ===")
    variants.append(("Classical (102)", sweep("Classical", cx, cy)))

    # Hybrid (BEATs-stats + Classical)
    hx = np.concatenate([sx, cx], axis=1).astype(np.float32)
    print(f"\n=== Hybrid: BEATs-stats + Classical (2406) ===")
    variants.append(("Hybrid (2406)", sweep("Hybrid", hx, sy)))

    print("\n" + "=" * 70)
    print("FINAL RANKING (5-fold CV best across sweep):")
    print("=" * 70)
    for name, b in sorted(variants, key=lambda v: -v[1]["acc"]):
        print(f"  {b['acc']:.3f}  {name:30}  hidden={b['hidden']:3d} dropout={b['dropout']} wd={b['wd']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
