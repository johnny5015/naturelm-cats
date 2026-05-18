"""5-fold group-aware CV with augmented training data.

Per fold:
  - All 5 augmentations of ~220 training clips → training pool (~1100 samples)
  - Only the ORIGINAL (aug_id=0) of ~55 test clips → test set (~55 samples)
  - Groups (source-clip IDs) don't cross folds, so augmentations of held-out
    clips never enter training.

This is the deployment-realistic eval: real clips arrive unaugmented.

Compares to:
  - BEATs-stats no-aug CV: 78.6% ± 1.5%
  - BEATs-mean no-aug CV: 76.5% ± 1.5%
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset

DATA = Path("/home/scott/datasets/cats")
N_FOLDS = 5
MAX_EPOCHS = 100
PATIENCE = 20
LR = 1e-3
BATCH = 64
SEED = 13

# Hyperparams from non-aug stats sweep — re-sweep optionally if these don't generalize
HIDDEN = 128
DROPOUT = 0.0
WEIGHT_DECAY = 1e-3


class Head(nn.Module):
    def __init__(self, in_dim, hidden, dropout, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_classes),
        )
    def forward(self, x): return self.net(x)


def load_all_aug():
    X = np.concatenate([np.load(DATA / "aug_features_train.npy"),
                        np.load(DATA / "aug_features_test.npy")], axis=0)
    y = np.concatenate([np.load(DATA / "aug_labels_train.npy"),
                        np.load(DATA / "aug_labels_test.npy")], axis=0)
    g = np.concatenate([np.load(DATA / "aug_groups_train.npy"),
                        np.load(DATA / "aug_groups_test.npy")], axis=0)

    # aug_id ∈ {0,1,2,3,4} is implicit in position within group. Since we wrote
    # them in order (aug0 first), aug_id = sample_idx % N_AUG
    n_aug = 5
    aug_ids = np.arange(len(X)) % n_aug

    return X.astype(np.float32), y.astype(np.int64), g.astype(np.int64), aug_ids


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
    print("Loading augmented features...")
    X, y, g, aug_ids = load_all_aug()
    print(f"Total samples: {len(X)} ({len(set(g))} unique source clips × 5 augs)")
    print(f"Hyperparams: hidden={HIDDEN}, dropout={DROPOUT}, wd={WEIGHT_DECAY}\n")

    # GroupKFold needs the *grouping* and stratification consistent. We stratify
    # on the per-group label. Since all augs of a clip share the same label,
    # using the per-sample label is fine for StratifiedGroupKFold.
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_accs = []
    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y, groups=g)):
        # Training: ALL augmentations of training groups
        X_tr = torch.from_numpy(X[tr_idx])
        y_tr = torch.from_numpy(y[tr_idx])

        # Test: ONLY original (aug_id == 0) of test groups
        te_orig_mask = aug_ids[te_idx] == 0
        te_idx_orig = te_idx[te_orig_mask]
        X_te = torch.from_numpy(X[te_idx_orig])
        y_te = torch.from_numpy(y[te_idx_orig])

        acc = train_one_fold(X_tr, y_tr, X_te, y_te, HIDDEN, DROPOUT, WEIGHT_DECAY)
        fold_accs.append(acc)
        n_train_groups = len(set(g[tr_idx]))
        n_test_groups = len(set(g[te_idx_orig]))
        print(f"  fold {fold_idx+1}/{N_FOLDS}: {acc:.3f}  "
              f"(train={len(tr_idx)} samples / {n_train_groups} clips, "
              f"test={len(te_idx_orig)} originals / {n_test_groups} clips)")

    mean = float(np.mean(fold_accs))
    std = float(np.std(fold_accs))
    sem = std / np.sqrt(N_FOLDS)

    print("\n" + "=" * 60)
    print("RESULT (augmented training, original-only test):")
    print(f"  {mean:.3f} ± {sem:.3f} (sem)  folds=[{', '.join(f'{a:.2f}' for a in fold_accs)}]")
    print("=" * 60)
    print(f"  Compare: BEATs-stats no-aug CV = 0.786 ± 0.015")
    print(f"           BEATs-mean  no-aug CV = 0.765 ± 0.015")
    print(f"  Delta vs no-aug stats: {mean - 0.786:+.3f}")

    out = DATA / "cv_results_augmented.json"
    out.write_text(json.dumps({
        "folds": fold_accs, "mean": mean, "std": std, "sem": sem,
        "config": {"hidden": HIDDEN, "dropout": DROPOUT, "wd": WEIGHT_DECAY,
                   "n_aug": 5, "test_uses_aug": False}
    }, indent=2))
    print(f"\nWrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
