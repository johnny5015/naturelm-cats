"""Train an MLP classifier on BEATs features → cat-vocalization context.

Three classes: brushing, isolation_unfamiliar_environment, waiting_for_food.

Two feature variants supported:
    --variant mean    (default) Mean-pooled BEATs, 768-dim
                      Saves to context_classifier.pt
    --variant stats   Mean+Std+Max-pooled BEATs, 2304-dim — 73.3% test acc
                      Saves to context_classifier_stats.pt

Loss: CrossEntropy. Optimizer: Adam(lr=1e-3). Hyperparams from grid sweep.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = Path("/home/scott/datasets/cats")

# Per-variant hyperparams (each won its own grid sweep)
VARIANT_CFG = {
    "mean": {
        "feature_prefix": "features",
        "save_name": "context_classifier.pt",
        "hidden": 128, "dropout": 0.7, "weight_decay": 1e-1,
    },
    "stats": {
        # Retuned via 5-fold CV: 81.5% with these hparams (was 78.6% with old defaults)
        "feature_prefix": "stats_features",
        "label_prefix": "stats_labels",
        "save_name": "context_classifier_stats.pt",
        "hidden": 256, "dropout": 0.3, "weight_decay": 1e-1,
    },
    "hybrid": {
        # BEATs-stats (2304) + classical (102) = 2406. 5-fold CV: 82.2%.
        "feature_prefix": "hybrid_features",
        "label_prefix": "stats_labels",  # same label order as BEATs-stats
        "save_name": "context_classifier_hybrid.pt",
        "hidden": 128, "dropout": 0.0, "weight_decay": 1e-1,
    },
}

LR = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 30
SEED = 13


class ContextHead(nn.Module):
    def __init__(self, in_dim: int = 768, hidden: int = 256, n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANT_CFG), default="hybrid",
                    help="Feature variant. 'hybrid' (BEATs-stats + classical) recommended — 82.2% 5-fold CV.")
    ap.add_argument("--all", action="store_true",
                    help="Train on train+test combined (no held-out). For final deployment classifier. "
                         "Use 5-fold CV results for honest accuracy estimate.")
    args = ap.parse_args()
    cfg = VARIANT_CFG[args.variant]
    feature_prefix = cfg["feature_prefix"]
    label_prefix = cfg.get("label_prefix", "labels")
    save_path = DATA_DIR / cfg["save_name"]
    hidden, dropout, weight_decay = cfg["hidden"], cfg["dropout"], cfg["weight_decay"]
    print(f"Variant: {args.variant} → {save_path.name}  (hidden={hidden}, dropout={dropout}, wd={weight_decay})")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    label_names = json.loads((DATA_DIR / "label_names.json").read_text())
    label_names = {int(k): v for k, v in label_names.items()}
    n_classes = len(label_names)
    print(f"Label names: {label_names}")

    X_train = torch.from_numpy(np.load(DATA_DIR / f"{feature_prefix}_train.npy")).float()
    y_train = torch.from_numpy(np.load(DATA_DIR / f"{label_prefix}_train.npy")).long()
    X_test = torch.from_numpy(np.load(DATA_DIR / f"{feature_prefix}_test.npy")).float()
    y_test = torch.from_numpy(np.load(DATA_DIR / f"{label_prefix}_test.npy")).long()

    if args.all:
        # Merge: train on all, "test" becomes same set for monitoring only
        X_train = torch.cat([X_train, X_test], dim=0)
        y_train = torch.cat([y_train, y_test], dim=0)
        X_test = X_train.clone()  # monitor only — true accuracy is from cv_results.json
        y_test = y_train.clone()
        save_path = save_path.with_stem(save_path.stem + "_all")
        print(f"--all: merged train+test → {X_train.shape}. Saving to {save_path.name}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Normalize features (per-feature z-score from train stats)
    mu = X_train.mean(0, keepdim=True)
    sd = X_train.std(0, keepdim=True).clamp(min=1e-6)
    X_train = (X_train - mu) / sd
    X_test = (X_test - mu) / sd

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = ContextHead(in_dim=X_train.shape[1], hidden=hidden, n_classes=n_classes, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)

    best_test_acc = 0.0
    best_state = None
    epochs_since_improvement = 0
    log = []

    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(yb)
            train_correct += (logits.argmax(-1) == yb).sum().item()
            train_total += len(yb)
        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        test_loss = 0.0
        test_correct = 0
        with torch.inference_mode():
            for xb, yb in test_loader:
                logits = model(xb)
                test_loss += F.cross_entropy(logits, yb).item() * len(yb)
                test_correct += (logits.argmax(-1) == yb).sum().item()
        test_loss /= len(X_test)
        test_acc = test_correct / len(X_test)

        log.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                    "test_loss": test_loss, "test_acc": test_acc})

        improvement = test_acc > best_test_acc
        if improvement:
            best_test_acc = test_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epoch <= 10 or epoch % 10 == 0 or improvement:
            mark = "*" if improvement else " "
            print(f"epoch {epoch:3d} {mark} train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                  f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}")

        if epochs_since_improvement >= PATIENCE:
            print(f"Early stop at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    elapsed = time.time() - t0
    print(f"\nTraining took {elapsed:.1f}s. Best test accuracy: {best_test_acc:.3f}")

    # Reload best state, save model + stats
    model.load_state_dict(best_state)
    torch.save({
        "state_dict": model.state_dict(),
        "feature_mu": mu,
        "feature_sd": sd,
        "label_names": label_names,
        "config": {"in_dim": X_train.shape[1], "hidden": hidden, "n_classes": n_classes, "dropout": dropout},
        "best_test_acc": best_test_acc,
    }, save_path)
    print(f"Saved classifier to {save_path}")

    # Confusion matrix
    model.eval()
    with torch.inference_mode():
        all_preds = model(X_test).argmax(-1).numpy()
        all_true = y_test.numpy()

    print("\n=== Confusion matrix (rows=true, cols=pred) ===")
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(all_true, all_preds):
        cm[t, p] += 1

    short = {i: label_names[i][:10] for i in range(n_classes)}
    print(f"{'':12}" + "".join(f"{short[i]:>12}" for i in range(n_classes)))
    for i in range(n_classes):
        print(f"{short[i]:12}" + "".join(f"{cm[i,j]:>12}" for j in range(n_classes)))

    print("\n=== Per-class accuracy ===")
    for i in range(n_classes):
        acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  {label_names[i]:35} {acc:.3f}  ({cm[i,i]}/{cm[i].sum()})")

    (DATA_DIR / "training_log.json").write_text(json.dumps(log, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
