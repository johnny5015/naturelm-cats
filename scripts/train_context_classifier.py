"""Train an MLP classifier on BEATs features → cat-vocalization context.

Three classes: brushing, isolation_unfamiliar_environment, waiting_for_food.

Architecture: [768] → Dropout(0.3) → Linear(256) → ReLU → Dropout(0.3) → Linear(3).
Loss: CrossEntropy. Optimizer: Adam(lr=1e-3, weight_decay=1e-4).
Stops when val accuracy stops improving for 10 epochs.

Outputs:
    /home/scott/datasets/cats/context_classifier.pt
    /home/scott/datasets/cats/training_log.json
    + printed confusion matrix on test set.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = Path("/home/scott/datasets/cats")
HIDDEN = 128
DROPOUT = 0.7
LR = 1e-3
WEIGHT_DECAY = 1e-1
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
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    label_names = json.loads((DATA_DIR / "label_names.json").read_text())
    label_names = {int(k): v for k, v in label_names.items()}
    n_classes = len(label_names)
    print(f"Label names: {label_names}")

    X_train = torch.from_numpy(np.load(DATA_DIR / "features_train.npy")).float()
    y_train = torch.from_numpy(np.load(DATA_DIR / "labels_train.npy")).long()
    X_test = torch.from_numpy(np.load(DATA_DIR / "features_test.npy")).float()
    y_test = torch.from_numpy(np.load(DATA_DIR / "labels_test.npy")).long()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Normalize features (per-feature z-score from train stats)
    mu = X_train.mean(0, keepdim=True)
    sd = X_train.std(0, keepdim=True).clamp(min=1e-6)
    X_train = (X_train - mu) / sd
    X_test = (X_test - mu) / sd

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = ContextHead(in_dim=X_train.shape[1], hidden=HIDDEN, n_classes=n_classes, dropout=DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

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
        "config": {"in_dim": X_train.shape[1], "hidden": HIDDEN, "n_classes": n_classes, "dropout": DROPOUT},
        "best_test_acc": best_test_acc,
    }, DATA_DIR / "context_classifier.pt")
    print(f"Saved classifier to {DATA_DIR/'context_classifier.pt'}")

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
