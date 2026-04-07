"""DeepLOB training loop with strict temporal split + early stopping.

Designed to be cheap enough for a smoke run (a couple of epochs on a
few hundred samples) yet structurally identical to a long production
run. AdamW + Cross-Entropy + LR plateau hook + early stopping on
validation loss.

Token-budget rule: only one summary line per epoch is printed; the
training tensors and gradients never leave the function.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from trade.research.lobframe.dataset import LOBDatasetConfig, LOBWindowDataset
from trade.research.lobframe.model import DeepLOB


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.3   # last 30% of samples = validation
    early_stop_patience: int = 3
    grad_clip: float | None = 1.0
    seed: int = 42
    device: str = "cpu"


@dataclass
class EpochSummary:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    seconds: float


def temporal_split(dataset_len: int, val_fraction: float) -> tuple[list[int], list[int]]:
    """Strictly chronological split: first (1-val_fraction) -> train,
    remaining tail -> validation. NO shuffle, NO interleaving.
    """
    n_train = int(math.floor(dataset_len * (1 - val_fraction)))
    train_idx = list(range(n_train))
    val_idx = list(range(n_train, dataset_len))
    return train_idx, val_idx


def _evaluate(model: nn.Module, loader: DataLoader, device: str,
              criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * x.shape[0]
            total_n += x.shape[0]
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum().item())
    return total_loss / max(total_n, 1), correct / max(total_n, 1)


def train_deeplob(
    tensor: np.ndarray,
    cfg: TrainConfig | None = None,
    dataset_cfg: LOBDatasetConfig | None = None,
    model: DeepLOB | None = None,
    log_fn=None,
) -> tuple[DeepLOB, list[EpochSummary]]:
    """Train DeepLOB on a (T, 40) LOBFrame tensor.

    Returns the trained model (best val loss state restored) and a
    list of EpochSummary records. Pure side effects: per-epoch one-line
    print via log_fn (or stdout).
    """
    cfg = cfg or TrainConfig()
    dataset_cfg = dataset_cfg or LOBDatasetConfig()
    log = log_fn or (lambda s: print(s, flush=True))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    full = LOBWindowDataset(tensor, dataset_cfg)
    train_idx, val_idx = temporal_split(len(full), cfg.val_fraction)
    if len(train_idx) < cfg.batch_size or len(val_idx) < cfg.batch_size:
        raise ValueError(
            f"dataset too small: train={len(train_idx)}, val={len(val_idx)}, "
            f"batch={cfg.batch_size}"
        )
    train_set = Subset(full, train_idx)
    val_set = Subset(full, val_idx)

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False, drop_last=False
    )

    if model is None:
        model = DeepLOB(n_levels=10, n_classes=3)
    model = model.to(cfg.device)

    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state: dict | None = None
    patience = 0
    summaries: list[EpochSummary] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_n = 0
        for x, y in train_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            optim.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            running_loss += float(loss.item()) * x.shape[0]
            running_n += x.shape[0]

        train_loss = running_loss / max(running_n, 1)
        val_loss, val_acc = _evaluate(model, val_loader, cfg.device, criterion)
        elapsed = time.time() - t0

        summary = EpochSummary(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_accuracy=val_acc,
            seconds=elapsed,
        )
        summaries.append(summary)
        log(
            f"  epoch {epoch:2d}/{cfg.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.3f} t={elapsed:.1f}s"
        )

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                log(f"  early stop at epoch {epoch} (no val_loss improvement)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, summaries


def save_model(model: DeepLOB, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: Path) -> DeepLOB:
    model = DeepLOB(n_levels=10, n_classes=3)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model
