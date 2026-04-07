"""End-to-end smoke for LOBFrame Phase 3 Part 2.

Runs entirely on the VPS and prints ONLY one-line summaries:
  - calibrated spread mean per symbol (institutional target check)
  - 11/11 paranoid leakage tests (10 from Part 1 + forward_impact)
  - tensor shape through bridge -> dataset -> DeepLOB
  - DeepLOB output shape (batch, n_classes)

Token-budget rule: NO dataframes are returned to chat.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trade.research.lobframe import (
    CALIBRATED_CONFIGS,
    LOBFrameBridge,
    calibrated_config,
    column_index,
    run_all,
    synthesize_lob,
    synthesize_snapshots,
)
from trade.research.lobframe.dataset import LOBDatasetConfig, LOBWindowDataset
from trade.research.lobframe.model import DeepLOB

DATA_DIR = Path("data/dukascopy")

# Institutional spread targets per symbol (used as a sanity check, not
# a hard assertion — the calibration aims for the correct order of
# magnitude even with synthetic-volume input).
SPREAD_TARGETS = {
    "EURUSD": (5e-5, 5e-4),  # 0.5 to 5 pips
    "USDJPY": (5e-3, 5e-2),  # 0.5 to 5 pips in JPY units
}


def load_sample(symbol: str, n: int = 1000) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{symbol}_1H.csv", parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df.iloc[:n].copy()


def section(title: str) -> None:
    print(f"\n  === {title} ===")


def main() -> int:
    overall_ok = True

    # ===============================================================
    # 1) Per-symbol calibrated spread / depth
    # ===============================================================
    section("Calibrated spread + depth (per symbol)")
    spread_results = {}
    paranoid_results = {}
    tensors = {}
    for sym, (target_lo, target_hi) in SPREAD_TARGETS.items():
        bars = load_sample(sym, n=1000)
        cfg = calibrated_config(sym)
        tensor = synthesize_lob(bars, cfg)
        ask1 = column_index(1, "ask_p")
        bid1 = column_index(1, "bid_p")
        ask_v1 = column_index(1, "ask_v")
        bid_v1 = column_index(1, "bid_v")
        spreads = tensor[:, ask1] - tensor[:, bid1]
        depth1 = (tensor[:, ask_v1] + tensor[:, bid_v1]) / 2

        in_target = bool(target_lo <= float(spreads.mean()) <= target_hi)
        spread_results[sym] = {
            "min": float(spreads.min()),
            "mean": float(spreads.mean()),
            "max": float(spreads.max()),
            "depth_mean": float(depth1.mean()),
            "in_target": in_target,
        }
        tensors[sym] = tensor
        mark = "PASS" if in_target else "FAIL"
        print(
            f"    [{mark}] {sym}: "
            f"spread min={spreads.min():.6f} mean={spreads.mean():.6f} "
            f"max={spreads.max():.6f} target=[{target_lo}, {target_hi}] "
            f"depth1={depth1.mean():.0f}"
        )
        if not in_target:
            overall_ok = False

    # ===============================================================
    # 2) Paranoid leakage suite (now 11 tests including forward_impact)
    # ===============================================================
    section("Paranoid leakage suite (incl. forward_impact)")
    for sym in SPREAD_TARGETS:
        bars = load_sample(sym, n=400)
        cfg = calibrated_config(sym)
        results = run_all(bars, cfg)
        passed = sum(int(v) for v in results.values())
        paranoid_results[sym] = (passed, len(results))
        for name, ok in results.items():
            mark = "PASS" if ok else "FAIL"
            print(f"    [{mark}] {sym}: {name}")
        if passed != len(results):
            overall_ok = False

    # ===============================================================
    # 3) Bridge round-trip
    # ===============================================================
    section("Nautilus bridge round-trip")
    for sym, tensor in tensors.items():
        bars = load_sample(sym, n=1000)
        snaps = synthesize_snapshots(bars, calibrated_config(sym))
        bridge = LOBFrameBridge(levels=10)
        round_trip = bridge.from_snapshots(snaps)
        ok = bool(np.array_equal(tensor, round_trip))
        mark = "PASS" if ok else "FAIL"
        print(f"    [{mark}] {sym}: bridge round-trip shape={round_trip.shape}")
        if not ok:
            overall_ok = False

    # ===============================================================
    # 4) Dataset + DataLoader + DeepLOB end-to-end
    # ===============================================================
    section("LOBFrame dataset -> DeepLOB end-to-end")
    sym = "EURUSD"
    tensor = tensors[sym]
    ds_cfg = LOBDatasetConfig(window=100, horizon=5, tau=1e-4, normalize=True)
    dataset = LOBWindowDataset(tensor, ds_cfg)
    print(f"    dataset: len={len(dataset)} window={ds_cfg.window} horizon={ds_cfg.horizon}")

    sample_x, sample_y = dataset[0]
    print(f"    sample x.shape={tuple(sample_x.shape)} y={int(sample_y)}")

    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    batch_x, batch_y = next(iter(loader))
    print(f"    batch  x.shape={tuple(batch_x.shape)} y.shape={tuple(batch_y.shape)}")

    model = DeepLOB(n_levels=10, n_classes=3)
    model.eval()
    with torch.no_grad():
        logits = model(batch_x)
    print(f"    DeepLOB output logits.shape={tuple(logits.shape)}")

    # Class distribution sanity (should not be a single class)
    label_dist = torch.bincount(batch_y, minlength=3).tolist()
    print(f"    label distribution in batch: down/flat/up = {label_dist}")

    expected_logits_shape = (32, 3)
    flow_ok = tuple(logits.shape) == expected_logits_shape
    print(f"    [{'PASS' if flow_ok else 'FAIL'}] DeepLOB shape matches {expected_logits_shape}")
    if not flow_ok:
        overall_ok = False

    # ===============================================================
    print()
    print(f"  OVERALL: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
