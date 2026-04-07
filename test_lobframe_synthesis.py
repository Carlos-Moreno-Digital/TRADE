"""Smoke test for the LOBFrame synthetic generator + Nautilus bridge.

Runs entirely on the VPS and prints ONLY:
  - dataset shape
  - generated tensor shape
  - per-test PASS/FAIL of the 10 paranoid leakage tests
  - aggregate spread / depth statistics (one line each, no arrays)

Token-budget rule: NO dataframes are returned to the chat.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trade.research.lobframe import (
    ContStoikovConfig,
    LOBFrameBridge,
    column_index,
    run_all,
    synthesize_lob,
    synthesize_snapshots,
)

DATA_DIR = Path("data/dukascopy")
SAMPLE_SYMBOL = "EURUSD"
SAMPLE_BARS = 500


def load_sample(symbol: str, n: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{symbol}_1H.csv", parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df.iloc[:n].copy()


def main() -> int:
    bars = load_sample(SAMPLE_SYMBOL, SAMPLE_BARS)
    print(f"  bars: symbol={SAMPLE_SYMBOL} shape={bars.shape}")

    cfg = ContStoikovConfig(
        levels=10,
        window=20,
        base_spread=8e-5,        # ~0.8 pip on EURUSD
        depth_scale=1e3,
    )

    # 1) Direct tensor synthesis
    tensor = synthesize_lob(bars, cfg)
    print(f"  synth tensor: shape={tensor.shape}")
    expected_shape = (SAMPLE_BARS, 4 * cfg.levels)
    shape_ok = tensor.shape == expected_shape
    print(f"  shape check: expected={expected_shape} ok={shape_ok}")

    # 2) Bridge from LOBSnapshot list (round trip)
    snapshots = synthesize_snapshots(bars, cfg)
    bridge = LOBFrameBridge(levels=cfg.levels)
    tensor_via_bridge = bridge.from_snapshots(snapshots)
    bridge_ok = bool(np.array_equal(tensor, tensor_via_bridge))
    print(f"  bridge round-trip: shape={tensor_via_bridge.shape} match={bridge_ok}")

    # 3) Aggregate spread sanity (one line)
    ask_1_col = column_index(1, "ask_p")
    bid_1_col = column_index(1, "bid_p")
    spreads = tensor[:, ask_1_col] - tensor[:, bid_1_col]
    print(
        f"  spread: min={spreads.min():.6f} mean={spreads.mean():.6f} "
        f"max={spreads.max():.6f} all_positive={bool((spreads > 0).all())}"
    )

    # 4) Aggregate depth sanity
    ask_v_1_col = column_index(1, "ask_v")
    bid_v_1_col = column_index(1, "bid_v")
    top_depth = (tensor[:, ask_v_1_col] + tensor[:, bid_v_1_col]) / 2
    print(
        f"  level-1 depth: min={top_depth.min():.2f} "
        f"mean={top_depth.mean():.2f} max={top_depth.max():.2f}"
    )

    # 5) Paranoid leakage tests
    print()
    print("  paranoid leakage suite:")
    results = run_all(bars, cfg)
    passed = 0
    for name, ok in results.items():
        mark = "PASS" if ok else "FAIL"
        passed += int(ok)
        print(f"    [{mark}] {name}")
    print(f"\n  {passed}/{len(results)} paranoid tests passed")

    overall = shape_ok and bridge_ok and passed == len(results)
    print(f"\n  OVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
