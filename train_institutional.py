"""Production-grade DeepLOB training script for the user's VPS.

This is the LONG version of run_deeplob_gauntlet.py and is intended
to be launched on the production VPS where:
  - The full Dukascopy 16-year tape is available
  - download_calibration_data.py has populated *_calib.csv with real
    tick volumes (so the depth calibration uses real liquidity)
  - There are CPU/RAM cores to throw at multi-hour training and
    a real grid search

Token-budget rule (still applies): the script prints only summary
lines per epoch and per gauntlet gate; no tensors, no dataframes,
no per-batch chatter.

Pipeline (single command, no manual intervention):

  1. Pre-flight checks
       - Required CSVs on disk
       - Leakage suite (12/12) on a slice of the data
       - Print: number of bars, leakage verdict
  2. Synthesize causal LOB tensor (calibrated_config)
  3. Train DeepLOB
       - 50 epochs, AdamW, weighted CE on triple-barrier labels
       - Temporal split 70/30, early stop patience 8
       - Print: per-epoch one-liner
  4. Hyperparameter grid search over the strategy adapter
       - Coarse grid by default (12 combos), can be widened by env var
       - For each combo: NautilusHarness + paranoid time_permutation
         (the two cheapest gates that catch the most failures)
       - Print: one summary line per combo
       - Persist the surviving combos
  5. Full gauntlet on the BEST surviving combo
       - NautilusHarness, CPCV (full param_grid), ParanoidSuite (full)
       - Print: final verdict table
  6. Promotion decision
       - VIABLE   -> save model + grid combo + verdict to data/models/
       - REJECTED -> save the verdict alongside for future inspection

Run:
    python train_institutional.py --symbol EURUSD --epochs 50

Defaults assume EURUSD; the user passes --symbol USDJPY (or others)
to repeat the run for additional pairs.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trade.research.lobframe import calibrated_config, synthesize_lob
from trade.research.lobframe.dataset import (
    LabelMethod,
    LOBDatasetConfig,
    LOBWindowDataset,
)
from trade.research.lobframe.leakage_tests import run_all as run_leakage
from trade.research.lobframe.train import (
    TrainConfig,
    save_model,
    train_deeplob,
)
from trade.research.strategies.deeplob_strategy import DeepLOBStrategy
from trade.validation.cpcv import CPCV
from trade.validation.nautilus_harness import NautilusHarness
from trade.validation.paranoid import ParanoidSuite

DATA_DIR = Path("data/dukascopy")
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PAIR_BY_SYMBOL = {"EURUSD": "EUR/USD", "USDJPY": "USD/JPY"}
SPREAD_BY_SYMBOL = {"EURUSD": 8e-5, "USDJPY": 8e-3}

DEFAULT_GRID = {
    "confidence_threshold": [0.40, 0.45, 0.50],
    "atr_sl_mult": [1.0, 1.5],
    "atr_tp_mult": [1.5, 2.0],
    "max_holding_bars": [12, 24],
}


def log(msg: str) -> None:
    print(msg, flush=True)


def load_full_bars(symbol: str) -> pd.DataFrame:
    """Prefer the calibrated Dukascopy file (real tick volume) when
    available, otherwise fall back to the yfinance dump."""
    calib = DATA_DIR / f"{symbol}_calib.csv"
    if calib.exists():
        df = pd.read_csv(calib, parse_dates=["timestamp"])
        log(f"  data: {symbol}_calib.csv (real Dukascopy ticks)")
    else:
        df = pd.read_csv(DATA_DIR / f"{symbol}_1H.csv", parse_dates=["timestamp"])
        log(f"  data: {symbol}_1H.csv (yfinance fallback, depth=floor)")
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    needed = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in needed if c in df.columns]]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df


def preflight(bars: pd.DataFrame, symbol: str) -> bool:
    log("\n=== Pre-flight ===")
    log(f"  bars: {symbol} shape={bars.shape} "
        f"{bars.index[0].date()} -> {bars.index[-1].date()}")
    cfg = calibrated_config(symbol)
    sample = bars.iloc[: min(2000, len(bars))]
    leak = run_leakage(sample, cfg)
    passed = sum(int(v) for v in leak.values())
    log(f"  leakage suite: {passed}/{len(leak)} passed")
    if passed != len(leak):
        for name, ok in leak.items():
            if not ok:
                log(f"    [FAIL] {name}")
        return False
    return True


def train_phase(
    bars: pd.DataFrame,
    symbol: str,
    epochs: int,
    window: int,
):
    log("\n=== Training ===")
    cfg = calibrated_config(symbol)
    tensor = synthesize_lob(bars, cfg)
    log(f"  tensor: {tensor.shape}")

    ds_cfg = LOBDatasetConfig(
        window=window,
        normalize=True,
        label_method=LabelMethod.TRIPLE_BARRIER,
        tb_atr_period=14,
        tb_atr_mult_tp=2.0,
        tb_atr_mult_sl=2.0,
        tb_vertical_bars=50,
    )
    label_ds = LOBWindowDataset(tensor, ds_cfg, bars=bars)
    dist = label_ds.label_distribution()
    weights = label_ds.class_weights()
    log(
        f"  triple barrier labels: down={dist[0]} flat={dist[1]} up={dist[2]} "
        f"weights=({float(weights[0]):.2f}, "
        f"{float(weights[1]):.2f}, {float(weights[2]):.2f})"
    )

    train_cfg = TrainConfig(
        epochs=epochs,
        batch_size=128,
        lr=5e-4,
        weight_decay=1e-4,
        val_fraction=0.3,
        early_stop_patience=8,
    )
    t0 = time.time()
    model, summaries = train_deeplob(
        tensor,
        cfg=train_cfg,
        dataset_cfg=ds_cfg,
        bars=bars,
        class_weights=weights,
        log_fn=log,
    )
    log(f"  training time: {time.time() - t0:.1f}s, epochs={len(summaries)}")
    return model, tensor, ds_cfg


def grid_search(
    strategy: DeepLOBStrategy,
    bars: pd.DataFrame,
    spread: float,
    symbol: str,
    pair: str,
    grid: dict,
) -> tuple[dict | None, list[dict]]:
    log("\n=== Hyperparameter grid search ===")
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    log(f"  combos: {len(combos)}")

    survivors: list[dict] = []
    harness = NautilusHarness(
        spread_abs=spread, commission_bps=0.5, slip_prob=0.3,
        account=10_000.0, risk_per_trade=0.003,
    )
    for i, vals in enumerate(combos, 1):
        params = dict(zip(keys, vals))
        strategy.default_params.update(params)
        # Cheap filters: nautilus + permutation only
        try:
            nautilus_res = harness.run(strategy, bars, symbol, pair)
        except Exception as e:
            log(f"  [{i:2d}/{len(combos)}] {params} -> harness ERROR {e}")
            continue
        if not nautilus_res.viable:
            log(
                f"  [{i:2d}/{len(combos)}] {params} -> "
                f"nautilus REJECT pnl={nautilus_res.nautilus_pnl:+.0f}"
            )
            continue

        # Quick permutation gate (50 perms is enough at coarse stage)
        suite = ParanoidSuite(strategy, bars, spread, n_trials=len(combos))
        perm = suite.time_permutation(n_permutations=50)
        if not perm["passed"]:
            log(
                f"  [{i:2d}/{len(combos)}] {params} -> "
                f"perm p={perm['p_value']:.3f} REJECT"
            )
            continue

        log(
            f"  [{i:2d}/{len(combos)}] {params} -> "
            f"SURVIVE pnl={nautilus_res.nautilus_pnl:+.0f} "
            f"perm_p={perm['p_value']:.3f}"
        )
        survivors.append({
            "params": params,
            "nautilus_pnl": nautilus_res.nautilus_pnl,
            "permutation_p": perm["p_value"],
        })

    log(f"  survivors: {len(survivors)}/{len(combos)}")
    if not survivors:
        return None, []
    best = max(survivors, key=lambda r: r["nautilus_pnl"])
    return best, survivors


def full_gauntlet(
    strategy: DeepLOBStrategy,
    bars: pd.DataFrame,
    spread: float,
    symbol: str,
    pair: str,
) -> dict:
    log("\n=== Full gauntlet ===")
    harness = NautilusHarness(
        spread_abs=spread, commission_bps=0.5, slip_prob=0.3,
        account=10_000.0, risk_per_trade=0.003,
    )
    nautilus_res = harness.run(strategy, bars, symbol, pair)
    log(f"  Nautilus: {nautilus_res.summary()}")

    cpcv = CPCV(strategy, n_folds=6, n_test_folds=2, embargo_bars=50)
    cpcv_res = cpcv.run(bars, spread)
    log(f"  CPCV:     {cpcv_res.summary()}")

    n_combos = 1
    for v in strategy.param_grid.values():
        n_combos *= len(v)
    suite = ParanoidSuite(strategy, bars, spread, n_trials=n_combos, seed=42)
    paranoid = {
        "future_shift": suite.future_shift(),
        "time_permutation": suite.time_permutation(n_permutations=500),
        "block_bootstrap": suite.block_bootstrap(n_bootstrap=1000),
        "deflated_sharpe": suite.deflated_sharpe(),
        "minbtl": suite.minbtl(),
        "param_stability": suite.param_stability(),
        "wfe": suite.wfe(n_splits=5),
    }
    paranoid_passed = sum(int(r.get("passed", False)) for r in paranoid.values())
    log(f"  Paranoid: {paranoid_passed}/{len(paranoid)} passed")

    viable = (
        nautilus_res.viable
        and cpcv_res.viable
        and paranoid_passed == len(paranoid)
    )
    return {
        "nautilus_pnl": nautilus_res.nautilus_pnl,
        "nautilus_viable": nautilus_res.viable,
        "cpcv_oos_sharpe": cpcv_res.mean_oos_sharpe,
        "cpcv_wfe": cpcv_res.wfe,
        "cpcv_pbo": cpcv_res.pbo,
        "cpcv_viable": cpcv_res.viable,
        "paranoid_passed": paranoid_passed,
        "paranoid_total": len(paranoid),
        "viable": viable,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()

    sym = args.symbol
    pair = PAIR_BY_SYMBOL.get(sym, "EUR/USD")
    spread = SPREAD_BY_SYMBOL.get(sym, 8e-5)

    log(f"=== train_institutional.py ===")
    log(f"  symbol={sym} pair={pair} spread={spread} epochs={args.epochs} window={args.window}")

    bars = load_full_bars(sym)
    if not preflight(bars, sym):
        log("ABORT: preflight failed")
        return 2

    model, tensor, _ds_cfg = train_phase(bars, sym, args.epochs, args.window)
    save_model(model, MODELS_DIR / f"deeplob_{sym}.pt")

    strategy = DeepLOBStrategy(model=model, symbol=sym, window=args.window)
    best, survivors = grid_search(strategy, bars, spread, sym, pair, DEFAULT_GRID)
    if best is None:
        verdict = {"viable": False, "reason": "no grid survivor"}
    else:
        log(f"  best combo: {best}")
        strategy.default_params.update(best["params"])
        verdict = full_gauntlet(strategy, bars, spread, sym, pair)
        verdict["best_params"] = best["params"]
        verdict["n_survivors"] = len(survivors)

    out = MODELS_DIR / f"deeplob_{sym}_verdict.json"
    out.write_text(json.dumps(verdict, indent=2))
    log(f"\n  verdict written -> {out}")
    log(f"  VIABLE={verdict.get('viable', False)}")
    return 0 if verdict.get("viable", False) else 1


if __name__ == "__main__":
    sys.exit(main())
