"""Diagnose signal frequency bottlenecks across the pipeline.

For each pair, reports:
  - How many bars exist in the test window
  - How many primary signals RegimeMomentum generates (and WHY
    the rest are rejected: regime filter, Donchian not triggered, etc.)
  - How many survive the meta-filter
  - The distribution of meta-predicted PnL (what threshold would
    keep more signals while staying profitable)

Token rule: summary lines only, no dataframes.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trade.research.indicators import ATR
from trade.research.strategies.meta_labelling import (
    _build_meta_features,
    _triple_barrier_pnl,
)
from trade.research.strategies.regime_momentum import RegimeMomentum
from trade.validation.sample_weights import average_uniqueness, normalize_to_sum_n

DATA_DIR = Path("data/dukascopy")
CUTOFF = pd.Timestamp("2024-01-01")

PAIRS = {
    "EURUSD": 8e-5,
    "USDJPY": 8e-3,
    "XAUUSD": 0.40,
}

BEST_PARAMS = {
    "EURUSD": {
        "donchian_period": 30, "atr_sl_mult": 1.0, "atr_tp_mult": 3.0,
        "max_holding": 24, "atr_period": 14,
        "mr_lookback": 20, "mr_z_entry": 2.0, "mr_z_exit": 0.5,
        "rf_n_estimators": 100, "rf_max_depth": 4, "rf_min_samples_leaf": 5,
        "vertical_bars": 50, "meta_threshold": 0.0003,
    },
    "USDJPY": {
        "donchian_period": 15, "atr_sl_mult": 1.5, "atr_tp_mult": 3.0,
        "max_holding": 24, "atr_period": 14,
        "mr_lookback": 20, "mr_z_entry": 2.0, "mr_z_exit": 0.5,
        "rf_n_estimators": 50, "rf_max_depth": 4, "rf_min_samples_leaf": 5,
        "vertical_bars": 100, "meta_threshold": 0.0,
    },
    "XAUUSD": {
        "donchian_period": 20, "atr_sl_mult": 1.0, "atr_tp_mult": 2.0,
        "max_holding": 12, "atr_period": 14,
        "mr_lookback": 20, "mr_z_entry": 2.0, "mr_z_exit": 0.5,
        "rf_n_estimators": 100, "rf_max_depth": 4, "rf_min_samples_leaf": 5,
        "vertical_bars": 50, "meta_threshold": 0.0,
    },
}


def log(msg: str) -> None:
    print(msg, flush=True)


def load_bars(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{symbol}_1H.csv", parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df[["open", "high", "low", "close", "volume"]]


def diagnose(symbol: str, spread: float):
    log(f"\n{'='*60}")
    log(f"  {symbol} — frequency bottleneck analysis")
    log(f"{'='*60}")

    bars = load_bars(symbol)
    train = bars[bars.index < CUTOFF]
    test = bars[bars.index >= CUTOFF]
    params = BEST_PARAMS[symbol]
    primary_params = {k: params[k] for k in [
        "donchian_period", "atr_period", "atr_sl_mult", "atr_tp_mult",
        "max_holding", "mr_lookback", "mr_z_entry", "mr_z_exit",
    ]}

    log(f"  test bars: {len(test):,} ({test.index[0].date()} -> {test.index[-1].date()})")

    # ---- Primary bottleneck ----
    primary = RegimeMomentum(symbol=symbol)
    regimes = primary._regime_per_bar(test)
    n_risk_on = int((regimes == primary.risk_on_state).sum())
    n_risk_off = int((regimes == primary.risk_off_state).sum())
    log(f"  regimes: risk_on={n_risk_on} ({n_risk_on/len(test)*100:.0f}%) "
        f"risk_off={n_risk_off} ({n_risk_off/len(test)*100:.0f}%)")

    test_signals = primary.signals(test, primary_params)
    log(f"  primary signals: {len(test_signals)}")
    log(f"  signal rate: 1 per {len(test)/max(len(test_signals),1):.0f} bars "
        f"= {len(test_signals)/len(test)*100:.1f}%")

    if test_signals.empty:
        log(f"  BOTTLENECK: primary generates ZERO signals")
        return

    # ---- Meta bottleneck: train RF on pre-cutoff, score test ----
    train_signals = primary.signals(train, primary_params)
    feats_train = _build_meta_features(train, atr_period=int(params["atr_period"]))
    high_t = train["high"].to_numpy(dtype=float)
    low_t = train["low"].to_numpy(dtype=float)
    close_t = train["close"].to_numpy(dtype=float)
    vertical_bars = int(params["vertical_bars"])

    rows, targets, starts, ends = [], [], [], []
    for s in train_signals.itertuples():
        t0 = int(s.bar_idx) - 1
        if t0 < 0 or t0 >= len(feats_train):
            continue
        f = feats_train.iloc[t0]
        if f.isna().any():
            continue
        entry = float(close_t[t0])
        pnl_pu, _, offset = _triple_barrier_pnl(
            t0, s.side, entry, float(s.sl), float(s.tp),
            high_t, low_t, close_t, vertical_bars,
        )
        rows.append(list(f.values) + [1.0 if s.side == "buy" else 0.0])
        targets.append(pnl_pu)
        starts.append(t0)
        ends.append(t0 + offset)

    if len(rows) < 30:
        log(f"  BOTTLENECK: too few train labels ({len(rows)}), cannot diagnose meta")
        return

    X_train = np.array(rows, dtype=float)
    y_train = np.array(targets, dtype=float)
    weights = average_uniqueness(
        np.array(starts), np.array(ends), n_bars=int(max(ends)) + 1,
    )
    weights = normalize_to_sum_n(weights)

    rf = RandomForestRegressor(
        n_estimators=int(params["rf_n_estimators"]),
        max_depth=int(params["rf_max_depth"]),
        min_samples_leaf=int(params["rf_min_samples_leaf"]),
        random_state=42, n_jobs=1,
    )
    rf.fit(X_train, y_train, sample_weight=weights)

    # Score test signals
    feats_test = _build_meta_features(test, atr_period=int(params["atr_period"]))
    preds = []
    for s in test_signals.itertuples():
        t0 = int(s.bar_idx) - 1
        if t0 < 0 or t0 >= len(feats_test):
            preds.append(float("nan"))
            continue
        f = feats_test.iloc[t0]
        if f.isna().any():
            preds.append(float("nan"))
            continue
        row = np.array(
            list(f.values) + [1.0 if s.side == "buy" else 0.0],
            dtype=float,
        ).reshape(1, -1)
        preds.append(float(rf.predict(row)[0]))

    preds = np.array(preds)
    valid = ~np.isnan(preds)
    current_threshold = max(float(params["meta_threshold"]), 2.0 * spread)
    n_above = int((preds[valid] >= current_threshold).sum())

    log(f"  meta predictions on test signals: {int(valid.sum())}")
    log(f"  current threshold: {current_threshold:.6f}")
    log(f"  kept (current): {n_above} ({n_above/max(int(valid.sum()),1)*100:.0f}%)")

    # Distribution of predictions
    p = preds[valid]
    log(f"  pred distribution: min={p.min():.6f} p25={np.percentile(p,25):.6f} "
        f"median={np.median(p):.6f} p75={np.percentile(p,75):.6f} max={p.max():.6f}")

    # What if we lowered the threshold?
    for t in [0.0, current_threshold * 0.5, current_threshold, current_threshold * 2]:
        n = int((p >= t).sum())
        log(f"    threshold={t:.6f} -> kept={n} ({n/len(p)*100:.0f}%)")

    # ---- Feature importance (what the RF considers most informative) ----
    from trade.research.strategies.meta_labelling import META_FEATURE_COLS
    fi = rf.feature_importances_
    cols = META_FEATURE_COLS + ["side_buy"]
    top3 = sorted(zip(cols, fi), key=lambda x: -x[1])[:3]
    log(f"  top3 features: {', '.join(f'{c}={v:.3f}' for c,v in top3)}")


def main():
    for sym, spread in PAIRS.items():
        diagnose(sym, spread)


if __name__ == "__main__":
    main()
