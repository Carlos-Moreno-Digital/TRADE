"""Threshold sweep on walk-forward test window.

For each viable pair, trains the RF on pre-2024 data and sweeps
meta_threshold from aggressive (0) to conservative (current) on
the 2024-2026 test window. Reports the Pareto frontier of
trades vs Sharpe vs PnL so we can pick the optimal tradeoff.

Token rule: one line per (pair, threshold) combo.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trade.research.strategies.meta_labelling import (
    _build_meta_features,
    _triple_barrier_pnl,
)
from trade.research.strategies.regime_momentum import RegimeMomentum
from trade.validation.sample_weights import average_uniqueness, normalize_to_sum_n

DATA_DIR = Path("data/dukascopy")
CUTOFF = pd.Timestamp("2024-01-01")
ACCOUNT = 10_000.0
RISK = 0.003

PAIRS = {
    "EURUSD": {"spread": 8e-5, "thresholds": [0.0, 5e-5, 1e-4, 1.5e-4, 2e-4, 3e-4]},
    "USDJPY": {"spread": 8e-3, "thresholds": [0.0, 2e-3, 4e-3, 8e-3, 1.2e-2, 1.6e-2]},
    "XAUUSD": {"spread": 0.40, "thresholds": [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]},
}

BEST_PARAMS = {
    "EURUSD": {
        "donchian_period": 30, "atr_sl_mult": 1.0, "atr_tp_mult": 3.0,
        "max_holding": 24, "atr_period": 14,
        "mr_lookback": 20, "mr_z_entry": 2.0, "mr_z_exit": 0.5,
        "rf_n_estimators": 100, "rf_max_depth": 4, "rf_min_samples_leaf": 5,
        "vertical_bars": 50,
    },
    "USDJPY": {
        "donchian_period": 15, "atr_sl_mult": 1.5, "atr_tp_mult": 3.0,
        "max_holding": 24, "atr_period": 14,
        "mr_lookback": 20, "mr_z_entry": 2.0, "mr_z_exit": 0.5,
        "rf_n_estimators": 50, "rf_max_depth": 4, "rf_min_samples_leaf": 5,
        "vertical_bars": 100,
    },
    "XAUUSD": {
        "donchian_period": 20, "atr_sl_mult": 1.0, "atr_tp_mult": 2.0,
        "max_holding": 12, "atr_period": 14,
        "mr_lookback": 20, "mr_z_entry": 2.0, "mr_z_exit": 0.5,
        "rf_n_estimators": 100, "rf_max_depth": 4, "rf_min_samples_leaf": 5,
        "vertical_bars": 50,
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


def walk_no_timeout(bars, spread, signals_kept):
    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    n = len(close)
    entries = {int(s["bar_idx"]) - 1: s for s in signals_kept}
    pnls, pos, equity = [], None, ACCOUNT
    for i in range(n):
        if pos is None:
            sig = entries.get(i)
            if sig is None:
                continue
            risk = (sig["entry"] - sig["sl"]) if sig["side"] == "buy" else (sig["sl"] - sig["entry"])
            if risk <= 0:
                continue
            qty = (equity * RISK) / risk
            pos = {**sig, "idx": i, "qty": qty}
            continue
        ep = None
        if pos["side"] == "buy":
            if low[i] <= pos["sl"]:
                ep = pos["sl"]
            elif high[i] >= pos["tp"]:
                ep = pos["tp"]
        else:
            if high[i] >= pos["sl"]:
                ep = pos["sl"]
            elif low[i] <= pos["tp"]:
                ep = pos["tp"]
        if ep is not None:
            pnl = ((ep - pos["entry"]) if pos["side"] == "buy" else (pos["entry"] - ep)) * pos["qty"] - spread * pos["qty"] * 2
            pnls.append(pnl)
            equity += pnl
            pos = None
    if pos is not None:
        last = close[-1]
        pnl = ((last - pos["entry"]) if pos["side"] == "buy" else (pos["entry"] - last)) * pos["qty"] - spread * pos["qty"] * 2
        pnls.append(pnl)
    return pnls


def sweep(symbol: str, spread: float, thresholds: list[float]):
    log(f"\n{'='*70}")
    log(f"  {symbol} — threshold sweep (walk-forward 2024-2026)")
    log(f"{'='*70}")

    bars = load_bars(symbol)
    train = bars[bars.index < CUTOFF]
    test = bars[bars.index >= CUTOFF]

    params = BEST_PARAMS[symbol]
    pp = {k: params[k] for k in [
        "donchian_period", "atr_period", "atr_sl_mult", "atr_tp_mult",
        "max_holding", "mr_lookback", "mr_z_entry", "mr_z_exit",
    ]}
    vb = int(params["vertical_bars"])

    primary = RegimeMomentum(symbol=symbol)

    # Train RF
    train_sigs = primary.signals(train, pp)
    feats_tr = _build_meta_features(train, atr_period=int(params["atr_period"]))
    h_t, l_t, c_t = train["high"].values, train["low"].values, train["close"].values

    rows, tgts, starts, ends = [], [], [], []
    for s in train_sigs.itertuples():
        t0 = int(s.bar_idx) - 1
        if t0 < 0 or t0 >= len(feats_tr):
            continue
        f = feats_tr.iloc[t0]
        if f.isna().any():
            continue
        pnl_pu, _, off = _triple_barrier_pnl(t0, s.side, float(c_t[t0]), float(s.sl), float(s.tp), h_t, l_t, c_t, vb)
        rows.append(list(f.values) + [1.0 if s.side == "buy" else 0.0])
        tgts.append(pnl_pu)
        starts.append(t0)
        ends.append(t0 + off)

    if len(rows) < 30:
        log(f"  insufficient training labels ({len(rows)})")
        return

    X = np.array(rows, dtype=float)
    y = np.array(tgts, dtype=float)
    w = average_uniqueness(np.array(starts), np.array(ends), n_bars=int(max(ends)) + 1)
    w = normalize_to_sum_n(w)
    rf = RandomForestRegressor(
        n_estimators=int(params["rf_n_estimators"]),
        max_depth=int(params["rf_max_depth"]),
        min_samples_leaf=int(params["rf_min_samples_leaf"]),
        random_state=42, n_jobs=1,
    )
    rf.fit(X, y, sample_weight=w)

    # Score test signals
    test_sigs = primary.signals(test, pp)
    feats_te = _build_meta_features(test, atr_period=int(params["atr_period"]))

    scored = []
    for s in test_sigs.itertuples():
        t0 = int(s.bar_idx) - 1
        if t0 < 0 or t0 >= len(feats_te):
            continue
        f = feats_te.iloc[t0]
        if f.isna().any():
            continue
        row = np.array(list(f.values) + [1.0 if s.side == "buy" else 0.0], dtype=float).reshape(1, -1)
        pred = float(rf.predict(row)[0])
        scored.append({
            "bar_idx": int(s.bar_idx), "side": s.side,
            "sl": float(s.sl), "tp": float(s.tp),
            "entry": float(test["close"].iloc[t0]),
            "pred": pred,
        })

    log(f"  test signals scored: {len(scored)}")
    log(f"  {'threshold':>12} {'kept':>6} {'trades':>7} {'PnL':>10} {'WR':>6} {'Sharpe':>8} {'MaxDD':>8} {'tr/mo':>6}")
    log(f"  {'-'*64}")

    for t in thresholds:
        kept = [s for s in scored if s["pred"] >= t]
        pnls = walk_no_timeout(test, spread, kept)
        n = len(pnls)
        if n == 0:
            log(f"  {t:>12.6f} {len(kept):>6} {0:>7} {'$0':>10} {'—':>6} {'—':>8} {'—':>8} {'0.0':>6}")
            continue
        arr = np.array(pnls)
        total = float(arr.sum())
        wr = float((arr > 0).mean())
        sharpe = float(arr.mean() / arr.std() * math.sqrt(220)) if arr.std() > 0 else 0.0
        eq = ACCOUNT + np.cumsum(arr)
        dd = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min())
        months = (test.index[-1] - test.index[0]).days / 30.44
        tpm = n / max(months, 1)
        log(f"  {t:>12.6f} {len(kept):>6} {n:>7} ${total:>+9,.0f} {wr:>5.1%} {sharpe:>+8.2f} {dd:>7.1%} {tpm:>6.1f}")


def main():
    for sym, cfg in PAIRS.items():
        sweep(sym, cfg["spread"], cfg["thresholds"])
    log(f"\n{'='*70}")
    log("Pick the threshold that maximizes trades while Sharpe stays > 0.5")


if __name__ == "__main__":
    main()
