"""Walk-forward validation on 5-minute bars.

Same Meta-Labelling architecture as validate_walk_forward.py but on
5-minute data. Expected impact:
  - 12x more bars per pair → 12x more primary signals
  - Same meta-RF filter rate → 12x more surviving trades
  - ~5-10 trades/day across 3 pairs (the target)

The primary (RegimeMomentum) and meta-RF are IDENTICAL in logic.
Only the bar timeframe changes. The SJM regime is still computed
on daily resampled data (same as before) and broadcast down to 5min.

Params are adapted for 5min:
  - Donchian and ATR periods stay the same (they now span 5min bars
    instead of 1H, so they cover a shorter calendar window — this is
    intentional: 5min breakouts are shorter-lived)
  - atr_tp_mult reduced to 2.0 (targets are closer on 5min)
  - vertical_bars increased to 100 (= 500min ≈ 8 hours of label
    horizon, comparable to the 1H vertical of 50-100 bars)

Token rule: summary lines only.
"""
from __future__ import annotations

import json
import math
import sys
import time
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
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CUTOFF = pd.Timestamp("2024-01-01")
ACCOUNT = 10_000.0
RISK = 0.003

PAIRS = {
    "EURUSD": {"spread": 8e-5, "file": "EURUSD_5M.csv"},
    "USDJPY": {"spread": 8e-3, "file": "USDJPY_5M.csv"},
    "XAUUSD": {"spread": 0.40, "file": "XAUUSD_5M.csv"},
}

# Params adapted for 5min timeframe
PARAMS_5M = {
    "EURUSD": {
        "donchian_period": 30, "atr_sl_mult": 1.0, "atr_tp_mult": 2.0,
        "max_holding": 24, "atr_period": 14,
        "mr_lookback": 20, "mr_z_entry": 2.0, "mr_z_exit": 0.5,
        "rf_n_estimators": 100, "rf_max_depth": 4, "rf_min_samples_leaf": 10,
        "vertical_bars": 100, "meta_threshold": 0.0001,
    },
    "USDJPY": {
        "donchian_period": 20, "atr_sl_mult": 1.5, "atr_tp_mult": 2.0,
        "max_holding": 24, "atr_period": 14,
        "mr_lookback": 20, "mr_z_entry": 2.0, "mr_z_exit": 0.5,
        "rf_n_estimators": 100, "rf_max_depth": 4, "rf_min_samples_leaf": 10,
        "vertical_bars": 100, "meta_threshold": 0.008,
    },
    "XAUUSD": {
        "donchian_period": 20, "atr_sl_mult": 1.0, "atr_tp_mult": 2.0,
        "max_holding": 24, "atr_period": 14,
        "mr_lookback": 20, "mr_z_entry": 2.0, "mr_z_exit": 0.5,
        "rf_n_estimators": 100, "rf_max_depth": 4, "rf_min_samples_leaf": 10,
        "vertical_bars": 100, "meta_threshold": 0.0,
    },
}


def log(msg: str) -> None:
    print(msg, flush=True)


def load_5min(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / PAIRS[symbol]["file"]
    if not path.exists():
        raise FileNotFoundError(f"missing {path} — run download_5min.py first")
    df = pd.read_csv(path, parse_dates=["timestamp"])
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


def run_pair(symbol: str) -> dict:
    cfg = PAIRS[symbol]
    spread = cfg["spread"]
    params = PARAMS_5M[symbol]
    vb = int(params["vertical_bars"])
    threshold = float(params["meta_threshold"])

    bars = load_5min(symbol)
    train = bars[bars.index < CUTOFF]
    test = bars[bars.index >= CUTOFF]
    log(f"    bars: train={len(train):,} test={len(test):,}")

    if len(train) < 50_000 or len(test) < 10_000:
        return {"symbol": symbol, "error": "insufficient data",
                "train": len(train), "test": len(test)}

    pp = {k: params[k] for k in [
        "donchian_period", "atr_period", "atr_sl_mult", "atr_tp_mult",
        "max_holding", "mr_lookback", "mr_z_entry", "mr_z_exit",
    ]}

    primary = RegimeMomentum(symbol=symbol)

    # ---- TRAIN ----
    train_sigs = primary.signals(train, pp)
    log(f"    train signals: {len(train_sigs)}")
    if len(train_sigs) < 100:
        return {"symbol": symbol, "error": f"too few train signals ({len(train_sigs)})"}

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
        pnl_pu, _, off = _triple_barrier_pnl(
            t0, s.side, float(c_t[t0]), float(s.sl), float(s.tp),
            h_t, l_t, c_t, vb,
        )
        rows.append(list(f.values) + [1.0 if s.side == "buy" else 0.0])
        tgts.append(pnl_pu)
        starts.append(t0)
        ends.append(t0 + off)

    if len(rows) < 100:
        return {"symbol": symbol, "error": f"too few labelled ({len(rows)})"}

    X = np.array(rows, dtype=float)
    y = np.array(tgts, dtype=float)
    w = average_uniqueness(np.array(starts), np.array(ends), n_bars=int(max(ends))+1)
    w = normalize_to_sum_n(w)
    log(f"    train labels: {len(X)} (uniqueness min={w.min():.3f} mean={w.mean():.3f})")

    rf = RandomForestRegressor(
        n_estimators=int(params["rf_n_estimators"]),
        max_depth=int(params["rf_max_depth"]),
        min_samples_leaf=int(params["rf_min_samples_leaf"]),
        random_state=42, n_jobs=1,
    )
    rf.fit(X, y, sample_weight=w)

    # ---- TEST ----
    test_sigs = primary.signals(test, pp)
    log(f"    test signals: {len(test_sigs)}")

    feats_te = _build_meta_features(test, atr_period=int(params["atr_period"]))
    min_pnl = max(threshold, 0.5 * spread)  # reduced floor (RF already includes spread)

    kept = []
    for s in test_sigs.itertuples():
        t0 = int(s.bar_idx) - 1
        if t0 < 0 or t0 >= len(feats_te):
            continue
        f = feats_te.iloc[t0]
        if f.isna().any():
            continue
        row = np.array(list(f.values) + [1.0 if s.side == "buy" else 0.0], dtype=float).reshape(1,-1)
        pred = float(rf.predict(row)[0])
        if pred >= min_pnl:
            kept.append({
                "bar_idx": int(s.bar_idx), "side": s.side,
                "sl": float(s.sl), "tp": float(s.tp),
                "entry": float(test["close"].iloc[t0]),
            })

    pnls = walk_no_timeout(test, spread, kept)
    n = len(pnls)
    if n == 0:
        return {"symbol": symbol, "test_signals": len(test_sigs),
                "kept": len(kept), "error": "no trades survived"}

    arr = np.array(pnls)
    total = float(arr.sum())
    wr = float((arr > 0).mean())
    sharpe = float(arr.mean()/arr.std()*math.sqrt(220*12)) if arr.std() > 0 else 0.0
    eq = ACCOUNT + np.cumsum(arr)
    dd = float(((eq - np.maximum.accumulate(eq))/np.maximum.accumulate(eq)).min())
    days = (test.index[-1] - test.index[0]).days
    tpd = n / max(days, 1)

    return {
        "symbol": symbol,
        "test_signals": len(test_sigs),
        "kept": len(kept),
        "n_trades": n,
        "pnl": round(total, 2),
        "wr": round(wr, 3),
        "sharpe": round(sharpe, 3),
        "max_dd": round(dd, 3),
        "trades_per_day": round(tpd, 1),
    }


def main():
    log("=" * 70)
    log("5-MINUTE WALK-FORWARD VALIDATION (2024-2026)")
    log("=" * 70)

    results = {}
    for sym in PAIRS:
        log(f"\n  {sym}:")
        t0 = time.time()
        try:
            r = run_pair(sym)
        except FileNotFoundError as e:
            r = {"symbol": sym, "error": str(e)}
        except Exception as e:
            r = {"symbol": sym, "error": str(e)}
        elapsed = time.time() - t0
        results[sym] = r

        if "error" in r:
            log(f"    ERROR: {r['error']}")
            continue
        log(f"    RESULT: trades={r['n_trades']} pnl=${r['pnl']:+,.0f} "
            f"wr={r['wr']:.1%} sharpe={r['sharpe']:+.2f} "
            f"max_dd={r['max_dd']:.1%} trades/day={r['trades_per_day']} ({elapsed:.0f}s)")

    log(f"\n{'='*70}")
    log(f"{'Symbol':<10} {'Trades':>7} {'PnL':>10} {'WR':>6} {'Sharpe':>8} {'MaxDD':>8} {'Tr/Day':>7}")
    log("-" * 60)
    combined_pnl, combined_tpd = 0.0, 0.0
    for sym, r in results.items():
        if "error" in r:
            log(f"{sym:<10} ERROR: {r['error']}")
            continue
        combined_pnl += r["pnl"]
        combined_tpd += r["trades_per_day"]
        log(f"{sym:<10} {r['n_trades']:>7} ${r['pnl']:>+9,.0f} "
            f"{r['wr']:>5.1%} {r['sharpe']:>+8.2f} {r['max_dd']:>7.1%} {r['trades_per_day']:>7.1f}")
    log("-" * 60)
    log(f"{'COMBINED':<10} {'':>7} ${combined_pnl:>+9,.0f} {'':>6} {'':>8} {'':>8} {combined_tpd:>7.1f}")

    out = MODELS_DIR / "walk_forward_5min_verdict.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    log(f"\n  verdict -> {out}")


if __name__ == "__main__":
    main()
