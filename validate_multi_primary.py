"""Walk-forward with Multi-Primary signal pool.

Same meta-RF architecture as validate_walk_forward.py but feeds it
3x more signals from: RegimeMomentum + Bollinger + RSI.

The RF sees the same 11 features regardless of which primary
generated the signal. It doesn't know (or care) about the source —
it just predicts whether the setup will be profitable.

Token rule: one summary line per pair.
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
from trade.research.strategies.multi_primary import generate_multi_primary
from trade.research.strategies.regime_momentum import RegimeMomentum
from trade.validation.sample_weights import average_uniqueness, normalize_to_sum_n

DATA_DIR = Path("data/dukascopy")
MODELS_DIR = Path("data/models")
CUTOFF = pd.Timestamp("2024-01-01")
ACCOUNT = 10_000.0
RISK = 0.003

PAIRS = {
    "EURUSD": {"spread": 8e-5, "threshold": 0.0001},
    "USDJPY": {"spread": 8e-3, "threshold": 0.016},
    "XAUUSD": {"spread": 0.40, "threshold": 0.0},
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


def run_pair(symbol: str, spread: float, threshold: float) -> dict:
    bars = load_bars(symbol)
    train = bars[bars.index < CUTOFF]
    test = bars[bars.index >= CUTOFF]

    if len(train) < 5000 or len(test) < 1000:
        return {"symbol": symbol, "error": "insufficient data"}

    params = BEST_PARAMS[symbol]
    pp = {k: params[k] for k in [
        "donchian_period", "atr_period", "atr_sl_mult", "atr_tp_mult",
        "max_holding", "mr_lookback", "mr_z_entry", "mr_z_exit",
    ]}
    vb = int(params["vertical_bars"])

    primary = RegimeMomentum(symbol=symbol)

    # ---- TRAIN: multi-primary pool ----
    rm_train = primary.signals(train, pp)
    multi_train = generate_multi_primary(train, rm_train, params)
    log(f"    train pool: RM={len(rm_train)} -> multi={len(multi_train)}")

    feats_tr = _build_meta_features(train, atr_period=int(params["atr_period"]))
    h_t, l_t, c_t = train["high"].values, train["low"].values, train["close"].values

    rows, tgts, starts, ends = [], [], [], []
    for s in multi_train.itertuples():
        t0 = int(s.bar_idx) - 1
        if t0 < 0 or t0 >= len(feats_tr):
            continue
        f = feats_tr.iloc[t0]
        if f.isna().any():
            continue
        entry = float(c_t[t0])
        pnl_pu, _, off = _triple_barrier_pnl(
            t0, s.side, entry, float(s.sl), float(s.tp),
            h_t, l_t, c_t, vb,
        )
        rows.append(list(f.values) + [1.0 if s.side == "buy" else 0.0])
        tgts.append(pnl_pu)
        starts.append(t0)
        ends.append(t0 + off)

    if len(rows) < 50:
        return {"symbol": symbol, "error": f"too few train labels ({len(rows)})"}

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

    # ---- TEST: multi-primary pool ----
    rm_test = primary.signals(test, pp)
    multi_test = generate_multi_primary(test, rm_test, params)
    log(f"    test pool:  RM={len(rm_test)} -> multi={len(multi_test)}")

    feats_te = _build_meta_features(test, atr_period=int(params["atr_period"]))
    kept = []
    source_counts = {"regime_momentum": 0, "bollinger": 0, "rsi": 0}
    for s in multi_test.itertuples():
        t0 = int(s.bar_idx) - 1
        if t0 < 0 or t0 >= len(feats_te):
            continue
        f = feats_te.iloc[t0]
        if f.isna().any():
            continue
        row = np.array(list(f.values) + [1.0 if s.side == "buy" else 0.0], dtype=float).reshape(1, -1)
        pred = float(rf.predict(row)[0])
        if pred >= threshold:
            kept.append({
                "bar_idx": int(s.bar_idx), "side": s.side,
                "sl": float(s.sl), "tp": float(s.tp),
                "entry": float(test["close"].iloc[t0]),
            })
            src = getattr(s, "source", "unknown")
            if src in source_counts:
                source_counts[src] += 1

    pnls = walk_no_timeout(test, spread, kept)
    n = len(pnls)
    if n == 0:
        return {"symbol": symbol, "n_pool": len(multi_test), "n_kept": 0,
                "metrics": {"n_trades": 0, "pnl": 0, "wr": 0, "sharpe": 0, "max_dd": 0},
                "sources": source_counts}

    arr = np.array(pnls)
    total = float(arr.sum())
    wr = float((arr > 0).mean())
    sharpe = float(arr.mean() / arr.std() * math.sqrt(220)) if arr.std() > 0 else 0.0
    eq = ACCOUNT + np.cumsum(arr)
    dd = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min())
    months = (test.index[-1] - test.index[0]).days / 30.44

    return {
        "symbol": symbol,
        "n_pool": len(multi_test),
        "n_kept": len(kept),
        "sources": source_counts,
        "metrics": {
            "n_trades": n, "pnl": round(total, 2), "wr": round(wr, 3),
            "sharpe": round(sharpe, 3), "max_dd": round(dd, 3),
        },
        "trades_per_month": round(n / max(months, 1), 1),
    }


def main():
    log("=" * 70)
    log("MULTI-PRIMARY WALK-FORWARD (2024-2026)")
    log("Sources: RegimeMomentum + Bollinger + RSI → meta-RF filter")
    log("=" * 70)

    results = {}
    for sym, cfg in PAIRS.items():
        log(f"\n  {sym} (spread={cfg['spread']}, threshold={cfg['threshold']})...")
        t0 = time.time()
        r = run_pair(sym, cfg["spread"], cfg["threshold"])
        elapsed = time.time() - t0
        results[sym] = r
        if "error" in r:
            log(f"    ERROR: {r['error']}")
            continue
        m = r["metrics"]
        log(f"    sources kept: {r['sources']}")
        log(f"    pool={r['n_pool']} kept={r['n_kept']} trades={m['n_trades']} "
            f"pnl=${m['pnl']:+,.0f} wr={m['wr']:.1%} sharpe={m['sharpe']:+.2f} "
            f"max_dd={m['max_dd']:.1%} tr/mo={r['trades_per_month']} ({elapsed:.0f}s)")

    log(f"\n{'='*70}")
    log("MULTI-PRIMARY vs SINGLE-PRIMARY comparison:")
    log(f"{'Symbol':<10} {'Single tr':>10} {'Multi tr':>10} {'Single PnL':>12} {'Multi PnL':>12} {'Multi Sharpe':>13}")
    log("-" * 70)
    # Single-primary results from the walk-forward verdict
    single = {"EURUSD": (11, 187), "USDJPY": (121, 660), "XAUUSD": (80, 380)}
    combined_s, combined_m = 0.0, 0.0
    for sym in PAIRS:
        r = results.get(sym, {})
        m = r.get("metrics", {})
        s_tr, s_pnl = single.get(sym, (0, 0))
        m_tr = m.get("n_trades", 0)
        m_pnl = m.get("pnl", 0)
        m_sh = m.get("sharpe", 0)
        combined_s += s_pnl
        combined_m += m_pnl
        log(f"{sym:<10} {s_tr:>10} {m_tr:>10} ${s_pnl:>+11,.0f} ${m_pnl:>+11,.0f} {m_sh:>+13.2f}")
    log("-" * 70)
    log(f"{'COMBINED':<10} {'':>10} {'':>10} ${combined_s:>+11,.0f} ${combined_m:>+11,.0f}")

    out = MODELS_DIR / "multi_primary_verdict.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    log(f"\n  verdict -> {out}")


if __name__ == "__main__":
    main()
