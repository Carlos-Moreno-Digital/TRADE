"""Strict walk-forward validation + multi-pair institutional run.

This script does what the Purged K-Fold did NOT do: it trains the
meta-model ONLY on data before a hard cutoff date and evaluates
EXCLUSIVELY on data after that date. This is the closest simulation
to going live.

Walk-forward protocol:
  1. Split the full bar history at CUTOFF_DATE (default 2024-01-01).
     TRAIN = everything before the cutoff (~14 years).
     TEST  = everything after the cutoff (~2 years).
  2. Run the primary (RegimeMomentum) on TRAIN to get training signals.
  3. Label with triple-barrier PnL, compute uniqueness, train RF.
  4. Run the primary on TEST to get test signals.
  5. Score test signals with the trained RF.
  6. Filter by meta_threshold + spread floor.
  7. Walk positions on TEST with pessimistic SL-first, no max_holding.
  8. Report metrics ONLY on the TEST window.

This runs on ALL 5 Dukascopy pairs:
  EURUSD, USDJPY, GBPNZD, AUDNZD, XAUUSD

Using the best params from the institutional grid search where
available, or defaults for pairs that haven't been grid-searched yet.

Token rule: one summary line per pair, final verdict table.
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

from trade.research.indicators import ATR
from trade.research.strategies.meta_labelling import (
    META_FEATURE_COLS,
    _build_meta_features,
    _triple_barrier_pnl,
)
from trade.research.strategies.regime_momentum import RegimeMomentum
from trade.validation.sample_weights import (
    average_uniqueness,
    normalize_to_sum_n,
)

DATA_DIR = Path("data/dukascopy")
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ACCOUNT = 10_000.0
RISK_PER_TRADE = 0.003
CUTOFF_DATE = "2024-01-01"

# All 5 Dukascopy pairs with their spread + SJM config
PAIRS = {
    "EURUSD": {"spread": 8e-5,  "pair_code": "EUR/USD"},
    "USDJPY": {"spread": 8e-3,  "pair_code": "USD/JPY"},
    "GBPNZD": {"spread": 3e-4,  "pair_code": "GBP/NZD"},
    "AUDNZD": {"spread": 2e-4,  "pair_code": "AUD/NZD"},
    "XAUUSD": {"spread": 0.40,  "pair_code": "XAU/USD"},
}

# Best params from the institutional grid search (EURUSD/USDJPY).
# Other pairs use a sensible default until they get their own grid search.
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
}
DEFAULT_PARAMS = {
    "donchian_period": 20, "atr_sl_mult": 1.5, "atr_tp_mult": 3.0,
    "max_holding": 24, "atr_period": 14,
    "mr_lookback": 20, "mr_z_entry": 2.0, "mr_z_exit": 0.5,
    "rf_n_estimators": 100, "rf_max_depth": 4, "rf_min_samples_leaf": 5,
    "vertical_bars": 50, "meta_threshold": 0.0,
}


def log(msg: str) -> None:
    print(msg, flush=True)


def load_bars(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_1H.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df[["open", "high", "low", "close", "volume"]]


def walk_positions_no_timeout(
    bars: pd.DataFrame, spread: float, kept_signals: list[dict],
) -> list[float]:
    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    n = len(close)
    entries = {int(s["bar_idx"]) - 1: s for s in kept_signals}
    pnls, pos, equity = [], None, ACCOUNT
    for i in range(n):
        if pos is None:
            sig = entries.get(i)
            if sig is None:
                continue
            risk = (sig["entry"] - sig["sl"]) if sig["side"] == "buy" else (sig["sl"] - sig["entry"])
            if risk <= 0:
                continue
            qty = (equity * RISK_PER_TRADE) / risk
            pos = {**sig, "idx": i, "qty": qty}
            continue
        exit_price = None
        if pos["side"] == "buy":
            if low[i] <= pos["sl"]:
                exit_price = pos["sl"]
            elif high[i] >= pos["tp"]:
                exit_price = pos["tp"]
        else:
            if high[i] >= pos["sl"]:
                exit_price = pos["sl"]
            elif low[i] <= pos["tp"]:
                exit_price = pos["tp"]
        if exit_price is not None:
            if pos["side"] == "buy":
                pnl = (exit_price - pos["entry"]) * pos["qty"] - spread * pos["qty"] * 2
            else:
                pnl = (pos["entry"] - exit_price) * pos["qty"] - spread * pos["qty"] * 2
            pnls.append(pnl)
            equity += pnl
            pos = None
    if pos is not None:
        last = close[-1]
        if pos["side"] == "buy":
            pnl = (last - pos["entry"]) * pos["qty"] - spread * pos["qty"] * 2
        else:
            pnl = (pos["entry"] - last) * pos["qty"] - spread * pos["qty"] * 2
        pnls.append(pnl)
    return pnls


def compute_metrics(pnls: list[float]) -> dict:
    arr = np.array(pnls, dtype=float) if pnls else np.array([])
    n = len(arr)
    if n == 0:
        return {"n_trades": 0, "pnl": 0.0, "wr": 0.0, "sharpe": 0.0, "max_dd": 0.0}
    total = float(arr.sum())
    wr = float((arr > 0).mean())
    sharpe = float(arr.mean() / arr.std() * math.sqrt(220)) if arr.std() > 0 else 0.0
    eq = ACCOUNT + np.cumsum(arr)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return {
        "n_trades": n, "pnl": round(total, 2), "wr": round(wr, 3),
        "sharpe": round(sharpe, 3), "max_dd": round(float(dd.min()), 3),
    }


def run_walk_forward(symbol: str, spread: float) -> dict:
    """Train pre-cutoff, test post-cutoff. Returns metrics on TEST only."""
    bars = load_bars(symbol)
    cutoff = pd.Timestamp(CUTOFF_DATE)
    train_bars = bars[bars.index < cutoff]
    test_bars = bars[bars.index >= cutoff]
    if len(train_bars) < 5000 or len(test_bars) < 1000:
        return {"symbol": symbol, "error": "insufficient split",
                "train_rows": len(train_bars), "test_rows": len(test_bars)}

    params = BEST_PARAMS.get(symbol, DEFAULT_PARAMS)
    primary_params = {k: params[k] for k in [
        "donchian_period", "atr_period", "atr_sl_mult", "atr_tp_mult",
        "max_holding", "mr_lookback", "mr_z_entry", "mr_z_exit",
    ]}
    vertical_bars = int(params["vertical_bars"])
    meta_threshold = float(params["meta_threshold"])

    # Check if SJM model exists for this symbol; create if not
    sjm_path = Path("data/regimes") / f"{symbol}.json"
    if not sjm_path.exists():
        log(f"    SJM model missing for {symbol}, using EURUSD params as proxy")
        # Copy EURUSD SJM as a proxy (won't be perfect but allows the run)
        proxy = Path("data/regimes/EURUSD.json")
        if proxy.exists():
            import shutil
            shutil.copy(proxy, sjm_path)

    try:
        primary = RegimeMomentum(symbol=symbol)
    except FileNotFoundError:
        return {"symbol": symbol, "error": "no SJM model"}

    # ---- TRAIN phase: signals + features + labels on pre-cutoff ----
    train_signals = primary.signals(train_bars, primary_params)
    if train_signals.empty or len(train_signals) < 30:
        return {"symbol": symbol, "error": "too few train signals",
                "n_train_signals": len(train_signals)}

    feats_train = _build_meta_features(train_bars, atr_period=int(params["atr_period"]))
    high_t = train_bars["high"].to_numpy(dtype=float)
    low_t = train_bars["low"].to_numpy(dtype=float)
    close_t = train_bars["close"].to_numpy(dtype=float)

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
        return {"symbol": symbol, "error": "too few labelled train signals"}

    X_train = np.array(rows, dtype=float)
    y_train = np.array(targets, dtype=float)
    s_arr = np.array(starts, dtype=int)
    e_arr = np.array(ends, dtype=int)
    weights = average_uniqueness(s_arr, e_arr, n_bars=int(e_arr.max()) + 1)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights = normalize_to_sum_n(weights)

    rf = RandomForestRegressor(
        n_estimators=int(params["rf_n_estimators"]),
        max_depth=int(params["rf_max_depth"]),
        min_samples_leaf=int(params["rf_min_samples_leaf"]),
        random_state=42, n_jobs=1,
    )
    rf.fit(X_train, y_train, sample_weight=weights)

    # ---- TEST phase: signals on post-cutoff, scored by trained RF ----
    test_signals = primary.signals(test_bars, primary_params)
    if test_signals.empty:
        return {"symbol": symbol, "error": "no test signals"}

    feats_test = _build_meta_features(test_bars, atr_period=int(params["atr_period"]))
    min_pnl_per_unit = max(meta_threshold, 2.0 * spread)

    kept = []
    for s in test_signals.itertuples():
        t0 = int(s.bar_idx) - 1
        if t0 < 0 or t0 >= len(feats_test):
            continue
        f = feats_test.iloc[t0]
        if f.isna().any():
            continue
        row = np.array(
            list(f.values) + [1.0 if s.side == "buy" else 0.0],
            dtype=float,
        ).reshape(1, -1)
        pred = float(rf.predict(row)[0])
        if pred >= min_pnl_per_unit:
            kept.append({
                "bar_idx": int(s.bar_idx), "side": s.side,
                "sl": float(s.sl), "tp": float(s.tp),
                "entry": float(test_bars["close"].iloc[t0]),
            })

    pnls = walk_positions_no_timeout(test_bars, spread, kept)
    metrics = compute_metrics(pnls)

    return {
        "symbol": symbol,
        "cutoff": CUTOFF_DATE,
        "train_rows": len(train_bars),
        "test_rows": len(test_bars),
        "train_signals": len(train_signals),
        "test_signals": len(test_signals),
        "test_kept": len(kept),
        "filter_rate": round(1 - len(kept) / max(len(test_signals), 1), 3),
        "metrics": metrics,
        "uniqueness_mean": round(float(weights.mean()), 3),
        "uniqueness_min": round(float(weights.min()), 3),
    }


def main() -> int:
    log("=" * 70)
    log(f"WALK-FORWARD VALIDATION — cutoff={CUTOFF_DATE}")
    log(f"Train: everything before {CUTOFF_DATE} (~14 years)")
    log(f"Test:  everything after  {CUTOFF_DATE} (~2 years)")
    log("=" * 70)

    results = {}
    for symbol, cfg in PAIRS.items():
        path = DATA_DIR / f"{symbol}_1H.csv"
        if not path.exists():
            log(f"\n  {symbol}: SKIP (no data file)")
            continue
        log(f"\n  {symbol} (spread={cfg['spread']})...")
        t0 = time.time()
        r = run_walk_forward(symbol, cfg["spread"])
        elapsed = time.time() - t0
        results[symbol] = r

        if "error" in r:
            log(f"    ERROR: {r['error']}")
            continue

        m = r["metrics"]
        log(
            f"    train={r['train_rows']:,} test={r['test_rows']:,} "
            f"signals={r['test_signals']} kept={r['test_kept']} "
            f"filter={r['filter_rate']:.0%}"
        )
        log(
            f"    TEST RESULT: trades={m['n_trades']} pnl=${m['pnl']:+,.0f} "
            f"wr={m['wr']:.1%} sharpe={m['sharpe']:+.2f} "
            f"max_dd={m['max_dd']:.1%} ({elapsed:.0f}s)"
        )

    # Summary table
    log("\n" + "=" * 70)
    log("WALK-FORWARD SUMMARY (TEST period only: 2024-01-01 → present)")
    log(f"{'Symbol':<10} {'Trades':>7} {'PnL':>10} {'WR':>6} {'Sharpe':>8} {'MaxDD':>8} {'Viable':>8}")
    log("-" * 70)
    combined_pnl = 0.0
    combined_trades = 0
    for symbol, r in results.items():
        if "error" in r:
            log(f"{symbol:<10} {'ERROR':>7}  {r['error']}")
            continue
        m = r["metrics"]
        viable = m["sharpe"] > 0.5 and m["pnl"] > 0 and m["n_trades"] >= 5
        combined_pnl += m["pnl"]
        combined_trades += m["n_trades"]
        log(
            f"{symbol:<10} {m['n_trades']:>7} {m['pnl']:>+10,.0f} "
            f"{m['wr']:>5.1%} {m['sharpe']:>+8.2f} {m['max_dd']:>7.1%} "
            f"{'YES' if viable else 'no':>8}"
        )
    log("-" * 70)
    log(f"{'COMBINED':<10} {combined_trades:>7} {combined_pnl:>+10,.0f}")

    # Persist
    verdict_path = MODELS_DIR / "walk_forward_verdict.json"
    verdict_path.write_text(json.dumps(results, indent=2, default=str))
    log(f"\n  verdict -> {verdict_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
