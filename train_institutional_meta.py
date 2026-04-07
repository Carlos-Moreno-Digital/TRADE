"""Institutional training run for the Triple-Barrier Meta-Labelling
strategy on the full Dukascopy 16-year tape.

Pipeline (single command, no manual intervention):

  1. Pre-flight
       - Locate the largest available CSV (calib > 1H > yfinance fallback)
       - 12-test LOBFrame leakage suite on a slice (sanity)
  2. Grid search over a coarse hyperparameter space
       Outer loop  = primary (RegimeMomentum) parameters
                     -> primary signals are recomputed once per outer
                        combo (the expensive step)
       Inner loop  = meta (RandomForestClassifier) parameters
                     -> meta is re-fit on each fold inside purged K-fold
                        with the inner combo's hyperparameters
       For each (outer x inner) combo:
         a) Get primary signals
         b) Build meta features at every signal bar (causal)
         c) Compute triple-barrier labels for every signal
         d) Run Purged K-Fold (Lopez de Prado AFML ch 7) to obtain
            STRICTLY OUT-OF-SAMPLE meta probabilities for every signal
         e) Filter signals by meta_threshold
         f) Walk positions on the kept signals (pessimistic SL-first
            ordering + explicit spread cost = same accounting as the
            Nautilus harness)
         g) Compute total P&L, Sharpe, win rate, max drawdown
       Best combo selected by Sharpe ratio.
  3. Final fit
       Train ONE RandomForestClassifier on ALL signals (no purge —
       this is the production model that will run forward) using the
       best meta hyperparameters from the grid search.
       Persist with joblib to data/models/meta_rf_{symbol}.joblib.
  4. Verdict
       Write data/models/meta_rf_{symbol}_verdict.json with:
         - n_bars, n_signals, n_kept, filter_rate
         - best primary + meta hyperparameters
         - OOS sharpe / pnl / win rate / max drawdown
         - viable boolean (sharpe > 0.5 AND positive PnL)

Token-budget rule (still in force):
  - One log line per primary combo
  - One log line per inner meta combo
  - One log line per K-fold inside purged K-fold
  - Final verdict as a small JSON
  No dataframes ever leave the script.

Run on the VPS:
    python train_institutional_meta.py --symbol EURUSD --kfolds 5
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).parent / "src"))

import joblib

from trade.research.lobframe import calibrated_config
from trade.research.lobframe.leakage_tests import run_all as run_leakage
from trade.research.strategies.meta_labelling import (
    META_FEATURE_COLS,
    _build_meta_features,
    _triple_barrier_label,
)
from trade.research.strategies.regime_momentum import RegimeMomentum
from trade.validation.purged_kfold import (
    PurgedKFoldFold,
    purged_kfold_predict_proba,
)

DATA_DIR = Path("data/dukascopy")
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ACCOUNT = 10_000.0
RISK_PER_TRADE = 0.003

PAIR_BY_SYMBOL = {"EURUSD": "EUR/USD", "USDJPY": "USD/JPY"}
SPREAD_BY_SYMBOL = {"EURUSD": 8e-5, "USDJPY": 8e-3}

# Coarse grid — small enough to fit in a few hours on 2 vCPU.
PRIMARY_GRID: dict = {
    "donchian_period": [15, 20, 30],
    "atr_sl_mult": [1.0, 1.5],
    "atr_tp_mult": [2.0, 3.0],
    "max_holding": [12, 24],
}
META_GRID: dict = {
    "rf_n_estimators": [50, 100],
    "rf_max_depth": [4, 6],
    "rf_min_samples_leaf": [5],
    "meta_threshold": [0.40, 0.50, 0.60],
}


def log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
# Bars are loaded with strict priority. The calib CSV is the small
# 1-month sample produced by download_calibration_data.py and is
# only useful for spread/depth calibration of the LOB synthesizer;
# it is NEVER used as the time series for the institutional run.
#
# Order of preference for the bar series:
#   1. data/dukascopy/{SYMBOL}_1H.csv (the historical Dukascopy tape
#      built by download_dukascopy.py - up to 16 years)
#   2. yfinance fallback (~2 years of 1H bars) if (1) is missing or
#      too short
#
# A hard minimum of MIN_BARS_REQUIRED rows is enforced — anything
# below that is considered insufficient for institutional grid
# search and the script aborts with a clear message.

PRIMARY_HISTORY_FILE = "{symbol}_1H.csv"
MIN_BARS_REQUIRED = 5_000  # ~7 months of 1H bars at minimum

YFINANCE_TICKERS = {
    "EURUSD": "EURUSD=X",
    "USDJPY": "JPY=X",
    "GBPUSD": "GBPUSD=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCAD": "CAD=X",
}


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df[["open", "high", "low", "close", "volume"]]


def _yfinance_fallback(symbol: str) -> pd.DataFrame | None:
    ticker = YFINANCE_TICKERS.get(symbol)
    if ticker is None:
        return None
    try:
        import yfinance as yf
    except ImportError:
        log("  yfinance not installed; cannot fall back")
        return None
    try:
        df = yf.download(
            ticker, period="2y", interval="1h",
            auto_adjust=False, progress=False,
        )
    except Exception as e:
        log(f"  yfinance fallback failed: {e}")
        return None
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df = df[[c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]]
    df.index.name = "timestamp"
    df.index = df.index.tz_localize(None)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df


def load_full_bars(symbol: str) -> pd.DataFrame:
    """Strict load: full 1H historical first, yfinance fallback second.

    Raises FileNotFoundError if both are missing or both produce
    fewer than MIN_BARS_REQUIRED rows.
    """
    history_path = DATA_DIR / PRIMARY_HISTORY_FILE.format(symbol=symbol)

    df: pd.DataFrame | None = None
    source: str = ""
    if history_path.exists():
        df = _load_csv(history_path)
        source = history_path.name
        log(f"  data: {source}  rows={len(df):,}  "
            f"{df.index[0].date()} -> {df.index[-1].date()}")
        if len(df) < MIN_BARS_REQUIRED:
            log(
                f"  WARN: {source} has only {len(df):,} rows "
                f"(< {MIN_BARS_REQUIRED:,} minimum). "
                "Trying yfinance fallback for a longer series."
            )
            df = None

    if df is None:
        log(f"  attempting yfinance fallback for {symbol} (~2y 1H)...")
        df = _yfinance_fallback(symbol)
        if df is not None:
            source = f"yfinance:{YFINANCE_TICKERS.get(symbol)}"
            log(
                f"  data: {source}  rows={len(df):,}  "
                f"{df.index[0].date()} -> {df.index[-1].date()}"
            )

    if df is None or len(df) < MIN_BARS_REQUIRED:
        n = 0 if df is None else len(df)
        raise FileNotFoundError(
            f"insufficient bars for {symbol}: only {n:,} rows available "
            f"(need >= {MIN_BARS_REQUIRED:,}). "
            f"Run `python download_dukascopy.py` to fetch the full historical "
            f"tape, or ensure {history_path} exists with the long series. "
            f"Note: {DATA_DIR / (symbol + '_calib.csv')} is the calibration "
            f"sample and is intentionally NOT used for institutional training."
        )

    return df


# ------------------------------------------------------------------
# Position walker (matches NautilusHarness pessimistic accounting)
# ------------------------------------------------------------------
def walk_positions(
    bars: pd.DataFrame,
    spread: float,
    kept_signals: list[dict],
    max_holding: int,
) -> list[float]:
    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    n = len(close)
    entries = {int(s["bar_idx"]) - 1: s for s in kept_signals}
    pnls: list[float] = []
    pos = None
    equity = ACCOUNT
    for i in range(n - 1):
        if pos is None:
            sig = entries.get(i)
            if sig is None:
                continue
            side = sig["side"]
            sl = float(sig["sl"])
            tp = float(sig["tp"])
            entry = close[i]
            risk = (entry - sl) if side == "buy" else (sl - entry)
            if risk <= 0:
                continue
            qty = (equity * RISK_PER_TRADE) / risk
            pos = {
                "side": side, "entry": entry, "idx": i,
                "sl": sl, "tp": tp, "qty": qty,
            }
            continue
        bars_held = i - pos["idx"]
        exit_price = None
        if pos["side"] == "buy":
            if low[i] <= pos["sl"]:
                exit_price = pos["sl"]
            elif high[i] >= pos["tp"]:
                exit_price = pos["tp"]
            elif bars_held >= max_holding:
                exit_price = close[i]
            if exit_price is not None:
                pnl = (exit_price - pos["entry"]) * pos["qty"] - spread * pos["qty"] * 2
                pnls.append(pnl)
                equity += pnl
                pos = None
        else:
            if high[i] >= pos["sl"]:
                exit_price = pos["sl"]
            elif low[i] <= pos["tp"]:
                exit_price = pos["tp"]
            elif bars_held >= max_holding:
                exit_price = close[i]
            if exit_price is not None:
                pnl = (pos["entry"] - exit_price) * pos["qty"] - spread * pos["qty"] * 2
                pnls.append(pnl)
                equity += pnl
                pos = None
    return pnls


# ------------------------------------------------------------------
# Metric helpers
# ------------------------------------------------------------------
def compute_metrics(pnls: list[float], periods_per_year: int = 220) -> dict:
    arr = np.array(pnls, dtype=float) if pnls else np.array([])
    n = len(arr)
    if n == 0:
        return {
            "n_trades": 0, "pnl": 0.0, "wr": 0.0,
            "sharpe": 0.0, "max_dd": 0.0,
        }
    total = float(arr.sum())
    wr = float((arr > 0).mean())
    if arr.std() > 0:
        sharpe = float(arr.mean() / arr.std() * math.sqrt(periods_per_year))
    else:
        sharpe = 0.0
    eq = ACCOUNT + np.cumsum(arr)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(dd.min())
    return {
        "n_trades": n,
        "pnl": round(total, 2),
        "wr": round(wr, 3),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd, 3),
    }


# ------------------------------------------------------------------
# Per-combo evaluation: signals -> labels -> purged kfold -> walk
# ------------------------------------------------------------------
def extract_signal_features_and_labels(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    feats: pd.DataFrame,
    max_holding: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the (X, y, signal_bar_idx) triple from a primary signals
    DataFrame. Drops signals whose feature row contains NaN.
    """
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)

    rows = []
    labels = []
    bar_idx = []
    for s in signals.itertuples():
        t0 = int(s.bar_idx) - 1
        if t0 < 0 or t0 >= len(feats):
            continue
        f = feats.iloc[t0]
        if f.isna().any():
            continue
        label = _triple_barrier_label(
            t0, s.side, float(s.sl), float(s.tp), high, low, max_holding,
        )
        row = list(f.values) + [1.0 if s.side == "buy" else 0.0]
        rows.append(row)
        labels.append(label)
        bar_idx.append(t0)
    if not rows:
        return (
            np.empty((0, len(META_FEATURE_COLS)), dtype=float),
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
        )
    return (
        np.array(rows, dtype=float),
        np.array(labels, dtype=int),
        np.array(bar_idx, dtype=int),
    )


def evaluate_combo(
    bars: pd.DataFrame,
    spread: float,
    primary: RegimeMomentum,
    primary_params: dict,
    meta_params: dict,
    n_splits: int,
    embargo_bars: int,
    min_labels: int = 50,
) -> dict:
    signals = primary.signals(bars, primary_params)
    n_primary = len(signals)
    if n_primary == 0:
        return {"n_signals": 0, "n_labelled": 0, "n_kept": 0,
                "metrics": compute_metrics([]),
                "valid": False, "reason": "no primary signals"}

    feats = _build_meta_features(bars, atr_period=int(primary_params["atr_period"]))
    X, y, sig_bar_idx = extract_signal_features_and_labels(
        bars, signals, feats, int(primary_params["max_holding"])
    )
    if len(X) < min_labels:
        return {"n_signals": n_primary, "n_labelled": int(len(X)),
                "n_kept": 0,
                "metrics": compute_metrics([]),
                "valid": False, "reason": f"<{min_labels} labelled signals"}

    def factory():
        return RandomForestClassifier(
            n_estimators=int(meta_params["rf_n_estimators"]),
            max_depth=int(meta_params["rf_max_depth"]),
            min_samples_leaf=int(meta_params["rf_min_samples_leaf"]),
            random_state=42,
            n_jobs=1,
            class_weight="balanced",
        )

    oos_probs, folds = purged_kfold_predict_proba(
        X=X, y=y,
        signal_bar_indices=sig_bar_idx,
        label_horizon=int(primary_params["max_holding"]),
        n_splits=n_splits,
        embargo_bars=embargo_bars,
        classifier_factory=factory,
        progress_fn=None,  # silent inside the inner loop
    )

    threshold = float(meta_params["meta_threshold"])
    keep_mask = (~np.isnan(oos_probs)) & (oos_probs >= threshold)

    valid_signals = signals.iloc[: len(X)]  # rows that produced labels
    kept_records = valid_signals.iloc[keep_mask].to_dict("records")
    pnls = walk_positions(
        bars, spread, kept_records, int(primary_params["max_holding"])
    )
    metrics = compute_metrics(pnls)
    return {
        "n_signals": n_primary,
        "n_labelled": len(X),
        "n_kept": int(keep_mask.sum()),
        "filter_rate": round(1 - (int(keep_mask.sum()) / max(len(X), 1)), 3),
        "metrics": metrics,
        "valid": True,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--embargo", type=int, default=20)
    parser.add_argument("--max-bars", type=int, default=0,
                        help="Cap on bars (0 = use full file)")
    parser.add_argument("--min-labels", type=int, default=50,
                        help="Skip a combo if it produced fewer labelled signals")
    parser.add_argument("--min-trades-viable", type=int, default=50,
                        help="Minimum kept trades for viability")
    args = parser.parse_args()

    sym = args.symbol
    pair = PAIR_BY_SYMBOL.get(sym, "EUR/USD")
    spread = SPREAD_BY_SYMBOL.get(sym, 8e-5)

    log(f"=== train_institutional_meta.py ===")
    log(f"  symbol={sym} pair={pair} spread={spread} kfolds={args.kfolds}")

    bars = load_full_bars(sym)
    if args.max_bars > 0:
        bars = bars.iloc[: args.max_bars].copy()
        log(f"  capped to first {args.max_bars} bars")

    # ---- Pre-flight ----
    log("\n=== Pre-flight: leakage suite ===")
    cfg = calibrated_config(sym)
    sample = bars.iloc[: min(2000, len(bars))]
    leak = run_leakage(sample, cfg)
    passed = sum(int(v) for v in leak.values())
    log(f"  leakage suite: {passed}/{len(leak)} passed")
    if passed != len(leak):
        for name, ok in leak.items():
            if not ok:
                log(f"    [FAIL] {name}")
        log("ABORT: leakage suite failed")
        return 2

    # ---- Grid search ----
    log("\n=== Grid search ===")
    primary_keys = list(PRIMARY_GRID.keys())
    meta_keys = list(META_GRID.keys())
    primary_combos = list(itertools.product(*PRIMARY_GRID.values()))
    meta_combos = list(itertools.product(*META_GRID.values()))
    log(f"  primary combos: {len(primary_combos)}  "
        f"meta combos: {len(meta_combos)}  "
        f"total fits: {len(primary_combos) * len(meta_combos) * args.kfolds}")

    primary = RegimeMomentum(symbol=sym)
    base_primary = dict(primary.default_params)

    results: list[dict] = []
    t0 = time.time()
    for pi, pcombo in enumerate(primary_combos, 1):
        primary_params = dict(base_primary)
        primary_params.update(dict(zip(primary_keys, pcombo)))
        log(f"\n  primary {pi}/{len(primary_combos)}: {dict(zip(primary_keys, pcombo))}")
        for mi, mcombo in enumerate(meta_combos, 1):
            meta_params = dict(zip(meta_keys, mcombo))
            r = evaluate_combo(
                bars=bars, spread=spread,
                primary=primary,
                primary_params=primary_params,
                meta_params=meta_params,
                n_splits=args.kfolds,
                embargo_bars=args.embargo,
                min_labels=args.min_labels,
            )
            r["primary_params"] = dict(zip(primary_keys, pcombo))
            r["meta_params"] = meta_params
            results.append(r)
            m = r["metrics"]
            log(
                f"    meta {mi:2d}/{len(meta_combos)} "
                f"{meta_params}  "
                f"kept={r.get('n_kept', 0):>4d}/{r.get('n_labelled', 0):>4d}  "
                f"pnl=${m['pnl']:+,.0f}  "
                f"sharpe={m['sharpe']:+.2f}  "
                f"wr={m['wr']:.2f}  "
                f"max_dd={m['max_dd']:+.2f}"
            )

    log(f"\n  grid search wall time: {time.time() - t0:.0f}s")

    # ---- Pick best ----
    valid = [r for r in results if r.get("valid")]
    if not valid:
        log("ABORT: no valid combo")
        return 3
    best = max(valid, key=lambda r: r["metrics"]["sharpe"])
    log(f"\n=== Best combo ===")
    log(f"  primary: {best['primary_params']}")
    log(f"  meta:    {best['meta_params']}")
    log(f"  metrics: {best['metrics']}")
    log(f"  signals: primary={best['n_signals']} labelled={best['n_labelled']} "
        f"kept={best['n_kept']} filter_rate={best['filter_rate']}")

    # ---- Final fit on all signals ----
    log("\n=== Final fit on all signals ===")
    final_primary_params = dict(base_primary)
    final_primary_params.update(best["primary_params"])
    signals = primary.signals(bars, final_primary_params)
    feats = _build_meta_features(bars, atr_period=int(final_primary_params["atr_period"]))
    X, y, _ = extract_signal_features_and_labels(
        bars, signals, feats, int(final_primary_params["max_holding"])
    )
    if len(X) >= 30 and len(np.unique(y)) >= 2:
        final_rf = RandomForestClassifier(
            n_estimators=int(best["meta_params"]["rf_n_estimators"]),
            max_depth=int(best["meta_params"]["rf_max_depth"]),
            min_samples_leaf=int(best["meta_params"]["rf_min_samples_leaf"]),
            random_state=42,
            n_jobs=1,
            class_weight="balanced",
        )
        final_rf.fit(X, y)
        model_path = MODELS_DIR / f"meta_rf_{sym}.joblib"
        joblib.dump({
            "rf": final_rf,
            "feature_cols": META_FEATURE_COLS,
            "primary_params": best["primary_params"],
            "meta_params": best["meta_params"],
            "n_train_samples": int(len(X)),
        }, model_path)
        log(f"  model -> {model_path}")
    else:
        log("  WARN: insufficient labels for final fit, skipping model dump")

    # ---- Verdict ----
    sharpe = best["metrics"]["sharpe"]
    pnl = best["metrics"]["pnl"]
    n_kept = best.get("n_kept", 0)
    viable = (
        sharpe > 0.5
        and pnl > 0
        and n_kept >= args.min_trades_viable
    )
    verdict = {
        "symbol": sym,
        "n_bars": int(len(bars)),
        "n_signals": int(best["n_signals"]),
        "n_labelled": int(best["n_labelled"]),
        "n_kept": int(best["n_kept"]),
        "filter_rate": float(best["filter_rate"]),
        "best_primary": best["primary_params"],
        "best_meta": best["meta_params"],
        "metrics": best["metrics"],
        "viable": bool(viable),
        "wall_time_seconds": int(time.time() - t0),
    }
    out = MODELS_DIR / f"meta_rf_{sym}_verdict.json"
    out.write_text(json.dumps(verdict, indent=2))
    log(f"\n  verdict -> {out}")
    log(f"  VIABLE={verdict['viable']}")
    return 0 if verdict["viable"] else 1


if __name__ == "__main__":
    sys.exit(main())
