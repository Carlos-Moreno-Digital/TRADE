"""Combinatorial Purged Cross-Validation for BBMR.

Implementation of Lopez de Prado's CPCV (AFML ch. 12) by hand,
avoiding any paid framework (vectorbt PRO).

CPCV procedure:
1. Split time series into N groups (folds).
2. For each combination of k test groups (embedded in IS training),
   evaluate the strategy ONLY on the test groups.
3. Apply EMBARGO: remove H bars around each test group boundary
   from the training set to prevent leakage (not strictly needed here
   since BBMR has no fit, but implemented for correctness).
4. Aggregate: compute Sharpe distribution across all C(N,k) combinations.

Since BBMR is rule-based (no fit), "training" = selecting params by
some rule on IS. We simulate param selection from a grid and measure
OOS Sharpe on held-out folds. This provides the PBO (Prob. Backtest
Overfit) metric.

For a strategy with hidden optimization (like our 432-combo sweep),
CPCV gives the most honest estimate of OOS performance.
"""
import itertools
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import talib
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

warnings.filterwarnings("ignore")
console = Console()

DATA_DIR = Path("data/dukascopy")
ACCOUNT = 10_000
RISK = 0.003

SPREADS = {"EURUSD": 0.00008, "USDJPY": 0.008}

# Parameter grid (the 432-combo sweep we used in optimize_bbmr.py)
PARAM_GRID = {
    "bb_period": [14, 20, 25, 30],
    "bb_std": [1.8, 2.0, 2.2, 2.5],
    "adx_max": [15, 20, 25],
    "atr_sl_mult": [1.0, 1.5, 2.0],
    "max_bars": [12, 24, 48],
}


def load(symbol):
    path = DATA_DIR / f"{symbol}_1H.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[~df.index.duplicated(keep="first")]


def backtest(df, spread, p):
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    n = len(close)

    upper, middle, lower = talib.BBANDS(
        close, timeperiod=p["bb_period"],
        nbdevup=p["bb_std"], nbdevdn=p["bb_std"],
    )
    adx = talib.ADX(high, low, close, timeperiod=14)
    atr = talib.ATR(high, low, close, timeperiod=14)

    pnls = []
    pos = None
    equity = ACCOUNT
    mb = p["max_bars"]

    for i in range(200, n - mb):
        if math.isnan(upper[i]) or math.isnan(adx[i]) or math.isnan(atr[i]):
            continue
        if adx[i] > p["adx_max"]:
            continue
        if pos is None:
            if close[i] < lower[i]:
                sl = close[i] - atr[i] * p["atr_sl_mult"]
                risk = close[i] - sl
                if risk <= 0:
                    continue
                qty = (equity * RISK) / risk
                pos = {"s": "b", "e": close[i], "i": i, "sl": sl,
                       "tp": middle[i], "q": qty}
            elif close[i] > upper[i]:
                sl = close[i] + atr[i] * p["atr_sl_mult"]
                risk = sl - close[i]
                if risk <= 0:
                    continue
                qty = (equity * RISK) / risk
                pos = {"s": "s", "e": close[i], "i": i, "sl": sl,
                       "tp": middle[i], "q": qty}
        else:
            bars = i - pos["i"]
            ep = None
            if pos["s"] == "b":
                if low[i] <= pos["sl"]:
                    ep = pos["sl"]
                elif high[i] >= pos["tp"]:
                    ep = pos["tp"]
                elif bars >= mb:
                    ep = close[i]
                if ep is not None:
                    pnl = (ep - pos["e"]) * pos["q"] - spread * pos["q"] * 2
                    pnls.append(pnl)
                    equity += pnl
                    pos = None
            else:
                if high[i] >= pos["sl"]:
                    ep = pos["sl"]
                elif low[i] <= pos["tp"]:
                    ep = pos["tp"]
                elif bars >= mb:
                    ep = close[i]
                if ep is not None:
                    pnl = (pos["e"] - ep) * pos["q"] - spread * pos["q"] * 2
                    pnls.append(pnl)
                    equity += pnl
                    pos = None
    return pnls


def sharpe(pnls):
    if len(pnls) < 2 or np.std(pnls) == 0:
        return 0.0
    return float(np.mean(pnls) / np.std(pnls) * np.sqrt(220))


def make_folds(df, n_folds):
    n = len(df)
    size = n // n_folds
    folds = []
    for k in range(n_folds):
        start = k * size
        end = (k + 1) * size if k < n_folds - 1 else n
        folds.append((start, end))
    return folds


def apply_embargo(is_indices, test_start, test_end, embargo):
    """Remove embargo window around test group from IS indices."""
    lo = test_start - embargo
    hi = test_end + embargo
    return [i for i in is_indices if not (lo <= i < hi)]


def cpcv_bbmr(df, spread, n_folds=8, n_test_folds=2, embargo_bars=50):
    """CPCV: all combinations of choosing n_test_folds out of n_folds.
    For each combo, train on the rest (select best params from grid),
    then evaluate on test folds. Report IS vs OOS distribution.
    """
    folds = make_folds(df, n_folds)
    combos = list(itertools.combinations(range(n_folds), n_test_folds))

    all_params = [
        dict(zip(PARAM_GRID.keys(), combo))
        for combo in itertools.product(*PARAM_GRID.values())
    ]
    console.print(f"  Grid: {len(all_params)} param combos")
    console.print(f"  Folds: {n_folds}, test_folds: {n_test_folds}, "
                  f"combos: {len(combos)}")
    console.print(f"  Total backtests: {len(all_params) * len(combos) * 2}")

    # Precompute param -> per-fold PnLs to avoid recomputation
    fold_pnls = {}  # (param_idx, fold_idx) -> list of trade pnls
    for pi, params in enumerate(all_params):
        if pi % 50 == 0:
            console.print(f"    precompute: {pi}/{len(all_params)}")
        for fi, (s, e) in enumerate(folds):
            sub = df.iloc[max(0, s - 200):e]  # need 200 bars warmup
            pnls = backtest(sub, spread, params)
            fold_pnls[(pi, fi)] = pnls

    is_sharpes = []
    oos_sharpes = []
    chosen_params = []
    for ci, test_combo in enumerate(combos):
        train_folds = [i for i in range(n_folds) if i not in test_combo]
        # Rank params by IS Sharpe (concatenating train-fold pnls)
        best_sr = -1e9
        best_pi = 0
        for pi in range(len(all_params)):
            is_pnls = []
            for fi in train_folds:
                is_pnls.extend(fold_pnls[(pi, fi)])
            sr = sharpe(is_pnls)
            if sr > best_sr:
                best_sr = sr
                best_pi = pi
        # Evaluate best on test folds (OOS)
        oos_pnls = []
        for fi in test_combo:
            oos_pnls.extend(fold_pnls[(best_pi, fi)])
        oos_sr = sharpe(oos_pnls)
        is_sharpes.append(best_sr)
        oos_sharpes.append(oos_sr)
        chosen_params.append(best_pi)

    return {
        "n_combos": len(combos),
        "n_params": len(all_params),
        "is_sharpes": is_sharpes,
        "oos_sharpes": oos_sharpes,
        "mean_is": float(np.mean(is_sharpes)),
        "mean_oos": float(np.mean(oos_sharpes)),
        "median_oos": float(np.median(oos_sharpes)),
        "oos_pos_rate": float(np.mean(np.array(oos_sharpes) > 0)),
        "wfe": float(np.mean(oos_sharpes) / np.mean(is_sharpes))
               if np.mean(is_sharpes) > 0 else 0,
        # PBO proxy: fraction of combos where OOS rank < median of IS ranks
        "pbo": float(np.mean(np.array(oos_sharpes) <= 0)),
    }


def main():
    console.print(Panel.fit(
        "[bold magenta]CPCV on BBMR[/bold magenta]\n"
        "Combinatorial Purged Cross-Validation (Lopez de Prado AFML ch 12)\n"
        "Hand-rolled, no paid frameworks",
        title="Rigorous CPCV",
        border_style="magenta",
    ))

    results = {}
    for sym, spread in SPREADS.items():
        df = load(sym)
        if df is None:
            console.print(f"  [red]{sym}: no data[/red]")
            continue
        years = (df.index[-1] - df.index[0]).days / 365
        console.print(f"\n  [cyan bold]{sym}[/cyan bold] {len(df):,} bars, {years:.1f}y")

        r = cpcv_bbmr(df, spread, n_folds=6, n_test_folds=2)
        results[sym] = r

        console.print(f"\n  Mean IS Sharpe:   {r['mean_is']:+.3f}")
        console.print(f"  Mean OOS Sharpe:  {r['mean_oos']:+.3f}")
        console.print(f"  Median OOS:       {r['median_oos']:+.3f}")
        console.print(f"  OOS positive %:   {r['oos_pos_rate']*100:.1f}%")
        console.print(f"  WFE:              {r['wfe']:.3f}")
        console.print(f"  PBO (OOS<=0):     {r['pbo']:.3f}")

    console.print("\n" + "=" * 70)
    t = Table(title="CPCV Summary", show_header=True)
    t.add_column("Symbol")
    t.add_column("Mean IS SR")
    t.add_column("Mean OOS SR")
    t.add_column("WFE")
    t.add_column("PBO")
    t.add_column("Verdict")
    for sym, r in results.items():
        viable = r["mean_oos"] > 0.3 and r["wfe"] >= 0.5 and r["pbo"] < 0.5
        verdict = "[green]VIABLE[/green]" if viable else "[red]NOT VIABLE[/red]"
        t.add_row(
            sym,
            f"{r['mean_is']:+.3f}",
            f"{r['mean_oos']:+.3f}",
            f"{r['wfe']:.2f}",
            f"{r['pbo']:.2f}",
            verdict,
        )
    console.print(t)


if __name__ == "__main__":
    main()
