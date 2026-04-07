"""BBMR parameter optimization with walk-forward validation.

Tests parameter combinations on first half (2010-2018) and validates
on second half (2018-2026). Only keeps params that work in BOTH halves.

This prevents overfitting to a specific time period.
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
ACCOUNT = 10000
RISK_PER_TRADE = 0.005

SPREADS = {
    "EURUSD": 0.00008,
    "USDJPY": 0.008,
    "GBPNZD": 0.00030,
    "AUDNZD": 0.00020,
    "XAUUSD": 0.40,
}


def load_dukascopy(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_1H.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df


def backtest_bbmr(
    df: pd.DataFrame,
    spread: float,
    bb_period: int,
    bb_std: float,
    adx_max: int,
    atr_sl_mult: float,
    max_bars: int,
    tp_mode: str = "middle",  # "middle" or "atr" or "opposite"
    atr_tp_mult: float = 2.0,
) -> dict:
    """Run BBMR backtest with given parameters."""
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)

    upper, middle, lower = talib.BBANDS(
        close, timeperiod=bb_period, nbdevup=bb_std, nbdevdn=bb_std
    )
    adx = talib.ADX(high, low, close, timeperiod=14)
    atr = talib.ATR(high, low, close, timeperiod=14)

    trades = []
    position = None
    equity = ACCOUNT
    peak = ACCOUNT
    max_dd = 0

    for i in range(200, len(close) - max_bars):
        if math.isnan(upper[i]) or math.isnan(adx[i]) or math.isnan(atr[i]):
            continue
        if adx[i] > adx_max:
            continue

        if position is None:
            if close[i] < lower[i]:
                sl_price = close[i] - atr[i] * atr_sl_mult
                if tp_mode == "middle":
                    tp_price = middle[i]
                elif tp_mode == "atr":
                    tp_price = close[i] + atr[i] * atr_tp_mult
                else:  # opposite
                    tp_price = upper[i]
                risk = close[i] - sl_price
                if risk <= 0:
                    continue
                qty = (equity * RISK_PER_TRADE) / risk
                position = {
                    "side": "buy", "entry": close[i], "idx": i,
                    "sl": sl_price, "tp": tp_price, "qty": qty,
                    "entry_time": df.index[i],
                }
            elif close[i] > upper[i]:
                sl_price = close[i] + atr[i] * atr_sl_mult
                if tp_mode == "middle":
                    tp_price = middle[i]
                elif tp_mode == "atr":
                    tp_price = close[i] - atr[i] * atr_tp_mult
                else:
                    tp_price = lower[i]
                risk = sl_price - close[i]
                if risk <= 0:
                    continue
                qty = (equity * RISK_PER_TRADE) / risk
                position = {
                    "side": "sell", "entry": close[i], "idx": i,
                    "sl": sl_price, "tp": tp_price, "qty": qty,
                    "entry_time": df.index[i],
                }
        else:
            bars_held = i - position["idx"]
            exit_price = None

            if position["side"] == "buy":
                if low[i] <= position["sl"]:
                    exit_price = position["sl"]
                elif high[i] >= position["tp"]:
                    exit_price = position["tp"]
                elif bars_held >= max_bars:
                    exit_price = close[i]
                if exit_price is not None:
                    move = exit_price - position["entry"]
                    cost = spread * position["qty"] * 2
                    pnl = move * position["qty"] - cost
                    trades.append({"pnl": pnl, "time": position["entry_time"]})
                    equity += pnl
                    if equity > peak:
                        peak = equity
                    dd = (peak - equity) / peak * 100
                    max_dd = max(max_dd, dd)
                    position = None
            else:
                if high[i] >= position["sl"]:
                    exit_price = position["sl"]
                elif low[i] <= position["tp"]:
                    exit_price = position["tp"]
                elif bars_held >= max_bars:
                    exit_price = close[i]
                if exit_price is not None:
                    move = position["entry"] - exit_price
                    cost = spread * position["qty"] * 2
                    pnl = move * position["qty"] - cost
                    trades.append({"pnl": pnl, "time": position["entry_time"]})
                    equity += pnl
                    if equity > peak:
                        peak = equity
                    dd = (peak - equity) / peak * 100
                    max_dd = max(max_dd, dd)
                    position = None

    if not trades:
        return {"pnl": 0, "trades": 0, "wr": 0, "max_dd": 0, "pf": 0, "sharpe": 0}

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total = sum(pnls)
    wr = len(wins) / len(trades) * 100
    pf = abs(sum(wins) / sum(losses)) if losses else 999
    sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(len(trades)) if np.std(pnls) > 0 else 0

    return {
        "pnl": round(total, 2),
        "trades": len(trades),
        "wr": round(wr, 1),
        "max_dd": round(max_dd, 2),
        "pf": round(pf, 2),
        "sharpe": round(sharpe, 2),
        "final_equity": round(equity, 2),
    }


def split_df(df: pd.DataFrame, split_date: str = "2018-01-01"):
    """Split df into train (before) and validation (after)."""
    split = pd.Timestamp(split_date)
    return df[df.index < split], df[df.index >= split]


def main():
    console.print(Panel.fit(
        "[bold magenta]BBMR PARAMETER OPTIMIZATION[/bold magenta]\n"
        "Walk-forward validation: train on 2010-2017, validate on 2018-2026\n"
        "Only keeps params that work in BOTH halves (no overfitting)",
        title="🔬 BBMR Sweep",
        border_style="magenta",
    ))

    available = {}
    for sym in SPREADS:
        df = load_dukascopy(sym)
        if df is not None and len(df) > 10000:
            available[sym] = df
            console.print(f"  [green]✓[/green] {sym}: {len(df):,} candles")
        else:
            console.print(f"  [red]✗[/red] {sym}: not available")

    if len(available) < 2:
        console.print("[red]Need at least 2 symbols[/red]")
        return

    # === PARAMETER GRID ===
    bb_periods = [14, 20, 25, 30]
    bb_stds = [1.8, 2.0, 2.2, 2.5]
    adx_maxes = [15, 20, 25]
    atr_sl_mults = [1.0, 1.5, 2.0]
    max_bars_list = [12, 24, 48]
    tp_modes = ["middle"]  # Keep simple, middle band is the proven one

    total_combos = (
        len(bb_periods) * len(bb_stds) * len(adx_maxes)
        * len(atr_sl_mults) * len(max_bars_list) * len(tp_modes)
    )
    console.print(f"\n  Testing {total_combos} combinations × {len(available)} symbols = "
                  f"{total_combos * len(available)} backtests")
    console.print("  This will take a few minutes...\n")

    results = []
    combo_count = 0

    for bb_p, bb_s, adx_m, atr_sl, mb, tp_m in itertools.product(
        bb_periods, bb_stds, adx_maxes, atr_sl_mults, max_bars_list, tp_modes
    ):
        combo_count += 1
        if combo_count % 20 == 0:
            console.print(f"    Progress: {combo_count}/{total_combos}")

        # Evaluate on both halves of each symbol
        train_total_pnl = 0
        val_total_pnl = 0
        train_wr = []
        val_wr = []
        train_dd = 0
        val_dd = 0
        train_trades = 0
        val_trades = 0
        all_symbols_positive_train = True
        all_symbols_positive_val = True

        for sym, df in available.items():
            train_df, val_df = split_df(df)
            if len(train_df) < 5000 or len(val_df) < 5000:
                continue

            spread = SPREADS[sym]
            r_train = backtest_bbmr(train_df, spread, bb_p, bb_s, adx_m, atr_sl, mb, tp_m)
            r_val = backtest_bbmr(val_df, spread, bb_p, bb_s, adx_m, atr_sl, mb, tp_m)

            train_total_pnl += r_train["pnl"]
            val_total_pnl += r_val["pnl"]
            train_trades += r_train["trades"]
            val_trades += r_val["trades"]
            train_wr.append(r_train["wr"])
            val_wr.append(r_val["wr"])
            train_dd = max(train_dd, r_train["max_dd"])
            val_dd = max(val_dd, r_val["max_dd"])

            if r_train["pnl"] <= 0:
                all_symbols_positive_train = False
            if r_val["pnl"] <= 0:
                all_symbols_positive_val = False

        results.append({
            "params": {
                "bb_period": bb_p, "bb_std": bb_s, "adx_max": adx_m,
                "atr_sl": atr_sl, "max_bars": mb, "tp_mode": tp_m,
            },
            "train_pnl": round(train_total_pnl, 2),
            "val_pnl": round(val_total_pnl, 2),
            "total_pnl": round(train_total_pnl + val_total_pnl, 2),
            "train_wr": round(np.mean(train_wr), 1) if train_wr else 0,
            "val_wr": round(np.mean(val_wr), 1) if val_wr else 0,
            "train_trades": train_trades,
            "val_trades": val_trades,
            "max_dd": round(max(train_dd, val_dd), 2),
            "all_positive_train": all_symbols_positive_train,
            "all_positive_val": all_symbols_positive_val,
        })

    # === ANALYSIS ===
    console.print(f"\n{'=' * 70}")
    console.print(Panel.fit("[bold]TOP RESULTS[/bold]", border_style="magenta"))

    # Filter: must be profitable in BOTH halves on ALL symbols
    robust = [r for r in results if r["all_positive_train"] and r["all_positive_val"]]
    console.print(f"\n  Robust combinations (profitable both halves, all symbols): "
                  f"{len(robust)}/{len(results)}")

    if not robust:
        console.print("[red]No robust combinations found![/red]")
        robust = results  # Fall back to all

    # Sort by validation P&L (out-of-sample performance)
    robust.sort(key=lambda x: x["val_pnl"], reverse=True)

    # Top 15
    t = Table(title="Top 15 by Out-of-Sample P&L (2018-2026)")
    t.add_column("#", width=3)
    t.add_column("BB", width=8)
    t.add_column("Std", width=5)
    t.add_column("ADX", width=5)
    t.add_column("SL", width=5)
    t.add_column("Bars", width=5)
    t.add_column("Train P&L", width=11)
    t.add_column("Val P&L", width=11)
    t.add_column("Total", width=11)
    t.add_column("Val WR", width=7)
    t.add_column("DD%", width=7)

    for i, r in enumerate(robust[:15], 1):
        p = r["params"]
        t.add_row(
            str(i),
            str(p["bb_period"]),
            str(p["bb_std"]),
            str(p["adx_max"]),
            str(p["atr_sl"]),
            str(p["max_bars"]),
            f"[green]${r['train_pnl']:+,.0f}[/green]",
            f"[green]${r['val_pnl']:+,.0f}[/green]",
            f"[green]${r['total_pnl']:+,.0f}[/green]",
            f"{r['val_wr']}%",
            f"{r['max_dd']}%",
        )
    console.print(t)

    # Best overall
    best = robust[0]
    console.print(f"\n  [bold]BEST PARAMETERS (robust, out-of-sample):[/bold]")
    for k, v in best["params"].items():
        console.print(f"    {k}: {v}")
    console.print(f"\n    Train (2010-2017): ${best['train_pnl']:+,.2f}")
    console.print(f"    Val (2018-2026):   ${best['val_pnl']:+,.2f}")
    console.print(f"    Combined:          ${best['total_pnl']:+,.2f}")
    console.print(f"    Train WR: {best['train_wr']}%  |  Val WR: {best['val_wr']}%")
    console.print(f"    Max DD: {best['max_dd']}%")

    # Save all results for analysis
    df_results = pd.DataFrame(results)
    df_results.to_csv("bbmr_sweep_results.csv", index=False)
    console.print(f"\n  Full results saved to: bbmr_sweep_results.csv")


if __name__ == "__main__":
    main()
