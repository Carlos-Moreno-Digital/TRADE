"""Rigorous backtest of Bollinger Band Mean Reversion strategy.

The ML model failed on 16 years of real data (-$26K).
The forensic analysis found that simple BB mean reversion works:
  EURUSD: +$4,249 (53% WR)
  USDJPY: +$3,931 (53% WR)

This script validates the strategy rigorously before changing the bot:
- Year-by-year breakdown (consistency check)
- All available Dukascopy instruments
- Proper prop firm risk management (0.5% risk per trade)
- Max drawdown tracking
- Sharpe, Calmar, profit factor
"""

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
RISK_PER_TRADE = 0.005  # 0.5% of account per trade

SYMBOL_CONFIG = {
    "EURUSD": {"spread": 0.00008, "pip_value": 10},
    "USDJPY": {"spread": 0.008, "pip_value": 9},
    "GBPNZD": {"spread": 0.00030, "pip_value": 6},
    "AUDNZD": {"spread": 0.00020, "pip_value": 6},
    "XAUUSD": {"spread": 0.40, "pip_value": 10},
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
    bb_period: int = 20,
    bb_std: float = 2.0,
    adx_max: int = 20,
    atr_sl_mult: float = 1.5,
    max_bars: int = 24,
) -> dict:
    """Bollinger Band Mean Reversion backtest with proper risk management."""
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)

    upper, middle, lower = talib.BBANDS(close, timeperiod=bb_period, nbdevup=bb_std, nbdevdn=bb_std)
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
        if adx[i] > adx_max:  # Only ranging markets
            continue

        if position is None:
            # BUY at lower band
            if close[i] < lower[i]:
                sl_price = close[i] - atr[i] * atr_sl_mult
                tp_price = middle[i]
                risk_per_unit = close[i] - sl_price
                if risk_per_unit <= 0:
                    continue
                # Position size: risk 0.5% of equity
                qty = (equity * RISK_PER_TRADE) / risk_per_unit
                position = {
                    "side": "buy",
                    "entry": close[i],
                    "idx": i,
                    "sl": sl_price,
                    "tp": tp_price,
                    "qty": qty,
                    "entry_time": df.index[i],
                }
            # SELL at upper band
            elif close[i] > upper[i]:
                sl_price = close[i] + atr[i] * atr_sl_mult
                tp_price = middle[i]
                risk_per_unit = sl_price - close[i]
                if risk_per_unit <= 0:
                    continue
                qty = (equity * RISK_PER_TRADE) / risk_per_unit
                position = {
                    "side": "sell",
                    "entry": close[i],
                    "idx": i,
                    "sl": sl_price,
                    "tp": tp_price,
                    "qty": qty,
                    "entry_time": df.index[i],
                }
        else:
            bars_held = i - position["idx"]
            exit_price = None
            exit_reason = None

            if position["side"] == "buy":
                if low[i] <= position["sl"]:
                    exit_price = position["sl"]
                    exit_reason = "SL"
                elif high[i] >= position["tp"]:
                    exit_price = position["tp"]
                    exit_reason = "TP"
                elif bars_held >= max_bars:
                    exit_price = close[i]
                    exit_reason = "TIME"
                if exit_price is not None:
                    price_move = exit_price - position["entry"]
                    cost = spread * position["qty"] * 2
                    pnl = price_move * position["qty"] - cost
                    trades.append({
                        "entry_time": position["entry_time"],
                        "exit_time": df.index[i],
                        "side": "buy",
                        "entry": position["entry"],
                        "exit": exit_price,
                        "pnl": pnl,
                        "reason": exit_reason,
                        "bars_held": bars_held,
                    })
                    equity += pnl
                    if equity > peak:
                        peak = equity
                    dd = (peak - equity) / peak * 100
                    max_dd = max(max_dd, dd)
                    position = None
            else:  # sell
                if high[i] >= position["sl"]:
                    exit_price = position["sl"]
                    exit_reason = "SL"
                elif low[i] <= position["tp"]:
                    exit_price = position["tp"]
                    exit_reason = "TP"
                elif bars_held >= max_bars:
                    exit_price = close[i]
                    exit_reason = "TIME"
                if exit_price is not None:
                    price_move = position["entry"] - exit_price
                    cost = spread * position["qty"] * 2
                    pnl = price_move * position["qty"] - cost
                    trades.append({
                        "entry_time": position["entry_time"],
                        "exit_time": df.index[i],
                        "side": "sell",
                        "entry": position["entry"],
                        "exit": exit_price,
                        "pnl": pnl,
                        "reason": exit_reason,
                        "bars_held": bars_held,
                    })
                    equity += pnl
                    if equity > peak:
                        peak = equity
                    dd = (peak - equity) / peak * 100
                    max_dd = max(max_dd, dd)
                    position = None

    if not trades:
        return {"error": "No trades"}

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    wr = len(wins) / len(trades) * 100
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    pf = abs(sum(wins) / sum(losses)) if losses else 999

    sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252 * 24) if np.std(pnls) > 0 else 0

    # Yearly breakdown
    yearly = {}
    for t in trades:
        year = str(t["entry_time"])[:4]
        if year not in yearly:
            yearly[year] = {"pnl": 0, "trades": 0, "wins": 0}
        yearly[year]["pnl"] += t["pnl"]
        yearly[year]["trades"] += 1
        if t["pnl"] > 0:
            yearly[year]["wins"] += 1

    return {
        "total_pnl": round(total_pnl, 2),
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(wr, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(pf, 2),
        "max_dd_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "final_equity": round(equity, 2),
        "return_pct": round((equity - ACCOUNT) / ACCOUNT * 100, 2),
        "yearly": yearly,
    }


def main():
    console.print(Panel.fit(
        "[bold green]BOLLINGER BAND MEAN REVERSION — Rigorous Backtest[/bold green]\n"
        f"Account: ${ACCOUNT:,} | Risk: {RISK_PER_TRADE*100:.1f}% per trade\n"
        "Only trade when ADX < 20 (ranging markets)\n"
        "Entry: price touches band | Exit: middle band / SL / 24 bars",
        title="🎯 BBMR Validation",
        border_style="green",
    ))

    # Load all available symbols
    available = {}
    for sym in SYMBOL_CONFIG:
        df = load_dukascopy(sym)
        if df is not None and len(df) > 10000:
            available[sym] = df
            years = (df.index[-1] - df.index[0]).days / 365
            console.print(f"  [green]✓[/green] {sym}: {len(df):,} candles, {years:.1f} years")
        else:
            console.print(f"  [red]✗[/red] {sym}: not available")

    console.print()

    all_results = {}
    combined_yearly = {}
    combined_total = 0

    for sym, df in available.items():
        console.print(f"  [cyan]Backtesting {sym}...[/cyan]", end=" ")
        config = SYMBOL_CONFIG[sym]
        result = backtest_bbmr(df, config["spread"])
        if "error" in result:
            console.print(f"[red]{result['error']}[/red]")
            continue

        color = "green" if result["total_pnl"] > 0 else "red"
        console.print(
            f"{result['trades']} trades | {result['win_rate']}% WR | "
            f"[{color}]${result['total_pnl']:+,.0f}[/{color}] | "
            f"PF {result['profit_factor']} | DD {result['max_dd_pct']}%"
        )

        all_results[sym] = result
        combined_total += result["total_pnl"]

        for year, data in result["yearly"].items():
            if year not in combined_yearly:
                combined_yearly[year] = {"pnl": 0, "trades": 0}
            combined_yearly[year]["pnl"] += data["pnl"]
            combined_yearly[year]["trades"] += data["trades"]

    # === SUMMARY TABLE ===
    console.print(f"\n{'=' * 70}")
    console.print(Panel.fit("[bold]DETAILED METRICS[/bold]", border_style="magenta"))

    t = Table(show_header=True)
    t.add_column("Symbol", width=10)
    t.add_column("Trades", width=8)
    t.add_column("WR%", width=6)
    t.add_column("P&L", width=12)
    t.add_column("Return%", width=10)
    t.add_column("Avg Win", width=10)
    t.add_column("Avg Loss", width=10)
    t.add_column("PF", width=6)
    t.add_column("Max DD%", width=8)
    t.add_column("Sharpe", width=8)

    for sym, r in all_results.items():
        pc = "green" if r["total_pnl"] >= 0 else "red"
        t.add_row(
            sym,
            str(r["trades"]),
            f"{r['win_rate']}%",
            f"[{pc}]${r['total_pnl']:+,.0f}[/{pc}]",
            f"[{pc}]{r['return_pct']:+.1f}%[/{pc}]",
            f"${r['avg_win']:+,.2f}",
            f"${r['avg_loss']:+,.2f}",
            str(r["profit_factor"]),
            f"{r['max_dd_pct']}%",
            str(r["sharpe"]),
        )
    console.print(t)

    # === COMBINED SUMMARY ===
    console.print(f"\n  [bold]COMBINED (all symbols):[/bold]")
    color = "green" if combined_total > 0 else "red"
    console.print(f"  Total P&L: [{color}]${combined_total:+,.2f}[/{color}]")
    years_avg = 16
    console.print(f"  Annual avg: ${combined_total / years_avg:+,.0f}")
    console.print(f"  Monthly avg: ${combined_total / years_avg / 12:+,.0f}")

    # === YEARLY CONSISTENCY ===
    console.print(f"\n  [bold]YEAR-BY-YEAR CONSISTENCY:[/bold]")
    yr_table = Table(show_header=True)
    yr_table.add_column("Year", width=6)
    for sym in all_results:
        yr_table.add_column(sym, width=10)
    yr_table.add_column("Total", width=12)

    pos_years = 0
    for year in sorted(combined_yearly.keys()):
        row = [year]
        year_total = 0
        for sym in all_results:
            yr_data = all_results[sym].get("yearly", {}).get(year, {})
            pnl = yr_data.get("pnl", 0)
            year_total += pnl
            c = "green" if pnl > 0 else "red" if pnl < 0 else "white"
            row.append(f"[{c}]${pnl:+,.0f}[/{c}]")
        tc = "green" if year_total > 0 else "red"
        row.append(f"[{tc}]${year_total:+,.0f}[/{tc}]")
        if year_total > 0:
            pos_years += 1
        yr_table.add_row(*row)
    console.print(yr_table)

    total_years = len(combined_yearly)
    console.print(f"\n  [bold]Profitable years: {pos_years}/{total_years} ({pos_years/total_years*100:.0f}%)[/bold]")

    # === VERDICT ===
    console.print(f"\n{'=' * 70}")
    if combined_total > 0 and pos_years / total_years >= 0.6:
        console.print(Panel.fit(
            "[bold green]✓ STRATEGY IS VIABLE[/bold green]\n"
            "Edge is real and consistent across years.\n"
            "Next step: deploy to paper trading",
            border_style="green",
        ))
    else:
        console.print(Panel.fit(
            "[bold red]✗ STRATEGY NOT VIABLE[/bold red]\n"
            "Either negative P&L or inconsistent across years.",
            border_style="red",
        ))


if __name__ == "__main__":
    main()
