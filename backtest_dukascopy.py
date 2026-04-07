"""Dukascopy 16-Year Backtest — validate ML model on 1H data from 2010.

Uses real Dukascopy 1H data instead of yfinance (which only has 2 years).
This is the ultimate validation before purchasing the prop firm challenge.
"""

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trade.ml_backtest import _build_features, _build_target

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import GradientBoostingClassifier

console = Console()

# Symbol config: spread + per-instrument horizon (from our optimization)
SYMBOL_CONFIG = {
    "EURUSD": {"spread": 0.00008, "horizon": 6},
    "USDJPY": {"spread": 0.008, "horizon": 10},
    "GBPNZD": {"spread": 0.00030, "horizon": 10},
    "AUDNZD": {"spread": 0.00020, "horizon": 4},
    "XAUUSD": {"spread": 0.40, "horizon": 8},  # Gold
}

SLIPPAGE = 0.5
DATA_DIR = Path("data/dukascopy")


def load_dukascopy(symbol: str) -> pd.DataFrame:
    """Load Dukascopy 1H CSV."""
    path = DATA_DIR / f"{symbol}_1H.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df


def backtest_symbol(symbol: str, df: pd.DataFrame, config: dict) -> dict:
    """Run walk-forward backtest on a symbol."""
    spread = config["spread"]
    horizon = config["horizon"]

    features = _build_features(df)
    target = _build_target(df, horizon=horizon, min_move_pct=0.001)

    data = features.copy()
    data["target"] = target
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    train_bars = 4000
    test_bars = 500
    purge_bars = 24

    if len(data) < train_bars + test_bars + purge_bars:
        return {"error": f"Insufficient data: {len(data)}"}

    y_map = {-1: 0, 0: 1, 1: 2}
    y_inv = {0: -1, 1: 0, 2: 1}

    pnl = 0
    trades = 0
    wins = 0
    max_dd = 0
    peak = 0
    equity = 0
    yearly = {}

    i = train_bars
    while i + test_bars <= len(data):
        tr_start = max(0, i - purge_bars - train_bars)
        train_data = data.iloc[tr_start:i - purge_bars]
        test_data = data.iloc[i:i + test_bars]

        X_train = train_data.drop(columns=["target"]).fillna(0)
        y_train = train_data["target"]
        X_test = test_data.drop(columns=["target"]).fillna(0)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        y_train_m = y_train.map(y_map)

        if HAS_XGB:
            model = xgb.XGBClassifier(
                n_estimators=250, max_depth=4, learning_rate=0.04,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
                eval_metric="mlogloss", verbosity=0,
                random_state=42, seed=42,
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=250, max_depth=4, learning_rate=0.04,
                subsample=0.8, min_samples_leaf=20, random_state=42,
            )

        try:
            model.fit(X_train_s, y_train_m)
            preds = model.predict(X_test_s)
            proba = model.predict_proba(X_test_s)
        except Exception:
            i += test_bars
            continue

        close_prices = df["close"].reindex(test_data.index).values.astype(float)
        atr_vals = features["atr_14"].reindex(test_data.index).values
        last_j = -10

        for j in range(len(preds)):
            pred_class = y_inv.get(int(preds[j]), 0)
            if pred_class == 0:
                continue
            if j - last_j < horizon:
                continue
            max_prob = float(proba[j].max())
            if max_prob < 0.53:
                continue
            if j + horizon >= len(close_prices):
                continue

            price = float(close_prices[j])
            atr_v = float(atr_vals[j]) if not math.isnan(atr_vals[j]) else price * 0.001
            if price <= 0 or atr_v <= 0:
                continue

            slip = spread * SLIPPAGE
            risk_amt = 10000 * 0.015  # 1.5% risk on $10K
            sl_dist = atr_v * 1.2
            qty = risk_amt / sl_dist if sl_dist > 0 else 0
            max_qty = 10000 * 5 / price
            qty = min(qty, max_qty)

            if qty <= 0:
                continue

            future_price = float(close_prices[j + horizon])
            cost = (spread + slip * 2) * qty

            if pred_class == 1:
                trade_pnl = (future_price - price) * qty - cost
            else:
                trade_pnl = (price - future_price) * qty - cost

            pnl += trade_pnl
            trades += 1
            if trade_pnl > 0:
                wins += 1
            last_j = j

            # Track DD
            equity += trade_pnl
            if equity > peak:
                peak = equity
            dd = (peak - equity) / 10000 * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)

            # Track yearly
            year = str(test_data.index[j])[:4]
            if year not in yearly:
                yearly[year] = {"pnl": 0, "trades": 0, "wins": 0}
            yearly[year]["pnl"] += trade_pnl
            yearly[year]["trades"] += 1
            if trade_pnl > 0:
                yearly[year]["wins"] += 1

        i += test_bars

    return {
        "pnl": round(pnl, 2),
        "trades": trades,
        "wins": wins,
        "win_rate": round(wins / trades * 100, 1) if trades > 0 else 0,
        "max_dd": round(max_dd, 2),
        "yearly": yearly,
    }


def main():
    console.print(Panel.fit(
        "[bold magenta]DUKASCOPY 16-YEAR 1H BACKTEST[/bold magenta]\n"
        "Ultimate validation with real Dukascopy data\n"
        "This is the decision point for prop firm purchase",
        title="🏆 Final Backtest",
        border_style="magenta",
    ))

    # Check which files exist
    available = {}
    for sym in SYMBOL_CONFIG:
        df = load_dukascopy(sym)
        if df is not None and len(df) > 5000:
            available[sym] = df
            years = (df.index[-1] - df.index[0]).days / 365
            console.print(f"  [green]✓[/green] {sym}: {len(df):,} candles, {years:.1f} years "
                          f"({str(df.index[0])[:10]} → {str(df.index[-1])[:10]})")
        else:
            console.print(f"  [red]✗[/red] {sym}: not available")

    if not available:
        console.print("[red]No data available. Run download_dukascopy.py first.[/red]")
        return

    console.print(f"\n  Running backtest on {len(available)} instruments...\n")

    all_results = {}
    total_pnl = 0
    total_trades = 0
    total_wins = 0

    for sym, df in available.items():
        console.print(f"  [cyan]Testing {sym}...[/cyan]", end=" ")
        result = backtest_symbol(sym, df, SYMBOL_CONFIG[sym])
        if "error" in result:
            console.print(f"[red]{result['error']}[/red]")
            continue

        color = "green" if result["pnl"] > 0 else "red"
        console.print(
            f"{result['trades']} trades, {result['win_rate']:.0f}% WR, "
            f"[{color}]${result['pnl']:+,.0f}[/{color}], DD: {result['max_dd']:.1f}%"
        )

        all_results[sym] = result
        total_pnl += result["pnl"]
        total_trades += result["trades"]
        total_wins += result["wins"]

    # === SUMMARY ===
    console.print(f"\n{'=' * 70}")
    console.print(Panel.fit("[bold]16-YEAR BACKTEST RESULTS[/bold]", border_style="magenta"))

    t = Table(show_header=True)
    t.add_column("Symbol", width=10)
    t.add_column("Trades", width=8)
    t.add_column("WR%", width=6)
    t.add_column("P&L", width=12)
    t.add_column("P&L/year", width=12)
    t.add_column("Max DD", width=8)

    for sym, r in all_results.items():
        years = 16  # Approximate
        pc = "green" if r["pnl"] >= 0 else "red"
        t.add_row(
            sym,
            str(r["trades"]),
            f"{r['win_rate']}%",
            f"[{pc}]${r['pnl']:+,.0f}[/{pc}]",
            f"${r['pnl']/years:+,.0f}",
            f"{r['max_dd']}%",
        )

    console.print(t)

    wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    pc = "green" if total_pnl >= 0 else "red"
    console.print(f"\n  [bold]COMBINED TOTAL:[/bold]")
    console.print(f"  Total P&L: [{pc}]${total_pnl:+,.2f}[/{pc}]")
    console.print(f"  Total Trades: {total_trades}")
    console.print(f"  Win Rate: {wr:.1f}%")
    console.print(f"  Monthly avg: ${total_pnl/(16*12):+,.0f}")
    console.print(f"  Annual avg: ${total_pnl/16:+,.0f}")

    # Year-by-year
    console.print(f"\n  [bold]YEAR-BY-YEAR BREAKDOWN:[/bold]")
    all_years = set()
    for r in all_results.values():
        all_years.update(r.get("yearly", {}).keys())

    yr_table = Table(show_header=True)
    yr_table.add_column("Year", width=6)
    for sym in all_results:
        yr_table.add_column(sym, width=10)
    yr_table.add_column("Total", width=12)

    pos_years = 0
    for year in sorted(all_years):
        row = [year]
        year_total = 0
        for sym in all_results:
            yr_data = all_results[sym].get("yearly", {}).get(year, {})
            pnl = yr_data.get("pnl", 0)
            year_total += pnl
            color = "green" if pnl > 0 else "red" if pnl < 0 else "white"
            row.append(f"[{color}]${pnl:+,.0f}[/{color}]")
        tc = "green" if year_total > 0 else "red"
        row.append(f"[{tc}]${year_total:+,.0f}[/{tc}]")
        if year_total > 0:
            pos_years += 1
        yr_table.add_row(*row)

    console.print(yr_table)
    console.print(f"\n  Profitable years: {pos_years}/{len(all_years)} ({pos_years/len(all_years)*100:.0f}%)")


if __name__ == "__main__":
    main()
