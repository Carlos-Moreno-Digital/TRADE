"""Forensic analysis of ML backtest failure + test alternative strategies.

Runs on Dukascopy CSVs (data/dukascopy/).
Analyzes WHY the ML model fails on 16 years of real data.
Tests 5 alternative approaches to find something that actually works.
"""

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import StandardScaler
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

warnings.filterwarnings("ignore")
console = Console()

DATA_DIR = Path("data/dukascopy")


def load_dukascopy(symbol: str) -> pd.DataFrame:
    """Load Dukascopy 1H CSV."""
    path = DATA_DIR / f"{symbol}_1H.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df


def analyze_by_regime(df: pd.DataFrame) -> dict:
    """Forensic analysis: which market regimes kill the ML model?"""
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)

    atr = talib.ATR(high, low, close, timeperiod=14)
    adx = talib.ADX(high, low, close, timeperiod=14)

    # Forward return
    returns = pd.Series(close).pct_change(8).shift(-8)  # 8-hour forward return

    # Segment by volatility regime
    vol_percentile = pd.Series(atr).rolling(500).rank(pct=True)
    adx_series = pd.Series(adx)

    regimes = {
        "low_vol_trending": (vol_percentile < 0.3) & (adx_series > 25),
        "low_vol_ranging": (vol_percentile < 0.3) & (adx_series < 20),
        "mid_vol_trending": (vol_percentile.between(0.3, 0.7)) & (adx_series > 25),
        "mid_vol_ranging": (vol_percentile.between(0.3, 0.7)) & (adx_series < 20),
        "high_vol_trending": (vol_percentile > 0.7) & (adx_series > 25),
        "high_vol_ranging": (vol_percentile > 0.7) & (adx_series < 20),
    }

    stats = {}
    for name, mask in regimes.items():
        rets = returns[mask].dropna()
        if len(rets) > 50:
            stats[name] = {
                "count": len(rets),
                "mean_return": rets.mean() * 100,
                "std_return": rets.std() * 100,
                "win_rate": (rets > 0).mean() * 100,
                "sharpe": rets.mean() / rets.std() * np.sqrt(252 * 24) if rets.std() > 0 else 0,
            }
    return stats


def strategy_1_simple_momentum(df: pd.DataFrame, spread: float) -> dict:
    """STRATEGY 1: Simple long-term momentum (trend following)."""
    close = df["close"].values.astype(float)
    # 200-bar SMA trend filter + pullback entry
    sma_200 = talib.SMA(close, timeperiod=200)
    rsi = talib.RSI(close, timeperiod=14)
    atr = talib.ATR(df["high"].values, df["low"].values, close, timeperiod=14)

    trades = []
    position = None
    for i in range(200, len(close) - 24):
        if math.isnan(sma_200[i]) or math.isnan(rsi[i]):
            continue

        # Entry rules
        if position is None:
            # Uptrend + pullback
            if close[i] > sma_200[i] and rsi[i] < 40:
                position = {
                    "side": "buy", "entry": close[i], "idx": i,
                    "sl": close[i] - atr[i] * 2, "tp": close[i] + atr[i] * 4,
                }
            # Downtrend + pullback
            elif close[i] < sma_200[i] and rsi[i] > 60:
                position = {
                    "side": "sell", "entry": close[i], "idx": i,
                    "sl": close[i] + atr[i] * 2, "tp": close[i] - atr[i] * 4,
                }
        else:
            # Exit: SL/TP/time (24 bars)
            bars_held = i - position["idx"]
            exit_price = None
            if position["side"] == "buy":
                if df["low"].iloc[i] <= position["sl"]:
                    exit_price = position["sl"]
                elif df["high"].iloc[i] >= position["tp"]:
                    exit_price = position["tp"]
                elif bars_held >= 48:
                    exit_price = close[i]
                if exit_price:
                    pnl = (exit_price - position["entry"]) - spread * 2
                    trades.append(pnl)
                    position = None
            else:
                if df["high"].iloc[i] >= position["sl"]:
                    exit_price = position["sl"]
                elif df["low"].iloc[i] <= position["tp"]:
                    exit_price = position["tp"]
                elif bars_held >= 48:
                    exit_price = close[i]
                if exit_price:
                    pnl = (position["entry"] - exit_price) - spread * 2
                    trades.append(pnl)
                    position = None

    if not trades:
        return {"pnl": 0, "trades": 0, "wr": 0}

    qty_scale = 10000 / close[0]  # Normalize
    total = sum(trades) * qty_scale
    wins = sum(1 for t in trades if t > 0)
    return {
        "pnl": round(total, 2),
        "trades": len(trades),
        "wr": round(wins / len(trades) * 100, 1),
        "avg": round(total / len(trades), 2) if trades else 0,
    }


def strategy_2_bollinger_mean_reversion(df: pd.DataFrame, spread: float) -> dict:
    """STRATEGY 2: Bollinger Band mean reversion (only in ranging markets)."""
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)

    upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    adx = talib.ADX(high, low, close, timeperiod=14)
    atr = talib.ATR(high, low, close, timeperiod=14)

    trades = []
    position = None
    for i in range(200, len(close) - 24):
        if math.isnan(upper[i]) or math.isnan(adx[i]) or adx[i] > 20:
            continue

        if position is None:
            # Mean reversion only in ranging markets (ADX < 20)
            if close[i] < lower[i]:
                position = {
                    "side": "buy", "entry": close[i], "idx": i,
                    "sl": close[i] - atr[i] * 1.5, "tp": middle[i],
                }
            elif close[i] > upper[i]:
                position = {
                    "side": "sell", "entry": close[i], "idx": i,
                    "sl": close[i] + atr[i] * 1.5, "tp": middle[i],
                }
        else:
            bars_held = i - position["idx"]
            exit_price = None
            if position["side"] == "buy":
                if low[i] <= position["sl"]:
                    exit_price = position["sl"]
                elif high[i] >= position["tp"]:
                    exit_price = position["tp"]
                elif bars_held >= 24:
                    exit_price = close[i]
                if exit_price:
                    pnl = (exit_price - position["entry"]) - spread * 2
                    trades.append(pnl)
                    position = None
            else:
                if high[i] >= position["sl"]:
                    exit_price = position["sl"]
                elif low[i] <= position["tp"]:
                    exit_price = position["tp"]
                elif bars_held >= 24:
                    exit_price = close[i]
                if exit_price:
                    pnl = (position["entry"] - exit_price) - spread * 2
                    trades.append(pnl)
                    position = None

    if not trades:
        return {"pnl": 0, "trades": 0, "wr": 0}

    qty_scale = 10000 / close[0]
    total = sum(trades) * qty_scale
    wins = sum(1 for t in trades if t > 0)
    return {
        "pnl": round(total, 2),
        "trades": len(trades),
        "wr": round(wins / len(trades) * 100, 1),
        "avg": round(total / len(trades), 2) if trades else 0,
    }


def strategy_3_volatility_breakout(df: pd.DataFrame, spread: float) -> dict:
    """STRATEGY 3: Volatility breakout (Donchian + ATR filter)."""
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)

    # Donchian channels
    high_20 = pd.Series(high).rolling(20).max().values
    low_20 = pd.Series(low).rolling(20).min().values
    atr = talib.ATR(high, low, close, timeperiod=14)
    atr_ma = pd.Series(atr).rolling(50).mean().values

    trades = []
    position = None
    for i in range(200, len(close) - 48):
        if math.isnan(high_20[i]) or math.isnan(atr[i]) or math.isnan(atr_ma[i]):
            continue

        # Only trade when volatility expanding
        if atr[i] < atr_ma[i]:
            continue

        if position is None:
            if close[i] > high_20[i - 1]:  # Breakout long
                position = {
                    "side": "buy", "entry": close[i], "idx": i,
                    "sl": close[i] - atr[i] * 2, "tp": close[i] + atr[i] * 6,
                }
            elif close[i] < low_20[i - 1]:  # Breakout short
                position = {
                    "side": "sell", "entry": close[i], "idx": i,
                    "sl": close[i] + atr[i] * 2, "tp": close[i] - atr[i] * 6,
                }
        else:
            bars_held = i - position["idx"]
            exit_price = None
            if position["side"] == "buy":
                if low[i] <= position["sl"]:
                    exit_price = position["sl"]
                elif high[i] >= position["tp"]:
                    exit_price = position["tp"]
                elif bars_held >= 96:  # 4 days
                    exit_price = close[i]
                if exit_price:
                    pnl = (exit_price - position["entry"]) - spread * 2
                    trades.append(pnl)
                    position = None
            else:
                if high[i] >= position["sl"]:
                    exit_price = position["sl"]
                elif low[i] <= position["tp"]:
                    exit_price = position["tp"]
                elif bars_held >= 96:
                    exit_price = close[i]
                if exit_price:
                    pnl = (position["entry"] - exit_price) - spread * 2
                    trades.append(pnl)
                    position = None

    if not trades:
        return {"pnl": 0, "trades": 0, "wr": 0}

    qty_scale = 10000 / close[0]
    total = sum(trades) * qty_scale
    wins = sum(1 for t in trades if t > 0)
    return {
        "pnl": round(total, 2),
        "trades": len(trades),
        "wr": round(wins / len(trades) * 100, 1),
        "avg": round(total / len(trades), 2) if trades else 0,
    }


def strategy_4_daily_timeframe_ml(df: pd.DataFrame, spread: float) -> dict:
    """STRATEGY 4: Aggregate to daily + simple trend+momentum combo."""
    try:
        import xgboost as xgb
    except ImportError:
        return {"pnl": 0, "trades": 0, "wr": 0}

    # Resample to daily
    daily = pd.DataFrame({
        "open": df["open"].resample("D").first(),
        "high": df["high"].resample("D").max(),
        "low": df["low"].resample("D").min(),
        "close": df["close"].resample("D").last(),
    }).dropna()

    if len(daily) < 500:
        return {"pnl": 0, "trades": 0, "wr": 0}

    close_d = daily["close"].values
    high_d = daily["high"].values
    low_d = daily["low"].values

    # Simple features on daily
    sma_20 = talib.SMA(close_d, 20)
    sma_50 = talib.SMA(close_d, 50)
    rsi = talib.RSI(close_d, 14)
    atr = talib.ATR(high_d, low_d, close_d, 14)

    trades = []
    position = None
    for i in range(100, len(close_d) - 10):
        if math.isnan(sma_50[i]) or math.isnan(rsi[i]):
            continue

        if position is None:
            # Long: uptrend + oversold pullback
            if close_d[i] > sma_50[i] and rsi[i] < 35:
                position = {
                    "side": "buy", "entry": close_d[i], "idx": i,
                    "sl": close_d[i] - atr[i] * 2, "tp": close_d[i] + atr[i] * 4,
                }
            # Short: downtrend + overbought bounce
            elif close_d[i] < sma_50[i] and rsi[i] > 65:
                position = {
                    "side": "sell", "entry": close_d[i], "idx": i,
                    "sl": close_d[i] + atr[i] * 2, "tp": close_d[i] - atr[i] * 4,
                }
        else:
            bars_held = i - position["idx"]
            exit_price = None
            if position["side"] == "buy":
                if low_d[i] <= position["sl"]:
                    exit_price = position["sl"]
                elif high_d[i] >= position["tp"]:
                    exit_price = position["tp"]
                elif bars_held >= 10:
                    exit_price = close_d[i]
                if exit_price:
                    pnl = (exit_price - position["entry"]) - spread * 2
                    trades.append(pnl)
                    position = None
            else:
                if high_d[i] >= position["sl"]:
                    exit_price = position["sl"]
                elif low_d[i] <= position["tp"]:
                    exit_price = position["tp"]
                elif bars_held >= 10:
                    exit_price = close_d[i]
                if exit_price:
                    pnl = (position["entry"] - exit_price) - spread * 2
                    trades.append(pnl)
                    position = None

    if not trades:
        return {"pnl": 0, "trades": 0, "wr": 0}

    qty_scale = 10000 / close_d[0]
    total = sum(trades) * qty_scale
    wins = sum(1 for t in trades if t > 0)
    return {
        "pnl": round(total, 2),
        "trades": len(trades),
        "wr": round(wins / len(trades) * 100, 1),
        "avg": round(total / len(trades), 2) if trades else 0,
    }


def main():
    console.print(Panel.fit(
        "[bold magenta]FORENSIC ANALYSIS + ALTERNATIVE STRATEGIES[/bold magenta]\n"
        "Why did the ML model fail? What actually works?\n"
        "Testing on 16 years of real Dukascopy 1H data",
        title="🔬 Failure Analysis",
        border_style="magenta",
    ))

    available = {}
    for sym in ["EURUSD", "USDJPY", "GBPNZD", "AUDNZD", "XAUUSD"]:
        df = load_dukascopy(sym)
        if df is not None and len(df) > 10000:
            available[sym] = df
            years = (df.index[-1] - df.index[0]).days / 365
            console.print(f"  [green]✓[/green] {sym}: {len(df):,} candles, {years:.1f} years")

    console.print()

    # === PART 1: Regime analysis ===
    console.print(Panel.fit("[bold cyan]PART 1: What regime kills the ML model?[/bold cyan]",
                            border_style="cyan"))

    for sym, df in available.items():
        console.print(f"\n  [bold]{sym}[/bold] regime analysis:")
        stats = analyze_by_regime(df)
        for regime, s in stats.items():
            sign = "+" if s["mean_return"] > 0 else ""
            color = "green" if s["mean_return"] > 0 else "red"
            console.print(
                f"    {regime:25s}: n={s['count']:5d} | "
                f"[{color}]μ={sign}{s['mean_return']:.3f}%[/{color}] | "
                f"σ={s['std_return']:.3f}% | WR={s['win_rate']:.0f}%"
            )

    # === PART 2: Alternative strategies ===
    console.print(f"\n{'=' * 70}")
    console.print(Panel.fit("[bold yellow]PART 2: Alternative strategies (16-year test)[/bold yellow]",
                            border_style="yellow"))

    SPREADS = {
        "EURUSD": 0.00008, "USDJPY": 0.008,
        "GBPNZD": 0.00030, "AUDNZD": 0.00020, "XAUUSD": 0.40,
    }

    strategies = {
        "1. Trend Following (SMA200+RSI)": strategy_1_simple_momentum,
        "2. Bollinger Mean Reversion (ranging only)": strategy_2_bollinger_mean_reversion,
        "3. Volatility Breakout (Donchian+ATR)": strategy_3_volatility_breakout,
        "4. Daily Timeframe (RSI+SMA50)": strategy_4_daily_timeframe_ml,
    }

    table = Table(title="Alternative Strategy Results (16 years)")
    table.add_column("Strategy", width=40)
    table.add_column("Symbol", width=8)
    table.add_column("Trades", width=8)
    table.add_column("WR%", width=6)
    table.add_column("P&L", width=12)

    best_strategy = None
    best_pnl = -99999

    for strat_name, strat_fn in strategies.items():
        for sym, df in available.items():
            spread = SPREADS.get(sym, 0.0001)
            result = strat_fn(df, spread)
            pc = "green" if result["pnl"] > 0 else "red"
            table.add_row(
                strat_name,
                sym,
                str(result["trades"]),
                f"{result['wr']}%",
                f"[{pc}]${result['pnl']:+,.0f}[/{pc}]",
            )
            if result["pnl"] > best_pnl:
                best_pnl = result["pnl"]
                best_strategy = f"{strat_name} on {sym}"

    console.print(table)
    console.print(f"\n  [bold]Best: {best_strategy} (${best_pnl:+,.0f})[/bold]")


if __name__ == "__main__":
    main()
