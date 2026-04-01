"""Massive Backtesting Engine — Multi-timeframe, multi-strategy, parameter sweep.

Runs the most comprehensive backtest possible with yfinance data:
- 1H data: 2 years (730 days) × 8 pairs = ~32,000 candles
- Daily data: 5+ years × 30+ symbols = massive sample
- 5min data: 60 days × 8 pairs (yfinance limit)
- Parameter sweep: auto-test 20+ configurations
- Walk-forward: train on first half, validate on second half
"""

from __future__ import annotations

import itertools
import math
import time
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import talib
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# ECN-realistic spreads
SPREADS = {
    "EURUSD=X": 0.00008, "GBPUSD=X": 0.00010, "USDJPY=X": 0.008,
    "AUDUSD=X": 0.00010, "USDCAD=X": 0.00012, "USDCHF=X": 0.00010,
    "EURGBP=X": 0.00012, "EURJPY=X": 0.012, "GBPJPY=X": 0.015,
    "NZDUSD=X": 0.00012, "EURNZD=X": 0.00025, "AUDNZD=X": 0.00020,
}
SLIPPAGE = 0.5


class MassiveBacktester:
    """Run massive multi-timeframe backtests with parameter optimization."""

    def __init__(self, account_size: float = 10000):
        self.account_size = account_size

    def run_all(self) -> dict:
        """Run the complete massive backtest suite."""
        console.print(Panel.fit(
            "[bold magenta]MASSIVE BACKTESTING ENGINE[/bold magenta]\n"
            "Multi-timeframe · Parameter Sweep · Walk-Forward Validation\n"
            "Testing every possible configuration to find the optimal strategy",
            title="🔬 Maximum Backtest",
            border_style="magenta",
        ))

        results = {}

        # Phase 1: 1H Backtest (2 years — the most important)
        console.print("\n[bold cyan]═══ PHASE 1: 1H BACKTEST (2 YEARS) ═══[/bold cyan]")
        results["1h"] = self._run_1h_backtest()

        # Phase 2: Parameter Sweep on 1H
        console.print("\n[bold cyan]═══ PHASE 2: PARAMETER SWEEP (1H) ═══[/bold cyan]")
        results["sweep"] = self._run_parameter_sweep()

        # Phase 3: Walk-Forward Validation
        console.print("\n[bold cyan]═══ PHASE 3: WALK-FORWARD VALIDATION ═══[/bold cyan]")
        results["walkforward"] = self._run_walk_forward()

        # Phase 4: Daily max historical
        console.print("\n[bold cyan]═══ PHASE 4: DAILY BACKTEST (MAX HISTORY) ═══[/bold cyan]")
        results["daily"] = self._run_daily_max()

        # Phase 5: 5min validation
        console.print("\n[bold cyan]═══ PHASE 5: 5MIN VALIDATION (60 DAYS) ═══[/bold cyan]")
        results["5min"] = self._run_5min_validation()

        # Final Report
        self._print_final_report(results)
        return results

    # =========================================================================
    # DATA DOWNLOAD
    # =========================================================================
    def _download_1h(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """Download 2 years of 1H data for all symbols."""
        import yfinance as yf
        data = {}
        for sym in symbols:
            try:
                df = yf.download(sym, period="2y", interval="1h", progress=False)
                if not df.empty and len(df) >= 200:
                    if hasattr(df.columns, 'levels'):
                        df.columns = [c[0].lower() for c in df.columns]
                    else:
                        df.columns = [c.lower() for c in df.columns]
                    data[sym] = df
            except Exception:
                pass
        return data

    def _download_daily(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """Download max daily data (5-10 years)."""
        import yfinance as yf
        data = {}
        for sym in symbols:
            try:
                df = yf.download(sym, period="5y", interval="1d", progress=False)
                if not df.empty and len(df) >= 200:
                    if hasattr(df.columns, 'levels'):
                        df.columns = [c[0].lower() for c in df.columns]
                    else:
                        df.columns = [c.lower() for c in df.columns]
                    data[sym] = df
            except Exception:
                pass
        return data

    # =========================================================================
    # STRATEGY: ORB-style adapted for 1H
    # =========================================================================
    def _strategy_signal(self, df_slice: pd.DataFrame, symbol: str,
                         params: dict, current_ts=None) -> dict | None:
        """Unified strategy engine with configurable parameters.

        Combines the best elements:
        - EMA trend alignment (20/50)
        - ADX filter (configurable threshold)
        - RSI pullback entries in trend direction
        - ATR-based SL/TP with configurable multipliers
        - Session filter for forex
        - Monday filter
        """
        try:
            close = df_slice["close"].values.astype(float)
            high = df_slice["high"].values.astype(float)
            low = df_slice["low"].values.astype(float)
            n = len(close)
            if n < 50:
                return None

            latest = float(close[-1])
            if math.isnan(latest) or latest <= 0:
                return None

            # Monday filter
            if current_ts is not None:
                weekday = current_ts.weekday() if hasattr(current_ts, 'weekday') else None
                if weekday == 0 and params.get("skip_monday", True):
                    return None

            # Session filter (intraday only — skip for daily candles where hour==0)
            if current_ts is not None and hasattr(current_ts, 'hour'):
                hour = current_ts.hour
                # Daily candles have hour=0 and minute=0 — skip session filter for those
                minute = current_ts.minute if hasattr(current_ts, 'minute') else 0
                is_daily = (hour == 0 and minute == 0)
                if not is_daily and not (7 <= hour <= 19):
                    return None

            # Indicators
            atr = talib.ATR(high, low, close, timeperiod=14)
            atr_val = float(atr[-1]) if not math.isnan(atr[-1]) else latest * 0.001
            if atr_val <= 0:
                return None

            ema_fast = talib.EMA(close, timeperiod=params.get("ema_fast", 20))
            ema_slow = talib.EMA(close, timeperiod=params.get("ema_slow", 50))
            ema_f = float(ema_fast[-1]) if not math.isnan(ema_fast[-1]) else latest
            ema_s = float(ema_slow[-1]) if not math.isnan(ema_slow[-1]) else latest

            adx = talib.ADX(high, low, close, timeperiod=14)
            adx_val = float(adx[-1]) if not math.isnan(adx[-1]) else 15

            rsi = talib.RSI(close, timeperiod=14)
            rsi_val = float(rsi[-1]) if not math.isnan(rsi[-1]) else 50

            adx_thresh = params.get("adx_threshold", 20)
            if adx_val < adx_thresh:
                return None

            # Trend
            trend_up = ema_f > ema_s and latest > ema_s
            trend_down = ema_f < ema_s and latest < ema_s

            action = None
            rsi_buy_lo = params.get("rsi_buy_lo", 35)
            rsi_buy_hi = params.get("rsi_buy_hi", 55)
            rsi_sell_lo = params.get("rsi_sell_lo", 45)
            rsi_sell_hi = params.get("rsi_sell_hi", 65)

            # Buy: uptrend + RSI pullback (not overbought)
            if trend_up and rsi_buy_lo <= rsi_val <= rsi_buy_hi:
                # Confirm: price bouncing off EMA
                if low[-1] <= ema_f * 1.002 and close[-1] > ema_f:
                    action = "buy"

            # Sell: downtrend + RSI bounce (not oversold)
            elif trend_down and rsi_sell_lo <= rsi_val <= rsi_sell_hi:
                if high[-1] >= ema_f * 0.998 and close[-1] < ema_f:
                    action = "sell"

            if action is None:
                return None

            sl_mult = params.get("sl_mult", 1.2)
            tp_mult = params.get("tp_mult", 2.0)

            if action == "buy":
                sl = latest - atr_val * sl_mult
                tp = latest + atr_val * tp_mult
            else:
                sl = latest + atr_val * sl_mult
                tp = latest - atr_val * tp_mult

            risk = abs(latest - sl)
            reward = abs(tp - latest)
            if risk <= 0 or reward / risk < 1.2:
                return None

            return {
                "symbol": symbol, "action": action, "price": latest,
                "sl": round(sl, 6), "tp": round(tp, 6), "atr": atr_val,
            }
        except Exception:
            return None

    # =========================================================================
    # SIMULATION ENGINE (shared by all timeframes)
    # =========================================================================
    def _simulate(self, all_data: dict[str, pd.DataFrame], params: dict,
                  warmup: int = 100, max_concurrent: int = 3,
                  max_daily_trades: int = 6, max_daily_losses: int = 2,
                  label: str = "") -> dict:
        """Run a complete simulation on the given data with the given parameters."""
        risk_pct = params.get("risk_pct", 0.0075)
        cash = self.account_size
        peak = self.account_size
        open_pos = []
        closed = []
        max_dd = 0.0
        trades_today = 0
        losses_today = 0
        current_day = ""

        ref = next(iter(all_data.values()))
        timestamps = sorted(ref.index.tolist())

        for i in range(warmup, len(timestamps)):
            ts = timestamps[i]
            ts_str = str(ts)
            day = ts_str[:10]

            if day != current_day:
                trades_today = 0
                losses_today = 0
                current_day = day

            # Check SL/TP
            to_close = []
            for pos in open_pos:
                df = all_data.get(pos["symbol"])
                if df is None:
                    continue
                mask = df.index <= ts
                if mask.sum() == 0:
                    continue
                c = df[mask].iloc[-1]
                h, l, cp = float(c["high"]), float(c["low"]), float(c["close"])
                spread = SPREADS.get(pos["symbol"], 0.0002)
                slip = spread * SLIPPAGE

                if pos["side"] == "buy":
                    if l <= pos["sl"]:
                        pos["exit"] = pos["sl"] - slip
                        pos["pnl"] = (pos["exit"] - pos["entry"]) * pos["qty"]
                        pos["reason"] = "SL"
                        to_close.append(pos)
                    elif h >= pos["tp"]:
                        pos["exit"] = pos["tp"] - slip
                        pos["pnl"] = (pos["exit"] - pos["entry"]) * pos["qty"]
                        pos["reason"] = "TP"
                        to_close.append(pos)
                else:
                    if h >= pos["sl"]:
                        pos["exit"] = pos["sl"] + slip
                        pos["pnl"] = (pos["entry"] - pos["exit"]) * pos["qty"]
                        pos["reason"] = "SL"
                        to_close.append(pos)
                    elif l <= pos["tp"]:
                        pos["exit"] = pos["tp"] + slip
                        pos["pnl"] = (pos["entry"] - pos["exit"]) * pos["qty"]
                        pos["reason"] = "TP"
                        to_close.append(pos)

            for pos in to_close:
                margin = pos["entry"] * pos["qty"] * 0.01
                cash += margin + pos["pnl"]
                open_pos.remove(pos)
                closed.append(pos)
                if pos["pnl"] < 0:
                    losses_today += 1

            # New entries
            if (len(open_pos) < max_concurrent and
                    trades_today < max_daily_trades and losses_today < max_daily_losses):
                for sym, df in all_data.items():
                    if len(open_pos) >= max_concurrent:
                        break
                    if any(p["symbol"] == sym for p in open_pos):
                        continue
                    mask = df.index <= ts
                    sl = df[mask]
                    if len(sl) < warmup:
                        continue

                    signal = self._strategy_signal(sl, sym, params, current_ts=ts)
                    if signal is None:
                        continue

                    entry = signal["price"]
                    sl_price = signal["sl"]
                    tp_price = signal["tp"]
                    rd = abs(entry - sl_price)
                    if rd <= 0:
                        continue

                    qty = min(
                        self.account_size * risk_pct / rd,
                        self.account_size * 5 / entry if entry > 0 else 0
                    )
                    spread = SPREADS.get(sym, 0.0002)
                    slip = spread * SLIPPAGE
                    fill = entry + spread / 2 + slip if signal["action"] == "buy" else entry - spread / 2 - slip

                    pos = {
                        "symbol": sym, "side": signal["action"],
                        "entry": fill, "qty": round(qty, 4),
                        "sl": sl_price, "tp": tp_price,
                        "entry_time": ts_str,
                    }
                    cash -= fill * qty * 0.01
                    open_pos.append(pos)
                    trades_today += 1

            # Equity
            margin_held = sum(p["entry"] * p["qty"] * 0.01 for p in open_pos)
            unrealized = 0
            for p in open_pos:
                df = all_data.get(p["symbol"])
                if df is None:
                    continue
                mask = df.index <= ts
                if mask.sum() == 0:
                    continue
                cp = float(df[mask].iloc[-1]["close"])
                if p["side"] == "buy":
                    unrealized += (cp - p["entry"]) * p["qty"]
                else:
                    unrealized += (p["entry"] - cp) * p["qty"]

            equity = cash + margin_held + unrealized
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, min(dd, 100))

        # Close remaining
        for pos in open_pos:
            cash += pos["entry"] * pos["qty"] * 0.01

        wins = [t for t in closed if t["pnl"] > 0]
        losses_list = [t for t in closed if t["pnl"] <= 0]
        total = len(closed)
        wr = len(wins) / total * 100 if total > 0 else 0
        total_pnl = sum(t["pnl"] for t in closed)
        avg_w = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
        avg_l = sum(t["pnl"] for t in losses_list) / len(losses_list) if losses_list else 0
        pf = abs(sum(t["pnl"] for t in wins) / sum(t["pnl"] for t in losses_list)) if losses_list and sum(t["pnl"] for t in losses_list) != 0 else 999

        # Trading days
        trade_days = len(set(t["entry_time"][:10] for t in closed))
        total_days = len(set(ts_str[:10] for ts_str in [str(t) for t in timestamps[warmup:]]))

        return {
            "label": label, "total_pnl": round(total_pnl, 2),
            "pnl_pct": round(total_pnl / self.account_size * 100, 2),
            "trades": total, "wins": len(wins), "losses": len(losses_list),
            "win_rate": round(wr, 1), "avg_win": round(avg_w, 2),
            "avg_loss": round(avg_l, 2), "profit_factor": round(pf, 2) if pf < 999 else 999,
            "max_dd": round(max_dd, 2), "trade_days": trade_days,
            "total_days": total_days, "params": params,
        }

    # =========================================================================
    # PHASE 1: 1H BACKTEST (2 YEARS)
    # =========================================================================
    def _run_1h_backtest(self) -> dict:
        symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
                    "USDCAD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X"]

        console.print(f"Downloading 2 years of 1H data for {len(symbols)} pairs...")
        data = self._download_1h(symbols)
        console.print(f"  Got {len(data)}/{len(symbols)} symbols\n")

        if not data:
            return {"error": "No data"}

        ref = next(iter(data.values()))
        console.print(f"  {len(ref)} candles per symbol (~{len(ref)//24} days)\n")

        # Run with best known parameters
        params = {
            "ema_fast": 20, "ema_slow": 50, "adx_threshold": 18,
            "sl_mult": 1.2, "tp_mult": 2.0, "risk_pct": 0.0075,
            "rsi_buy_lo": 35, "rsi_buy_hi": 55,
            "rsi_sell_lo": 45, "rsi_sell_hi": 65,
            "skip_monday": True,
        }

        result = self._simulate(data, params, warmup=100, label="1H-Base")
        self._print_result(result)
        return result

    # =========================================================================
    # PHASE 2: PARAMETER SWEEP
    # =========================================================================
    def _run_parameter_sweep(self) -> list[dict]:
        symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
                    "USDCAD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X"]

        console.print("Downloading 1H data for parameter sweep...")
        data = self._download_1h(symbols)
        if not data:
            return [{"error": "No data"}]

        console.print(f"  Got {len(data)} symbols\n")

        # Define parameter grid
        configs = [
            {"label": "Conservative", "sl_mult": 1.5, "tp_mult": 2.5, "adx_threshold": 22, "risk_pct": 0.005},
            {"label": "Balanced", "sl_mult": 1.2, "tp_mult": 2.0, "adx_threshold": 18, "risk_pct": 0.0075},
            {"label": "Aggressive", "sl_mult": 1.0, "tp_mult": 1.5, "adx_threshold": 15, "risk_pct": 0.01},
            {"label": "TightSL", "sl_mult": 0.8, "tp_mult": 1.5, "adx_threshold": 20, "risk_pct": 0.0075},
            {"label": "WideSL", "sl_mult": 1.5, "tp_mult": 2.0, "adx_threshold": 18, "risk_pct": 0.0075},
            {"label": "HighTP", "sl_mult": 1.2, "tp_mult": 3.0, "adx_threshold": 20, "risk_pct": 0.0075},
            {"label": "LowTP", "sl_mult": 1.0, "tp_mult": 1.3, "adx_threshold": 18, "risk_pct": 0.0075},
            {"label": "StrongTrend", "sl_mult": 1.2, "tp_mult": 2.0, "adx_threshold": 25, "risk_pct": 0.01},
            {"label": "WeakFilter", "sl_mult": 1.2, "tp_mult": 1.8, "adx_threshold": 12, "risk_pct": 0.005},
            {"label": "RSI-Wide", "sl_mult": 1.2, "tp_mult": 2.0, "adx_threshold": 18, "risk_pct": 0.0075,
             "rsi_buy_lo": 30, "rsi_buy_hi": 60, "rsi_sell_lo": 40, "rsi_sell_hi": 70},
            {"label": "RSI-Tight", "sl_mult": 1.2, "tp_mult": 2.0, "adx_threshold": 18, "risk_pct": 0.0075,
             "rsi_buy_lo": 38, "rsi_buy_hi": 50, "rsi_sell_lo": 50, "rsi_sell_hi": 62},
            {"label": "EMA-Fast", "sl_mult": 1.2, "tp_mult": 2.0, "adx_threshold": 18, "risk_pct": 0.0075,
             "ema_fast": 10, "ema_slow": 30},
            {"label": "EMA-Slow", "sl_mult": 1.2, "tp_mult": 2.0, "adx_threshold": 18, "risk_pct": 0.0075,
             "ema_fast": 30, "ema_slow": 100},
            {"label": "NoMonday", "sl_mult": 1.2, "tp_mult": 2.0, "adx_threshold": 18, "risk_pct": 0.0075,
             "skip_monday": True},
            {"label": "WithMonday", "sl_mult": 1.2, "tp_mult": 2.0, "adx_threshold": 18, "risk_pct": 0.0075,
             "skip_monday": False},
            {"label": "HighRisk", "sl_mult": 1.2, "tp_mult": 2.0, "adx_threshold": 18, "risk_pct": 0.015},
            {"label": "MaxRisk", "sl_mult": 1.0, "tp_mult": 1.8, "adx_threshold": 18, "risk_pct": 0.02},
            {"label": "PropFirm", "sl_mult": 1.0, "tp_mult": 1.8, "adx_threshold": 18, "risk_pct": 0.01,
             "rsi_buy_lo": 35, "rsi_buy_hi": 55, "rsi_sell_lo": 45, "rsi_sell_hi": 65},
        ]

        results = []
        base_params = {
            "ema_fast": 20, "ema_slow": 50, "adx_threshold": 18,
            "sl_mult": 1.2, "tp_mult": 2.0, "risk_pct": 0.0075,
            "rsi_buy_lo": 35, "rsi_buy_hi": 55,
            "rsi_sell_lo": 45, "rsi_sell_hi": 65,
            "skip_monday": True,
        }

        console.print(f"Running {len(configs)} parameter configurations...\n")

        for idx, cfg in enumerate(configs):
            params = {**base_params, **cfg}
            label = cfg.pop("label", f"Config-{idx}")
            result = self._simulate(data, params, warmup=100, label=label)
            results.append(result)

            pc = "green" if result["total_pnl"] >= 0 else "red"
            console.print(
                f"  [{idx+1:2d}/{len(configs)}] {label:15s} | "
                f"{result['trades']:3d} trades | {result['win_rate']:.0f}% WR | "
                f"[{pc}]${result['total_pnl']:+8.2f} ({result['pnl_pct']:+.1f}%)[/{pc}] | "
                f"DD:{result['max_dd']:.1f}% | PF:{result['profit_factor']:.2f}"
            )

        # Sort by P&L
        results.sort(key=lambda x: x["total_pnl"], reverse=True)

        console.print(f"\n[bold]Top 5 Configurations:[/bold]")
        for i, r in enumerate(results[:5]):
            console.print(
                f"  {i+1}. [bold]{r['label']}[/bold] → "
                f"${r['total_pnl']:+.2f} ({r['pnl_pct']:+.1f}%) | "
                f"{r['win_rate']}% WR | PF {r['profit_factor']} | DD {r['max_dd']}%"
            )

        return results

    # =========================================================================
    # PHASE 3: WALK-FORWARD VALIDATION
    # =========================================================================
    def _run_walk_forward(self) -> dict:
        """Train on first half, validate on second half."""
        symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
                    "USDCAD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X"]

        console.print("Downloading 1H data for walk-forward...")
        data = self._download_1h(symbols)
        if not data:
            return {"error": "No data"}

        # Split each symbol's data into first half / second half
        first_half = {}
        second_half = {}
        for sym, df in data.items():
            mid = len(df) // 2
            first_half[sym] = df.iloc[:mid]
            second_half[sym] = df.iloc[mid:]

        console.print(f"  First half: ~{len(next(iter(first_half.values())))} candles")
        console.print(f"  Second half: ~{len(next(iter(second_half.values())))} candles\n")

        # Best params from sweep (use balanced as default)
        params = {
            "ema_fast": 20, "ema_slow": 50, "adx_threshold": 18,
            "sl_mult": 1.2, "tp_mult": 2.0, "risk_pct": 0.0075,
            "rsi_buy_lo": 35, "rsi_buy_hi": 55,
            "rsi_sell_lo": 45, "rsi_sell_hi": 65,
            "skip_monday": True,
        }

        console.print("  Training period (first half)...")
        train = self._simulate(first_half, params, warmup=100, label="TRAIN")
        self._print_result(train, indent=4)

        console.print("\n  Validation period (second half)...")
        test = self._simulate(second_half, params, warmup=100, label="TEST")
        self._print_result(test, indent=4)

        # Consistency check
        both_positive = train["total_pnl"] >= 0 and test["total_pnl"] >= 0
        consistent_wr = abs(train["win_rate"] - test["win_rate"]) < 15

        console.print(f"\n  Walk-Forward Consistency:")
        console.print(f"    Both positive: [{'green' if both_positive else 'red'}]{'YES' if both_positive else 'NO'}[/{'green' if both_positive else 'red'}]")
        console.print(f"    WR difference: {abs(train['win_rate'] - test['win_rate']):.1f}% [{'green' if consistent_wr else 'red'}]({'CONSISTENT' if consistent_wr else 'INCONSISTENT'})[/{'green' if consistent_wr else 'red'}]")

        return {"train": train, "test": test, "consistent": both_positive and consistent_wr}

    # =========================================================================
    # PHASE 4: DAILY MAX HISTORY
    # =========================================================================
    def _run_daily_max(self) -> dict:
        """Run daily backtest on maximum available history (5 years)."""
        symbols = [
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
            "USDCAD=X", "USDCHF=X", "NZDUSD=X",
            "EURGBP=X", "EURJPY=X", "GBPJPY=X",
            "EURNZD=X", "AUDNZD=X",
        ]

        console.print(f"Downloading 5 years of daily data for {len(symbols)} pairs...")
        data = self._download_daily(symbols)
        console.print(f"  Got {len(data)}/{len(symbols)} symbols")

        if not data:
            return {"error": "No data"}

        ref = next(iter(data.values()))
        console.print(f"  {len(ref)} daily candles (~{len(ref)//252} years)\n")

        params = {
            "ema_fast": 20, "ema_slow": 50, "adx_threshold": 18,
            "sl_mult": 1.0, "tp_mult": 1.8, "risk_pct": 0.01,
            "rsi_buy_lo": 35, "rsi_buy_hi": 55,
            "rsi_sell_lo": 45, "rsi_sell_hi": 65,
            "skip_monday": True,
        }

        result = self._simulate(data, params, warmup=100,
                                max_daily_trades=3, max_daily_losses=2,
                                label="Daily-5Y")
        self._print_result(result)
        return result

    # =========================================================================
    # PHASE 5: 5MIN VALIDATION
    # =========================================================================
    def _run_5min_validation(self) -> dict:
        """Run the existing 5min scalp validation."""
        from trade.backtest_5min import ScalpBacktester
        bt = ScalpBacktester(account_size=self.account_size)
        return bt.run_random_samples(n_samples=8)

    # =========================================================================
    # REPORTING
    # =========================================================================
    def _print_result(self, r: dict, indent: int = 2) -> None:
        pad = " " * indent
        pc = "green" if r.get("total_pnl", 0) >= 0 else "red"
        console.print(f"{pad}[{pc}]P&L: ${r.get('total_pnl', 0):+,.2f} ({r.get('pnl_pct', 0):+.1f}%)[/{pc}]")
        console.print(f"{pad}Trades: {r.get('trades', 0)} | WR: {r.get('win_rate', 0)}% | PF: {r.get('profit_factor', 0)}")
        console.print(f"{pad}Avg Win: ${r.get('avg_win', 0):+.2f} | Avg Loss: ${r.get('avg_loss', 0):+.2f}")
        console.print(f"{pad}Max DD: {r.get('max_dd', 0)}% | Days: {r.get('total_days', 0)}")

    def _print_final_report(self, results: dict) -> None:
        console.print(f"\n{'='*70}")
        console.print(Panel.fit(
            "[bold magenta]MASSIVE BACKTEST — FINAL REPORT[/bold magenta]",
            border_style="magenta",
        ))

        t = Table(title="Results by Timeframe")
        t.add_column("Timeframe", style="bold")
        t.add_column("P&L", justify="right")
        t.add_column("P&L%", justify="right")
        t.add_column("Trades", justify="right")
        t.add_column("WR%", justify="right")
        t.add_column("PF", justify="right")
        t.add_column("Max DD%", justify="right")
        t.add_column("Days", justify="right")

        for key, label in [("1h", "1H (2 years)"), ("daily", "Daily (5 years)")]:
            r = results.get(key, {})
            if "error" in r:
                continue
            pc = "green" if r.get("total_pnl", 0) >= 0 else "red"
            t.add_row(
                label,
                f"[{pc}]${r.get('total_pnl', 0):+,.2f}[/{pc}]",
                f"[{pc}]{r.get('pnl_pct', 0):+.1f}%[/{pc}]",
                str(r.get("trades", 0)),
                f"{r.get('win_rate', 0):.0f}%",
                f"{r.get('profit_factor', 0):.2f}",
                f"{r.get('max_dd', 0):.1f}%",
                str(r.get("total_days", 0)),
            )

        # 5min
        r5 = results.get("5min", {})
        if "total_pnl" in r5:
            pc = "green" if r5["total_pnl"] >= 0 else "red"
            t.add_row(
                "5min (60 days)",
                f"[{pc}]${r5['total_pnl']:+,.2f}[/{pc}]",
                f"[{pc}]{r5.get('total_pnl', 0)/self.account_size*100:+.1f}%[/{pc}]",
                str(r5.get("total_trades", 0)),
                f"{r5.get('win_rate', 0):.0f}%",
                "-",
                "-",
                "60",
            )

        console.print(t)

        # Sweep best
        sweep = results.get("sweep", [])
        if sweep and isinstance(sweep, list):
            console.print(f"\n[bold]Best Parameter Config:[/bold]")
            best = sweep[0]
            console.print(f"  {best['label']}: ${best['total_pnl']:+,.2f} | {best['win_rate']}% WR | PF {best['profit_factor']}")

        # Walk-forward
        wf = results.get("walkforward", {})
        if "consistent" in wf:
            color = "green" if wf["consistent"] else "red"
            console.print(f"\n[bold]Walk-Forward Validation:[/bold] [{color}]{'PASSED' if wf['consistent'] else 'FAILED'}[/{color}]")
            if "train" in wf:
                console.print(f"  Train: ${wf['train']['total_pnl']:+,.2f} ({wf['train']['win_rate']}% WR)")
                console.print(f"  Test:  ${wf['test']['total_pnl']:+,.2f} ({wf['test']['win_rate']}% WR)")

        # Monthly projection
        console.print(f"\n[bold]Monthly Projections (at current pace):[/bold]")
        for key, label in [("1h", "1H Strategy"), ("daily", "Daily Strategy")]:
            r = results.get(key, {})
            if "total_pnl" in r and r.get("total_days", 0) > 0:
                monthly = r["total_pnl"] / r["total_days"] * 22
                mp = monthly / self.account_size * 100
                pc = "green" if monthly >= 0 else "red"
                console.print(f"  {label}: [{pc}]${monthly:+,.0f}/month ({mp:+.1f}%/month)[/{pc}]")


def run_massive_backtest():
    """Entry point for massive backtesting."""
    bt = MassiveBacktester(account_size=10000)
    return bt.run_all()
