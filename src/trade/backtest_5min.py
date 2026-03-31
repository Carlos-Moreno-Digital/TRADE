"""5-Minute Scalping Backtester - Intraday strategy optimized for prop firms.

Strategy: EMA Crossover + RSI Filter + ATR Stops
- Entry: EMA8 crosses EMA21 with RSI(7) confirmation
- Exit: ATR-based SL (1x) and TP (2x)
- Sessions: Only London (07:00-12:00 UTC) and NY (12:00-17:00 UTC)
- Pairs: EUR/USD, GBP/USD, USD/JPY (tightest spreads)
- Risk: 0.5% per trade (tighter for scalping)

yfinance limitation: only provides 5-day history at 5min interval.
So this backtester simulates ~5 trading days of 5min scalping.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import talib
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trade.data.providers import MarketDataProvider

console = Console()

# Tight spreads for scalping (in price units)
SCALP_SPREADS = {
    "EURUSD=X": 0.00010, "GBPUSD=X": 0.00013, "USDJPY=X": 0.010,
    "AUDUSD=X": 0.00013, "USDCAD=X": 0.00015, "USDCHF=X": 0.00013,
    "EURGBP=X": 0.00015, "EURJPY=X": 0.015, "GBPJPY=X": 0.020,
}
SLIPPAGE = 0.5  # 50% of spread


class ScalpPosition:
    def __init__(self, symbol, side, entry, qty, sl, tp, entry_time):
        self.symbol = symbol
        self.side = side
        self.entry_price = entry
        self.quantity = qty
        self.stop_loss = sl
        self.take_profit = tp
        self.entry_time = entry_time
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        self.pnl = 0.0


class ScalpBacktester:
    def __init__(self, account_size: float = 10000):
        self.account_size = account_size
        self.risk_per_trade = 0.005  # 0.5% per scalp (conservative)
        self.max_trades = 2          # Max 2 concurrent
        self.max_trades_per_day = 10 # Don't overtrade
        self.provider = MarketDataProvider()

    def run(self, symbols: list[str] | None = None) -> dict[str, Any]:
        if symbols is None:
            symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
                       "USDCAD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X"]

        console.print(Panel.fit(
            f"[bold cyan]5-Minute Scalping Backtester[/bold cyan]\n"
            f"Account: ${self.account_size:,.0f} | Risk: {self.risk_per_trade*100:.1f}%/trade | "
            f"Max {self.max_trades_per_day} trades/day\n"
            f"Strategy: EMA8/21 crossover + RSI(7) + ATR stops\n"
            f"Sessions: London (07-12 UTC) + NY (12-17 UTC) only",
            title="Scalp Backtest",
            border_style="magenta",
        ))

        # Download 5min data (yfinance: max 5 days)
        console.print(f"[dim]Downloading 5min data for {len(symbols)} pairs...[/dim]")
        all_data = {}
        for sym in symbols:
            try:
                df = self.provider.get_historical(sym, period="5d", interval="5m")
                if not df.empty and len(df) >= 100:
                    all_data[sym] = df
            except Exception:
                pass
        console.print(f"  Got {len(all_data)}/{len(symbols)} symbols\n")

        if not all_data:
            return {"error": "No data"}

        # Simulation
        cash = self.account_size
        equity = self.account_size
        peak = self.account_size
        open_pos: list[ScalpPosition] = []
        closed: list[ScalpPosition] = []
        max_dd = 0.0
        trades_today = 0
        current_day = ""

        # Get all timestamps from first symbol
        ref = next(iter(all_data.values()))
        timestamps = sorted(ref.index.tolist())
        warmup = 50  # Need 50 candles for indicators

        console.print(f"Simulating {len(timestamps) - warmup} candles of 5min data...\n")

        for i in range(warmup, len(timestamps)):
            ts = timestamps[i]
            ts_str = str(ts)
            day = ts_str[:10]
            hour = ts.hour if hasattr(ts, 'hour') else int(ts_str[11:13]) if len(ts_str) > 13 else 12

            # Reset daily counter
            if day != current_day:
                trades_today = 0
                current_day = day

            # SESSION FILTER: Only trade London (07-12) and NY (12-17) UTC
            in_session = 7 <= hour <= 16
            if not in_session:
                continue

            # 1. CHECK SL/TP on open positions
            to_close = []
            for pos in open_pos:
                df = all_data.get(pos.symbol)
                if df is None:
                    continue
                mask = df.index <= ts
                if mask.sum() == 0:
                    continue
                candle = df[mask].iloc[-1]
                high = float(candle["high"])
                low = float(candle["low"])
                close_p = float(candle["close"])

                spread = SCALP_SPREADS.get(pos.symbol, 0.0002)
                slip = spread * SLIPPAGE

                if pos.side == "long":
                    if low <= pos.stop_loss:
                        pos.exit_price = pos.stop_loss - slip
                        pos.pnl = (pos.exit_price - pos.entry_price) * pos.quantity
                        pos.exit_reason = "SL"
                        to_close.append(pos)
                    elif high >= pos.take_profit:
                        pos.exit_price = pos.take_profit - slip
                        pos.pnl = (pos.exit_price - pos.entry_price) * pos.quantity
                        pos.exit_reason = "TP"
                        to_close.append(pos)
                else:
                    if high >= pos.stop_loss:
                        pos.exit_price = pos.stop_loss + slip
                        pos.pnl = (pos.entry_price - pos.exit_price) * pos.quantity
                        pos.exit_reason = "SL"
                        to_close.append(pos)
                    elif low <= pos.take_profit:
                        pos.exit_price = pos.take_profit + slip
                        pos.pnl = (pos.entry_price - pos.exit_price) * pos.quantity
                        pos.exit_reason = "TP"
                        to_close.append(pos)

                # Time exit: close after 2 hours (24 candles) if no SL/TP
                if pos not in to_close:
                    entry_idx = next((j for j, t in enumerate(timestamps) if str(t) >= pos.entry_time), i)
                    candles_held = i - entry_idx
                    if candles_held >= 24:  # 2 hours
                        pos.exit_price = close_p
                        if pos.side == "long":
                            pos.pnl = (close_p - pos.entry_price) * pos.quantity
                        else:
                            pos.pnl = (pos.entry_price - close_p) * pos.quantity
                        pos.exit_reason = "TIME"
                        to_close.append(pos)

            for pos in to_close:
                pos.exit_time = ts_str
                cash += pos.pnl + (pos.entry_price * pos.quantity if pos.side == "long" else 0)
                open_pos.remove(pos)
                closed.append(pos)

            # 2. FIND NEW SCALP ENTRIES
            if len(open_pos) < self.max_trades and trades_today < self.max_trades_per_day:
                for sym, df in all_data.items():
                    if len(open_pos) >= self.max_trades:
                        break
                    if any(p.symbol == sym for p in open_pos):
                        continue

                    mask = df.index <= ts
                    df_slice = df[mask]
                    if len(df_slice) < 50:
                        continue

                    signal = self._scalp_signal(df_slice, sym)
                    if signal is None:
                        continue

                    entry = signal["price"]
                    sl = signal["sl"]
                    tp = signal["tp"]
                    risk_dist = abs(entry - sl)
                    if risk_dist <= 0:
                        continue

                    risk_amt = equity * self.risk_per_trade
                    qty = risk_amt / risk_dist

                    # Apply spread + slippage
                    spread = SCALP_SPREADS.get(sym, 0.0002)
                    slip = spread * SLIPPAGE
                    if signal["action"] == "buy":
                        fill = entry + spread / 2 + slip
                    else:
                        fill = entry - spread / 2 - slip

                    pos = ScalpPosition(sym, signal["action"], fill, round(qty, 4),
                                        sl, tp, ts_str)
                    if signal["action"] == "buy":
                        cash -= fill * qty
                    open_pos.append(pos)
                    trades_today += 1

            # 3. UPDATE equity
            unrealized = 0
            for pos in open_pos:
                df = all_data.get(pos.symbol)
                if df is None:
                    continue
                mask = df.index <= ts
                if mask.sum() == 0:
                    continue
                cp = float(df[mask].iloc[-1]["close"])
                if pos.side == "buy":
                    unrealized += (cp - pos.entry_price) * pos.quantity
                else:
                    unrealized += (pos.entry_price - cp) * pos.quantity

            equity = cash + sum(
                p.entry_price * p.quantity for p in open_pos if p.side == "buy"
            ) + unrealized

            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)

            # Progress (every 20 candles)
            if i % 20 == 0:
                pct = (i - warmup) / (len(timestamps) - warmup) * 100
                filled = int(30 * pct / 100)
                bar = "█" * filled + "░" * (30 - filled)
                w = sum(1 for t in closed if t.pnl > 0)
                l = sum(1 for t in closed if t.pnl <= 0)
                console.print(f"\r  [{bar}] {pct:.0f}% | {day} {hour:02d}:00 | ${equity:,.0f} | W:{w} L:{l}", end="")

        console.print("\n")
        return self._report(closed, open_pos, max_dd)

    def _scalp_signal(self, df: pd.DataFrame, symbol: str) -> dict | None:
        """5-minute scalping: momentum + mean reversion hybrid."""
        try:
            close = df["close"].values.astype(float)
            high = df["high"].values.astype(float)
            low = df["low"].values.astype(float)

            latest = float(close[-1])
            if math.isnan(latest) or latest <= 0:
                return None

            # Ultra-fast scalping indicators
            ema5 = talib.EMA(close, timeperiod=5)
            ema13 = talib.EMA(close, timeperiod=13)
            ema50 = talib.EMA(close, timeperiod=50)
            rsi = talib.RSI(close, timeperiod=7)
            atr = talib.ATR(high, low, close, timeperiod=10)
            macd, macd_sig, macd_hist = talib.MACD(close, fastperiod=5, slowperiod=13, signalperiod=4)
            upper, middle, lower_bb = talib.BBANDS(close, timeperiod=14, nbdevup=2, nbdevdn=2)
            stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)

            ema5_now = float(ema5[-1]) if not math.isnan(ema5[-1]) else latest
            ema13_now = float(ema13[-1]) if not math.isnan(ema13[-1]) else latest
            ema50_now = float(ema50[-1]) if not math.isnan(ema50[-1]) else latest
            rsi_val = float(rsi[-1]) if not math.isnan(rsi[-1]) else 50
            atr_val = float(atr[-1]) if not math.isnan(atr[-1]) else latest * 0.001
            macd_val = float(macd_hist[-1]) if not math.isnan(macd_hist[-1]) else 0
            macd_prev = float(macd_hist[-2]) if len(macd_hist) > 1 and not math.isnan(macd_hist[-2]) else 0
            bb_up = float(upper[-1]) if not math.isnan(upper[-1]) else latest * 1.01
            bb_lo = float(lower_bb[-1]) if not math.isnan(lower_bb[-1]) else latest * 0.99
            stoch = float(stoch_k[-1]) if not math.isnan(stoch_k[-1]) else 50

            if atr_val <= 0:
                return None

            # Trend direction from EMA50
            trend = "up" if latest > ema50_now else "down"

            # === MULTI-SIGNAL SCORING (not just crossover) ===
            bull = 0
            bear = 0

            # 1. EMA alignment (fast above slow = bullish momentum)
            if ema5_now > ema13_now:
                bull += 1
            else:
                bear += 1

            # 2. Price position vs EMA50 (trend context)
            if latest > ema50_now:
                bull += 1
            else:
                bear += 1

            # 3. RSI zones (adjusted for 5min - wider zones)
            if rsi_val < 30:
                bull += 2  # Oversold bounce
            elif rsi_val < 40:
                bull += 1
            if rsi_val > 70:
                bear += 2  # Overbought fade
            elif rsi_val > 60:
                bear += 1

            # 4. MACD momentum flip (most important for 5min)
            if macd_prev <= 0 < macd_val:
                bull += 2  # Just flipped bullish
            if macd_prev >= 0 > macd_val:
                bear += 2  # Just flipped bearish
            elif macd_val > 0:
                bull += 1
            elif macd_val < 0:
                bear += 1

            # 5. Bollinger Band touch (mean reversion)
            if latest <= bb_lo:
                bull += 2  # At lower band
            if latest >= bb_up:
                bear += 2  # At upper band

            # 6. Stochastic extremes
            if stoch < 20:
                bull += 1
            if stoch > 80:
                bear += 1

            # 7. 3-candle momentum
            if len(close) >= 3:
                mom3 = (latest - float(close[-3])) / float(close[-3]) * 100
                if mom3 > 0.05:
                    bull += 1
                elif mom3 < -0.05:
                    bear += 1

            # === DECISION: need 3+ score with trend alignment ===
            action = None
            net = bull - bear

            # With trend (safer)
            if trend == "up" and net >= 3 and bull >= 4:
                action = "buy"
            elif trend == "down" and net <= -3 and bear >= 4:
                action = "sell"
            # Counter-trend only on extreme signals (mean reversion)
            elif net >= 5 and bull >= 5:
                action = "buy"
            elif net <= -5 and bear >= 5:
                action = "sell"

            if action is None:
                return None

            # SL/TP for scalping: tight and fast
            sl_mult = 1.0
            tp_mult = 1.8  # Slightly less than 2:1 for more TP hits

            if action == "buy":
                sl = latest - atr_val * sl_mult
                tp = latest + atr_val * tp_mult
            else:
                sl = latest + atr_val * sl_mult
                tp = latest - atr_val * tp_mult

            return {
                "symbol": symbol, "action": action, "price": latest,
                "sl": round(sl, 5), "tp": round(tp, 5), "atr": atr_val,
                "rsi": rsi_val, "bull": bull, "bear": bear,
            }
        except Exception:
            return None

    def _report(self, closed, open_pos, max_dd):
        initial = self.account_size
        total_pnl = sum(t.pnl for t in closed)
        final = initial + total_pnl
        pnl_pct = total_pnl / initial * 100

        wins = [t for t in closed if t.pnl > 0]
        losses = [t for t in closed if t.pnl <= 0]
        total = len(closed)
        wr = len(wins) / total * 100 if total > 0 else 0
        avg_w = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_l = sum(t.pnl for t in losses) / len(losses) if losses else 0
        pf = abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else 999

        # Extrapolate to 30 days
        trading_days = 5  # yfinance gives ~5 days of 5min data
        monthly_pnl = total_pnl * (22 / trading_days) if trading_days > 0 else 0
        monthly_pct = monthly_pnl / initial * 100

        passed = pnl_pct >= 2.0  # 2% in 5 days = ~8-10% monthly pace

        console.print(Panel.fit(
            f"[bold]5-Min Scalp Results - ~{trading_days} Trading Days[/bold]",
            border_style="green" if passed else "red",
        ))

        t = Table(show_header=False, box=None)
        t.add_column("", style="bold", width=28)
        t.add_column("", width=25)

        pc = "green" if total_pnl >= 0 else "red"
        t.add_row("Initial Balance", f"${initial:,.2f}")
        t.add_row("Final Balance", f"${final:,.2f}")
        t.add_row("P&L (5 days)", f"[{pc}]${total_pnl:+,.2f} ({pnl_pct:+.2f}%)[/{pc}]")
        t.add_row("Projected Monthly P&L", f"[{pc}]${monthly_pnl:+,.0f} ({monthly_pct:+.1f}%)[/{pc}]")
        t.add_row("Max Drawdown", f"{max_dd:.2f}%")
        t.add_row("", "")
        t.add_row("Total Trades", str(total))
        t.add_row("Wins / Losses", f"{len(wins)} / {len(losses)}")
        wc = "green" if wr >= 50 else "red"
        t.add_row("Win Rate", f"[{wc}]{wr:.1f}%[/{wc}]")
        t.add_row("Avg Win", f"${avg_w:+,.2f}")
        t.add_row("Avg Loss", f"${avg_l:+,.2f}")
        t.add_row("Profit Factor", f"{pf:.2f}" if pf < 999 else "∞")
        t.add_row("Trades/Day", f"{total/max(1,trading_days):.1f}")
        t.add_row("", "")
        t.add_row("10% Challenge Pace?", f"[{'green bold' if monthly_pct >= 10 else 'red'}]{'YES' if monthly_pct >= 10 else 'NO'} ({monthly_pct:.1f}%/month)[/{'green bold' if monthly_pct >= 10 else 'red'}]")

        console.print(t)

        if closed:
            console.print(f"\n[bold]Last 20 Trades[/bold]")
            tt = Table()
            tt.add_column("#", width=3)
            tt.add_column("Symbol", width=10)
            tt.add_column("Side", width=5)
            tt.add_column("Entry", width=10)
            tt.add_column("Exit", width=10)
            tt.add_column("P&L", width=10)
            tt.add_column("Reason", width=5)
            tt.add_column("Time", width=20)

            for idx, trade in enumerate(closed[-20:], 1):
                c = "green" if trade.pnl > 0 else "red"
                entry_t = trade.entry_time[11:16] if len(trade.entry_time) > 15 else ""
                exit_t = trade.exit_time[11:16] if trade.exit_time and len(trade.exit_time) > 15 else ""
                tt.add_row(
                    str(idx), trade.symbol[:8], trade.side.upper()[:4],
                    f"{trade.entry_price:.4f}"[:10],
                    f"{trade.exit_price:.4f}"[:10] if trade.exit_price else "",
                    f"[{c}]${trade.pnl:+,.2f}[/{c}]",
                    trade.exit_reason or "", f"{entry_t}→{exit_t}",
                )
            console.print(tt)

        return {
            "pnl": round(total_pnl, 2), "pnl_pct": round(pnl_pct, 2),
            "monthly_projected": round(monthly_pnl, 2),
            "monthly_pct": round(monthly_pct, 1),
            "trades": total, "wins": len(wins), "losses": len(losses),
            "win_rate": round(wr, 1), "max_dd": round(max_dd, 2),
            "profit_factor": round(pf, 2) if pf < 999 else 999,
        }
