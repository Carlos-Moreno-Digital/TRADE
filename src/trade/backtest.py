"""Real Backtester v2 - Proper historical simulation with no lookahead bias.

Key differences from v1:
- Downloads ALL data upfront, then SLICES it per day (no future data leak)
- Feeds sliced DataFrames directly to agents (bypasses yfinance calls during sim)
- Tracks open positions with real price updates each day
- Simulates SL/TP hits with actual daily high/low prices
- Properly tracks P&L, drawdown, and win/loss per trade

Usage:
    python -m trade.main --backtest           # 30 days default
    python -m trade.main --backtest 60        # 60 days
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
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


class Position:
    """A simulated open position."""

    def __init__(self, symbol: str, side: str, entry_price: float, quantity: float,
                 stop_loss: float, take_profit: float, entry_date: str):
        self.symbol = symbol
        self.side = side  # "long" or "short"
        self.entry_price = entry_price
        self.quantity = quantity
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.entry_date = entry_date
        self.exit_price: float | None = None
        self.exit_date: str | None = None
        self.exit_reason: str | None = None
        self.pnl: float = 0.0


class Backtester:
    """Proper backtester that slices data per day and simulates real trading."""

    def __init__(self, prop_firm: str = "funderpro_classic_10k", top_n: int = 5):
        from trade.risk.prop_firm import load_prop_firm_config
        prop_config = load_prop_firm_config(prop_firm)
        self.account_size = prop_config.account_size
        self.risk_per_trade = 0.01    # 1% per trade - aggressive but within limits
        self.max_trades = 4           # Up to 4 concurrent positions
        self.min_rr = 1.5             # Minimum 1.5:1 risk:reward
        self.top_n = top_n

        self.provider = MarketDataProvider()

    def run(self, days: int = 30, symbols: list[str] | None = None) -> dict[str, Any]:
        """Run backtest."""
        if symbols is None:
            symbols = self._get_default_symbols()

        console.print(Panel.fit(
            f"[bold cyan]Backtester v2 - {days} Day Simulation[/bold cyan]\n"
            f"Account: ${self.account_size:,.0f} | Risk/trade: {self.risk_per_trade*100:.1f}% | "
            f"R:R min: {self.min_rr} | Max positions: {self.max_trades}",
            title="Backtest",
            border_style="cyan",
        ))

        # Download all data upfront
        console.print(f"[dim]Downloading {len(symbols)} symbols...[/dim]")
        all_data = {}
        for sym in symbols:
            try:
                df = self.provider.get_historical(sym, period="6mo", interval="1d")
                if not df.empty and len(df) >= 50:
                    all_data[sym] = df
            except Exception:
                pass
        console.print(f"  Got data for {len(all_data)}/{len(symbols)} symbols\n")

        if not all_data:
            return {"error": "No data"}

        # Get trading days
        ref_df = next(iter(all_data.values()))
        all_dates = sorted(ref_df.index.tolist())
        warmup = 40  # Need 40 days for indicators
        sim_dates = all_dates[warmup:][-days:]

        # Simulation state
        cash = self.account_size
        equity = self.account_size
        peak = self.account_size
        open_positions: list[Position] = []
        closed_trades: list[Position] = []
        daily_equity: list[dict] = []
        max_dd = 0.0

        console.print(f"Simulating {len(sim_dates)} trading days...\n")

        for i, date in enumerate(sim_dates):
            date_str = date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date)[:10]

            # 1. CHECK SL/TP + TRAILING STOP on open positions
            positions_to_close = []
            for pos in open_positions:
                df = all_data.get(pos.symbol)
                if df is None:
                    continue
                mask = df.index <= date
                if mask.sum() == 0:
                    continue
                today = df[mask].iloc[-1]
                high = float(today["high"])
                low = float(today["low"])
                close_price = float(today["close"])

                # TRAILING STOP: move SL to breakeven when price moves 1x risk in our favor
                risk_dist = abs(pos.entry_price - pos.stop_loss)
                if pos.side == "long":
                    # If price is 1x risk above entry, move SL to entry (breakeven)
                    if close_price >= pos.entry_price + risk_dist and pos.stop_loss < pos.entry_price:
                        pos.stop_loss = pos.entry_price + risk_dist * 0.1  # Tiny profit lock
                    # If price is 1.5x risk above, trail SL to 0.5x risk above entry
                    if close_price >= pos.entry_price + risk_dist * 1.5 and pos.stop_loss < pos.entry_price + risk_dist * 0.5:
                        pos.stop_loss = pos.entry_price + risk_dist * 0.5
                elif pos.side == "short":
                    if close_price <= pos.entry_price - risk_dist and pos.stop_loss > pos.entry_price:
                        pos.stop_loss = pos.entry_price - risk_dist * 0.1
                    if close_price <= pos.entry_price - risk_dist * 1.5 and pos.stop_loss > pos.entry_price - risk_dist * 0.5:
                        pos.stop_loss = pos.entry_price - risk_dist * 0.5

                if pos.side == "long":
                    if low <= pos.stop_loss:
                        pos.exit_price = pos.stop_loss
                        pos.exit_reason = "SL" if pos.stop_loss <= pos.entry_price else "TSL"
                        pos.pnl = (pos.exit_price - pos.entry_price) * pos.quantity
                        positions_to_close.append(pos)
                    elif high >= pos.take_profit:
                        pos.exit_price = pos.take_profit
                        pos.exit_reason = "TP"
                        pos.pnl = (pos.exit_price - pos.entry_price) * pos.quantity
                        positions_to_close.append(pos)
                elif pos.side == "short":
                    if high >= pos.stop_loss:
                        pos.exit_price = pos.stop_loss
                        pos.exit_reason = "SL" if pos.stop_loss >= pos.entry_price else "TSL"
                        pos.pnl = (pos.entry_price - pos.exit_price) * pos.quantity
                        positions_to_close.append(pos)
                    elif low <= pos.take_profit:
                        pos.exit_price = pos.take_profit
                        pos.exit_reason = "TP"
                        pos.pnl = (pos.entry_price - pos.exit_price) * pos.quantity
                        positions_to_close.append(pos)

            for pos in positions_to_close:
                pos.exit_date = date_str
                cash += pos.pnl + (pos.entry_price * pos.quantity if pos.side == "long" else 0)
                open_positions.remove(pos)
                closed_trades.append(pos)

            # 2. ANALYZE each symbol for new entries (if we have room)
            if len(open_positions) < self.max_trades:
                scored = []
                for sym, df in all_data.items():
                    # Skip if already have position
                    if any(p.symbol == sym for p in open_positions):
                        continue

                    mask = df.index <= date
                    df_slice = df[mask]
                    if len(df_slice) < 40:
                        continue

                    signal = self._analyze_symbol(df_slice, sym)
                    if signal and signal["action"] != "hold":
                        scored.append(signal)

                # Sort by confidence, take top N
                scored.sort(key=lambda s: s["confidence"], reverse=True)

                for signal in scored[:self.top_n - len(open_positions)]:
                    if len(open_positions) >= self.max_trades:
                        break

                    entry_price = signal["price"]
                    sl = signal["stop_loss"]
                    tp = signal["take_profit"]
                    risk_per_unit = abs(entry_price - sl)

                    if risk_per_unit <= 0:
                        continue

                    # Position size based on risk
                    risk_amount = equity * self.risk_per_trade
                    quantity = risk_amount / risk_per_unit

                    # Check R:R
                    reward = abs(tp - entry_price)
                    rr = reward / risk_per_unit if risk_per_unit > 0 else 0
                    if rr < self.min_rr:
                        continue

                    side = "long" if signal["action"] == "buy" else "short"

                    pos = Position(
                        symbol=signal["symbol"],
                        side=side,
                        entry_price=entry_price,
                        quantity=round(quantity, 4),
                        stop_loss=sl,
                        take_profit=tp,
                        entry_date=date_str,
                    )

                    if side == "long":
                        cash -= entry_price * quantity

                    open_positions.append(pos)

            # 3. UPDATE equity
            unrealized = 0
            for pos in open_positions:
                df = all_data.get(pos.symbol)
                if df is None:
                    continue
                mask = df.index <= date
                if mask.sum() == 0:
                    continue
                current_price = float(df[mask].iloc[-1]["close"])
                if pos.side == "long":
                    unrealized += (current_price - pos.entry_price) * pos.quantity
                else:
                    unrealized += (pos.entry_price - current_price) * pos.quantity

            equity = cash + sum(
                pos.entry_price * pos.quantity for pos in open_positions if pos.side == "long"
            ) + unrealized

            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)

            daily_equity.append({
                "date": date_str,
                "equity": round(equity, 2),
                "cash": round(cash, 2),
                "positions": len(open_positions),
                "unrealized": round(unrealized, 2),
            })

            # Progress
            pct = (i + 1) / len(sim_dates) * 100
            bar_len = 30
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            trades_str = f"W:{sum(1 for t in closed_trades if t.pnl > 0)} L:{sum(1 for t in closed_trades if t.pnl <= 0)}"
            console.print(
                f"\r  [{bar}] {pct:.0f}% | {date_str} | ${equity:,.0f} | "
                f"Open:{len(open_positions)} | {trades_str}",
                end="",
            )

        console.print("\n")

        # Generate report
        return self._report(closed_trades, open_positions, daily_equity, max_dd, sim_dates)

    def _analyze_symbol(self, df: pd.DataFrame, symbol: str) -> dict | None:
        """Analyze a symbol for prop firm trading. More signals, tighter SL/TP."""
        try:
            close = df["close"].values.astype(float)
            high = df["high"].values.astype(float)
            low = df["low"].values.astype(float)

            if len(close) < 30:
                return None

            latest = float(close[-1])
            if math.isnan(latest) or latest <= 0:
                return None

            # === INDICATORS ===
            rsi = talib.RSI(close, timeperiod=14)
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            ema9 = talib.EMA(close, timeperiod=9)
            ema21 = talib.EMA(close, timeperiod=21)
            ema50 = talib.EMA(close, timeperiod=50)
            atr = talib.ATR(high, low, close, timeperiod=14)
            stoch_k, stoch_d = talib.STOCH(high, low, close)
            upper, middle, lower = talib.BBANDS(close, timeperiod=20)

            rsi_val = float(rsi[-1]) if not math.isnan(rsi[-1]) else 50
            macd_val = float(macd_hist[-1]) if not math.isnan(macd_hist[-1]) else 0
            macd_prev = float(macd_hist[-2]) if len(macd_hist) > 1 and not math.isnan(macd_hist[-2]) else 0
            ema9_val = float(ema9[-1]) if not math.isnan(ema9[-1]) else latest
            ema21_val = float(ema21[-1]) if not math.isnan(ema21[-1]) else latest
            ema50_val = float(ema50[-1]) if not math.isnan(ema50[-1]) else latest
            atr_val = float(atr[-1]) if not math.isnan(atr[-1]) else latest * 0.02
            stoch_val = float(stoch_k[-1]) if not math.isnan(stoch_k[-1]) else 50
            bb_upper = float(upper[-1]) if not math.isnan(upper[-1]) else latest * 1.02
            bb_lower = float(lower[-1]) if not math.isnan(lower[-1]) else latest * 0.98

            if atr_val <= 0:
                return None

            # === MAJOR TREND FILTER (THE KEY RULE) ===
            # Price vs EMA50 determines allowed direction
            # If price > EMA50: ONLY LONGS (buy pullbacks in uptrend)
            # If price < EMA50: ONLY SHORTS (sell rallies in downtrend)
            major_trend = "bullish" if latest > ema50_val else "bearish"

            # 20-day momentum for trend confirmation
            if len(close) >= 20:
                mom20 = (latest - float(close[-20])) / float(close[-20]) * 100
            else:
                mom20 = 0

            # === SCORING ===
            bull_score = 0
            bear_score = 0

            # Trend alignment (EMA stack)
            if ema9_val > ema21_val > ema50_val:
                bull_score += 3  # Perfect uptrend alignment
            elif ema9_val > ema21_val:
                bull_score += 1
            if ema9_val < ema21_val < ema50_val:
                bear_score += 3
            elif ema9_val < ema21_val:
                bear_score += 1

            # RSI - look for pullbacks within trend, not reversals
            if major_trend == "bullish":
                # In uptrend: buy when RSI pulls back to 40-50 (not just oversold)
                if 35 <= rsi_val <= 50:
                    bull_score += 2  # Pullback in uptrend = best entry
                if rsi_val < 35:
                    bull_score += 1  # Oversold in uptrend
            else:
                # In downtrend: sell when RSI bounces to 50-65
                if 50 <= rsi_val <= 65:
                    bear_score += 2
                if rsi_val > 65:
                    bear_score += 1

            # MACD
            if macd_val > 0:
                bull_score += 1
            if macd_val < 0:
                bear_score += 1
            if macd_prev <= 0 < macd_val:
                bull_score += 2  # Bullish crossover
            if macd_prev >= 0 > macd_val:
                bear_score += 2

            # Stochastic - confirming pullback entries
            if major_trend == "bullish" and stoch_val < 30:
                bull_score += 1  # Oversold in uptrend
            if major_trend == "bearish" and stoch_val > 70:
                bear_score += 1

            # Bollinger
            if major_trend == "bullish" and latest <= bb_lower:
                bull_score += 2  # Price hit lower band in uptrend = strong buy
            if major_trend == "bearish" and latest >= bb_upper:
                bear_score += 2

            # Short-term momentum (for entry timing)
            if len(close) >= 3:
                mom3 = (latest - float(close[-3])) / float(close[-3]) * 100
                if mom3 > 0.2:
                    bull_score += 1
                elif mom3 < -0.2:
                    bear_score += 1

            # === DECISION (with trend filter) ===
            action = "hold"
            confidence = 0.0
            net_score = bull_score - bear_score

            # ONLY trade WITH the major trend (lowered thresholds for more trades)
            if major_trend == "bullish" and net_score >= 2 and bull_score >= 2:
                action = "buy"
                confidence = min(0.9, bull_score * 0.12)
            elif major_trend == "bearish" and net_score <= -2 and bear_score >= 2:
                action = "sell"
                confidence = min(0.9, bear_score * 0.12)
            # Counter-trend only on extreme signals
            elif major_trend == "bullish" and net_score <= -4 and bear_score >= 5:
                action = "sell"
                confidence = 0.35
            elif major_trend == "bearish" and net_score >= 4 and bull_score >= 5:
                action = "buy"
                confidence = 0.35

            if action == "hold":
                return None

            # === SL/TP optimized for prop firm ===
            sl_mult = 1.2   # Tight SL = controlled risk
            tp_mult = 2.5   # TP at 2.5x risk = good R:R, still achievable

            if action == "buy":
                sl = latest - atr_val * sl_mult
                tp = latest + atr_val * tp_mult
            else:
                sl = latest + atr_val * sl_mult
                tp = latest - atr_val * tp_mult

            return {
                "symbol": symbol,
                "action": action,
                "price": latest,
                "stop_loss": round(sl, 5),
                "take_profit": round(tp, 5),
                "confidence": confidence,
                "atr": atr_val,
                "rsi": rsi_val,
                "bull_score": bull_score,
                "bear_score": bear_score,
                "net_score": net_score,
            }
        except Exception:
            return None

    def _report(
        self,
        closed: list[Position],
        open_pos: list[Position],
        daily_eq: list[dict],
        max_dd: float,
        sim_dates: list,
    ) -> dict:
        """Generate backtest report."""
        initial = self.account_size
        final = daily_eq[-1]["equity"] if daily_eq else initial
        total_pnl = final - initial
        pnl_pct = (total_pnl / initial * 100) if initial > 0 else 0

        wins = [t for t in closed if t.pnl > 0]
        losses = [t for t in closed if t.pnl <= 0]
        total_trades = len(closed)
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0

        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
        profit_factor = abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else float('inf')

        passed = pnl_pct >= 10.0 and max_dd < 10.0

        # Print report
        console.print(Panel.fit(
            f"[bold]Backtest Results - {len(sim_dates)} Trading Days[/bold]",
            border_style="green" if passed else "red",
        ))

        t = Table(show_header=False, box=None)
        t.add_column("", style="bold", width=25)
        t.add_column("", width=25)

        pnl_color = "green" if total_pnl >= 0 else "red"
        t.add_row("Initial Balance", f"${initial:,.2f}")
        t.add_row("Final Balance", f"${final:,.2f}")
        t.add_row("Total P&L", f"[{pnl_color}]${total_pnl:+,.2f} ({pnl_pct:+.2f}%)[/{pnl_color}]")
        t.add_row("Max Drawdown", f"{max_dd:.2f}%")
        t.add_row("", "")
        t.add_row("Total Trades", str(total_trades))
        t.add_row("Wins / Losses", f"{len(wins)} / {len(losses)}")
        wr_color = "green" if win_rate >= 50 else "red"
        t.add_row("Win Rate", f"[{wr_color}]{win_rate:.1f}%[/{wr_color}]")
        t.add_row("Avg Win", f"${avg_win:+,.2f}")
        t.add_row("Avg Loss", f"${avg_loss:+,.2f}")
        pf_str = f"{profit_factor:.2f}" if profit_factor < 999 else "∞"
        t.add_row("Profit Factor", pf_str)
        t.add_row("Open Positions", str(len(open_pos)))
        t.add_row("", "")
        pass_color = "green bold" if passed else "red bold"
        t.add_row("Challenge Target", "10% ($1,000)")
        t.add_row("Challenge Result", f"[{pass_color}]{'PASSED' if passed else 'NOT PASSED'}[/{pass_color}]")

        console.print(t)

        # Trade log
        if closed:
            console.print(f"\n[bold]Trade History ({len(closed)} trades)[/bold]")
            tt = Table()
            tt.add_column("#", width=3)
            tt.add_column("Symbol", width=12)
            tt.add_column("Side", width=6)
            tt.add_column("Entry", width=12)
            tt.add_column("Exit", width=12)
            tt.add_column("P&L", width=12)
            tt.add_column("Reason", width=6)
            tt.add_column("Dates", width=25)

            for i, trade in enumerate(closed, 1):
                pnl_c = "green" if trade.pnl > 0 else "red"
                tt.add_row(
                    str(i),
                    trade.symbol,
                    trade.side.upper(),
                    f"{trade.entry_price:.2f}",
                    f"{trade.exit_price:.2f}" if trade.exit_price else "OPEN",
                    f"[{pnl_c}]${trade.pnl:+,.2f}[/{pnl_c}]",
                    trade.exit_reason or "",
                    f"{trade.entry_date} → {trade.exit_date or ''}",
                )
            console.print(tt)

        # Equity curve
        if daily_eq:
            console.print(f"\n[bold]Equity Curve (last 20 days)[/bold]")
            min_eq = min(e["equity"] for e in daily_eq)
            max_eq = max(e["equity"] for e in daily_eq)
            eq_range = max_eq - min_eq if max_eq > min_eq else 1

            for eq in daily_eq[-20:]:
                normalized = (eq["equity"] - min_eq) / eq_range if eq_range > 0 else 0.5
                bar_len = int(normalized * 40)
                bar = "█" * max(1, bar_len)
                diff = eq["equity"] - initial
                color = "green" if diff >= 0 else "red"
                console.print(
                    f"  {eq['date']} | ${eq['equity']:>10,.2f} [{color}]({diff:+,.0f})[/{color}] | [{color}]{bar}[/{color}]"
                )

        return {
            "days": len(sim_dates),
            "initial": initial,
            "final": round(final, 2),
            "pnl": round(total_pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "max_drawdown": round(max_dd, 2),
            "total_trades": total_trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 1),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor < 999 else 999,
            "passed": passed,
        }

    def _get_default_symbols(self) -> list[str]:
        return [
            # Forex majors
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X", "USDCAD=X", "USDCHF=X",
            # Forex crosses - more opportunities
            "EURGBP=X", "EURJPY=X", "GBPJPY=X", "EURNZD=X", "EURAUD=X", "EURCAD=X", "EURCHF=X",
            "GBPAUD=X", "GBPNZD=X", "GBPCAD=X", "GBPCHF=X",
            "AUDJPY=X", "NZDJPY=X", "CADJPY=X", "CHFJPY=X",
            "AUDNZD=X", "AUDCAD=X",
            # Commodities
            "GC=F", "SI=F", "CL=F", "PL=F",
            # Crypto
            "BTC-USD", "ETH-USD",
        ]
