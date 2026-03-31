"""5-Minute Scalping Backtester v3 - Opening Range Breakout (ORB).

Strategy: ORB + EMA50 Trend + ADX Filter
- Opening Range: High/Low of first 30min of London (07:00-07:30) and NY (13:00-13:30)
- Entry: Breakout above/below range in direction of EMA50 trend
- Filter: ADX > 18 (market must be moving)
- Exit: SL at opposite side of range, TP at 2x range width
- Sessions: London (07:30-11:00 UTC) and NY (13:30-16:00 UTC) only
- Max 1 trade per session per symbol (prevents overtrading)
- Risk: 0.5% per trade
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

# Spreads for scalping (in price units)
# These are CONSERVATIVE estimates — real ECN/STP may be 30-50% tighter
SCALP_SPREADS = {
    "EURUSD=X": 0.00008, "GBPUSD=X": 0.00010, "USDJPY=X": 0.008,
    "AUDUSD=X": 0.00010, "USDCAD=X": 0.00012, "USDCHF=X": 0.00010,
    "EURGBP=X": 0.00012, "EURJPY=X": 0.012, "GBPJPY=X": 0.015,
}
SLIPPAGE = 0.5  # 50% of spread


class ScalpPosition:
    def __init__(self, symbol, side, entry, qty, sl, tp, entry_time, breakeven_trigger=None):
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
        # Breakeven: when price reaches this level, move SL to entry
        self.breakeven_trigger = breakeven_trigger
        self.breakeven_active = False


class ScalpBacktester:
    def __init__(self, account_size: float = 10000):
        self.account_size = account_size
        self.risk_per_trade = 0.005  # 0.5% per scalp
        self.max_trades = 2          # Max 2 concurrent
        self.max_trades_per_day = 4  # ORB: max 1 per session per symbol = very selective
        self.provider = MarketDataProvider()
        # Track opening ranges per day per symbol per session
        self._opening_ranges: dict[str, dict] = {}  # key: "SYMBOL_DATE_SESSION"

    def run(self, symbols: list[str] | None = None) -> dict[str, Any]:
        if symbols is None:
            symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
                       "USDCAD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X"]

        console.print(Panel.fit(
            f"[bold cyan]5-Minute Scalping Backtester[/bold cyan]\n"
            f"Account: ${self.account_size:,.0f} | Risk: {self.risk_per_trade*100:.1f}%/trade | "
            f"Max {self.max_trades_per_day} trades/day\n"
            f"Strategy: Opening Range Breakout (ORB) + EMA50 + ADX\n"
            f"Sessions: London (07:30-11:00) + NY (13:30-16:00) UTC",
            title="Scalp Backtest v3",
            border_style="magenta",
        ))
        self._opening_ranges = {}  # Reset for fresh run

        # Download 5min data (yfinance: max 5 days)
        console.print(f"[dim]Downloading 5min data for {len(symbols)} pairs...[/dim]")
        all_data = {}
        import yfinance as yf
        from datetime import timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        for sym in symbols:
            try:
                df = yf.download(sym, start=start_date, end=end_date,
                                interval="5m", progress=False)
                if not df.empty and len(df) >= 200:
                    # Flatten MultiIndex columns from yf.download
                    if hasattr(df.columns, 'levels'):
                        df.columns = [c[0].lower() for c in df.columns]
                    else:
                        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                    all_data[sym] = df
            except Exception as e:
                logger.debug(f"Failed {sym}: {e}")
        console.print(f"  Got {len(all_data)}/{len(symbols)} symbols (5min, 30 days)\n")

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

                # Breakeven stop: move SL to entry when trigger is hit
                if not pos.breakeven_active and pos.breakeven_trigger is not None:
                    if pos.side == "buy" and high >= pos.breakeven_trigger:
                        pos.stop_loss = pos.entry_price + spread  # Lock in tiny profit
                        pos.breakeven_active = True
                    elif pos.side == "sell" and low <= pos.breakeven_trigger:
                        pos.stop_loss = pos.entry_price - spread
                        pos.breakeven_active = True

                if pos.side == "buy":
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
                # Return margin + P&L
                margin = pos.entry_price * pos.quantity * 0.01
                cash += margin + pos.pnl
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

                    signal = self._scalp_signal(df_slice, sym, current_ts=ts)
                    if signal is None:
                        continue

                    entry = signal["price"]
                    sl = signal["sl"]
                    tp = signal["tp"]
                    risk_dist = abs(entry - sl)
                    if risk_dist <= 0:
                        continue

                    # Use INITIAL balance for sizing (not equity - prevents runaway)
                    risk_amt = self.account_size * self.risk_per_trade
                    qty = risk_amt / risk_dist
                    # Cap quantity to prevent absurd sizes
                    max_notional = self.account_size * 5  # Max 5x leverage
                    max_qty = max_notional / entry if entry > 0 else 0
                    qty = min(qty, max_qty)

                    # Apply spread + slippage
                    spread = SCALP_SPREADS.get(sym, 0.0002)
                    slip = spread * SLIPPAGE
                    if signal["action"] == "buy":
                        fill = entry + spread / 2 + slip
                    else:
                        fill = entry - spread / 2 - slip

                    pos = ScalpPosition(sym, signal["action"], fill, round(qty, 4),
                                        sl, tp, ts_str,
                                        breakeven_trigger=signal.get("breakeven_trigger"))
                    # Forex: don't subtract notional, only track margin
                    # Margin = ~1% of notional (1:100 leverage)
                    margin = fill * qty * 0.01
                    cash -= margin
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

            # Equity = cash (with margin reserved) + unrealized P&L + margin held
            margin_held = sum(
                p.entry_price * p.quantity * 0.01 for p in open_pos
            )
            equity = cash + margin_held + unrealized

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

    def run_random_samples(self, n_samples: int = 8, symbols: list[str] | None = None) -> dict:
        """Test strategy on N non-overlapping 5-day windows from 60 days of REAL 5min data."""
        import yfinance as yf
        from datetime import timedelta

        if symbols is None:
            symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
                       "USDCAD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X"]

        console.print(Panel.fit(
            f"[bold magenta]5-Min Validation - {n_samples} Independent Periods[/bold magenta]\n"
            f"Downloads 60 days of REAL 5min data\n"
            f"Splits into {n_samples} non-overlapping 5-day windows\n"
            f"Tests scalping strategy on each independently",
            title="Scalp Validation",
            border_style="magenta",
        ))

        # Download 60 days of 5min data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=59)

        console.print(f"[dim]Downloading 60 days of 5min data for {len(symbols)} pairs...[/dim]")
        all_5min = {}
        for sym in symbols:
            try:
                df = yf.download(sym, start=start_date, end=end_date,
                                interval="5m", progress=False)
                if not df.empty and len(df) >= 500:
                    if hasattr(df.columns, 'levels'):
                        df.columns = [c[0].lower() for c in df.columns]
                    else:
                        df.columns = [c.lower() for c in df.columns]
                    all_5min[sym] = df
            except Exception:
                pass
        console.print(f"  Got {len(all_5min)}/{len(symbols)} symbols\n")

        if not all_5min:
            return {"error": "No data"}

        # Get trading days
        ref = next(iter(all_5min.values()))
        all_days = sorted(set(str(d)[:10] for d in ref.index))
        console.print(f"  {len(all_days)} trading days available\n")

        # Split into non-overlapping 5-day blocks
        # Skip first 3 days (warmup for indicators)
        usable_days = all_days[3:]
        blocks = []
        for i in range(0, len(usable_days) - 4, 5):
            block = usable_days[i:i+5]
            if len(block) == 5:
                blocks.append(block)

        n_samples = min(n_samples, len(blocks))
        console.print(f"Testing {n_samples} independent 5-day periods:\n")

        results = []
        total_trades = 0
        total_wins = 0
        total_losses = 0
        total_pnl = 0.0

        for idx, block_days in enumerate(blocks[:n_samples]):
            block_start = block_days[0]
            block_end = block_days[-1]

            # Run strategy on this block
            self._opening_ranges = {}  # Reset ORB state for each block
            cash = self.account_size
            open_pos = []
            closed = []
            max_dd = 0.0
            peak = self.account_size
            trades_today = 0
            current_day = ""

            ref_df = next(iter(all_5min.values()))
            all_ts = sorted(ref_df.index.tolist())
            warmup = 50

            for i in range(warmup, len(all_ts)):
                ts = all_ts[i]
                ts_str = str(ts)
                day = ts_str[:10]

                if day < block_start or day > block_end:
                    continue

                hour = ts.hour if hasattr(ts, 'hour') else 12
                if not (7 <= hour <= 16):
                    continue

                if day != current_day:
                    trades_today = 0
                    current_day = day

                # Check SL/TP
                to_close = []
                for pos in open_pos:
                    df = all_5min.get(pos.symbol)
                    if df is None:
                        continue
                    mask = df.index <= ts
                    if mask.sum() == 0:
                        continue
                    c = df[mask].iloc[-1]
                    h, l = float(c["high"]), float(c["low"])
                    spread = SCALP_SPREADS.get(pos.symbol, 0.0002)
                    slip = spread * SLIPPAGE

                    # Breakeven stop
                    if not pos.breakeven_active and pos.breakeven_trigger is not None:
                        if pos.side == "buy" and h >= pos.breakeven_trigger:
                            pos.stop_loss = pos.entry_price + spread
                            pos.breakeven_active = True
                        elif pos.side == "sell" and l <= pos.breakeven_trigger:
                            pos.stop_loss = pos.entry_price - spread
                            pos.breakeven_active = True

                    if pos.side == "buy":
                        if l <= pos.stop_loss:
                            pos.exit_price = pos.stop_loss - slip
                            pos.pnl = (pos.exit_price - pos.entry_price) * pos.quantity
                            pos.exit_reason = "SL"; to_close.append(pos)
                        elif h >= pos.take_profit:
                            pos.exit_price = pos.take_profit - slip
                            pos.pnl = (pos.exit_price - pos.entry_price) * pos.quantity
                            pos.exit_reason = "TP"; to_close.append(pos)
                    else:
                        if h >= pos.stop_loss:
                            pos.exit_price = pos.stop_loss + slip
                            pos.pnl = (pos.entry_price - pos.exit_price) * pos.quantity
                            pos.exit_reason = "SL"; to_close.append(pos)
                        elif l <= pos.take_profit:
                            pos.exit_price = pos.take_profit + slip
                            pos.pnl = (pos.entry_price - pos.exit_price) * pos.quantity
                            pos.exit_reason = "TP"; to_close.append(pos)

                for pos in to_close:
                    cash += pos.entry_price * pos.quantity * 0.01 + pos.pnl
                    open_pos.remove(pos)
                    closed.append(pos)

                # New entries
                if len(open_pos) < self.max_trades and trades_today < self.max_trades_per_day:
                    for sym, df in all_5min.items():
                        if len(open_pos) >= self.max_trades:
                            break
                        if any(p.symbol == sym for p in open_pos):
                            continue
                        mask = df.index <= ts
                        sl = df[mask]
                        if len(sl) < 50:
                            continue
                        signal = self._scalp_signal(sl, sym, current_ts=ts)
                        if signal is None:
                            continue

                        entry = signal["price"]
                        rd = abs(entry - signal["sl"])
                        if rd <= 0:
                            continue
                        qty = min(self.account_size * self.risk_per_trade / rd,
                                  self.account_size * 5 / entry if entry > 0 else 0)

                        spread = SCALP_SPREADS.get(sym, 0.0002)
                        slip = spread * SLIPPAGE
                        fill = entry + spread/2 + slip if signal["action"] == "buy" else entry - spread/2 - slip

                        pos = ScalpPosition(sym, signal["action"], fill, round(qty, 4),
                                            signal["sl"], signal["tp"], ts_str,
                                            breakeven_trigger=signal.get("breakeven_trigger"))
                        cash -= fill * qty * 0.01
                        open_pos.append(pos)
                        trades_today += 1

                # Equity
                margin_held = sum(p.entry_price * p.quantity * 0.01 for p in open_pos)
                unrealized = 0
                for p in open_pos:
                    df = all_5min.get(p.symbol)
                    if df is None: continue
                    m = df.index <= ts
                    if m.sum() == 0: continue
                    cp = float(df[m].iloc[-1]["close"])
                    unrealized += (cp - p.entry_price) * p.quantity if p.side == "buy" else (p.entry_price - cp) * p.quantity

                equity = cash + margin_held + unrealized
                if equity > peak: peak = equity
                dd = min((peak - equity) / peak * 100 if peak > 0 else 0, 100)
                max_dd = max(max_dd, dd)

            # Close remaining
            for pos in open_pos:
                cash += pos.entry_price * pos.quantity * 0.01

            w = sum(1 for t in closed if t.pnl > 0)
            l_count = sum(1 for t in closed if t.pnl <= 0)
            pnl = sum(t.pnl for t in closed)
            pnl_pct = pnl / self.account_size * 100

            total_trades += len(closed)
            total_wins += w
            total_losses += l_count
            total_pnl += pnl

            pc = "green" if pnl >= 0 else "red"
            wr = w / len(closed) * 100 if closed else 0
            console.print(
                f"  [{idx+1:2d}/{n_samples}] {block_start} → {block_end} | "
                f"{len(closed):3d} trades | W:{w} L:{l_count} ({wr:.0f}%) | "
                f"[{pc}]${pnl:+.2f} ({pnl_pct:+.1f}%)[/{pc}] | DD:{max_dd:.1f}%"
            )
            results.append({"start": block_start, "end": block_end, "trades": len(closed),
                           "wins": w, "losses": l_count, "pnl": round(pnl, 2),
                           "pnl_pct": round(pnl_pct, 2), "win_rate": round(wr, 1), "max_dd": round(max_dd, 2)})

        # Summary
        console.print(f"\n{'='*70}")
        profitable = sum(1 for r in results if r["pnl"] > 0)
        avg_pnl = total_pnl / max(1, n_samples)
        overall_wr = total_wins / max(1, total_wins + total_losses) * 100

        console.print(Panel.fit(f"[bold]Validation Summary - {n_samples} Periods[/bold]", border_style="magenta"))

        st = Table(show_header=False, box=None)
        st.add_column("", style="bold", width=28)
        st.add_column("", width=25)
        st.add_row("Profitable Periods", f"{profitable}/{n_samples} ({profitable/max(1,n_samples)*100:.0f}%)")
        st.add_row("Total Trades", str(total_trades))
        st.add_row("Overall Win Rate", f"{overall_wr:.1f}%")
        pc = "green" if total_pnl >= 0 else "red"
        st.add_row("Total P&L", f"[{pc}]${total_pnl:+,.2f}[/{pc}]")
        st.add_row("Avg P&L per 5-day period", f"[{pc}]${avg_pnl:+,.2f} ({avg_pnl/self.account_size*100:+.2f}%)[/{pc}]")
        mp = avg_pnl * (22 / 5)
        mc = "green" if mp >= 0 else "red"
        st.add_row("Projected Monthly P&L", f"[{mc}]${mp:+,.0f} ({mp/self.account_size*100:+.1f}%)[/{mc}]")
        st.add_row("", "")
        pace = mp / self.account_size * 100 >= 10
        st.add_row("10% Challenge Pace?", f"[{'green bold' if pace else 'red'}]{'YES' if pace else 'NO'}[/{'green bold' if pace else 'red'}]")
        console.print(st)

        return {"samples": n_samples, "profitable": profitable, "total_trades": total_trades,
                "win_rate": round(overall_wr, 1), "total_pnl": round(total_pnl, 2),
                "avg_pnl": round(avg_pnl, 2), "monthly_projected": round(mp, 2), "results": results}

    def _build_opening_range(self, df: pd.DataFrame, symbol: str, current_ts) -> None:
        """Build opening ranges for London and NY sessions.

        Opening Range = High/Low of first 30 minutes of each session.
        London: 07:00-07:30 UTC → trade breakouts 07:30-11:00
        NY: 13:00-13:30 UTC → trade breakouts 13:30-16:00
        """
        ts_str = str(current_ts)
        day = ts_str[:10]
        hour = current_ts.hour if hasattr(current_ts, 'hour') else int(ts_str[11:13])
        minute = current_ts.minute if hasattr(current_ts, 'minute') else int(ts_str[14:16])

        # London opening range: collect candles from 07:00-07:30
        london_key = f"{symbol}_{day}_london"
        if hour == 7 and minute < 30:
            if london_key not in self._opening_ranges:
                self._opening_ranges[london_key] = {"highs": [], "lows": [], "ready": False}
            h = float(df.iloc[-1]["high"])
            l = float(df.iloc[-1]["low"])
            if not math.isnan(h) and not math.isnan(l):
                self._opening_ranges[london_key]["highs"].append(h)
                self._opening_ranges[london_key]["lows"].append(l)
        elif hour == 7 and minute >= 30 and london_key in self._opening_ranges:
            r = self._opening_ranges[london_key]
            if not r["ready"] and r["highs"] and r["lows"]:
                r["range_high"] = max(r["highs"])
                r["range_low"] = min(r["lows"])
                r["range_width"] = r["range_high"] - r["range_low"]
                r["ready"] = True
                r["traded"] = False

        # NY opening range: collect candles from 13:00-13:30
        ny_key = f"{symbol}_{day}_ny"
        if hour == 13 and minute < 30:
            if ny_key not in self._opening_ranges:
                self._opening_ranges[ny_key] = {"highs": [], "lows": [], "ready": False}
            h = float(df.iloc[-1]["high"])
            l = float(df.iloc[-1]["low"])
            if not math.isnan(h) and not math.isnan(l):
                self._opening_ranges[ny_key]["highs"].append(h)
                self._opening_ranges[ny_key]["lows"].append(l)
        elif hour == 13 and minute >= 30 and ny_key in self._opening_ranges:
            r = self._opening_ranges[ny_key]
            if not r["ready"] and r["highs"] and r["lows"]:
                r["range_high"] = max(r["highs"])
                r["range_low"] = min(r["lows"])
                r["range_width"] = r["range_high"] - r["range_low"]
                r["ready"] = True
                r["traded"] = False

    def _scalp_signal(self, df: pd.DataFrame, symbol: str, current_ts=None) -> dict | None:
        """v3: Opening Range Breakout (ORB) + Trend Filter.

        Proven institutional strategy:
        - Define range from first 30min of London/NY session
        - Trade breakouts in the direction of EMA50 trend
        - SL: opposite side of range (structural)
        - TP: 1.5x range width
        - Max 1 trade per session per symbol (no overtrading)
        - ADX filter: only trade when market is moving (ADX > 18)
        - Candle momentum filter: body > 40% of range (no dojis)
        """
        try:
            close = df["close"].values.astype(float)
            high = df["high"].values.astype(float)
            low = df["low"].values.astype(float)

            n = len(close)
            if n < 50:
                return None

            latest = float(close[-1])
            if math.isnan(latest) or latest <= 0:
                return None

            if current_ts is None:
                return None
            ts_str = str(current_ts)
            day = ts_str[:10]
            hour = current_ts.hour if hasattr(current_ts, 'hour') else int(ts_str[11:13])
            minute = current_ts.minute if hasattr(current_ts, 'minute') else int(ts_str[14:16])

            # Build opening ranges
            self._build_opening_range(df, symbol, current_ts)

            # Determine active session range
            active_range = None
            if 7 <= hour <= 10 and (hour > 7 or minute >= 30):
                key = f"{symbol}_{day}_london"
                if key in self._opening_ranges:
                    r = self._opening_ranges[key]
                    if r.get("ready") and not r.get("traded"):
                        active_range = r
            elif 13 <= hour <= 15 and (hour > 13 or minute >= 30):
                key = f"{symbol}_{day}_ny"
                if key in self._opening_ranges:
                    r = self._opening_ranges[key]
                    if r.get("ready") and not r.get("traded"):
                        active_range = r

            if active_range is None:
                return None

            range_high = active_range["range_high"]
            range_low = active_range["range_low"]
            range_width = active_range["range_width"]

            # ATR filter
            atr = talib.ATR(high, low, close, timeperiod=14)
            atr_val = float(atr[-1]) if not math.isnan(atr[-1]) else latest * 0.001
            if atr_val <= 0:
                return None

            if range_width < atr_val * 0.3 or range_width > atr_val * 4:
                return None

            # Trend filter
            ema50 = talib.EMA(close, timeperiod=50)
            ema50_val = float(ema50[-1]) if not math.isnan(ema50[-1]) else latest
            trend_up = latest > ema50_val
            trend_down = latest < ema50_val

            # ADX filter
            adx = talib.ADX(high, low, close, timeperiod=14)
            adx_val = float(adx[-1]) if not math.isnan(adx[-1]) else 15
            if adx_val < 18:
                return None

            # Candle momentum filter
            candle_body = abs(close[-1] - (close[-2] if n >= 2 else close[-1]))
            candle_range = high[-1] - low[-1]
            if candle_range <= 0:
                return None
            if candle_body / candle_range < 0.4:
                return None

            # RSI momentum confirmation
            rsi = talib.RSI(close, timeperiod=14)
            rsi_val = float(rsi[-1]) if not math.isnan(rsi[-1]) else 50

            # Breakout detection
            action = None
            # Buy: breakout above range + uptrend + RSI confirms momentum (not overbought)
            if close[-1] > range_high + range_width * 0.05 and trend_up and 45 < rsi_val < 75:
                action = "buy"
            # Sell: breakout below range + downtrend + RSI confirms weakness (not oversold)
            elif close[-1] < range_low - range_width * 0.05 and trend_down and 25 < rsi_val < 55:
                action = "sell"

            if action is None:
                return None

            active_range["traded"] = True

            # SL/TP
            if action == "buy":
                sl = range_low - atr_val * 0.3
                tp = latest + range_width * 1.5
                risk = latest - sl
                reward = tp - latest
            else:
                sl = range_high + atr_val * 0.3
                tp = latest - range_width * 1.5
                risk = sl - latest
                reward = latest - tp

            if risk <= 0 or reward / risk < 1.2:
                active_range["traded"] = False
                return None

            return {
                "symbol": symbol, "action": action, "price": latest,
                "sl": round(sl, 6), "tp": round(tp, 6), "atr": atr_val,
                "range_high": range_high, "range_low": range_low,
                "range_width": range_width,
            }
        except Exception:
            return None

    def _report(self, closed, open_pos, max_dd):
        # Cap max_dd to realistic levels (equity calc can overshoot with forex margin)
        max_dd = min(max_dd, 100.0)
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

        # Calculate actual trading days from trade dates
        trade_dates = set()
        for t in closed:
            if t.entry_time:
                trade_dates.add(t.entry_time[:10])
        trading_days = max(len(trade_dates), 1)

        # Extrapolate to 22 trading days (1 month)
        if trading_days < 22:
            monthly_pnl = total_pnl * (22 / trading_days)
        else:
            monthly_pnl = total_pnl
        monthly_pct = monthly_pnl / initial * 100

        passed = monthly_pct >= 10.0

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
