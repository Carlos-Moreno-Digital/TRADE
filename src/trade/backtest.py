"""Backtester v3 - Professional-grade strategy with regime detection, SMC, correlations.

Implements 6 phases of improvements:
1. Higher entry thresholds (quality over quantity)
2. Regime detection (trending vs ranging - different rules)
3. Smart Money Concepts (order blocks, FVG, market structure)
4. Correlation filtering (no duplicate risk)
5. Dynamic position sizing (compound winners)
6. Day-of-week bias (avoid bad days)
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

# Correlation pairs that should NOT be traded simultaneously
CORRELATED_PAIRS = {
    frozenset(["EURUSD=X", "GBPUSD=X"]): 0.85,
    frozenset(["EURUSD=X", "USDCHF=X"]): 0.95,
    frozenset(["AUDUSD=X", "NZDUSD=X"]): 0.90,
    frozenset(["EURJPY=X", "GBPJPY=X"]): 0.80,
    frozenset(["AUDJPY=X", "NZDJPY=X"]): 0.85,
    frozenset(["EURCAD=X", "GBPCAD=X"]): 0.80,
    frozenset(["EURAUD=X", "GBPAUD=X"]): 0.82,
    frozenset(["GC=F", "SI=F"]): 0.75,
}


class Position:
    def __init__(self, symbol: str, side: str, entry_price: float, quantity: float,
                 stop_loss: float, take_profit: float, entry_date: str):
        self.symbol = symbol
        self.side = side
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
    def __init__(self, prop_firm: str = "funderpro_classic_10k", top_n: int = 5):
        from trade.risk.prop_firm import load_prop_firm_config
        prop_config = load_prop_firm_config(prop_firm)
        self.account_size = prop_config.account_size
        self.base_risk = 0.01        # 1% base risk per trade
        self.max_trades = 3          # Max 3 concurrent (less = more focused)
        self.min_rr = 1.5
        self.top_n = top_n
        self.provider = MarketDataProvider()

    def run(self, days: int = 30, symbols: list[str] | None = None) -> dict[str, Any]:
        if symbols is None:
            symbols = self._get_default_symbols()

        console.print(Panel.fit(
            f"[bold cyan]Backtester v3 - {days} Day Pro Simulation[/bold cyan]\n"
            f"Account: ${self.account_size:,.0f} | Base risk: {self.base_risk*100:.1f}% | "
            f"R:R min: {self.min_rr} | Max positions: {self.max_trades}\n"
            f"Features: Regime detection, SMC, Correlations, Compounding, Day bias",
            title="Backtest",
            border_style="cyan",
        ))

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

        ref_df = next(iter(all_data.values()))
        all_dates = sorted(ref_df.index.tolist())
        warmup = 50
        sim_dates = all_dates[warmup:][-days:]

        # State
        cash = self.account_size
        equity = self.account_size
        peak = self.account_size
        open_positions: list[Position] = []
        closed_trades: list[Position] = []
        daily_equity: list[dict] = []
        max_dd = 0.0
        consecutive_wins = 0

        console.print(f"Simulating {len(sim_dates)} trading days...\n")

        for i, date in enumerate(sim_dates):
            date_str = date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date)[:10]

            # === PHASE 6: DAY OF WEEK BIAS ===
            try:
                day_of_week = datetime.strptime(date_str, "%Y-%m-%d").weekday()
            except Exception:
                day_of_week = 2  # Default to Wednesday

            skip_day = day_of_week == 0  # Monday = avoid (range forming)
            friday = day_of_week == 4    # Friday = smaller size

            # 1. CHECK SL/TP + TRAILING STOP
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

                risk_dist = abs(pos.entry_price - pos.stop_loss)

                # Trailing stop
                if pos.side == "long":
                    if close_price >= pos.entry_price + risk_dist * 1.5 and pos.stop_loss < pos.entry_price:
                        pos.stop_loss = pos.entry_price + risk_dist * 0.1
                    if close_price >= pos.entry_price + risk_dist * 2.0 and pos.stop_loss < pos.entry_price + risk_dist * 0.5:
                        pos.stop_loss = pos.entry_price + risk_dist * 0.5
                elif pos.side == "short":
                    if close_price <= pos.entry_price - risk_dist * 1.5 and pos.stop_loss > pos.entry_price:
                        pos.stop_loss = pos.entry_price - risk_dist * 0.1
                    if close_price <= pos.entry_price - risk_dist * 2.0 and pos.stop_loss > pos.entry_price - risk_dist * 0.5:
                        pos.stop_loss = pos.entry_price - risk_dist * 0.5

                # SL/TP check
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

                # Time exit: only close LOSERS after 15 days, let winners run
                if pos not in positions_to_close:
                    days_held = sum(1 for d in sim_dates[:i+1] if str(d)[:10] >= pos.entry_date)
                    if pos.side == "long":
                        current_pnl = (close_price - pos.entry_price) * pos.quantity
                    else:
                        current_pnl = (pos.entry_price - close_price) * pos.quantity

                    if days_held >= 15 and current_pnl <= 0:
                        pos.exit_price = close_price
                        pos.pnl = current_pnl
                        pos.exit_reason = "TIME"
                        positions_to_close.append(pos)

            for pos in positions_to_close:
                pos.exit_date = date_str
                cash += pos.pnl + (pos.entry_price * pos.quantity if pos.side == "long" else 0)
                open_positions.remove(pos)
                closed_trades.append(pos)
                # Track consecutive wins for compounding
                if pos.pnl > 0:
                    consecutive_wins += 1
                else:
                    consecutive_wins = 0

            # 2. COOLDOWN + DRAWDOWN PROTECTION
            recent = closed_trades[-3:] if closed_trades else []
            consec_losses = sum(1 for t in reversed(recent) if t.pnl < 0)
            if recent and recent[-1].pnl >= 0:
                consec_losses = 0

            days_since = 0
            if closed_trades:
                last_exit = closed_trades[-1].exit_date or ""
                days_since = sum(1 for d in sim_dates[:i+1] if str(d)[:10] > last_exit)
            cooldown = consec_losses >= 3 and days_since < 2

            dd_from_peak = (peak - equity) / peak * 100 if peak > 0 else 0
            risk_mult = 1.0
            if dd_from_peak > 4.0:
                risk_mult = 0.25  # Quarter size when DD > 4%
            elif dd_from_peak > 3.0:
                risk_mult = 0.5
            elif dd_from_peak > 2.0:
                risk_mult = 0.75

            # === PHASE 5: COMPOUNDING ===
            if consecutive_wins >= 3:
                risk_mult *= 1.5  # 50% more after 3 wins
            elif consecutive_wins >= 2:
                risk_mult *= 1.25  # 25% more after 2 wins

            # Friday = reduce size
            if friday:
                risk_mult *= 0.75

            # 3. FIND NEW TRADES
            if not cooldown and not skip_day and len(open_positions) < self.max_trades:
                scored = []
                open_syms = [p.symbol for p in open_positions]

                for sym, df in all_data.items():
                    if sym in [p.symbol for p in open_positions]:
                        continue

                    # === PHASE 4: CORRELATION CHECK ===
                    correlated = False
                    for open_sym in open_syms:
                        pair = frozenset([sym, open_sym])
                        if pair in CORRELATED_PAIRS and CORRELATED_PAIRS[pair] >= 0.75:
                            correlated = True
                            break
                    if correlated:
                        continue

                    mask = df.index <= date
                    df_slice = df[mask]
                    if len(df_slice) < 50:
                        continue

                    signal = self._analyze_symbol(df_slice, sym, day_of_week)
                    if signal and signal["action"] != "hold":
                        scored.append(signal)

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

                    risk_amount = equity * self.base_risk * risk_mult
                    quantity = risk_amount / risk_per_unit

                    reward = abs(tp - entry_price)
                    rr = reward / risk_per_unit if risk_per_unit > 0 else 0
                    if rr < self.min_rr:
                        continue

                    side = "long" if signal["action"] == "buy" else "short"
                    pos = Position(signal["symbol"], side, entry_price, round(quantity, 4),
                                   sl, tp, date_str)
                    if side == "long":
                        cash -= entry_price * quantity
                    open_positions.append(pos)

            # 4. UPDATE equity
            unrealized = 0
            for pos in open_positions:
                df = all_data.get(pos.symbol)
                if df is None:
                    continue
                mask = df.index <= date
                if mask.sum() == 0:
                    continue
                cp = float(df[mask].iloc[-1]["close"])
                if pos.side == "long":
                    unrealized += (cp - pos.entry_price) * pos.quantity
                else:
                    unrealized += (pos.entry_price - cp) * pos.quantity

            equity = cash + sum(
                pos.entry_price * pos.quantity for pos in open_positions if pos.side == "long"
            ) + unrealized

            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)

            daily_equity.append({
                "date": date_str, "equity": round(equity, 2),
                "cash": round(cash, 2), "positions": len(open_positions),
                "unrealized": round(unrealized, 2),
            })

            pct = (i + 1) / len(sim_dates) * 100
            filled = int(30 * pct / 100)
            bar = "█" * filled + "░" * (30 - filled)
            ts = f"W:{sum(1 for t in closed_trades if t.pnl > 0)} L:{sum(1 for t in closed_trades if t.pnl <= 0)}"
            console.print(f"\r  [{bar}] {pct:.0f}% | {date_str} | ${equity:,.0f} | Open:{len(open_positions)} | {ts}", end="")

        console.print("\n")
        return self._report(closed_trades, open_positions, daily_equity, max_dd, sim_dates)

    def _analyze_symbol(self, df: pd.DataFrame, symbol: str, day_of_week: int = 2) -> dict | None:
        """Professional-grade analysis with regime detection and SMC."""
        try:
            close = df["close"].values.astype(float)
            high = df["high"].values.astype(float)
            low = df["low"].values.astype(float)

            if len(close) < 50:
                return None
            latest = float(close[-1])
            if math.isnan(latest) or latest <= 0:
                return None

            # === INDICATORS ===
            rsi = talib.RSI(close, timeperiod=14)
            macd, macd_signal, macd_hist = talib.MACD(close)
            ema9 = talib.EMA(close, timeperiod=9)
            ema21 = talib.EMA(close, timeperiod=21)
            ema50 = talib.EMA(close, timeperiod=50)
            atr = talib.ATR(high, low, close, timeperiod=14)
            adx = talib.ADX(high, low, close, timeperiod=14)
            stoch_k, stoch_d = talib.STOCH(high, low, close)
            upper, middle, lower_bb = talib.BBANDS(close, timeperiod=20)

            rsi_val = float(rsi[-1]) if not math.isnan(rsi[-1]) else 50
            macd_val = float(macd_hist[-1]) if not math.isnan(macd_hist[-1]) else 0
            macd_prev = float(macd_hist[-2]) if len(macd_hist) > 1 and not math.isnan(macd_hist[-2]) else 0
            ema9_v = float(ema9[-1]) if not math.isnan(ema9[-1]) else latest
            ema21_v = float(ema21[-1]) if not math.isnan(ema21[-1]) else latest
            ema50_v = float(ema50[-1]) if not math.isnan(ema50[-1]) else latest
            atr_val = float(atr[-1]) if not math.isnan(atr[-1]) else latest * 0.02
            adx_val = float(adx[-1]) if not math.isnan(adx[-1]) else 20
            stoch_v = float(stoch_k[-1]) if not math.isnan(stoch_k[-1]) else 50
            bb_up = float(upper[-1]) if not math.isnan(upper[-1]) else latest * 1.02
            bb_lo = float(lower_bb[-1]) if not math.isnan(lower_bb[-1]) else latest * 0.98

            if atr_val <= 0:
                return None

            # === PHASE 2: REGIME DETECTION ===
            atr_avg = float(np.nanmean(atr[-20:])) if len(atr) >= 20 else atr_val

            if adx_val > 25 and atr_val > atr_avg:
                regime = "TRENDING"
            elif adx_val < 20:
                regime = "RANGING"
            else:
                regime = "TRANSITIONING"

            # Don't trade in transitioning markets
            if regime == "TRANSITIONING":
                return None

            # === PHASE 1: ADX FILTER (raised to 22) ===
            if regime == "TRENDING" and adx_val < 22:
                return None

            major_trend = "bullish" if latest > ema50_v else "bearish"

            # === PHASE 3: SMART MONEY CONCEPTS ===
            smc_boost = 0
            try:
                from trade.analysis.smart_money import get_smc_analysis
                smc = get_smc_analysis(df)
                if smc.get("available"):
                    structure = smc.get("market_structure", "unknown")
                    active_obs = smc.get("order_blocks", [])
                    active_fvgs = smc.get("fair_value_gaps", [])

                    # Structure alignment bonus
                    if structure == "bullish" and major_trend == "bullish":
                        smc_boost += 2
                    elif structure == "bearish" and major_trend == "bearish":
                        smc_boost += 2

                    # Order Block confluence
                    for ob in active_obs:
                        ob_low = ob.get("low", 0)
                        ob_high = ob.get("high", 0)
                        if ob_low > 0 and abs(latest - ob_low) / latest < 0.005:
                            if ob.get("type") == "bullish_ob":
                                smc_boost += 3  # Price at bullish OB = strong buy
                        if ob_high > 0 and abs(latest - ob_high) / latest < 0.005:
                            if ob.get("type") == "bearish_ob":
                                smc_boost += 3

                    # FVG as target confirmation
                    for fvg in active_fvgs:
                        if fvg.get("type") == "bullish_fvg" and fvg.get("low", 0) > latest:
                            smc_boost += 1  # Unfilled FVG above = magnet for price
                        elif fvg.get("type") == "bearish_fvg" and fvg.get("high", 0) < latest:
                            smc_boost += 1
            except Exception:
                pass

            # === SCORING (regime-adaptive) ===
            bull_score = 0
            bear_score = 0

            if regime == "TRENDING":
                # TREND FOLLOWING: EMA alignment is king
                if ema9_v > ema21_v > ema50_v:
                    bull_score += 4
                elif ema9_v > ema21_v:
                    bull_score += 1
                if ema9_v < ema21_v < ema50_v:
                    bear_score += 4
                elif ema9_v < ema21_v:
                    bear_score += 1

                # RSI pullbacks in trend
                if major_trend == "bullish" and 35 <= rsi_val <= 50:
                    bull_score += 2
                if major_trend == "bearish" and 50 <= rsi_val <= 65:
                    bear_score += 2

                # MACD crossover = strong in trends
                if macd_prev <= 0 < macd_val:
                    bull_score += 3
                if macd_prev >= 0 > macd_val:
                    bear_score += 3
                elif macd_val > 0:
                    bull_score += 1
                elif macd_val < 0:
                    bear_score += 1

            elif regime == "RANGING":
                # MEAN REVERSION: Bollinger + RSI extremes
                if rsi_val > 75:
                    bear_score += 3
                elif rsi_val > 65:
                    bear_score += 1
                if rsi_val < 25:
                    bull_score += 3
                elif rsi_val < 35:
                    bull_score += 1

                if latest >= bb_up:
                    bear_score += 3  # At upper band = sell
                if latest <= bb_lo:
                    bull_score += 3  # At lower band = buy

                if stoch_v > 80:
                    bear_score += 2
                if stoch_v < 20:
                    bull_score += 2

            # Momentum (both regimes)
            if len(close) >= 5:
                mom5 = (latest - float(close[-5])) / float(close[-5]) * 100
                if mom5 > 0.5:
                    bull_score += 1
                elif mom5 < -0.5:
                    bear_score += 1

            # Add SMC boost
            if major_trend == "bullish":
                bull_score += smc_boost
            else:
                bear_score += smc_boost

            # === PHASE 6: DAY BIAS ===
            day_boost = 0
            if day_of_week in (2, 3):  # Wed/Thu = best trending days
                day_boost = 1
            if day_of_week == 1:  # Tuesday = good for reversals
                if regime == "RANGING":
                    day_boost = 1

            bull_score += day_boost if bull_score > bear_score else 0
            bear_score += day_boost if bear_score > bull_score else 0

            # === DECISION (PHASE 1: higher thresholds) ===
            action = "hold"
            confidence = 0.0
            net_score = bull_score - bear_score

            if regime == "TRENDING":
                # Require strong confluence in trends
                if major_trend == "bullish" and net_score >= 3 and bull_score >= 4:
                    action = "buy"
                    confidence = min(0.95, bull_score * 0.1)
                elif major_trend == "bearish" and net_score <= -3 and bear_score >= 4:
                    action = "sell"
                    confidence = min(0.95, bear_score * 0.1)
            elif regime == "RANGING":
                # Mean reversion needs extreme signals
                if net_score >= 4 and bull_score >= 5:
                    action = "buy"
                    confidence = min(0.85, bull_score * 0.1)
                elif net_score <= -4 and bear_score >= 5:
                    action = "sell"
                    confidence = min(0.85, bear_score * 0.1)

            if action == "hold":
                return None

            # SL/TP based on regime
            if regime == "TRENDING":
                sl_mult = 1.2
                tp_mult = 2.5  # Wider TP in trends (let it run)
            else:
                sl_mult = 1.0  # Tighter in ranges
                tp_mult = 1.8  # Shorter TP in ranges (mean reversion)

            if action == "buy":
                sl = latest - atr_val * sl_mult
                tp = latest + atr_val * tp_mult
            else:
                sl = latest + atr_val * sl_mult
                tp = latest - atr_val * tp_mult

            return {
                "symbol": symbol, "action": action, "price": latest,
                "stop_loss": round(sl, 5), "take_profit": round(tp, 5),
                "confidence": confidence, "atr": atr_val, "regime": regime,
                "bull_score": bull_score, "bear_score": bear_score,
            }
        except Exception:
            return None

    def _report(self, closed, open_pos, daily_eq, max_dd, sim_dates):
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

            for idx, trade in enumerate(closed, 1):
                pc = "green" if trade.pnl > 0 else "red"
                tt.add_row(
                    str(idx), trade.symbol, trade.side.upper(),
                    f"{trade.entry_price:.2f}", f"{trade.exit_price:.2f}" if trade.exit_price else "OPEN",
                    f"[{pc}]${trade.pnl:+,.2f}[/{pc}]", trade.exit_reason or "",
                    f"{trade.entry_date} → {trade.exit_date or ''}",
                )
            console.print(tt)

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
                console.print(f"  {eq['date']} | ${eq['equity']:>10,.2f} [{color}]({diff:+,.0f})[/{color}] | [{color}]{bar}[/{color}]")

        return {
            "days": len(sim_dates), "initial": initial, "final": round(final, 2),
            "pnl": round(total_pnl, 2), "pnl_pct": round(pnl_pct, 2),
            "max_drawdown": round(max_dd, 2), "total_trades": total_trades,
            "wins": len(wins), "losses": len(losses), "win_rate": round(win_rate, 1),
            "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor < 999 else 999,
            "passed": passed,
        }

    def _get_default_symbols(self) -> list[str]:
        return [
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X", "USDCAD=X", "USDCHF=X",
            "EURGBP=X", "EURJPY=X", "GBPJPY=X", "EURNZD=X", "EURAUD=X", "EURCAD=X", "EURCHF=X",
            "GBPAUD=X", "GBPNZD=X", "GBPCAD=X", "GBPCHF=X",
            "AUDJPY=X", "NZDJPY=X", "CADJPY=X", "CHFJPY=X",
            "AUDNZD=X", "AUDCAD=X",
            "GC=F", "SI=F", "CL=F", "PL=F",
            "BTC-USD", "ETH-USD",
        ]
