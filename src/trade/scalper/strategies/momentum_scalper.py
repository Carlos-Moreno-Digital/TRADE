"""Momentum Scalper — catches micro-impulses on 1-minute bars.

Signal logic (all computed on 1-minute OHLCV):

  1. VWAP deviation: price vs the session VWAP. When price is
     significantly above/below VWAP, momentum is stretched.

  2. EMA crossover (fast=5, slow=20): direction filter. Only
     take BUY if fast > slow, SELL if fast < slow.

  3. RSI filter (period=7): avoid entering overbought/oversold
     against the trend. BUY needs RSI < 70, SELL needs RSI > 30.

  4. Momentum burst: the last 3 candles must show consistent
     directional movement (2/3 candles in the signal direction).

  5. ATR-based SL/TP:
     SL = 1.0 × ATR(14) on 1min bars (~5-8 pips on EURUSD)
     TP = 1.2 × ATR(14) (~6-10 pips) → slight R:R > 1.0

Entry:
  All 4 conditions (VWAP, EMA, RSI, momentum) must align.
  Enter at market price with SL and TP pre-set.

Exit:
  TP or SL only. No time stop (the risk manager handles the
  session boundary). If the daily kill switch fires, the risk
  manager closes all positions — the strategy doesn't need to
  know about that.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from trade.scalper.connectors.base import Bar, Side


@dataclass
class ScalpSignal:
    side: Side
    entry_price: float
    sl: float
    tp: float
    strength: float  # 0-1 confidence proxy
    reason: str


@dataclass
class MomentumConfig:
    ema_fast: int = 5
    ema_slow: int = 20
    rsi_period: int = 7
    atr_period: int = 14
    atr_sl_mult: float = 1.0
    atr_tp_mult: float = 1.2
    vwap_threshold: float = 0.0002  # min distance from VWAP (as % of price)
    min_bars_for_signal: int = 30
    momentum_lookback: int = 3     # candles to check for directional consistency
    momentum_min_ratio: float = 0.66  # 2/3 candles must agree


def _ema(values: list[float], period: int) -> float:
    """Compute EMA of the last `period` values."""
    if len(values) < period:
        return values[-1] if values else 0.0
    k = 2.0 / (period + 1)
    ema = values[-period]
    for v in values[-period + 1:]:
        ema = v * k + ema * (1 - k)
    return ema


def _rsi(closes: list[float], period: int) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(-period, 0)]
    gains = [d for d in deltas if d > 0]
    losses = [-d for d in deltas if d < 0]
    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.001
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(bars: list[Bar], period: int) -> float:
    if len(bars) < period + 1:
        return 0.0
    trs = []
    for i in range(-period, 0):
        h = bars[i].high
        l = bars[i].low
        pc = bars[i - 1].close
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    return sum(trs) / len(trs)


def _vwap(bars: list[Bar]) -> float:
    """Session VWAP from all bars provided (assumed to be today's session)."""
    total_pv = 0.0
    total_vol = 0.0
    for b in bars:
        typical = (b.high + b.low + b.close) / 3.0
        vol = max(b.volume, 1.0)
        total_pv += typical * vol
        total_vol += vol
    return total_pv / total_vol if total_vol > 0 else bars[-1].close


class MomentumScalper:
    """Generates scalp signals from 1-minute bars."""

    def __init__(self, config: MomentumConfig | None = None):
        self.cfg = config or MomentumConfig()

    def evaluate(self, bars: list[Bar]) -> ScalpSignal | None:
        """Evaluate the current bar for a scalp signal.

        Parameters
        ----------
        bars : list[Bar]
            Recent 1-minute bars. Must be at least min_bars_for_signal
            in length. The LAST bar is the current bar (just closed).

        Returns
        -------
        ScalpSignal or None
        """
        if len(bars) < self.cfg.min_bars_for_signal:
            return None

        closes = [b.close for b in bars]
        current = closes[-1]

        # 1. EMA crossover direction
        ema_fast = _ema(closes, self.cfg.ema_fast)
        ema_slow = _ema(closes, self.cfg.ema_slow)
        bullish_ema = ema_fast > ema_slow
        bearish_ema = ema_fast < ema_slow
        if not (bullish_ema or bearish_ema):
            return None

        # 2. RSI filter
        rsi = _rsi(closes, self.cfg.rsi_period)
        if bullish_ema and rsi >= 70:
            return None  # overbought in bullish trend
        if bearish_ema and rsi <= 30:
            return None  # oversold in bearish trend

        # 3. VWAP deviation
        vwap = _vwap(bars)
        vwap_dev = (current - vwap) / vwap if vwap > 0 else 0.0
        # BUY: price above VWAP (momentum up)
        # SELL: price below VWAP (momentum down)
        bullish_vwap = vwap_dev > self.cfg.vwap_threshold
        bearish_vwap = vwap_dev < -self.cfg.vwap_threshold

        # 4. Momentum burst (directional consistency)
        lb = self.cfg.momentum_lookback
        recent = closes[-lb:]
        up_count = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i - 1])
        down_count = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i - 1])
        bullish_momentum = up_count / max(lb - 1, 1) >= self.cfg.momentum_min_ratio
        bearish_momentum = down_count / max(lb - 1, 1) >= self.cfg.momentum_min_ratio

        # 5. Combine all conditions
        buy_signal = bullish_ema and bullish_vwap and bullish_momentum
        sell_signal = bearish_ema and bearish_vwap and bearish_momentum

        if not (buy_signal or sell_signal):
            return None

        # 6. ATR-based SL/TP
        atr = _atr(bars, self.cfg.atr_period)
        if atr <= 0:
            return None

        side = Side.BUY if buy_signal else Side.SELL
        if side == Side.BUY:
            sl = current - atr * self.cfg.atr_sl_mult
            tp = current + atr * self.cfg.atr_tp_mult
        else:
            sl = current + atr * self.cfg.atr_sl_mult
            tp = current - atr * self.cfg.atr_tp_mult

        # Strength proxy: how many confirmations are strong
        strength = 0.0
        strength += 0.25 * min(abs(ema_fast - ema_slow) / max(atr, 1e-8), 1.0)
        strength += 0.25 * (abs(vwap_dev) / max(self.cfg.vwap_threshold * 3, 1e-8))
        strength += 0.25 * (abs(rsi - 50) / 50)
        strength += 0.25 * (max(up_count, down_count) / max(lb - 1, 1))
        strength = min(strength, 1.0)

        return ScalpSignal(
            side=side,
            entry_price=current,
            sl=sl,
            tp=tp,
            strength=round(strength, 3),
            reason=f"EMA({'bull' if bullish_ema else 'bear'}) "
                   f"VWAP({vwap_dev:+.4f}) "
                   f"RSI({rsi:.0f}) "
                   f"Mom({up_count}/{lb-1})",
        )
