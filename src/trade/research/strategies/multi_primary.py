"""Multi-Primary signal generator.

Combines multiple independent primary signal sources into a single
pool that feeds the Meta-Labelling RF. The RF doesn't care WHERE a
signal came from — it scores each one by its features and predicted
PnL. More primaries = larger pool = more surviving trades after the
meta-filter, WITHOUT lowering the quality threshold.

Primary sources:
  1. RegimeMomentum (Donchian breakout + MR z-score dispatch by SJM)
     → the proven primary that passed walk-forward
  2. BollingerReversion — mean-reversion entries when price touches
     the Bollinger lower/upper band AND ADX < 25 (ranging market).
     This generates signals in the OPPOSITE market condition from
     Donchian breakout, so the two pools are complementary.
  3. RSI extremes — entry when RSI crosses below 30 (buy) or above
     70 (sell). Classic momentum-exhaustion signal.

Each primary generates signals independently. The combined pool is
deduplicated by bar_idx (if two primaries fire on the same bar, keep
the first). The meta-RF scores the entire pool and the same threshold
applies.

The strategy itself implements StrategyProtocol so it can be driven
through the gauntlet, walk-forward, and institutional grid search
with no changes to the validation infrastructure.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from trade.research.indicators import ATR, BollingerBands, RSI

ACCOUNT = 10_000.0
RISK_PER_TRADE = 0.003


def _bollinger_signals(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_std: float = 2.0,
    adx_max: int = 25,
    atr_period: int = 14,
    atr_sl_mult: float = 1.5,
    atr_tp_mult: float = 3.0,
) -> pd.DataFrame:
    """Mean-reversion signals from Bollinger Band touches in ranging markets."""
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(close)
    upper, middle, lower = BollingerBands(close, period=bb_period, nbdev=bb_std)
    atr = ATR(high, low, close, period=atr_period)
    # Simple ADX proxy: use ATR/close ratio as a ranging filter
    # (true ADX requires DM computation; this is faster and directionally correct)
    from trade.research.indicators import ADX
    adx = ADX(high, low, close, period=14)

    signals = []
    for i in range(max(bb_period, atr_period) + 1, n):
        if math.isnan(upper[i]) or math.isnan(atr[i]) or math.isnan(adx[i]):
            continue
        if adx[i] >= adx_max:
            continue
        side = None
        if close[i] <= lower[i]:
            side = "buy"
        elif close[i] >= upper[i]:
            side = "sell"
        if side is None:
            continue
        entry = close[i]
        if side == "buy":
            sl = entry - atr[i] * atr_sl_mult
            tp = entry + atr[i] * atr_tp_mult
        else:
            sl = entry + atr[i] * atr_sl_mult
            tp = entry - atr[i] * atr_tp_mult
        risk = abs(entry - sl)
        if risk <= 0:
            continue
        signals.append({
            "bar_idx": i + 1, "side": side,
            "sl": float(sl), "tp": float(tp),
            "source": "bollinger",
        })
    return pd.DataFrame(signals) if signals else pd.DataFrame(
        columns=["bar_idx", "side", "sl", "tp", "source"]
    )


def _rsi_signals(
    df: pd.DataFrame,
    rsi_period: int = 14,
    rsi_buy: float = 30.0,
    rsi_sell: float = 70.0,
    atr_period: int = 14,
    atr_sl_mult: float = 1.5,
    atr_tp_mult: float = 3.0,
) -> pd.DataFrame:
    """Momentum-exhaustion signals from RSI extremes."""
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(close)
    rsi = RSI(close, period=rsi_period)
    atr = ATR(high, low, close, period=atr_period)

    signals = []
    for i in range(max(rsi_period, atr_period) + 1, n):
        if math.isnan(rsi[i]) or math.isnan(atr[i]) or atr[i] <= 0:
            continue
        side = None
        if rsi[i] <= rsi_buy:
            side = "buy"
        elif rsi[i] >= rsi_sell:
            side = "sell"
        if side is None:
            continue
        entry = close[i]
        if side == "buy":
            sl = entry - atr[i] * atr_sl_mult
            tp = entry + atr[i] * atr_tp_mult
        else:
            sl = entry + atr[i] * atr_sl_mult
            tp = entry - atr[i] * atr_tp_mult
        risk = abs(entry - sl)
        if risk <= 0:
            continue
        signals.append({
            "bar_idx": i + 1, "side": side,
            "sl": float(sl), "tp": float(tp),
            "source": "rsi",
        })
    return pd.DataFrame(signals) if signals else pd.DataFrame(
        columns=["bar_idx", "side", "sl", "tp", "source"]
    )


def generate_multi_primary(
    df: pd.DataFrame,
    regime_momentum_signals: pd.DataFrame,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Combine all primary sources into one deduplicated pool.

    RegimeMomentum signals are passed in pre-computed (they require
    the SJM model which is symbol-specific). Bollinger and RSI are
    computed here.
    """
    atr_sl = float(params.get("atr_sl_mult", 1.5))
    atr_tp = float(params.get("atr_tp_mult", 3.0))
    atr_period = int(params.get("atr_period", 14))

    # Source 1: RegimeMomentum (pre-computed)
    rm = regime_momentum_signals.copy()
    if "source" not in rm.columns:
        rm["source"] = "regime_momentum"

    # Source 2: Bollinger reversion
    bb = _bollinger_signals(
        df,
        bb_period=int(params.get("bb_period", 20)),
        bb_std=float(params.get("bb_std", 2.0)),
        adx_max=int(params.get("adx_max", 25)),
        atr_period=atr_period,
        atr_sl_mult=atr_sl,
        atr_tp_mult=atr_tp,
    )

    # Source 3: RSI extremes
    rs = _rsi_signals(
        df,
        rsi_period=int(params.get("rsi_period", 14)),
        rsi_buy=float(params.get("rsi_buy", 30.0)),
        rsi_sell=float(params.get("rsi_sell", 70.0)),
        atr_period=atr_period,
        atr_sl_mult=atr_sl,
        atr_tp_mult=atr_tp,
    )

    # Combine and deduplicate by bar_idx (first signal wins)
    combined = pd.concat([rm, bb, rs], ignore_index=True)
    combined = combined.sort_values("bar_idx")
    combined = combined.drop_duplicates(subset=["bar_idx"], keep="first")
    combined = combined.reset_index(drop=True)
    return combined
