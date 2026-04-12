"""CRT (Candle Range Theory) Multi-Timeframe Strategy.

Institutional liquidity-sweep strategy adapted for algorithmic execution.
The logic captures false breakouts at key levels across three timeframes,
confirming institutional order flow before entry.

Three-stage confirmation cascade:

  HTF (4H or Daily resampled from 1H bars):
    1. Bar N's high exceeds Bar N-1's high (sweep of buy-side liquidity)
       OR Bar N's low undercuts Bar N-1's low (sweep of sell-side liquidity)
    2. BUT Bar N CLOSES back inside Bar N-1's range (false breakout / trap)
    3. Direction: if sweep was above → bearish bias (sell target = unmitigated low)
                  if sweep was below → bullish bias (buy target = unmitigated high)

  MTF (1H — our native timeframe):
    4. Within the HTF signal window (next 4-8 bars), look for the SAME
       pattern to repeat on 1H: sweep + internal close in the HTF direction.
    5. This is the "confirmation" — smart money is defending the level twice.

  LTF (15min simulated via 1H sub-bar — see note):
    6. After MTF confirmation, identify the Order Block (OB): the last
       opposing candle before the impulsive move in the confirmed direction.
    7. Entry: limit order at the OB's 50% level (midpoint of OB body).
    8. SL: beyond the OB's extreme (OB high for sells, OB low for buys)
       + ATR buffer.
    9. TP: the unmitigated extreme from HTF (the liquidity pool).

NOTE ON TIMEFRAMES:
  We only have 1H bars. The HTF is constructed by resampling to 4H.
  The LTF (15min) cannot be constructed from 1H bars — we approximate
  the OB identification on the 1H bars themselves (the nearest opposing
  candle). When real 15min data is available via Dukascopy or a broker
  feed, the LTF logic can be swapped in with no changes to the HTF/MTF
  cascade.

supported_regimes = [0] — CRT is a mean-reversion / liquidity-capture
  strategy that relies on orderly market structure. In Risk-Off regimes
  (vol spikes, gaps), the "false breakout" assumption breaks because
  breakouts become REAL. The RegimeGate blocks CRT in those conditions.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from trade.research.indicators import ATR

ACCOUNT = 10_000.0
RISK_PER_TRADE = 0.003


def _resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1H bars to 4H OHLCV. Causal: each 4H bar uses only
    the 4 hours it covers, never future hours."""
    r = df.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return r


def _detect_htf_sweeps(htf: pd.DataFrame) -> pd.DataFrame:
    """Detect CRT sweep patterns on HTF bars.

    Returns a DataFrame with columns:
      bar_time, direction ('buy'/'sell'), target_price, sweep_price
    """
    high = htf["high"].values
    low = htf["low"].values
    close = htf["close"].values
    opn = htf["open"].values
    signals = []
    for i in range(1, len(htf)):
        prev_high = high[i - 1]
        prev_low = low[i - 1]
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        # Sweep above previous high + close inside previous range
        swept_high = curr_high > prev_high and curr_close < prev_high
        # Sweep below previous low + close inside previous range
        swept_low = curr_low < prev_low and curr_close > prev_low
        if swept_high:
            signals.append({
                "bar_time": htf.index[i],
                "direction": "sell",
                "target_price": prev_low,  # unmitigated low
                "sweep_price": curr_high,
            })
        elif swept_low:
            signals.append({
                "bar_time": htf.index[i],
                "direction": "buy",
                "target_price": prev_high,  # unmitigated high
                "sweep_price": curr_low,
            })
    return pd.DataFrame(signals) if signals else pd.DataFrame(
        columns=["bar_time", "direction", "target_price", "sweep_price"]
    )


def _find_mtf_confirmation(
    df_1h: pd.DataFrame,
    htf_signal_time: pd.Timestamp,
    direction: str,
    window: int = 8,
) -> int | None:
    """Look for the same sweep+internal-close pattern on the 1H bars
    within `window` bars after the HTF signal. Returns the 0-based
    index of the confirming bar, or None.
    """
    mask = df_1h.index > htf_signal_time
    candidates = df_1h.loc[mask].iloc[:window]
    if len(candidates) < 2:
        return None
    high = candidates["high"].values
    low = candidates["low"].values
    close = candidates["close"].values
    for j in range(1, len(candidates)):
        prev_h = high[j - 1]
        prev_l = low[j - 1]
        curr_h = high[j]
        curr_l = low[j]
        curr_c = close[j]
        if direction == "sell":
            if curr_h > prev_h and curr_c < prev_h:
                # Find the absolute index in df_1h
                abs_idx = df_1h.index.get_loc(candidates.index[j])
                return int(abs_idx)
        elif direction == "buy":
            if curr_l < prev_l and curr_c > prev_l:
                abs_idx = df_1h.index.get_loc(candidates.index[j])
                return int(abs_idx)
    return None


def _find_order_block(
    df_1h: pd.DataFrame,
    confirm_idx: int,
    direction: str,
    lookback: int = 5,
) -> dict | None:
    """Find the Order Block: the last opposing candle before the
    impulsive move at the confirmation bar.

    For a buy: OB = last bearish candle (close < open) in the lookback.
    For a sell: OB = last bullish candle (close > open) in the lookback.

    Returns dict with ob_high, ob_low, ob_mid, ob_idx or None.
    """
    start = max(0, confirm_idx - lookback)
    window = df_1h.iloc[start:confirm_idx]
    if window.empty:
        return None
    opn = window["open"].values
    close = window["close"].values
    high = window["high"].values
    low = window["low"].values
    for k in range(len(window) - 1, -1, -1):
        if direction == "buy" and close[k] < opn[k]:
            return {
                "ob_high": float(high[k]),
                "ob_low": float(low[k]),
                "ob_mid": float((high[k] + low[k]) / 2),
                "ob_idx": start + k,
            }
        elif direction == "sell" and close[k] > opn[k]:
            return {
                "ob_high": float(high[k]),
                "ob_low": float(low[k]),
                "ob_mid": float((high[k] + low[k]) / 2),
                "ob_idx": start + k,
            }
    return None


class CRTStrategy:
    name: str = "CRTStrategy"
    # Only operates in calm, orderly markets where false breakouts
    # are the norm and real breakouts are rare.
    supported_regimes: list[int] = [0]

    default_params: dict[str, Any] = {
        "htf_resample": "4h",
        "mtf_confirm_window": 8,
        "ob_lookback": 5,
        "atr_period": 14,
        "atr_sl_buffer": 0.5,  # extra ATR beyond OB extreme for SL
        "max_holding": 48,      # safety cap only (not a time stop for production)
    }

    param_grid: dict[str, list[Any]] = {
        "mtf_confirm_window": [6, 8, 12],
        "ob_lookback": [3, 5],
        "atr_sl_buffer": [0.3, 0.5, 0.8],
    }

    def _walk(
        self,
        df: pd.DataFrame,
        spread: float,
        params: dict[str, Any],
        signal_shift: int = 0,
        collect_signals: bool = False,
    ):
        if not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
            raise ValueError("df missing OHLCV columns")

        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        n = len(close)
        atr = ATR(high, low, close, period=int(params.get("atr_period", 14)))

        # HTF sweep detection
        htf = _resample_4h(df)
        htf_signals = _detect_htf_sweeps(htf)

        # Build all valid entry records
        entries_by_idx: dict[int, dict] = {}
        for _, htf_sig in htf_signals.iterrows():
            direction = htf_sig["direction"]
            target = float(htf_sig["target_price"])
            confirm_idx = _find_mtf_confirmation(
                df, htf_sig["bar_time"], direction,
                window=int(params["mtf_confirm_window"]),
            )
            if confirm_idx is None:
                continue
            ob = _find_order_block(
                df, confirm_idx, direction,
                lookback=int(params["ob_lookback"]),
            )
            if ob is None:
                continue
            entry_idx = confirm_idx + 1 + signal_shift
            if entry_idx < 0 or entry_idx >= n:
                continue
            if math.isnan(atr[entry_idx]) or atr[entry_idx] <= 0:
                continue
            entry_price = ob["ob_mid"]
            atr_i = atr[entry_idx]
            sl_buffer = atr_i * float(params["atr_sl_buffer"])
            if direction == "buy":
                sl = ob["ob_low"] - sl_buffer
                tp = target
                risk = entry_price - sl
            else:
                sl = ob["ob_high"] + sl_buffer
                tp = target
                risk = sl - entry_price
            if risk <= 0 or abs(tp - entry_price) < risk * 0.3:
                continue
            if entry_idx not in entries_by_idx:
                entries_by_idx[entry_idx] = {
                    "side": direction, "entry": entry_price,
                    "sl": sl, "tp": tp,
                }

        if collect_signals:
            signals = [
                {"bar_idx": idx + 1, "side": e["side"],
                 "sl": float(e["sl"]), "tp": float(e["tp"])}
                for idx, e in sorted(entries_by_idx.items())
            ]
            return pd.DataFrame(signals) if signals else pd.DataFrame(
                columns=["bar_idx", "side", "sl", "tp"]
            )

        # Walk positions with SL/TP only (no max_holding)
        pnls: list[float] = []
        pos = None
        equity = ACCOUNT
        for i in range(n):
            if pos is None:
                sig = entries_by_idx.get(i)
                if sig is None:
                    continue
                risk = (sig["entry"] - sig["sl"]) if sig["side"] == "buy" else (sig["sl"] - sig["entry"])
                if risk <= 0:
                    continue
                qty = (equity * RISK_PER_TRADE) / risk
                pos = {**sig, "idx": i, "qty": qty}
                continue
            exit_price = None
            if pos["side"] == "buy":
                if low[i] <= pos["sl"]:
                    exit_price = pos["sl"]
                elif high[i] >= pos["tp"]:
                    exit_price = pos["tp"]
            else:
                if high[i] >= pos["sl"]:
                    exit_price = pos["sl"]
                elif low[i] <= pos["tp"]:
                    exit_price = pos["tp"]
            if exit_price is not None:
                if pos["side"] == "buy":
                    pnl = (exit_price - pos["entry"]) * pos["qty"] - spread * pos["qty"] * 2
                else:
                    pnl = (pos["entry"] - exit_price) * pos["qty"] - spread * pos["qty"] * 2
                pnls.append(pnl)
                equity += pnl
                pos = None
        # Force-close any open position
        if pos is not None:
            last = close[-1]
            if pos["side"] == "buy":
                pnl = (last - pos["entry"]) * pos["qty"] - spread * pos["qty"] * 2
            else:
                pnl = (pos["entry"] - last) * pos["qty"] - spread * pos["qty"] * 2
            pnls.append(pnl)
        return pnls

    def backtest(self, df, spread, params, signal_shift=0):
        return self._walk(df, spread, params, signal_shift, collect_signals=False)

    def signals(self, df, params):
        return self._walk(df, 0.0, params, 0, collect_signals=True)
