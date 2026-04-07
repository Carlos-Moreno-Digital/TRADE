"""Pure numpy/pandas technical indicators.

Replaces the heavy ta-lib C dependency for the institutional VPS
deployment. Every function returns a numpy array of the same length
as the inputs with NaNs for the warmup region, matching ta-lib's
output convention so the rest of the project can swap imports
without touching call sites.

Implemented:
  - true_range(high, low, close)
  - ATR(high, low, close, period=14)        Wilder smoothing (RMA)
  - BollingerBands(close, period=20, nbdev=2.0)
  - ADX(high, low, close, period=14)        Wilder Average Directional Index
  - EMA(close, period)                       Exponential moving average
  - SMA(close, period)                       Simple moving average
  - RSI(close, period=14)                    Wilder Relative Strength Index

Mathematical references:
  - Wilder, J. Welles (1978). New Concepts in Technical Trading Systems.
  - Bollinger, John (2001). Bollinger on Bollinger Bands.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(high, low, close) -> np.ndarray:
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    n = len(close)
    if n == 0:
        return np.empty(0, dtype=float)
    prev_close = np.empty(n, dtype=float)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    return np.maximum(np.maximum(tr1, tr2), tr3)


def ATR(high, low, close, period: int = 14) -> np.ndarray:
    """Wilder's Average True Range (RMA smoothing).

    Output[i] = NaN for i < period - 1
    Output[period-1] = simple mean of TR[0..period-1]
    Output[i>=period] = (out[i-1] * (period-1) + TR[i]) / period
    """
    tr = true_range(high, low, close)
    n = len(tr)
    out = np.full(n, np.nan, dtype=float)
    if n < period:
        return out
    out[period - 1] = float(np.mean(tr[:period]))
    inv = 1.0 / period
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) * inv
    return out


def BollingerBands(close, period: int = 20, nbdev: float = 2.0):
    """Returns (upper, middle, lower) numpy arrays. ddof=0 to match ta-lib."""
    s = pd.Series(close, dtype=float)
    middle = s.rolling(period).mean()
    std = s.rolling(period).std(ddof=0)
    upper = middle + nbdev * std
    lower = middle - nbdev * std
    return upper.to_numpy(), middle.to_numpy(), lower.to_numpy()


def _wilder_sum(x: np.ndarray, period: int) -> np.ndarray:
    """Wilder's running sum used inside ADX. NaN until period-1."""
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    if n < period:
        return out
    out[period - 1] = float(np.sum(x[:period]))
    for i in range(period, n):
        out[i] = out[i - 1] - (out[i - 1] / period) + x[i]
    return out


def ADX(high, low, close, period: int = 14) -> np.ndarray:
    """Wilder's Average Directional Index.

    Returns array of length len(close) with NaN until 2*period - 2.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    n = len(close)
    if n < 2 * period:
        return np.full(n, np.nan, dtype=float)

    up_move = np.zeros(n, dtype=float)
    down_move = np.zeros(n, dtype=float)
    up_move[1:] = high[1:] - high[:-1]
    down_move[1:] = low[:-1] - low[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)

    smooth_tr = _wilder_sum(tr, period)
    smooth_plus_dm = _wilder_sum(plus_dm, period)
    smooth_minus_dm = _wilder_sum(minus_dm, period)

    with np.errstate(invalid="ignore", divide="ignore"):
        plus_di = 100.0 * smooth_plus_dm / smooth_tr
        minus_di = 100.0 * smooth_minus_dm / smooth_tr
        denom = plus_di + minus_di
        dx = 100.0 * np.abs(plus_di - minus_di) / denom

    adx = np.full(n, np.nan, dtype=float)
    init_idx = 2 * period - 2
    if init_idx >= n:
        return adx
    first_dx_window = dx[period - 1 : 2 * period - 1]
    if not np.isnan(first_dx_window).any():
        adx[init_idx] = float(np.mean(first_dx_window))
        for i in range(init_idx + 1, n):
            if not np.isnan(dx[i]) and not np.isnan(adx[i - 1]):
                adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx


def EMA(close, period: int) -> np.ndarray:
    s = pd.Series(close, dtype=float)
    return s.ewm(span=period, adjust=False).mean().to_numpy()


def SMA(close, period: int) -> np.ndarray:
    s = pd.Series(close, dtype=float)
    return s.rolling(period).mean().to_numpy()


def RSI(close, period: int = 14) -> np.ndarray:
    """Wilder's Relative Strength Index, ta-lib compatible.

    Seeds the first average gain/loss with a simple mean over the
    first `period` deltas, then applies Wilder smoothing recursively.
    Output is NaN until index `period`.
    """
    close = np.asarray(close, dtype=float)
    n = len(close)
    out = np.full(n, np.nan, dtype=float)
    if n <= period:
        return out
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = float(np.mean(gain[:period]))
    avg_loss = float(np.mean(loss[:period]))
    if avg_loss == 0.0:
        out[period] = 100.0 if avg_gain > 0 else 50.0
    else:
        rs = avg_gain / avg_loss
        out[period] = 100.0 - (100.0 / (1.0 + rs))
    inv = 1.0 / period
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gain[i - 1]) * inv
        avg_loss = (avg_loss * (period - 1) + loss[i - 1]) * inv
        if avg_loss == 0.0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out
