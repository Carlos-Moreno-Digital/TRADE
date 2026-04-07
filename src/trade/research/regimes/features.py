"""Regime features for the Statistical Jump Model.

All features are scale-invariant rolling statistics so the same model
can be applied across symbols without per-symbol re-fitting:

  - realized_vol_5    : 5-bar rolling std of log returns
  - realized_vol_20   : 20-bar rolling std of log returns
  - vol_ratio         : 5/20 vol ratio (regime acceleration)
  - skew_20           : 20-bar rolling skewness of log returns
  - drawdown_60       : peak-to-current drawdown over 60 bars

These match Nystrup et al. 2020 Section 4.1 and similar regime-detection
papers (Bulla & Bulla 2006, Hardy 2001).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(close: pd.Series) -> pd.DataFrame:
    """Compute the regime feature panel from a close-price series.

    Returns a DataFrame indexed like `close` with the features above.
    Drops rows with any NaN at the head so callers can fit straight away.
    """
    log_ret = np.log(close / close.shift(1))

    realized_vol_5 = log_ret.rolling(5).std()
    realized_vol_20 = log_ret.rolling(20).std()
    vol_ratio = realized_vol_5 / realized_vol_20

    skew_20 = log_ret.rolling(20).skew()

    rolling_max = close.rolling(60).max()
    drawdown_60 = (close - rolling_max) / rolling_max

    feats = pd.DataFrame({
        "realized_vol_5": realized_vol_5,
        "realized_vol_20": realized_vol_20,
        "vol_ratio": vol_ratio,
        "skew_20": skew_20,
        "drawdown_60": drawdown_60,
    })
    return feats.dropna()
