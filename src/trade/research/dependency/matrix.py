"""Build the NMI N×N matrix over an aligned universe of return series."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from trade.research.dependency.nmi import nmi


def align_returns(
    series_by_name: dict[str, pd.Series],
    how: str = "inner",
    resample: str | None = "1D",
) -> pd.DataFrame:
    """Inner-join all series by timestamp and return a DataFrame of
    aligned log-returns (one column per asset).

    Parameters
    ----------
    series_by_name : dict[str, Series]
        Close-price series keyed by asset name, each with a
        DatetimeIndex.
    how : str, default 'inner'
        Pandas join method after alignment.
    resample : str or None, default '1D'
        If not None, each series is resampled via .last() to the given
        pandas frequency BEFORE joining. This is required for
        cross-asset analysis when the universe mixes FX (24/5) with
        NYSE-hour indices (SPX) whose native intraday timestamps
        never match. Daily is the statistically clean default.
    """
    frames = {}
    for name, s in series_by_name.items():
        if resample:
            s = s.resample(resample).last().dropna()
        frames[name] = s
    df = pd.concat(frames, axis=1, join=how).dropna()
    log_rets = np.log(df / df.shift(1)).dropna()
    return log_rets


def build_nmi_matrix(
    returns: pd.DataFrame,
    k: int = 5,
    symmetric: bool = True,
) -> pd.DataFrame:
    """Pairwise NMI matrix on a DataFrame of return series.

    Diagonal is 1.0 by construction. Off-diagonal is clipped to [0,1].
    """
    names = list(returns.columns)
    n = len(names)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        M[i, i] = 1.0
        for j in range(i + 1, n):
            v = nmi(returns.iloc[:, i].values, returns.iloc[:, j].values, k=k)
            M[i, j] = v
            if symmetric:
                M[j, i] = v
    return pd.DataFrame(M, index=names, columns=names)
