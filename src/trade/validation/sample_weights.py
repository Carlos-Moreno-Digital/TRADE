"""Sample weighting utilities — Lopez de Prado AFML ch. 4.

Implements average sample uniqueness for triple-barrier labels. When
labels look forward in time their horizons overlap, which means
adjacent training samples are NOT independent. A naive RF that treats
them as iid will:
  - over-weight regions of the time series that have many overlapping
    labels (because they appear repeatedly in the same bag)
  - learn redundant correlations of the overlap pattern instead of
    genuine economic structure
  - report inflated training accuracy that does NOT translate to OOS

The fix is to compute, for each sample i, a scalar in (0, 1] called
the "average uniqueness" u_i. It captures how MUCH of sample i's
label horizon belongs to samples that aren't sharing it with anyone
else. Then sample_weight = u_i is passed to .fit() so the estimator
focuses on independent observations.

Reference:
  Lopez de Prado, M. (2018). Advances in Financial Machine Learning,
  ch. 4 ("Sample Weights"), Algorithm 4.1 (concurrent-labels indicator)
  and Algorithm 4.2 (average uniqueness).

Implementation note: pure numpy, O(N * H) where H is the average
label horizon. For our institutional run with N=6646 signals and
H=100 bars this is ~660K operations — under 100ms. No mlfinlab
dependency.
"""
from __future__ import annotations

import numpy as np


def concurrency(
    signal_starts: np.ndarray,
    signal_ends: np.ndarray,
    n_bars: int | None = None,
) -> np.ndarray:
    """Number of concurrent labels at every bar (Algorithm 4.1).

    Returns an array c of shape (n_bars,) where c[t] is the count of
    signals whose label horizon [start_i, end_i] contains bar t.
    """
    if len(signal_starts) != len(signal_ends):
        raise ValueError("starts and ends must have the same length")
    if len(signal_starts) == 0:
        n = n_bars or 0
        return np.zeros(n, dtype=float)

    if n_bars is None:
        n_bars = int(signal_ends.max()) + 1

    c = np.zeros(n_bars, dtype=float)
    for s, e in zip(signal_starts, signal_ends):
        s_i = int(max(0, s))
        e_i = int(min(n_bars - 1, e))
        if e_i < s_i:
            continue
        c[s_i:e_i + 1] += 1.0
    return c


def average_uniqueness(
    signal_starts: np.ndarray,
    signal_ends: np.ndarray,
    n_bars: int | None = None,
) -> np.ndarray:
    """Average uniqueness u_i per signal (Algorithm 4.2).

    For signal i:
        u_i = mean over t in [start_i, end_i] of  1 / c[t]

    Returns u of shape (N,) with values in (0, 1]. u_i = 1 means the
    signal's horizon never overlaps with any other; u_i close to 0
    means the horizon is fully shared with many concurrent labels.
    """
    if len(signal_starts) != len(signal_ends):
        raise ValueError("starts and ends must have the same length")
    n = len(signal_starts)
    if n == 0:
        return np.array([], dtype=float)

    if n_bars is None:
        n_bars = int(signal_ends.max()) + 1

    c = concurrency(signal_starts, signal_ends, n_bars=n_bars)
    inv_c = np.where(c > 0, 1.0 / np.maximum(c, 1.0), 0.0)

    out = np.zeros(n, dtype=float)
    for i in range(n):
        s_i = int(max(0, signal_starts[i]))
        e_i = int(min(n_bars - 1, signal_ends[i]))
        if e_i < s_i:
            out[i] = 0.0
            continue
        window = inv_c[s_i:e_i + 1]
        out[i] = float(window.mean()) if window.size > 0 else 0.0
    return out


def normalize_to_sum_n(weights: np.ndarray) -> np.ndarray:
    """Rescale a weight array so it sums to len(weights).

    sklearn estimators tolerate any positive scale, but matching
    sum(weights) == n keeps the loss magnitudes comparable to the
    unweighted case (useful for early-stopping thresholds and
    log diagnostics).
    """
    n = len(weights)
    s = float(weights.sum())
    if s <= 0:
        return np.ones(n, dtype=float)
    return weights * (n / s)
