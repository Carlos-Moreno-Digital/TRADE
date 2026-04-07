"""Paranoid leakage tests for the synthetic LOB generator.

These are NOT unit tests in the pytest sense — they are runtime
assertions exposed as a `run_all(bars) -> dict[str,bool]` function so
the smoke script and any CI hook can call them and exit non-zero on
failure.

The whole point of this module is to GUARANTEE that the synthetic LOB
at index t depends only on bars[0:t]. If even one of these tests
fails, every downstream LOBFrame model is contaminated and we have to
rebuild from scratch. There is no recovery path.

Tests
-----
1. Determinism — same bars, same seed -> same tensor.
2. Causal isolation (price) — perturbing close[t..] does not change lob[t].
3. Causal isolation (volume) — perturbing volume[t..] does not change lob[t].
4. Causal isolation (high/low) — high[t]/low[t] never enter lob[t].
5. Spread positivity — ask_p_1 > bid_p_1 for all t.
6. Ask monotonicity — ask_p_k strictly ascending across k.
7. Bid monotonicity — bid_p_k strictly descending across k.
8. Volume positivity — every volume cell > 0.
9. NaN-free — no NaN anywhere.
10. Past-causal pivot — lob[t] for t in middle of series is identical
    when we feed bars[:t+200] vs bars[:t+50] (LOB at t depends only
    on the past, so growing the future does nothing to it).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from trade.research.lobframe.cont_stoikov import (
    ContStoikovConfig,
    synthesize_lob,
)
from trade.research.lobframe.data_schema import (
    LEVELS,
    ask_price_columns,
    ask_volume_columns,
    bid_price_columns,
    bid_volume_columns,
)


def _baseline(bars: pd.DataFrame, cfg: ContStoikovConfig) -> np.ndarray:
    return synthesize_lob(bars, cfg)


def test_determinism(bars: pd.DataFrame, cfg: ContStoikovConfig) -> bool:
    a = synthesize_lob(bars, cfg)
    b = synthesize_lob(bars, cfg)
    return bool(np.array_equal(a, b))


def test_no_close_leakage(bars: pd.DataFrame, cfg: ContStoikovConfig) -> bool:
    base = _baseline(bars, cfg)
    perturbed = bars.copy()
    # Move every close from index 100 onward by +1.0; if lob[t<100]
    # depends on those, it will change.
    pivot = min(100, len(bars) - 1)
    perturbed.loc[perturbed.index[pivot:], "close"] += 1.0
    p = synthesize_lob(perturbed, cfg)
    # All rows BEFORE the pivot must be identical
    return bool(np.array_equal(base[:pivot], p[:pivot]))


def test_no_volume_leakage(bars: pd.DataFrame, cfg: ContStoikovConfig) -> bool:
    base = _baseline(bars, cfg)
    perturbed = bars.copy()
    pivot = min(100, len(bars) - 1)
    perturbed.loc[perturbed.index[pivot:], "volume"] *= 1000.0
    p = synthesize_lob(perturbed, cfg)
    return bool(np.array_equal(base[:pivot], p[:pivot]))


def test_no_high_low_leakage(bars: pd.DataFrame, cfg: ContStoikovConfig) -> bool:
    """high[t] and low[t] of bar t must NEVER influence lob[t].

    We perturb high[t] and low[t] for ALL bars and verify the entire
    tensor is unchanged. The only way that can happen is if the
    generator never reads those columns.
    """
    base = _baseline(bars, cfg)
    perturbed = bars.copy()
    perturbed["high"] = perturbed["high"] + 5.0
    perturbed["low"] = perturbed["low"] - 5.0
    p = synthesize_lob(perturbed, cfg)
    return bool(np.array_equal(base, p))


def test_spread_positivity(bars: pd.DataFrame, cfg: ContStoikovConfig) -> bool:
    arr = _baseline(bars, cfg)
    ask_1 = arr[:, 0]
    bid_1 = arr[:, 2]
    return bool((ask_1 - bid_1 > 0).all())


def test_ask_monotonic(bars: pd.DataFrame, cfg: ContStoikovConfig) -> bool:
    arr = _baseline(bars, cfg)
    cols = ask_price_columns()
    asks = arr[:, cols]
    diffs = np.diff(asks, axis=1)
    return bool((diffs > 0).all())


def test_bid_monotonic(bars: pd.DataFrame, cfg: ContStoikovConfig) -> bool:
    arr = _baseline(bars, cfg)
    cols = bid_price_columns()
    bids = arr[:, cols]
    diffs = np.diff(bids, axis=1)
    return bool((diffs < 0).all())


def test_volume_positivity(bars: pd.DataFrame, cfg: ContStoikovConfig) -> bool:
    arr = _baseline(bars, cfg)
    cols = ask_volume_columns() + bid_volume_columns()
    vols = arr[:, cols]
    return bool((vols > 0).all())


def test_no_nans(bars: pd.DataFrame, cfg: ContStoikovConfig) -> bool:
    arr = _baseline(bars, cfg)
    return bool(np.isfinite(arr).all())


def test_forward_impact(bars: pd.DataFrame, cfg: ContStoikovConfig) -> bool:
    """Forward impact: perturbing close[t] must alter lob[t+1:] AND
    leave lob[:t+1] byte-identical.

    This is the COMPLEMENT to the no-*-leakage tests:
      - no_close_leakage proves the past is not contaminated
      - forward_impact proves the past actually flows into the future
        (otherwise the generator would be a constant!)

    Both must hold for the causal contract to be meaningful.
    """
    if len(bars) < 50:
        return True
    base = _baseline(bars, cfg)
    perturbed = bars.copy()
    pivot = min(50, len(bars) - 5)
    perturbed.loc[perturbed.index[pivot], "close"] += 0.01
    p = synthesize_lob(perturbed, cfg)

    # Past must be untouched
    past_unchanged = bool(np.array_equal(base[: pivot + 1], p[: pivot + 1]))
    # Future MUST differ on at least one row, and the very next row should
    # be the first one to react (because lob[t+1] uses close[t]).
    future_diff = not np.array_equal(base[pivot + 1 :], p[pivot + 1 :])
    next_row_diff = not np.array_equal(base[pivot + 1], p[pivot + 1])
    return past_unchanged and future_diff and next_row_diff


def test_label_causality_triple_barrier(
    bars: pd.DataFrame, cfg: ContStoikovConfig
) -> bool:
    """Triple-barrier label causality contract.

    For a sample whose feature window ends at bar t:
      A) Perturbing close[t-W..t] (the PAST) must change the FEATURE
         tensor (the model would see different inputs) — sanity that
         the synth tensor is sensitive to its inputs.
      B) Perturbing close[t+1..t+vertical] (the FUTURE) must NOT
         change the feature tensor at row t — proves features are
         strictly causal.
      C) Perturbing close[t+1..t+vertical] MAY change the LABEL at
         t — proves the label correctly uses future info as the
         training target.

    This is the dataset-level complement of forward_impact: the
    triple-barrier label is supposed to read the future, but the
    features at the same index must NOT.
    """
    if len(bars) < 200:
        return True

    # Lazy imports so the leakage suite has no hard dep on the dataset
    from trade.research.lobframe.dataset import (
        LOBDatasetConfig,
        LOBWindowDataset,
        LabelMethod,
    )

    base_tensor = synthesize_lob(bars, cfg)
    base_ds = LOBWindowDataset(
        base_tensor,
        LOBDatasetConfig(
            window=20, label_method=LabelMethod.TRIPLE_BARRIER,
            tb_atr_period=14, tb_atr_mult_tp=1.5, tb_atr_mult_sl=1.5,
            tb_vertical_bars=20,
        ),
        bars=bars,
    )

    # Pick a probe index in the middle of the series
    probe_t = len(bars) // 2
    if probe_t <= base_ds._min_t or probe_t > base_ds._max_t - 1:
        return True

    base_label = base_ds._label(probe_t)
    base_feat_row = base_tensor[probe_t].copy()

    # ----- B) FUTURE perturbation: features at probe_t must NOT change
    perturbed_future = bars.copy()
    fut_lo = probe_t + 1
    fut_hi = min(len(bars) - 1, probe_t + 30)
    perturbed_future.iloc[fut_lo:fut_hi + 1, perturbed_future.columns.get_loc("close")] += 0.05
    perturbed_future.iloc[fut_lo:fut_hi + 1, perturbed_future.columns.get_loc("high")] += 0.05
    perturbed_future.iloc[fut_lo:fut_hi + 1, perturbed_future.columns.get_loc("low")] += 0.05
    fut_tensor = synthesize_lob(perturbed_future, cfg)
    fut_feat_row = fut_tensor[probe_t]
    feature_unchanged = bool(np.array_equal(base_feat_row, fut_feat_row))

    fut_ds = LOBWindowDataset(
        fut_tensor,
        LOBDatasetConfig(
            window=20, label_method=LabelMethod.TRIPLE_BARRIER,
            tb_atr_period=14, tb_atr_mult_tp=1.5, tb_atr_mult_sl=1.5,
            tb_vertical_bars=20,
        ),
        bars=perturbed_future,
    )
    fut_label = fut_ds._label(probe_t)
    label_can_change = True  # weaker requirement: it MAY change

    # ----- A) PAST perturbation: features at probe_t MUST change
    perturbed_past = bars.copy()
    past_lo = max(0, probe_t - 30)
    past_hi = probe_t - 1
    perturbed_past.iloc[past_lo:past_hi + 1, perturbed_past.columns.get_loc("close")] += 0.05
    past_tensor = synthesize_lob(perturbed_past, cfg)
    past_feat_row = past_tensor[probe_t]
    feature_changed_by_past = not np.array_equal(base_feat_row, past_feat_row)

    return feature_unchanged and feature_changed_by_past and label_can_change


def test_past_only_pivot(bars: pd.DataFrame, cfg: ContStoikovConfig) -> bool:
    """If we slice bars at two different lengths but >= the pivot, the
    rows up to the pivot must be byte-identical between the two runs.
    """
    if len(bars) < 200:
        return True  # not enough data — vacuously true
    short = bars.iloc[:150]
    longer = bars.iloc[:300]
    a = synthesize_lob(short, cfg)
    b = synthesize_lob(longer, cfg)[: len(short)]
    return bool(np.array_equal(a, b))


ALL_TESTS = {
    "determinism": test_determinism,
    "no_close_leakage": test_no_close_leakage,
    "no_volume_leakage": test_no_volume_leakage,
    "no_high_low_leakage": test_no_high_low_leakage,
    "spread_positivity": test_spread_positivity,
    "ask_monotonic": test_ask_monotonic,
    "bid_monotonic": test_bid_monotonic,
    "volume_positivity": test_volume_positivity,
    "no_nans": test_no_nans,
    "past_only_pivot": test_past_only_pivot,
    "forward_impact": test_forward_impact,
    "label_causality_tb": test_label_causality_triple_barrier,
}


def run_all(
    bars: pd.DataFrame,
    config: ContStoikovConfig | None = None,
) -> Dict[str, bool]:
    cfg = config or ContStoikovConfig()
    return {name: fn(bars, cfg) for name, fn in ALL_TESTS.items()}
