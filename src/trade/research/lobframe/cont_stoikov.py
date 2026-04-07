"""Causal synthetic LOB generator (Cont-Stoikov 2014 inspired).

CRITICAL CONTRACT (NO LOOK-AHEAD):
================================================================
The synthetic LOB at index t is built using ONLY the bars [0, t).
The OHLC of bar t is NEVER read when building lob[t]. This is the
single most important property of this module — every other test
in the project assumes it.

Why this matters: in a trading research pipeline, the LOB at the
START of bar t represents the state of the market right before our
strategy makes a decision for bar t. Any price/volume from bar t
itself would be future information at decision time, contaminating
all downstream evaluation.

Construction (per bar t):
  mid_t      = close[t-1]                               # last known close
  vol_t      = std(log_returns[t-W:t])                  # rolling realized vol
  vol_t      = max(vol_t, vol_floor)                    # lower bound
  spread_t   = base_spread * (1 + alpha * vol_t / scale_vol)
  tick_t     = spread_t / 2                             # half-spread per level step
  depth_unit = mean(volume[t-W:t]) / depth_scale        # rolling volume proxy

For each level k = 1..L:
  ask_p_k = mid_t + spread_t/2 + (k-1) * tick_t
  bid_p_k = mid_t - spread_t/2 - (k-1) * tick_t
  ask_v_k = depth_unit * exp(-decay * (k-1)) * (1 + jitter[t,k])
  bid_v_k = depth_unit * exp(-decay * (k-1)) * (1 + jitter[t,k])

The jitter is drawn from a deterministic per-bar PRNG seeded by
(seed, t) so the output is reproducible without contaminating across
bars.

This is intentionally a SHAPE reconstruction, not a tick-by-tick
microstructure simulation. It is sufficient as a tensor input for
LOBFrame-style models when real LOB tapes are unavailable, and the
absolute spread/depth scale is calibrated separately per symbol.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trade.research.lobframe.data_schema import (
    LEVELS,
    LOBSnapshot,
    N_FEATURES,
)


@dataclass
class ContStoikovConfig:
    levels: int = LEVELS
    window: int = 20            # bars used for vol/volume rolling
    base_spread: float = 1e-4   # ~1 pip for FX major
    alpha_vol: float = 5.0      # spread sensitivity to vol
    scale_vol: float = 1e-3     # vol normalizer
    vol_floor: float = 1e-5
    depth_scale: float = 1.0    # divide rolling volume by this
    depth_unit_floor: float = 1.0
    decay: float = 0.25         # exponential book decay across levels
    jitter_amplitude: float = 0.05  # +/- 5% per-level volume noise
    seed: int = 42


def _check_bars(bars: pd.DataFrame) -> None:
    needed = {"open", "high", "low", "close", "volume"}
    missing = needed - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing columns: {missing}")
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("bars must have a DatetimeIndex")


def synthesize_lob(
    bars: pd.DataFrame,
    config: ContStoikovConfig | None = None,
) -> np.ndarray:
    """Build a causal synthetic LOB tensor of shape (T, 4*levels).

    Bars before the warmup window get NaN-free zero books — the caller
    is responsible for slicing them off if it wants strictly valid rows.
    """
    cfg = config or ContStoikovConfig()
    _check_bars(bars)
    T = len(bars)
    L = cfg.levels
    n_features = 4 * L
    out = np.zeros((T, n_features), dtype=float)

    # Convert to numpy once for speed and to avoid pandas indexing surprises
    closes = bars["close"].to_numpy(dtype=float)
    volumes = bars["volume"].to_numpy(dtype=float)
    log_rets = np.diff(np.log(closes), prepend=closes[0])

    rng_master = np.random.default_rng(cfg.seed)
    # Pre-generate per-bar seed offsets so output is fully deterministic
    bar_seeds = rng_master.integers(0, 2**31 - 1, size=T)

    for t in range(T):
        # CAUSAL: at bar t, only data up to t-1 is allowed
        if t == 0:
            mid = closes[0]
            vol_t = cfg.vol_floor
            depth_unit = cfg.depth_unit_floor
        else:
            mid = closes[t - 1]
            lookback_lo = max(0, t - cfg.window)
            lookback_hi = t  # exclusive — STOP at t-1
            vol_window = log_rets[lookback_lo:lookback_hi]
            if vol_window.size >= 2:
                vol_t = float(np.std(vol_window, ddof=1))
            else:
                vol_t = cfg.vol_floor
            vol_t = max(vol_t, cfg.vol_floor)
            vol_window_volumes = volumes[lookback_lo:lookback_hi]
            if vol_window_volumes.size > 0:
                depth_unit = float(np.mean(vol_window_volumes)) / cfg.depth_scale
            else:
                depth_unit = cfg.depth_unit_floor
            depth_unit = max(depth_unit, cfg.depth_unit_floor)

        spread_t = cfg.base_spread * (1.0 + cfg.alpha_vol * vol_t / cfg.scale_vol)
        tick_t = spread_t / 2.0
        half_spread = spread_t / 2.0

        bar_rng = np.random.default_rng(int(bar_seeds[t]))
        jitter = bar_rng.uniform(
            -cfg.jitter_amplitude, cfg.jitter_amplitude, size=L
        )
        decay_factor = np.exp(-cfg.decay * np.arange(L))
        per_level_volume = depth_unit * decay_factor * (1.0 + jitter)
        per_level_volume = np.maximum(per_level_volume, 1e-6)

        for k in range(L):
            ask_p = mid + half_spread + k * tick_t
            bid_p = mid - half_spread - k * tick_t
            row_off = k * 4
            out[t, row_off + 0] = ask_p              # ask_p_k
            out[t, row_off + 1] = per_level_volume[k]  # ask_v_k
            out[t, row_off + 2] = bid_p              # bid_p_k
            out[t, row_off + 3] = per_level_volume[k]  # bid_v_k

    return out


def synthesize_snapshots(
    bars: pd.DataFrame,
    config: ContStoikovConfig | None = None,
) -> list[LOBSnapshot]:
    """Same as synthesize_lob but returns LOBSnapshot objects keyed by
    the bar timestamp. Useful for the bridge tests that need a typed
    view rather than a flat tensor.
    """
    cfg = config or ContStoikovConfig()
    tensor = synthesize_lob(bars, cfg)
    L = cfg.levels
    out: list[LOBSnapshot] = []
    for t, ts in enumerate(bars.index):
        ask_p = tensor[t, [k * 4 + 0 for k in range(L)]]
        ask_v = tensor[t, [k * 4 + 1 for k in range(L)]]
        bid_p = tensor[t, [k * 4 + 2 for k in range(L)]]
        bid_v = tensor[t, [k * 4 + 3 for k in range(L)]]
        out.append(
            LOBSnapshot(
                timestamp_ns=int(ts.value),
                ask_prices=ask_p,
                ask_volumes=ask_v,
                bid_prices=bid_p,
                bid_volumes=bid_v,
            )
        )
    return out
