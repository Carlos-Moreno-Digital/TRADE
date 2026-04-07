"""PyTorch Dataset/DataLoader on top of the LOBFrame tensor.

Slides a fixed-length window across the (T, 40) tensor and emits
samples (window, 40) paired with a 3-class label.

Two labelling methods are supported:

1. MID_HORIZON (legacy):
     label 0  -> mid_{t+H} < mid_t * (1 - tau)
     label 1  -> within band (FLAT)
     label 2  -> mid_{t+H} > mid_t * (1 + tau)

2. TRIPLE_BARRIER (López de Prado AFML, ch. 3):
     Walk forward from bar t up to a vertical barrier of N bars.
     Whichever horizontal barrier (entry +/- k * ATR) is touched
     first determines the label. If neither is touched, the
     vertical barrier (timeout) wins.
       label 2  -> upper barrier hit  (would-be profitable LONG)
       label 0  -> lower barrier hit  (would-be profitable SHORT)
       label 1  -> timeout / no decisive move
     This is the canonical AFML labelling, dynamically scaled by
     local volatility (ATR), which solves the class collapse you
     get from a fixed-tau labeller on quiet data.

Causality:
  - The features cover bars [t-W+1, t] of the LOB tensor (which is
    itself causal: row i of the tensor depends on bars[0:i] only).
  - The label at index t uses bars [t : t+vertical_bars] (i.e. the
    future of the feature window). This is correct for supervised
    training because the label is the target, never an input.
  - At INFERENCE time the model only sees the feature window; the
    label is never read. Enforced by the dataset returning labels
    only via __getitem__ for training, while the strategy adapter
    calls _logits_for_bars() which never touches labels.

Per-sample feature normalisation is z-score over the WINDOW only
(no global statistics) so the dataset has zero leakage from outside
the window.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from trade.research.indicators import ATR
from trade.research.lobframe.data_schema import column_index


class LabelMethod(str, Enum):
    MID_HORIZON = "mid_horizon"
    TRIPLE_BARRIER = "triple_barrier"


@dataclass
class LOBDatasetConfig:
    window: int = 100
    horizon: int = 5            # used by MID_HORIZON
    tau: float = 1e-4           # 1 bp deadband for the FLAT class
    normalize: bool = True
    label_method: LabelMethod = LabelMethod.MID_HORIZON
    # Triple-barrier specific:
    tb_atr_period: int = 14
    tb_atr_mult_tp: float = 2.0
    tb_atr_mult_sl: float = 2.0
    tb_vertical_bars: int = 50


class LOBWindowDataset(Dataset):
    def __init__(
        self,
        tensor: np.ndarray,
        config: LOBDatasetConfig | None = None,
        bars: pd.DataFrame | None = None,
    ):
        if tensor.ndim != 2:
            raise ValueError(f"tensor must be 2-D, got {tensor.shape}")
        self.tensor = tensor.astype(np.float32)
        self.cfg = config or LOBDatasetConfig()
        self._ask1 = column_index(1, "ask_p")
        self._bid1 = column_index(1, "bid_p")

        T = len(self.tensor)

        # Triple-barrier needs the original OHLC (the LOB tensor doesn't
        # carry intra-bar high/low). The bars dataframe is REQUIRED for
        # this method.
        if self.cfg.label_method == LabelMethod.TRIPLE_BARRIER:
            if bars is None:
                raise ValueError(
                    "label_method=TRIPLE_BARRIER requires the original "
                    "bars dataframe (with high/low/close)."
                )
            if len(bars) != T:
                raise ValueError(
                    f"bars length {len(bars)} != tensor length {T}"
                )
            self._high = bars["high"].to_numpy(dtype=float)
            self._low = bars["low"].to_numpy(dtype=float)
            self._close = bars["close"].to_numpy(dtype=float)
            self._atr = ATR(
                self._high, self._low, self._close,
                period=self.cfg.tb_atr_period,
            )
            forward_bars = self.cfg.tb_vertical_bars
        else:
            self._high = None
            self._low = None
            self._close = None
            self._atr = None
            forward_bars = self.cfg.horizon

        # Last valid window-end index t such that t + forward_bars < T
        self._max_t = T - forward_bars - 1
        # First valid window-end index (must have window-1 prior bars)
        self._min_t = self.cfg.window - 1
        # For triple-barrier we also need ATR to be defined
        if self.cfg.label_method == LabelMethod.TRIPLE_BARRIER:
            self._min_t = max(self._min_t, self.cfg.tb_atr_period + 1)
        if self._max_t < self._min_t:
            raise ValueError(
                f"tensor too short: T={T}, window={self.cfg.window}, "
                f"forward={forward_bars}"
            )

        # Precompute mid prices (used by mid_horizon label and by
        # diagnostic helpers)
        self._mids = 0.5 * (self.tensor[:, self._ask1] + self.tensor[:, self._bid1])

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._max_t - self._min_t + 1

    def _label_mid_horizon(self, t: int) -> int:
        m_now = float(self._mids[t])
        m_fut = float(self._mids[t + self.cfg.horizon])
        if m_now <= 0:
            return 1
        ret = (m_fut - m_now) / m_now
        if ret > self.cfg.tau:
            return 2
        if ret < -self.cfg.tau:
            return 0
        return 1

    def _label_triple_barrier(self, t: int) -> int:
        atr = self._atr[t]
        if math.isnan(atr) or atr <= 0:
            return 1
        entry = self._close[t]
        upper = entry + atr * self.cfg.tb_atr_mult_tp
        lower = entry - atr * self.cfg.tb_atr_mult_sl
        end = min(t + self.cfg.tb_vertical_bars, len(self._close) - 1)
        for i in range(t + 1, end + 1):
            high_i = self._high[i]
            low_i = self._low[i]
            # Pessimistic ordering: when both barriers fall inside the
            # same bar, take the LOSING side (SL on a long context, TP
            # on the opposite). For the labelling step we just check
            # which barrier was actually touched first across bars; if
            # the same bar touches both, we can't distinguish without
            # tick data, so we default to the FLAT class to avoid an
            # optimistic bias.
            up_hit = high_i >= upper
            down_hit = low_i <= lower
            if up_hit and down_hit:
                return 1  # ambiguous -> conservative neutral
            if up_hit:
                return 2
            if down_hit:
                return 0
        return 1

    def _label(self, t: int) -> int:
        if self.cfg.label_method == LabelMethod.TRIPLE_BARRIER:
            return self._label_triple_barrier(t)
        return self._label_mid_horizon(t)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = self._min_t + idx
        win = self.tensor[t - self.cfg.window + 1 : t + 1]
        if self.cfg.normalize:
            mu = win.mean(axis=0, keepdims=True)
            sigma = win.std(axis=0, keepdims=True) + 1e-8
            win = (win - mu) / sigma
        return (
            torch.from_numpy(win.copy()),
            torch.tensor(self._label(t), dtype=torch.long),
        )

    # ------------------------------------------------------------------
    # Diagnostics — small dicts/tensors only, no dataframes
    # ------------------------------------------------------------------
    def label_distribution(self) -> dict[int, int]:
        counts = {0: 0, 1: 0, 2: 0}
        for t in range(self._min_t, self._max_t + 1):
            counts[self._label(t)] += 1
        return counts

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights normalised so they sum to n_classes.
        Use as `nn.CrossEntropyLoss(weight=ds.class_weights())`.
        """
        dist = self.label_distribution()
        counts = np.array([dist[i] for i in (0, 1, 2)], dtype=float)
        inv = 1.0 / np.maximum(counts, 1.0)
        weights = inv / inv.sum() * len(counts)
        return torch.tensor(weights, dtype=torch.float32)
