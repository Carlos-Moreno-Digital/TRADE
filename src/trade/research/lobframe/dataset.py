"""PyTorch Dataset/DataLoader on top of the LOBFrame tensor.

Slides a fixed-length window across the (T, 40) tensor and emits
samples (window, 40) paired with a 3-class label derived from the
mid-price move `horizon` bars after the end of the window:

  label 0  -> mid_{t+H} < mid_t * (1 - tau)   (DOWN)
  label 1  -> within band                      (FLAT)
  label 2  -> mid_{t+H} > mid_t * (1 + tau)   (UP)

Causality:
  - The features cover bars [t-W+1, t]
  - The label uses mid at t and t+H, both INSIDE the future of the
    feature window. The label is therefore future-information, which
    is correct for supervised training (we want the model to learn
    to predict the future from the past).
  - At inference time the model only ever sees the feature window;
    the label is never used.

Per-sample feature normalization is z-score over the WINDOW only
(no global statistics) so the dataset has zero leakage from outside
the window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from trade.research.lobframe.data_schema import column_index


@dataclass
class LOBDatasetConfig:
    window: int = 100
    horizon: int = 5
    tau: float = 1e-4   # 1 bp deadband for the FLAT class
    normalize: bool = True


class LOBWindowDataset(Dataset):
    def __init__(self, tensor: np.ndarray, config: LOBDatasetConfig | None = None):
        if tensor.ndim != 2:
            raise ValueError(f"tensor must be 2-D, got {tensor.shape}")
        self.tensor = tensor.astype(np.float32)
        self.cfg = config or LOBDatasetConfig()
        self._ask1 = column_index(1, "ask_p")
        self._bid1 = column_index(1, "bid_p")
        T = len(self.tensor)
        # Last valid window-end index t such that t+horizon < T
        self._max_t = T - self.cfg.horizon - 1
        # First valid window-end index (must have window-1 prior bars)
        self._min_t = self.cfg.window - 1
        if self._max_t < self._min_t:
            raise ValueError(
                f"tensor too short: T={T}, window={self.cfg.window}, "
                f"horizon={self.cfg.horizon}"
            )
        # Precompute mid prices for label lookup
        self._mids = 0.5 * (self.tensor[:, self._ask1] + self.tensor[:, self._bid1])

    def __len__(self) -> int:
        return self._max_t - self._min_t + 1

    def _label(self, t: int) -> int:
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
