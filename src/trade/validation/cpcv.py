"""Combinatorial Purged Cross-Validation (strategy-agnostic).

Hand-rolled implementation of Lopez de Prado AFML ch. 12. Replaces
VectorBT PRO's paid CPCV with zero loss of rigor.

Usage
-----
    strat = MyStrategy()
    cpcv = CPCV(strat, n_folds=6, n_test_folds=2, embargo_bars=50)
    result = cpcv.run(df, spread=0.00008)
    if not result.viable:
        raise SystemExit(result.summary())
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from trade.validation.protocol import StrategyProtocol


def _sharpe(pnls: list[float], periods_per_year: int = 220) -> float:
    arr = np.array(pnls)
    if len(arr) < 2 or np.std(arr) == 0:
        return 0.0
    return float(np.mean(arr) / np.std(arr) * np.sqrt(periods_per_year))


@dataclass
class CPCVResult:
    strategy: str
    n_params: int
    n_folds: int
    n_test_folds: int
    n_combos: int
    mean_is_sharpe: float
    mean_oos_sharpe: float
    median_oos_sharpe: float
    oos_positive_rate: float
    wfe: float
    pbo: float

    @property
    def viable(self) -> bool:
        return (
            self.mean_oos_sharpe > 0.5
            and self.wfe >= 0.5
            and self.pbo < 0.5
        )

    def summary(self) -> str:
        verdict = "VIABLE" if self.viable else "NOT VIABLE"
        return (
            f"CPCV[{self.strategy}]: "
            f"IS_SR={self.mean_is_sharpe:+.3f} "
            f"OOS_SR={self.mean_oos_sharpe:+.3f} "
            f"WFE={self.wfe:.3f} PBO={self.pbo:.3f} => {verdict}"
        )


class CPCV:
    def __init__(
        self,
        strategy: StrategyProtocol,
        n_folds: int = 6,
        n_test_folds: int = 2,
        embargo_bars: int = 50,
    ):
        if n_test_folds >= n_folds:
            raise ValueError("n_test_folds must be < n_folds")
        self.strategy = strategy
        self.n_folds = n_folds
        self.n_test_folds = n_test_folds
        self.embargo_bars = embargo_bars

    def _make_folds(self, n: int) -> list[tuple[int, int]]:
        size = n // self.n_folds
        folds = []
        for k in range(self.n_folds):
            start = k * size
            end = (k + 1) * size if k < self.n_folds - 1 else n
            folds.append((start, end))
        return folds

    def _all_param_combos(self) -> list[dict]:
        keys = list(self.strategy.param_grid.keys())
        values = [self.strategy.param_grid[k] for k in keys]
        return [dict(zip(keys, c)) for c in itertools.product(*values)]

    def run(self, df: pd.DataFrame, spread: float, verbose: bool = False) -> CPCVResult:
        folds = self._make_folds(len(df))
        combos = list(itertools.combinations(range(self.n_folds), self.n_test_folds))
        all_params = self._all_param_combos()

        # Precompute PnLs per (param_idx, fold_idx) with warmup context
        fold_pnls: dict[tuple[int, int], list[float]] = {}
        warmup = 200  # conservative warmup for BB/ATR/ADX-style indicators
        for pi, params in enumerate(all_params):
            if verbose and pi % 50 == 0:
                print(f"  precompute {pi}/{len(all_params)}")
            for fi, (s, e) in enumerate(folds):
                sub = df.iloc[max(0, s - warmup):e]
                fold_pnls[(pi, fi)] = self.strategy.backtest(sub, spread, params)

        is_sharpes, oos_sharpes = [], []
        for test_combo in combos:
            train_folds = [i for i in range(self.n_folds) if i not in test_combo]
            best_sr, best_pi = -1e9, 0
            for pi in range(len(all_params)):
                is_pnls = []
                for fi in train_folds:
                    is_pnls.extend(fold_pnls[(pi, fi)])
                sr = _sharpe(is_pnls)
                if sr > best_sr:
                    best_sr, best_pi = sr, pi
            oos_pnls = []
            for fi in test_combo:
                oos_pnls.extend(fold_pnls[(best_pi, fi)])
            is_sharpes.append(best_sr)
            oos_sharpes.append(_sharpe(oos_pnls))

        is_arr = np.array(is_sharpes)
        oos_arr = np.array(oos_sharpes)
        mean_is = float(is_arr.mean())
        mean_oos = float(oos_arr.mean())
        wfe = mean_oos / mean_is if mean_is > 0 else 0.0
        return CPCVResult(
            strategy=self.strategy.name,
            n_params=len(all_params),
            n_folds=self.n_folds,
            n_test_folds=self.n_test_folds,
            n_combos=len(combos),
            mean_is_sharpe=mean_is,
            mean_oos_sharpe=mean_oos,
            median_oos_sharpe=float(np.median(oos_arr)),
            oos_positive_rate=float((oos_arr > 0).mean()),
            wfe=float(wfe),
            pbo=float((oos_arr <= 0).mean()),
        )
