"""StrategyProtocol — the contract every candidate strategy must satisfy.

A strategy is a callable bundle of:
  - a name (for logging)
  - a declared parameter grid (for CPCV + DSR N_trials correction)
  - a deterministic per-bar backtest function that returns per-trade P&L

No state is allowed in the backtest function — it must be a pure
function of (df, spread, params). This is essential for reproducibility
under CPCV and time-permutation tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd


class StrategyProtocol(Protocol):
    """Every strategy that wants to be validated implements this."""

    name: str
    param_grid: dict[str, list[Any]]
    default_params: dict[str, Any]
    # Regimes (Statistical Jump Model state ids) in which the strategy
    # is permitted to trade. Default is ALL: every strategy must
    # explicitly declare which regimes it was validated in via CPCV
    # before being allowed in production. Empty list = strategy is
    # currently not authorized in any regime.
    supported_regimes: list[int]

    def signals(
        self, df: pd.DataFrame, params: dict[str, Any]
    ) -> pd.DataFrame:
        """Return a DataFrame with columns [bar_idx, side, sl, tp] listing
        every entry signal the strategy would have emitted while walking
        the df bar-by-bar (no lookahead). `bar_idx` is 1-based index into
        df, `side` is 'buy' or 'sell'. Used by NautilusHarness to replay
        the strategy with realistic fills.
        """
        ...

    def backtest(
        self,
        df: pd.DataFrame,
        spread: float,
        params: dict[str, Any],
        signal_shift: int = 0,
    ) -> list[float]:
        """Run the strategy on OHLCV df and return per-trade P&L (USD).

        Parameters
        ----------
        df : DataFrame
            OHLCV with columns ['open','high','low','close','volume'] and a
            DatetimeIndex named 'timestamp'. Must be sorted ascending.
        spread : float
            Absolute spread in price units (e.g. 0.00008 for EURUSD).
        params : dict
            Strategy parameters (subset of param_grid keys).
        signal_shift : int, default 0
            For future-shift tests: >0 lags execution N bars after signal,
            <0 cheats by executing N bars before signal bar. Strategies
            must honour this flag; 0 is the normal run.

        Returns
        -------
        list[float]
            Per-trade realized P&L in USD. Empty list if no trades.
        """
        ...


@dataclass
class PipelineResult:
    """Aggregated output of the full validation pipeline."""

    strategy: str
    symbol: str
    paranoid_pass: bool = False
    cpcv_pass: bool = False
    nautilus_pass: bool = False
    details: dict = field(default_factory=dict)

    @property
    def viable(self) -> bool:
        return self.paranoid_pass and self.cpcv_pass and self.nautilus_pass

    def summary(self) -> str:
        gates = [
            ("paranoid", self.paranoid_pass),
            ("cpcv", self.cpcv_pass),
            ("nautilus", self.nautilus_pass),
        ]
        line = " | ".join(
            f"{name}={'PASS' if ok else 'FAIL'}" for name, ok in gates
        )
        verdict = "VIABLE" if self.viable else "REJECTED"
        return f"{self.strategy} on {self.symbol}: {line} => {verdict}"
