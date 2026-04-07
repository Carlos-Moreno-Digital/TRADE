"""ConcentrationGate — block redundant positions via NMI dependency.

Uses the cached NMI matrix produced by analyze_nmi.py to detect when a
new candidate position would just stack risk on top of an already-open
position from the same statistical cluster.

Two assets with NMI above the threshold (default 0.15) are considered
"the same bet under a different ticker". Going long both is a
concentration trap, not a diversification.

Designed to be cheap: loads the matrix once, lookup is O(1) per pair.
If the matrix file is missing the gate fails OPEN with a warning, so
the trading pipeline still works during research without NMI data.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_MATRIX_PATH = Path("data/nmi_matrix.csv")
DEFAULT_THRESHOLD = 0.15  # Anything above the noise floor (~0.01) and
                          # below the "same currency" floor (EUR/GBP=0.16)


@dataclass
class ConcentrationDecision:
    allowed: bool
    reason: str
    conflicting_symbol: str | None = None
    nmi_value: float | None = None


def _normalize(symbol: str) -> str:
    """Map orchestrator/yfinance tickers to NMI matrix names."""
    s = symbol.upper().replace("=X", "").replace("^", "")
    aliases = {
        "JPY": "USDJPY",
        "CAD": "USDCAD",
        "GC=F": "XAUUSD",
        "GC": "XAUUSD",
        "GSPC": "SPX",
    }
    return aliases.get(s, s)


class ConcentrationGate:
    def __init__(
        self,
        matrix_path: Path = DEFAULT_MATRIX_PATH,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.threshold = threshold
        self._matrix: pd.DataFrame | None = None
        self._loaded_from: Path | None = None
        if matrix_path.exists():
            try:
                self._matrix = pd.read_csv(matrix_path, index_col=0)
                self._loaded_from = matrix_path
            except Exception:
                self._matrix = None

    @property
    def is_armed(self) -> bool:
        return self._matrix is not None and not self._matrix.empty

    def nmi(self, a: str, b: str) -> float | None:
        """Return NMI(a, b) or None if either symbol is not in the matrix."""
        if not self.is_armed:
            return None
        na, nb = _normalize(a), _normalize(b)
        if na == nb:
            return 1.0
        if na not in self._matrix.index or nb not in self._matrix.columns:
            return None
        return float(self._matrix.loc[na, nb])

    def check(
        self,
        candidate_symbol: str,
        open_symbols: list[str],
    ) -> ConcentrationDecision:
        """Decide whether opening candidate_symbol is allowed given the
        already-open book.
        """
        if not self.is_armed:
            return ConcentrationDecision(
                allowed=True,
                reason="ConcentrationGate disarmed (no NMI matrix on disk)",
            )
        if not open_symbols:
            return ConcentrationDecision(
                allowed=True, reason="No open positions"
            )

        worst_pair: tuple[str, float] | None = None
        for sym in open_symbols:
            v = self.nmi(candidate_symbol, sym)
            if v is None:
                continue
            if worst_pair is None or v > worst_pair[1]:
                worst_pair = (sym, v)

        if worst_pair is None:
            return ConcentrationDecision(
                allowed=True,
                reason="No NMI data for any open symbol vs candidate",
            )

        sym, v = worst_pair
        if v >= self.threshold:
            return ConcentrationDecision(
                allowed=False,
                reason=(
                    f"Concentration: NMI({_normalize(candidate_symbol)},"
                    f"{_normalize(sym)})={v:.3f} >= {self.threshold:.2f}"
                ),
                conflicting_symbol=sym,
                nmi_value=v,
            )
        return ConcentrationDecision(
            allowed=True,
            reason=(
                f"Max NMI vs open book = {v:.3f} < {self.threshold:.2f}"
            ),
            conflicting_symbol=sym,
            nmi_value=v,
        )
