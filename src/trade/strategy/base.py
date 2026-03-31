"""Base strategy class for custom trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from trade.data.models import Signal, AssetType


class BaseStrategy(ABC):
    """Base class for custom trading strategies."""

    name: str = "base_strategy"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def generate_signals(
        self,
        symbol: str,
        asset_type: AssetType,
        df: pd.DataFrame,
        indicators: dict[str, Any],
        sentiment: dict[str, Any] | None = None,
    ) -> list[Signal]:
        """Generate trading signals based on data and indicators.

        Args:
            symbol: Trading symbol.
            asset_type: Type of asset.
            df: DataFrame with OHLCV + indicators.
            indicators: Technical indicator summary.
            sentiment: Optional sentiment analysis result.

        Returns:
            List of trading signals.
        """
        ...
