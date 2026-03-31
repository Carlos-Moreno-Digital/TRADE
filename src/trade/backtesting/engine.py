"""Simple backtesting engine for strategy evaluation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger

from trade.data.models import AssetType, PortfolioState
from trade.execution.paper import PaperTradingEngine
from trade.agents.orchestrator import Orchestrator
from trade.config import TradingConfig


class BacktestEngine:
    """Backtesting engine that replays historical data through the trading system."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.results: list[dict[str, Any]] = []

    def run(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        asset_type: AssetType = AssetType.STOCK,
    ) -> dict[str, Any]:
        """Run a backtest on historical data.

        Args:
            symbol: Trading symbol.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            asset_type: Type of asset.

        Returns:
            Backtest results dictionary.
        """
        logger.info(f"Starting backtest: {symbol} from {start_date} to {end_date}")

        orchestrator = Orchestrator(self.config)

        # Run analysis (in backtest mode, this uses historical data)
        result = orchestrator.analyze_symbol(symbol, asset_type)

        # Get paper trading summary
        summary = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "analysis": result,
            "portfolio": {
                "final_value": orchestrator.portfolio.total_value,
                "total_pnl": orchestrator.portfolio.total_pnl,
                "max_drawdown_pct": orchestrator.portfolio.max_drawdown_pct,
                "positions": len(orchestrator.portfolio.positions),
            },
        }

        self.results.append(summary)
        logger.info(f"Backtest complete: {symbol}")
        return summary

    def get_report(self) -> dict[str, Any]:
        """Generate a summary report of all backtests."""
        if not self.results:
            return {"message": "No backtests run yet"}

        return {
            "total_backtests": len(self.results),
            "results": self.results,
        }
