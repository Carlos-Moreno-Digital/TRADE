"""Orchestrator - Coordinates all agents in a pipeline to make trading decisions."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from loguru import logger
from rich.console import Console
from rich.table import Table

from trade.agents.base import BaseAgent, AgentSignal
from trade.agents.market_data import MarketDataAgent
from trade.agents.technical import TechnicalAgent
from trade.agents.sentiment import SentimentAgent
from trade.agents.fundamental import FundamentalAgent
from trade.agents.risk_manager import RiskManagerAgent
from trade.agents.portfolio import PortfolioManagerAgent
from trade.agents.execution import ExecutionAgent
from trade.config import TradingConfig
from trade.data.models import AnalysisContext, AssetType, PortfolioState


console = Console()


class Orchestrator:
    """Coordinates all trading agents in sequence."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.portfolio = PortfolioState(
            cash=config.portfolio.initial_capital,
            total_value=config.portfolio.initial_capital,
            peak_value=config.portfolio.initial_capital,
        )

        # Initialize agents
        self.agents: list[BaseAgent] = [
            MarketDataAgent(config={"provider": "yfinance"}),
            TechnicalAgent(technical_config=config.technical),
            SentimentAgent(config={
                "engine": config.sentiment.engine,
                "max_articles": 10,
            }),
            FundamentalAgent(),
            RiskManagerAgent(risk_config=config.risk),
            PortfolioManagerAgent(weights=config.agent_weights),
            ExecutionAgent(config={
                "mode": config.mode,
                "risk_per_trade_pct": config.risk.max_per_trade_loss_pct,
                "max_position_pct": config.risk.max_position_pct,
            }),
        ]

        self._analysis_history: list[dict] = []

    def analyze_symbol(self, symbol: str, asset_type: AssetType | None = None) -> dict:
        """Run full analysis pipeline for a single symbol."""
        start = time.time()

        if asset_type is None:
            asset_type = self._detect_asset_type(symbol)

        logger.info(f"=== Starting analysis for {symbol} ({asset_type.value}) ===")

        # Create analysis context
        context = AnalysisContext(
            symbol=symbol,
            asset_type=asset_type,
            portfolio=self.portfolio,
        )

        # Run each agent in sequence
        results: list[AgentSignal] = []
        for agent in self.agents:
            try:
                result = agent.analyze(context)
                results.append(result)
                # Add signal to context for downstream agents
                context.signals.append(result.signal)
                logger.info(
                    f"  [{agent.name}] {result.signal.action.value} "
                    f"({result.signal.strength.value}) "
                    f"conf={result.signal.confidence:.2f} | {result.signal.reasoning}"
                )
            except Exception as e:
                logger.error(f"  [{agent.name}] FAILED: {e}")

        # Update portfolio state
        self.portfolio.update_drawdown()
        self.portfolio.total_value = self.portfolio.cash + sum(
            pos.quantity * pos.current_price for pos in self.portfolio.positions
        )

        total_time = (time.time() - start) * 1000

        # Build result
        analysis = {
            "symbol": symbol,
            "asset_type": asset_type.value,
            "timestamp": datetime.utcnow().isoformat(),
            "agents": [
                {
                    "name": r.agent_name,
                    "action": r.signal.action.value,
                    "strength": r.signal.strength.value,
                    "confidence": r.signal.confidence,
                    "reasoning": r.signal.reasoning,
                    "execution_time_ms": r.execution_time_ms,
                }
                for r in results
            ],
            "final_decision": results[-1].signal.action.value if results else "hold",
            "portfolio": {
                "cash": round(self.portfolio.cash, 2),
                "total_value": round(self.portfolio.total_value, 2),
                "positions": len(self.portfolio.positions),
                "drawdown_pct": round(self.portfolio.max_drawdown_pct, 2),
                "daily_pnl": round(self.portfolio.daily_pnl, 2),
            },
            "total_time_ms": round(total_time, 2),
        }

        self._analysis_history.append(analysis)
        return analysis

    def analyze_watchlist(self) -> list[dict]:
        """Run analysis on all symbols in the configured watchlist."""
        results = []

        # Collect all symbols from all markets
        watchlist = []
        markets = self.config.markets

        if markets.stocks.enabled:
            for s in markets.stocks.watchlist:
                watchlist.append((s, AssetType.STOCK))
        if markets.crypto.enabled:
            for s in markets.crypto.watchlist:
                watchlist.append((s, AssetType.CRYPTO))
        if markets.forex.enabled:
            for s in markets.forex.watchlist:
                watchlist.append((s, AssetType.FOREX))

        logger.info(f"Analyzing {len(watchlist)} symbols across all markets")

        for symbol, asset_type in watchlist:
            try:
                result = self.analyze_symbol(symbol, asset_type)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")

        return results

    def print_summary(self, results: list[dict]) -> None:
        """Print a rich summary table of analysis results."""
        table = Table(title="Trading Agent Analysis Summary")
        table.add_column("Symbol", style="cyan")
        table.add_column("Type", style="blue")
        table.add_column("Decision", style="bold")
        table.add_column("Confidence")
        table.add_column("Technical")
        table.add_column("Sentiment")
        table.add_column("Risk Score")
        table.add_column("Time (ms)")

        for result in results:
            agents = {a["name"]: a for a in result["agents"]}

            decision = result["final_decision"]
            decision_style = (
                "green" if "buy" in decision else "red" if "sell" in decision else "yellow"
            )

            tech = agents.get("technical", {})
            sent = agents.get("sentiment", {})
            risk = agents.get("risk_manager", {})

            table.add_row(
                result["symbol"],
                result["asset_type"],
                f"[{decision_style}]{decision}[/{decision_style}]",
                f"{agents.get('portfolio_manager', {}).get('confidence', 0):.2f}",
                tech.get("action", "N/A"),
                sent.get("action", "N/A"),
                f"{risk.get('confidence', 0):.2f}",
                f"{result['total_time_ms']:.0f}",
            )

        console.print(table)

        # Portfolio summary
        if results:
            portfolio = results[-1]["portfolio"]
            console.print(f"\n[bold]Portfolio:[/bold] ${portfolio['total_value']:,.2f} "
                         f"| Cash: ${portfolio['cash']:,.2f} "
                         f"| Positions: {portfolio['positions']} "
                         f"| Drawdown: {portfolio['drawdown_pct']:.1f}%")

    @staticmethod
    def _detect_asset_type(symbol: str) -> AssetType:
        if symbol.endswith("=X"):
            return AssetType.FOREX
        if symbol.endswith("-USD") or symbol.endswith("-USDT"):
            return AssetType.CRYPTO
        return AssetType.STOCK
