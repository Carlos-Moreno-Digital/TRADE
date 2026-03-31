"""Orchestrator - Coordinates all agents with PropFirmRiskEngine protection.

This is the BRAIN of the trading system. It:
1. Checks if trading is allowed (risk engine, session, calendar)
2. Runs analysis agents (market data, technical, sentiment)
3. Runs risk manager (ABSOLUTE veto power)
4. Runs portfolio manager (aggregates signals)
5. Runs execution agent (only if all checks pass)
6. Logs everything for audit
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
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
from trade.cache.store import CacheStore
from trade.config import TradingConfig
from trade.data.models import AnalysisContext, AssetType, PortfolioState
from trade.knowledge.market_sessions import (
    get_current_session,
    get_active_kill_zone,
    is_optimal_trading_time,
)
from trade.knowledge.correlations import check_correlation_conflict
from trade.risk.prop_firm import PropFirmRiskEngine, PropFirmConfig


console = Console()


class Orchestrator:
    """Coordinates all trading agents with full safety integration."""

    def __init__(
        self,
        config: TradingConfig,
        risk_engine: PropFirmRiskEngine | None = None,
        cache: CacheStore | None = None,
    ):
        self.config = config

        # Portfolio state
        initial_capital = config.portfolio.initial_capital
        self.portfolio = PortfolioState()
        self.portfolio.initialize(initial_capital)

        # PropFirmRiskEngine - THE critical safety layer
        self.risk_engine = risk_engine
        if self.risk_engine:
            self.risk_engine.initialize(initial_capital)
            self.risk_engine.start_trading_day(initial_capital)

        # Cache for API data and decision logging
        self.cache = cache

        # Initialize agents
        self._analysis_agents: list[BaseAgent] = [
            MarketDataAgent(config={"provider": "yfinance"}),
            TechnicalAgent(technical_config=config.technical),
            SentimentAgent(config={
                "engine": config.sentiment.engine,
                "max_articles": 10,
            }),
        ]
        # Fundamental only for stocks
        self._fundamental_agent = FundamentalAgent()

        self._risk_agent = RiskManagerAgent(risk_config=config.risk)
        self._portfolio_agent = PortfolioManagerAgent(weights=config.agent_weights)
        self._execution_agent = ExecutionAgent(config={
            "mode": config.mode,
            "risk_per_trade_pct": config.risk.max_per_trade_loss_pct,
            "max_position_pct": config.risk.max_position_pct,
        })

        self._analysis_history: list[dict] = []
        self._day_initialized = False

    def analyze_symbol(self, symbol: str, asset_type: AssetType | None = None) -> dict:
        """Run full analysis pipeline for a single symbol with all safety checks."""
        start = time.time()

        if asset_type is None:
            asset_type = self._detect_asset_type(symbol)

        logger.info(f"=== Starting analysis for {symbol} ({asset_type.value}) ===")

        # =================================================================
        # PRE-FLIGHT CHECKS (before any agent runs)
        # =================================================================

        # Check 1: PropFirmRiskEngine - can we trade at all?
        if self.risk_engine:
            can_trade, reason = self.risk_engine.can_open_trade(self.portfolio)
            if not can_trade:
                logger.warning(f"PROP FIRM BLOCK: {reason}")
                return self._make_skip_result(symbol, asset_type, f"Prop firm: {reason}", start)

        # Check 2: Session and kill zone awareness
        now_utc = datetime.now(timezone.utc)
        session = get_current_session(now_utc)
        kill_zone = get_active_kill_zone(now_utc)
        is_optimal, time_reason = is_optimal_trading_time(now_utc)

        # Check 3: Correlation with existing positions
        open_symbols = [pos.symbol for pos in self.portfolio.positions]
        if open_symbols:
            correlation_warnings = check_correlation_conflict(open_symbols + [symbol])
        else:
            correlation_warnings = []

        # =================================================================
        # CREATE ANALYSIS CONTEXT
        # =================================================================
        context = AnalysisContext(
            symbol=symbol,
            asset_type=asset_type,
            portfolio=self.portfolio,
            metadata={
                "session": session.name.value if session else "no_session",
                "kill_zone": kill_zone.name if kill_zone else None,
                "optimal_time": is_optimal,
                "time_reason": time_reason,
                "correlation_warnings": correlation_warnings,
            },
        )

        # =================================================================
        # RUN ANALYSIS AGENTS
        # =================================================================
        results: list[AgentSignal] = []

        # Analysis agents (market data, technical, sentiment)
        for agent in self._analysis_agents:
            try:
                result = agent.analyze(context)
                results.append(result)
                context.signals.append(result.signal)
                logger.info(
                    f"  [{agent.name}] {result.signal.action.value} "
                    f"({result.signal.strength.value}) "
                    f"conf={result.signal.confidence:.2f}"
                )
            except Exception as e:
                logger.error(f"  [{agent.name}] FAILED: {e}")

        # Fundamental agent only for stocks
        if asset_type == AssetType.STOCK:
            try:
                result = self._fundamental_agent.analyze(context)
                results.append(result)
                context.signals.append(result.signal)
            except Exception as e:
                logger.error(f"  [fundamental] FAILED: {e}")

        # =================================================================
        # RISK MANAGER (ABSOLUTE VETO POWER)
        # =================================================================
        try:
            risk_result = self._risk_agent.analyze(context)
            results.append(risk_result)
            context.signals.append(risk_result.signal)

            # Risk veto is set in context.metadata by RiskManagerAgent
            if context.metadata.get("risk_veto", False):
                logger.warning(f"  [risk_manager] VETO ACTIVE - blocking trade")
        except Exception as e:
            logger.error(f"  [risk_manager] FAILED: {e} - FORCING HOLD for safety")
            context.metadata["risk_veto"] = True

        # Log correlation warnings
        if correlation_warnings:
            for warn in correlation_warnings:
                logger.warning(f"  [correlations] {warn}")
            # Block if high correlation
            context.metadata["risk_veto"] = True
            context.metadata.setdefault("risk_flags", []).extend(correlation_warnings)

        # =================================================================
        # PORTFOLIO MANAGER (aggregates signals, respects veto)
        # =================================================================
        try:
            portfolio_result = self._portfolio_agent.analyze(context)
            results.append(portfolio_result)
            context.signals.append(portfolio_result.signal)
            logger.info(
                f"  [portfolio_manager] {portfolio_result.signal.action.value} "
                f"conf={portfolio_result.signal.confidence:.2f} | "
                f"{portfolio_result.signal.reasoning}"
            )
        except Exception as e:
            logger.error(f"  [portfolio_manager] FAILED: {e}")
            # If portfolio manager fails, we MUST NOT trade
            return self._make_skip_result(symbol, asset_type, "Portfolio manager failed", start)

        # =================================================================
        # EXECUTION (only if portfolio manager says BUY or SELL)
        # =================================================================
        final_action = portfolio_result.signal.action
        executed = False
        pnl = 0.0

        if final_action in (TradeAction.BUY, TradeAction.SELL, TradeAction.SHORT):
            # Final risk engine validation before execution
            if self.risk_engine:
                can_trade, reason = self.risk_engine.can_open_trade(self.portfolio)
                if not can_trade:
                    logger.warning(f"  [prop_firm] LAST-SECOND BLOCK: {reason}")
                    final_action = TradeAction.HOLD
                else:
                    try:
                        exec_result = self._execution_agent.analyze(context)
                        results.append(exec_result)
                        executed = exec_result.raw_data.get("executed", False)

                        if executed:
                            pnl = exec_result.raw_data.get("pnl", 0.0)
                            self.portfolio.record_trade(pnl)
                            if self.risk_engine:
                                self.risk_engine.record_trade_result(pnl)
                    except Exception as e:
                        logger.error(f"  [execution] FAILED: {e}")
            else:
                try:
                    exec_result = self._execution_agent.analyze(context)
                    results.append(exec_result)
                    executed = exec_result.raw_data.get("executed", False)
                    if executed:
                        pnl = exec_result.raw_data.get("pnl", 0.0)
                        self.portfolio.record_trade(pnl)
                except Exception as e:
                    logger.error(f"  [execution] FAILED: {e}")

        # =================================================================
        # UPDATE PORTFOLIO STATE
        # =================================================================
        position_value = sum(
            pos.quantity * pos.current_price for pos in self.portfolio.positions
        )
        self.portfolio.total_value = self.portfolio.cash + position_value
        self.portfolio.update_drawdown()

        total_time = (time.time() - start) * 1000

        # =================================================================
        # BUILD RESULT + LOG TO CACHE
        # =================================================================
        analysis = {
            "symbol": symbol,
            "asset_type": asset_type.value,
            "timestamp": datetime.utcnow().isoformat(),
            "session": session.name.value if session else "none",
            "kill_zone": kill_zone.name if kill_zone else "none",
            "optimal_time": is_optimal,
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
            "final_decision": final_action.value,
            "executed": executed,
            "correlation_warnings": correlation_warnings,
            "portfolio": {
                "cash": round(self.portfolio.cash, 2),
                "total_value": round(self.portfolio.total_value, 2),
                "positions": len(self.portfolio.positions),
                "drawdown_pct": round(self.portfolio.max_drawdown_pct, 2),
                "daily_pnl": round(self.portfolio.daily_pnl, 2),
                "consecutive_losses": self.portfolio.consecutive_losses,
            },
            "total_time_ms": round(total_time, 2),
        }

        # Log decision to cache
        if self.cache:
            self.cache.log_decision(
                symbol=symbol,
                action=final_action.value,
                confidence=portfolio_result.signal.confidence if portfolio_result else 0,
                asset_type=asset_type.value,
                strength=portfolio_result.signal.strength.value if portfolio_result else "",
                risk_score=context.metadata.get("risk_score", 0),
                prop_firm_check="pass" if not context.metadata.get("risk_veto") else "veto",
                executed=executed,
                notes=f"Session: {analysis['session']} | KZ: {analysis['kill_zone']}",
            )

        self._analysis_history.append(analysis)
        return analysis

    def analyze_watchlist(self) -> list[dict]:
        """Run analysis on all symbols in the configured watchlist."""
        # Start new trading day if needed
        if self.risk_engine and not self._day_initialized:
            self.risk_engine.start_trading_day(self.portfolio.total_value)
            self.portfolio.reset_daily()
            self._day_initialized = True

        results = []
        watchlist = self._build_watchlist()

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
        table.add_column("Session")
        table.add_column("Decision", style="bold")
        table.add_column("Executed")
        table.add_column("Risk")
        table.add_column("Time (ms)")

        for result in results:
            decision = result["final_decision"]
            decision_style = (
                "green" if "buy" in decision else "red" if "sell" in decision else "yellow"
            )
            executed = "[green]YES[/green]" if result.get("executed") else "[dim]no[/dim]"

            agents = {a["name"]: a for a in result["agents"]}
            risk_conf = agents.get("risk_manager", {}).get("confidence", 0)

            table.add_row(
                result["symbol"],
                result["asset_type"],
                result.get("session", "?"),
                f"[{decision_style}]{decision}[/{decision_style}]",
                executed,
                f"{risk_conf:.2f}",
                f"{result['total_time_ms']:.0f}",
            )

        console.print(table)

        if results:
            portfolio = results[-1]["portfolio"]
            console.print(
                f"\n[bold]Portfolio:[/bold] ${portfolio['total_value']:,.2f} "
                f"| Cash: ${portfolio['cash']:,.2f} "
                f"| Positions: {portfolio['positions']} "
                f"| DD: {portfolio['drawdown_pct']:.1f}% "
                f"| Daily P&L: ${portfolio['daily_pnl']:,.2f}"
            )

            # Show prop firm progress if available
            if self.risk_engine:
                progress = self.risk_engine.get_challenge_progress(
                    self.portfolio.total_value
                )
                status = progress.get("safety_status", "?")
                status_color = {
                    "SAFE": "green", "CAUTION": "yellow",
                    "DANGER": "red", "CRITICAL": "red bold", "DEAD": "red bold",
                }.get(status, "white")

                console.print(
                    f"[bold]Challenge:[/bold] {progress.get('progress_pct', 0):.1f}% "
                    f"(${progress.get('current_equity', 0):,.2f}) "
                    f"| Target: {progress.get('target_pct', 0)}% "
                    f"| Status: [{status_color}]{status}[/{status_color}]"
                )

    def _build_watchlist(self) -> list[tuple[str, AssetType]]:
        """Build watchlist from config."""
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

        return watchlist

    def _make_skip_result(
        self, symbol: str, asset_type: AssetType, reason: str, start_time: float
    ) -> dict:
        """Create a result for skipped analysis."""
        total_time = (time.time() - start_time) * 1000
        return {
            "symbol": symbol,
            "asset_type": asset_type.value,
            "timestamp": datetime.utcnow().isoformat(),
            "session": "skipped",
            "kill_zone": "none",
            "optimal_time": False,
            "agents": [],
            "final_decision": "hold",
            "executed": False,
            "skip_reason": reason,
            "correlation_warnings": [],
            "portfolio": {
                "cash": round(self.portfolio.cash, 2),
                "total_value": round(self.portfolio.total_value, 2),
                "positions": len(self.portfolio.positions),
                "drawdown_pct": round(self.portfolio.max_drawdown_pct, 2),
                "daily_pnl": round(self.portfolio.daily_pnl, 2),
                "consecutive_losses": self.portfolio.consecutive_losses,
            },
            "total_time_ms": round(total_time, 2),
        }

    @staticmethod
    def _detect_asset_type(symbol: str) -> AssetType:
        if symbol.endswith("=X"):
            return AssetType.FOREX
        if symbol.endswith("-USD") or symbol.endswith("-USDT"):
            return AssetType.CRYPTO
        return AssetType.STOCK


# Keep backward compatibility
from trade.data.models import TradeAction  # noqa: E402
