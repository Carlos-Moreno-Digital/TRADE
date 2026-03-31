"""Tests that the risk veto is ABSOLUTE and cannot be overridden.

This is the single most important test file in the entire project.
If these tests fail, the bot WILL lose money.
"""

import pytest

from trade.agents.portfolio import PortfolioManagerAgent, STRENGTH_SCORES
from trade.config import AgentWeightsConfig
from trade.data.models import (
    AnalysisContext,
    AssetType,
    PortfolioState,
    Signal,
    TradeAction,
    SignalStrength,
)


@pytest.fixture
def context_with_veto():
    """Context where risk manager has issued a VETO."""
    portfolio = PortfolioState(cash=10000, total_value=10000, peak_value=10000)
    ctx = AnalysisContext(
        symbol="EURUSD",
        asset_type=AssetType.FOREX,
        portfolio=portfolio,
    )
    ctx.metadata["risk_veto"] = True
    ctx.metadata["risk_score"] = 0.9
    ctx.metadata["risk_flags"] = ["Daily loss 3.5% >= hard stop 3%"]

    # Add strong BUY signals from multiple agents
    for agent_name in ["market_data", "technical", "sentiment"]:
        ctx.signals.append(Signal(
            symbol="EURUSD",
            asset_type=AssetType.FOREX,
            action=TradeAction.BUY,
            strength=SignalStrength.STRONG_BUY,
            confidence=0.95,
            source_agent=agent_name,
            reasoning="Strong buy signal",
        ))

    return ctx


@pytest.fixture
def context_high_risk():
    """Context where risk score is high but not full veto."""
    portfolio = PortfolioState(cash=10000, total_value=10000, peak_value=10000)
    ctx = AnalysisContext(
        symbol="EURUSD",
        asset_type=AssetType.FOREX,
        portfolio=portfolio,
    )
    ctx.metadata["risk_veto"] = False
    ctx.metadata["risk_score"] = 0.6  # High but not veto

    for agent_name in ["market_data", "technical"]:
        ctx.signals.append(Signal(
            symbol="EURUSD",
            asset_type=AssetType.FOREX,
            action=TradeAction.BUY,
            strength=SignalStrength.STRONG_BUY,
            confidence=0.9,
            source_agent=agent_name,
        ))

    return ctx


class TestAbsoluteVeto:
    """These tests verify that NO combination of buy signals can override a risk veto."""

    def test_veto_forces_hold_despite_strong_buys(self, context_with_veto):
        """Even with 3 STRONG_BUY at 0.95 confidence, veto = HOLD."""
        agent = PortfolioManagerAgent()
        result = agent.analyze(context_with_veto)

        assert result.signal.action == TradeAction.HOLD
        assert result.signal.confidence == 1.0  # Maximum confidence in the HOLD
        assert "VETO" in result.signal.reasoning.upper()

    def test_veto_returns_immediately(self, context_with_veto):
        """Veto should short-circuit - no weighted calculation at all."""
        agent = PortfolioManagerAgent()
        result = agent.analyze(context_with_veto)

        # Should have "veto" in raw data
        assert result.raw_data.get("decision") == "veto"

    def test_high_risk_forces_hold(self, context_high_risk):
        """High risk score (>=0.5) should also force HOLD."""
        agent = PortfolioManagerAgent()
        result = agent.analyze(context_high_risk)

        assert result.signal.action == TradeAction.HOLD

    def test_risk_manager_excluded_from_voting(self):
        """Risk manager signals should NOT participate in weighted voting."""
        portfolio = PortfolioState(cash=10000, total_value=10000, peak_value=10000)
        ctx = AnalysisContext(
            symbol="EURUSD",
            asset_type=AssetType.FOREX,
            portfolio=portfolio,
        )
        ctx.metadata["risk_veto"] = False
        ctx.metadata["risk_score"] = 0.2  # Low risk

        # Only risk_manager signal (should be excluded)
        ctx.signals.append(Signal(
            symbol="EURUSD",
            asset_type=AssetType.FOREX,
            action=TradeAction.HOLD,
            strength=SignalStrength.STRONG_SELL,
            confidence=0.95,
            source_agent="risk_manager",
        ))

        agent = PortfolioManagerAgent()
        result = agent.analyze(ctx)

        # With no non-risk signals, should return HOLD
        assert result.signal.action == TradeAction.HOLD

    def test_higher_decision_thresholds(self):
        """Weak signals should NOT trigger trades (thresholds raised to 0.4)."""
        portfolio = PortfolioState(cash=10000, total_value=10000, peak_value=10000)
        ctx = AnalysisContext(
            symbol="EURUSD",
            asset_type=AssetType.FOREX,
            portfolio=portfolio,
        )
        ctx.metadata["risk_score"] = 0.1

        # Add a weak buy signal
        ctx.signals.append(Signal(
            symbol="EURUSD",
            asset_type=AssetType.FOREX,
            action=TradeAction.BUY,
            strength=SignalStrength.WEAK_BUY,
            confidence=0.4,
            source_agent="technical",
        ))

        agent = PortfolioManagerAgent()
        result = agent.analyze(ctx)

        # Weak buy with low confidence should NOT trigger a BUY
        # final_score = 0.3 * 0.30 * 0.4 = 0.036 (way below 0.4 threshold)
        assert result.signal.action == TradeAction.HOLD
