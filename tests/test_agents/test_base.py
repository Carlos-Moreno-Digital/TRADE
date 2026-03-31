"""Tests for base agent functionality."""

from trade.agents.base import BaseAgent, AgentSignal
from trade.data.models import (
    AnalysisContext,
    AssetType,
    PortfolioState,
    TradeAction,
    SignalStrength,
)


class ConcreteAgent(BaseAgent):
    """Concrete implementation for testing."""
    name = "test_agent"

    def analyze(self, context):
        signal = self._make_signal(
            context,
            action=TradeAction.BUY,
            strength=SignalStrength.BUY,
            confidence=0.8,
            reasoning="Test reasoning",
        )
        return self._make_output(signal, {"test": True}, notes="Test", exec_time=1.5)


class TestBaseAgent:
    def test_create_agent(self):
        agent = ConcreteAgent()
        assert agent.name == "test_agent"
        assert agent.config == {}

    def test_create_agent_with_config(self):
        agent = ConcreteAgent(config={"key": "value"})
        assert agent.config["key"] == "value"

    def test_analyze(self, analysis_context):
        agent = ConcreteAgent()
        result = agent.analyze(analysis_context)

        assert isinstance(result, AgentSignal)
        assert result.agent_name == "test_agent"
        assert result.signal.action == TradeAction.BUY
        assert result.signal.confidence == 0.8
        assert result.signal.symbol == "AAPL"
        assert result.raw_data["test"] is True
        assert result.execution_time_ms == 1.5

    def test_make_signal(self, analysis_context):
        agent = ConcreteAgent()
        signal = agent._make_signal(
            analysis_context,
            TradeAction.SELL,
            SignalStrength.STRONG_SELL,
            0.9,
            "Strong sell signal",
        )
        assert signal.action == TradeAction.SELL
        assert signal.strength == SignalStrength.STRONG_SELL
        assert signal.confidence == 0.9
        assert signal.source_agent == "test_agent"
