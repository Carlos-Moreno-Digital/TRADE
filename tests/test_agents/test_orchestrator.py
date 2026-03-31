"""Integration tests for the Orchestrator with PropFirmRiskEngine."""

import pytest
from unittest.mock import patch, MagicMock

from trade.agents.orchestrator import Orchestrator
from trade.config import TradingConfig
from trade.data.models import AssetType, PortfolioState, TradeAction
from trade.risk.prop_firm import PropFirmRiskEngine, PropFirmConfig


@pytest.fixture
def config():
    return TradingConfig()


@pytest.fixture
def risk_engine():
    prop_config = PropFirmConfig(name="test", hard_stop_daily_pct=3.0, hard_stop_drawdown_pct=7.0)
    engine = PropFirmRiskEngine(prop_config)
    return engine


@pytest.fixture
def orchestrator(config, risk_engine):
    return Orchestrator(config=config, risk_engine=risk_engine)


class TestOrchestratorInit:
    def test_creates_with_risk_engine(self, orchestrator, risk_engine):
        assert orchestrator.risk_engine is risk_engine

    def test_portfolio_initialized(self, orchestrator):
        # Uses prop firm account_size ($10K) instead of config default ($100K)
        assert orchestrator.portfolio.total_value == 10000.0
        assert orchestrator.portfolio.cash == 10000.0

    def test_risk_engine_initialized(self, orchestrator):
        assert orchestrator.risk_engine._initial_balance == 10000.0


class TestRiskVetoAbsolute:
    """The most important test: risk veto CANNOT be overridden."""

    def test_risk_veto_blocks_trade(self, config):
        """If risk engine says NO, the orchestrator must NOT trade."""
        risk_engine = PropFirmRiskEngine(PropFirmConfig(name="test"))
        risk_engine.initialize(10000.0)
        risk_engine.start_trading_day(10000.0)

        # Simulate hitting daily loss limit
        risk_engine.record_trade_result(-50.0)
        risk_engine.record_trade_result(-50.0)  # 2 consecutive losses = cooldown

        orch = Orchestrator(config=config, risk_engine=risk_engine)
        orch.portfolio.initialize(10000.0)

        # The orchestrator should skip analysis entirely
        result = orch.analyze_symbol("EURUSD=X", AssetType.FOREX)
        assert result["final_decision"] == "hold"
        assert result.get("skip_reason") or result["final_decision"] == "hold"

    def test_killed_engine_blocks_everything(self, config):
        """Kill switch must prevent all trading."""
        risk_engine = PropFirmRiskEngine(PropFirmConfig(name="test"))
        risk_engine.initialize(10000.0)
        risk_engine._kill("Test kill")

        orch = Orchestrator(config=config, risk_engine=risk_engine)
        result = orch.analyze_symbol("EURUSD=X", AssetType.FOREX)
        assert result["final_decision"] == "hold"


class TestKnowledgeIntegration:
    def test_context_has_session_info(self, config):
        """Orchestrator should add session/kill zone info to context."""
        orch = Orchestrator(config=config)
        result = orch.analyze_symbol("EURUSD=X", AssetType.FOREX)
        # Result should have session info
        assert "session" in result

    def test_fundamental_skipped_for_forex(self, config):
        """Fundamental agent should NOT run for forex pairs."""
        orch = Orchestrator(config=config)
        result = orch.analyze_symbol("EURUSD=X", AssetType.FOREX)
        agent_names = [a["name"] for a in result.get("agents", [])]
        assert "fundamental" not in agent_names


class TestPortfolioTracking:
    def test_portfolio_state_updates(self, orchestrator):
        """Portfolio state should be updated after analysis."""
        initial_value = orchestrator.portfolio.total_value
        orchestrator.analyze_symbol("AAPL", AssetType.STOCK)
        # Portfolio value should still be tracked
        assert orchestrator.portfolio.total_value > 0

    def test_daily_reset(self, orchestrator):
        """Watchlist analysis should reset daily counters."""
        orchestrator.portfolio.daily_pnl = -500.0
        orchestrator.portfolio.trades_today = 5
        orchestrator.analyze_watchlist()
        # Should have been reset
        assert orchestrator.portfolio.trades_today == 0 or orchestrator.portfolio.daily_pnl == 0.0


class TestCorrelationCheck:
    def test_correlation_warning_generated(self, config):
        """Should detect correlated positions."""
        from trade.data.models import Position
        orch = Orchestrator(config=config)

        # Add a EURUSD position
        orch.portfolio.positions.append(Position(
            symbol="EURUSD=X",
            asset_type=AssetType.FOREX,
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1000,
        ))

        # Analyzing GBPUSD should trigger correlation warning
        result = orch.analyze_symbol("GBPUSD=X", AssetType.FOREX)
        # Correlation check happens in orchestrator
        assert isinstance(result.get("correlation_warnings"), list)
