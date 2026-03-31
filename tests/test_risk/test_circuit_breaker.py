"""Tests for the circuit breaker."""

import pytest

from trade.config import RiskConfig
from trade.data.models import PortfolioState
from trade.risk.circuit_breaker import CircuitBreaker


@pytest.fixture
def breaker():
    return CircuitBreaker(RiskConfig())


class TestCircuitBreaker:
    def test_initial_state(self, breaker):
        assert not breaker.is_active
        status = breaker.status
        assert status["is_active"] is False
        assert status["is_tripped"] is False

    def test_allows_trading_healthy_portfolio(self, breaker, portfolio):
        assert breaker.check(portfolio) is True

    def test_trips_on_max_drawdown(self, breaker):
        portfolio = PortfolioState(
            cash=85000.0,
            total_value=85000.0,
            peak_value=100000.0,
            max_drawdown_pct=11.0,  # Exceeds 10% limit
        )
        assert breaker.check(portfolio) is False
        assert breaker.is_active
        assert "drawdown" in breaker.status["reason"].lower()

    def test_trips_on_daily_loss(self, breaker):
        portfolio = PortfolioState(
            cash=95000.0,
            total_value=95000.0,
            peak_value=100000.0,
            daily_pnl=-4000.0,  # 4.2% > 3% limit
        )
        assert breaker.check(portfolio) is False

    def test_cooldown_on_consecutive_losses(self, breaker):
        portfolio = PortfolioState(
            cash=98000.0,
            total_value=98000.0,
            peak_value=100000.0,
            consecutive_losses=3,  # Equals threshold
        )
        assert breaker.check(portfolio) is False

    def test_blocks_on_trade_limit(self, breaker):
        portfolio = PortfolioState(
            cash=98000.0,
            total_value=98000.0,
            peak_value=100000.0,
            trades_today=10,  # Equals limit
        )
        assert breaker.check(portfolio) is False

    def test_manual_reset(self, breaker):
        portfolio = PortfolioState(
            cash=85000.0,
            total_value=85000.0,
            peak_value=100000.0,
            max_drawdown_pct=11.0,
        )
        breaker.check(portfolio)
        assert breaker.is_active

        breaker.reset()
        assert not breaker.is_active

    def test_kill_switch(self, breaker):
        breaker.kill_switch()
        assert breaker.is_active
        assert "KILL SWITCH" in breaker.status["reason"]

        # Kill switch requires manual reset
        portfolio = PortfolioState()
        assert breaker.check(portfolio) is False

    def test_trip_history(self, breaker):
        portfolio = PortfolioState(
            max_drawdown_pct=11.0,
            total_value=89000.0,
            peak_value=100000.0,
        )
        breaker.check(portfolio)
        assert breaker.status["trip_count"] == 1


class TestRiskConfig:
    def test_default_values(self):
        config = RiskConfig()
        assert config.max_portfolio_drawdown_pct == 10.0
        assert config.max_daily_loss_pct == 3.0
        assert config.max_per_trade_loss_pct == 2.0
        assert config.max_position_pct == 5.0
        assert config.max_trades_per_hour == 10
        assert config.consecutive_loss_threshold == 3
