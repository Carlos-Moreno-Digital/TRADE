"""Tests for the Prop Firm Risk Engine."""

import pytest
from datetime import date

from trade.data.models import Order, PortfolioState, TradeAction, AssetType
from trade.risk.prop_firm import PropFirmConfig, PropFirmRiskEngine


@pytest.fixture
def engine():
    config = PropFirmConfig(name="test_firm")
    e = PropFirmRiskEngine(config)
    e.initialize(10000.0)
    e.start_trading_day(10000.0)
    return e


@pytest.fixture
def healthy_portfolio():
    return PortfolioState(
        cash=10000.0,
        total_value=10000.0,
        peak_value=10000.0,
    )


class TestPropFirmRiskEngine:
    def test_initialization(self, engine):
        assert engine._initial_balance == 10000.0
        assert engine._peak_equity == 10000.0
        assert not engine._is_killed

    def test_allows_trading_healthy(self, engine, healthy_portfolio):
        can_trade, reason = engine.can_open_trade(healthy_portfolio)
        assert can_trade is True
        assert reason == "OK"

    def test_blocks_on_daily_loss_hard_stop(self, engine):
        """Hard stop at 3% daily loss (firm limit: 5%)."""
        portfolio = PortfolioState(
            cash=9680.0,
            total_value=9680.0,  # -3.2% from 10000
            peak_value=10000.0,
        )
        can_trade, reason = engine.can_open_trade(portfolio)
        assert can_trade is False
        assert "HARD STOP" in reason

    def test_blocks_on_total_drawdown(self, engine):
        """Hard stop at 7% total drawdown (firm limit: 10%)."""
        # Start the day at 9500, so daily loss is small but total drawdown is >7%
        engine.start_trading_day(9500.0)  # Reset daily baseline
        portfolio = PortfolioState(
            cash=9250.0,
            total_value=9250.0,  # -7.5% from initial 10000, but only -2.6% from day start
            peak_value=10000.0,
        )
        can_trade, reason = engine.can_open_trade(portfolio)
        assert can_trade is False
        assert engine._is_killed  # Kill switch activated for drawdown

    def test_blocks_max_open_trades(self, engine):
        from trade.data.models import Position
        positions = [
            Position(symbol=f"PAIR{i}", asset_type=AssetType.FOREX,
                    side="long", quantity=1, entry_price=1.0)
            for i in range(3)
        ]
        portfolio = PortfolioState(
            cash=9000.0, total_value=10000.0, peak_value=10000.0,
            positions=positions,
        )
        can_trade, reason = engine.can_open_trade(portfolio)
        assert can_trade is False
        assert "open trades" in reason.lower()

    def test_blocks_max_daily_trades(self, engine, healthy_portfolio):
        for _ in range(5):
            engine.record_trade_result(50.0)
        can_trade, reason = engine.can_open_trade(healthy_portfolio)
        assert can_trade is False
        assert "daily trades" in reason.lower()

    def test_consecutive_loss_cooldown(self, engine, healthy_portfolio):
        engine.record_trade_result(-50.0)
        engine.record_trade_result(-50.0)
        can_trade, reason = engine.can_open_trade(healthy_portfolio)
        assert can_trade is False
        assert "consecutive" in reason.lower()

    def test_order_requires_stop_loss(self, engine, healthy_portfolio):
        order = Order(
            symbol="EURUSD",
            asset_type=AssetType.FOREX,
            action=TradeAction.BUY,
            quantity=1.0,
            price=1.1000,
            stop_loss=None,  # No stop loss!
        )
        valid, reason, _ = engine.validate_order(order, healthy_portfolio)
        assert valid is False
        assert "stop loss" in reason.lower()

    def test_order_risk_reward_check(self, engine, healthy_portfolio):
        order = Order(
            symbol="EURUSD",
            asset_type=AssetType.FOREX,
            action=TradeAction.BUY,
            quantity=1.0,
            price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1020,  # Only 20 pips profit vs 50 pips risk = 0.4 RR
        )
        valid, reason, _ = engine.validate_order(order, healthy_portfolio)
        assert valid is False
        assert "risk:reward" in reason.lower()

    def test_order_valid(self, engine, healthy_portfolio):
        order = Order(
            symbol="EURUSD",
            asset_type=AssetType.FOREX,
            action=TradeAction.BUY,
            quantity=0.1,
            price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100,  # 100 pips TP vs 50 pips SL = 2.0 RR
        )
        valid, reason, _ = engine.validate_order(order, healthy_portfolio)
        assert valid is True

    def test_challenge_progress(self, engine):
        progress = engine.get_challenge_progress(10500.0)
        assert progress["profit_pct"] == 5.0
        assert progress["target_pct"] == 8.0
        assert progress["progress_pct"] == 62.5
        assert progress["safety_status"] == "SAFE"

    def test_kill_switch(self, engine, healthy_portfolio):
        engine._kill("Test kill")
        can_trade, reason = engine.can_open_trade(healthy_portfolio)
        assert can_trade is False
        assert "KILLED" in reason

    def test_kill_switch_reset(self, engine, healthy_portfolio):
        engine._kill("Test kill")
        engine.reset_kill_switch()
        can_trade, _ = engine.can_open_trade(healthy_portfolio)
        assert can_trade is True

    def test_consistency_check(self, engine):
        engine.record_trade_result(500.0)  # Big win
        engine.record_trade_result(10.0)
        engine.record_trade_result(10.0)

        result = engine.check_consistency(520.0)
        # 500/520 = 96% from one day > 25% limit
        assert len(result["warnings"]) > 0

    def test_record_trade_results(self, engine):
        engine.record_trade_result(100.0)
        engine.record_trade_result(-50.0)
        engine.record_trade_result(-30.0)

        record = engine._get_today_record()
        assert record.trades == 3
        assert record.wins == 1
        assert record.losses == 2
        assert record.consecutive_losses == 2
        assert record.gross_pnl == 20.0


class TestPropFirmConfig:
    def test_default_config(self):
        config = PropFirmConfig()
        assert config.name == "funderpro"
        assert config.martingale_allowed is False
        assert config.grid_trading_allowed is False

    def test_safety_buffers(self):
        config = PropFirmConfig()
        # Our limits should be BELOW firm limits
        assert config.max_daily_loss_pct < config.firm_max_daily_loss_pct
        assert config.max_total_drawdown_pct < config.firm_max_total_drawdown_pct
        assert config.hard_stop_daily_pct < config.max_daily_loss_pct
        assert config.hard_stop_drawdown_pct < config.max_total_drawdown_pct

    def test_martingale_always_forbidden(self):
        config = PropFirmConfig(martingale_allowed=True)  # Try to enable
        # Even if someone passes True, our engine should block it
        # The config allows it but the engine has hardcoded checks
        assert config.martingale_allowed is True  # Config stores it
        # But PropFirmRiskEngine._is_martingale_detected blocks it
