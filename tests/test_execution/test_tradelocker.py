"""Tests for the TradeLocker broker integration.

These tests verify the broker logic WITHOUT connecting to a real account.
"""

import pytest
from unittest.mock import MagicMock, patch

from trade.data.models import (
    AssetType,
    Order,
    PortfolioState,
    TradeAction,
)
from trade.execution.tradelocker import TradeLockerBroker, SYMBOL_MAP
from trade.risk.prop_firm import PropFirmRiskEngine, PropFirmConfig


class TestSymbolMapping:
    def test_forex_mapping(self):
        assert SYMBOL_MAP["EURUSD"] == "EURUSD"
        assert SYMBOL_MAP["GBPUSD"] == "GBPUSD"

    def test_yfinance_conversion(self):
        assert SYMBOL_MAP["EURUSD=X"] == "EURUSD"
        assert SYMBOL_MAP["GBPUSD=X"] == "GBPUSD"
        assert SYMBOL_MAP["BTC-USD"] == "BTCUSD"

    def test_commodities(self):
        assert SYMBOL_MAP["XAUUSD"] == "XAUUSD"

    def test_indices(self):
        assert SYMBOL_MAP["US30"] == "US30"
        assert SYMBOL_MAP["NAS100"] == "NAS100"


class TestTradeLockerBroker:
    def test_init_demo(self):
        broker = TradeLockerBroker(demo=True)
        assert "demo" in broker.environment
        assert broker.demo is True
        assert broker._connected is False

    def test_init_with_credentials(self):
        broker = TradeLockerBroker(
            environment="https://test.tradelocker.com",
            username="test@test.com",
            password="password",
            server="TestServer",
        )
        assert broker.username == "test@test.com"
        assert broker.server == "TestServer"

    def test_map_side_buy(self):
        assert TradeLockerBroker._map_side(TradeAction.BUY) == "buy"

    def test_map_side_sell(self):
        assert TradeLockerBroker._map_side(TradeAction.SELL) == "sell"

    def test_map_side_short(self):
        assert TradeLockerBroker._map_side(TradeAction.SHORT) == "sell"

    def test_map_side_cover(self):
        assert TradeLockerBroker._map_side(TradeAction.COVER) == "buy"

    def test_map_side_hold(self):
        assert TradeLockerBroker._map_side(TradeAction.HOLD) is None

    def test_extract_balance_dict(self):
        state = {"balance": 10000.0, "equity": 10500.0}
        assert TradeLockerBroker._extract_balance(state) == 10000.0

    def test_extract_equity_dict(self):
        state = {"balance": 10000.0, "equity": 10500.0}
        assert TradeLockerBroker._extract_equity(state) == 10500.0

    def test_detect_asset_type(self):
        assert TradeLockerBroker._detect_asset_type("EURUSD") == AssetType.FOREX
        assert TradeLockerBroker._detect_asset_type("XAUUSD") == AssetType.FOREX
        assert TradeLockerBroker._detect_asset_type("US30") == AssetType.STOCK
        assert TradeLockerBroker._detect_asset_type("BTCUSD") == AssetType.CRYPTO

    def test_connect_fails_without_credentials(self):
        broker = TradeLockerBroker(username="", password="")
        result = broker.connect()
        assert result is False

    def test_disconnect(self):
        broker = TradeLockerBroker()
        broker._connected = True
        broker._tl = MagicMock()
        broker.disconnect()
        assert broker._connected is False
        assert broker._tl is None


class TestTradeLockerWithRiskEngine:
    def test_order_blocked_by_risk(self):
        """Risk engine should block orders when limits are breached."""
        config = PropFirmConfig(name="test")
        risk = PropFirmRiskEngine(config)
        risk.initialize(10000.0)
        risk.start_trading_day(10000.0)

        # Simulate losses to trigger cooldown
        risk.record_trade_result(-50.0)
        risk.record_trade_result(-50.0)

        broker = TradeLockerBroker(risk_engine=risk)
        broker._connected = True
        broker._tl = MagicMock()

        # Mock portfolio
        broker.get_portfolio_state = MagicMock(return_value=PortfolioState(
            cash=9900, total_value=9900, peak_value=10000,
        ))

        order = Order(
            symbol="EURUSD",
            asset_type=AssetType.FOREX,
            action=TradeAction.BUY,
            quantity=0.1,
            price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100,
        )

        result = broker.submit_order(order)
        assert result.status == "rejected"

    def test_order_without_stop_loss_rejected(self):
        """Orders without stop loss must be rejected."""
        config = PropFirmConfig(name="test")
        risk = PropFirmRiskEngine(config)
        risk.initialize(10000.0)
        risk.start_trading_day(10000.0)

        broker = TradeLockerBroker(risk_engine=risk)
        broker._connected = True
        broker._tl = MagicMock()

        broker.get_portfolio_state = MagicMock(return_value=PortfolioState(
            cash=10000, total_value=10000, peak_value=10000,
        ))

        order = Order(
            symbol="EURUSD",
            asset_type=AssetType.FOREX,
            action=TradeAction.BUY,
            quantity=0.1,
            price=1.1000,
            stop_loss=None,  # No stop loss!
        )

        result = broker.submit_order(order)
        assert result.status == "rejected"
