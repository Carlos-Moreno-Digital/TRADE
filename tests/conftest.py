"""Shared test fixtures."""

import pytest

from trade.config import RiskConfig, TradingConfig, TechnicalConfig, AgentWeightsConfig
from trade.data.models import (
    AnalysisContext,
    AssetType,
    PortfolioState,
    Position,
    Signal,
    TradeAction,
    SignalStrength,
)


@pytest.fixture
def risk_config():
    return RiskConfig()


@pytest.fixture
def trading_config():
    return TradingConfig()


@pytest.fixture
def portfolio():
    return PortfolioState(
        cash=100000.0,
        total_value=100000.0,
        peak_value=100000.0,
    )


@pytest.fixture
def portfolio_with_positions():
    positions = [
        Position(
            symbol="AAPL",
            asset_type=AssetType.STOCK,
            side="long",
            quantity=10,
            entry_price=150.0,
            current_price=155.0,
        ),
        Position(
            symbol="BTC-USD",
            asset_type=AssetType.CRYPTO,
            side="long",
            quantity=0.5,
            entry_price=40000.0,
            current_price=42000.0,
        ),
    ]
    return PortfolioState(
        cash=75000.0,
        positions=positions,
        total_value=97550.0,
        peak_value=100000.0,
    )


@pytest.fixture
def analysis_context(portfolio):
    return AnalysisContext(
        symbol="AAPL",
        asset_type=AssetType.STOCK,
        portfolio=portfolio,
    )


@pytest.fixture
def buy_signal():
    return Signal(
        symbol="AAPL",
        asset_type=AssetType.STOCK,
        action=TradeAction.BUY,
        strength=SignalStrength.BUY,
        confidence=0.7,
        source_agent="test",
        reasoning="Test buy signal",
    )


@pytest.fixture
def sell_signal():
    return Signal(
        symbol="AAPL",
        asset_type=AssetType.STOCK,
        action=TradeAction.SELL,
        strength=SignalStrength.SELL,
        confidence=0.7,
        source_agent="test",
        reasoning="Test sell signal",
    )
