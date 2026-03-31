"""Tests for data models."""

from trade.data.models import (
    AssetType,
    OHLCV,
    Quote,
    Signal,
    Order,
    Position,
    PortfolioState,
    TradeAction,
    SignalStrength,
    AnalysisContext,
)


class TestAssetType:
    def test_values(self):
        assert AssetType.STOCK == "stock"
        assert AssetType.CRYPTO == "crypto"
        assert AssetType.FOREX == "forex"


class TestPosition:
    def test_update_price_long(self):
        pos = Position(
            symbol="AAPL",
            asset_type=AssetType.STOCK,
            side="long",
            quantity=10,
            entry_price=150.0,
            current_price=150.0,
        )
        pos.update_price(160.0)
        assert pos.current_price == 160.0
        assert pos.unrealized_pnl == 100.0  # (160-150) * 10
        assert abs(pos.unrealized_pnl_pct - 6.6667) < 0.01

    def test_update_price_short(self):
        pos = Position(
            symbol="AAPL",
            asset_type=AssetType.STOCK,
            side="short",
            quantity=10,
            entry_price=150.0,
            current_price=150.0,
        )
        pos.update_price(140.0)
        assert pos.unrealized_pnl == 100.0  # (150-140) * 10

    def test_update_price_loss(self):
        pos = Position(
            symbol="AAPL",
            asset_type=AssetType.STOCK,
            side="long",
            quantity=10,
            entry_price=150.0,
            current_price=150.0,
        )
        pos.update_price(140.0)
        assert pos.unrealized_pnl == -100.0


class TestPortfolioState:
    def test_update_drawdown_new_peak(self):
        portfolio = PortfolioState(
            cash=110000.0,
            total_value=110000.0,
            peak_value=100000.0,
        )
        portfolio.update_drawdown()
        assert portfolio.peak_value == 110000.0
        assert portfolio.max_drawdown_pct == 0.0

    def test_update_drawdown_decline(self):
        portfolio = PortfolioState(
            cash=90000.0,
            total_value=90000.0,
            peak_value=100000.0,
        )
        portfolio.update_drawdown()
        assert portfolio.max_drawdown_pct == 10.0

    def test_initial_state(self):
        portfolio = PortfolioState()
        assert portfolio.cash == 100000.0
        assert portfolio.total_value == 100000.0
        assert portfolio.positions == []
        assert portfolio.consecutive_losses == 0


class TestSignal:
    def test_create_signal(self, buy_signal):
        assert buy_signal.symbol == "AAPL"
        assert buy_signal.action == TradeAction.BUY
        assert buy_signal.confidence == 0.7

    def test_signal_defaults(self):
        signal = Signal(
            symbol="AAPL",
            asset_type=AssetType.STOCK,
            action=TradeAction.HOLD,
        )
        assert signal.strength == SignalStrength.NEUTRAL
        assert signal.confidence == 0.0
        assert signal.source_agent == ""


class TestOrder:
    def test_create_market_order(self):
        order = Order(
            symbol="AAPL",
            asset_type=AssetType.STOCK,
            action=TradeAction.BUY,
            quantity=10,
        )
        assert order.order_type == "market"
        assert order.price is None
        assert order.status == "pending"

    def test_create_limit_order(self):
        order = Order(
            symbol="BTC-USD",
            asset_type=AssetType.CRYPTO,
            action=TradeAction.BUY,
            quantity=0.5,
            price=40000.0,
            order_type="limit",
            stop_loss=38000.0,
            take_profit=45000.0,
        )
        assert order.price == 40000.0
        assert order.stop_loss == 38000.0
        assert order.take_profit == 45000.0


class TestAnalysisContext:
    def test_create_context(self, analysis_context):
        assert analysis_context.symbol == "AAPL"
        assert analysis_context.asset_type == AssetType.STOCK
        assert analysis_context.signals == []
        assert analysis_context.metadata == {}
