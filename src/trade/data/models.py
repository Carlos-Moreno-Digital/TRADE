"""Core data models for the trading system."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AssetType(str, Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"


class TradeAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    SHORT = "short"
    COVER = "cover"


class SignalStrength(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    NEUTRAL = "neutral"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class OHLCV(BaseModel):
    """Open-High-Low-Close-Volume data point."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""
    asset_type: AssetType = AssetType.STOCK


class Quote(BaseModel):
    """Real-time quote."""

    symbol: str
    asset_type: AssetType
    price: float
    bid: float | None = None
    ask: float | None = None
    volume: float | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    change_pct: float | None = None


class NewsItem(BaseModel):
    """A news article or social media post."""

    title: str
    source: str
    published: datetime | None = None
    url: str | None = None
    summary: str | None = None
    sentiment_score: float | None = None  # -1.0 to 1.0


class Signal(BaseModel):
    """Trading signal from an agent."""

    symbol: str
    asset_type: AssetType
    action: TradeAction
    strength: SignalStrength = SignalStrength.NEUTRAL
    confidence: float = 0.0  # 0.0 to 1.0
    source_agent: str = ""
    reasoning: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Order(BaseModel):
    """A trading order."""

    id: str = ""
    symbol: str
    asset_type: AssetType
    action: TradeAction
    quantity: float
    price: float | None = None  # None = market order
    order_type: str = "market"  # market | limit | stop
    stop_loss: float | None = None
    take_profit: float | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str = "pending"  # pending | filled | cancelled | rejected
    fill_price: float | None = None
    fill_timestamp: datetime | None = None


class Position(BaseModel):
    """An open position."""

    symbol: str
    asset_type: AssetType
    side: str  # long | short
    quantity: float
    entry_price: float
    current_price: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    opened_at: datetime = Field(default_factory=datetime.utcnow)
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    def update_price(self, price: float) -> None:
        """Update current price and recalculate P&L."""
        self.current_price = price
        if self.side == "long":
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity
        if self.entry_price > 0:
            self.unrealized_pnl_pct = (
                (price - self.entry_price) / self.entry_price * 100
                if self.side == "long"
                else (self.entry_price - price) / self.entry_price * 100
            )


class PortfolioState(BaseModel):
    """Current portfolio state."""

    cash: float = 100000.0
    positions: list[Position] = Field(default_factory=list)
    total_value: float = 100000.0
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    peak_value: float = 100000.0
    trades_today: int = 0
    consecutive_losses: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def update_drawdown(self) -> None:
        """Update peak value and max drawdown."""
        if self.total_value > self.peak_value:
            self.peak_value = self.total_value
        if self.peak_value > 0:
            current_dd = (self.peak_value - self.total_value) / self.peak_value * 100
            self.max_drawdown_pct = max(self.max_drawdown_pct, current_dd)


class AnalysisContext(BaseModel):
    """Context passed to each agent for analysis."""

    symbol: str
    asset_type: AssetType
    portfolio: PortfolioState
    signals: list[Signal] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
