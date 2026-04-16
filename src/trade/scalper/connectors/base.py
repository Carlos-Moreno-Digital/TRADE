"""Base connector interface — all brokers implement this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Tick:
    timestamp: datetime
    bid: float
    ask: float
    volume: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Position:
    id: str
    symbol: str
    side: Side
    entry_price: float
    quantity: float
    sl: float | None = None
    tp: float | None = None
    open_time: datetime | None = None
    pnl: float = 0.0


@dataclass
class OrderResult:
    order_id: str
    status: OrderStatus
    fill_price: float = 0.0
    message: str = ""


@dataclass
class AccountInfo:
    balance: float
    equity: float
    margin_used: float
    margin_free: float
    currency: str = "USD"


class BaseConnector(ABC):
    """Every broker connector implements this interface.

    The scalper engine calls ONLY these methods — it never touches
    broker-specific internals. This lets us swap TradeLocker for
    Paper or MT5 without changing the engine.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Authenticate and establish connection."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Clean shutdown."""
        ...

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """Current account state."""
        ...

    @abstractmethod
    def get_tick(self, symbol: str) -> Tick | None:
        """Latest bid/ask for a symbol."""
        ...

    @abstractmethod
    def get_bars(self, symbol: str, timeframe: str, count: int) -> list[Bar]:
        """Recent OHLCV bars. timeframe: '1m', '5m', '15m', '1h'."""
        ...

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """All open positions."""
        ...

    @abstractmethod
    def open_position(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        sl: float | None = None,
        tp: float | None = None,
        order_type: OrderType = OrderType.MARKET,
    ) -> OrderResult:
        """Open a new position."""
        ...

    @abstractmethod
    def close_position(self, position_id: str) -> OrderResult:
        """Close a specific position by ID."""
        ...

    @abstractmethod
    def close_all_positions(self) -> int:
        """Close everything. Returns count of positions closed."""
        ...

    @abstractmethod
    def modify_position(
        self,
        position_id: str,
        sl: float | None = None,
        tp: float | None = None,
    ) -> bool:
        """Modify SL/TP of an open position."""
        ...
