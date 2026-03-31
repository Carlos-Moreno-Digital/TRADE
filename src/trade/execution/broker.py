"""Broker interface - Abstract base for live trading connections."""

from __future__ import annotations

from abc import ABC, abstractmethod

from trade.data.models import Order, Position


class BaseBroker(ABC):
    """Abstract interface for broker connections.

    Implement this for live trading with specific brokers
    (Alpaca, Interactive Brokers, Binance, etc.)
    """

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the broker."""
        ...

    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        """Submit an order to the broker."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        ...

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Get all current positions."""
        ...

    @abstractmethod
    def get_account_balance(self) -> float:
        """Get current account balance."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Close the broker connection."""
        ...
