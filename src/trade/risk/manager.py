"""Risk Manager - Integrates all risk management components."""

from __future__ import annotations

from typing import Any

from loguru import logger

from trade.config import RiskConfig
from trade.data.models import Order, PortfolioState
from trade.risk.circuit_breaker import CircuitBreaker


class RiskManager:
    """Central risk management system."""

    def __init__(self, config: RiskConfig):
        self.config = config
        self.circuit_breaker = CircuitBreaker(config)

    def can_trade(self, portfolio: PortfolioState) -> bool:
        """Check if trading is currently allowed."""
        return self.circuit_breaker.check(portfolio)

    def validate_order(self, order: Order, portfolio: PortfolioState) -> tuple[bool, str]:
        """Validate an order against risk rules.

        Returns:
            Tuple of (is_valid, reason).
        """
        # Check circuit breaker first
        if not self.can_trade(portfolio):
            return False, "Circuit breaker active"

        # Check position size
        order_value = order.quantity * (order.price or 0)
        if portfolio.total_value > 0:
            position_pct = order_value / portfolio.total_value * 100
            if position_pct > self.config.max_position_pct:
                return False, (
                    f"Position too large: {position_pct:.1f}% > {self.config.max_position_pct}%"
                )

        # Check cash availability for buys
        if order.action.value == "buy" and order_value > portfolio.cash:
            return False, f"Insufficient cash: ${portfolio.cash:.2f} < ${order_value:.2f}"

        # Check per-trade risk
        if order.stop_loss and order.price:
            trade_risk = abs(order.price - order.stop_loss) * order.quantity
            max_risk = portfolio.total_value * (self.config.max_per_trade_loss_pct / 100)
            if trade_risk > max_risk:
                return False, f"Trade risk too high: ${trade_risk:.2f} > ${max_risk:.2f}"

        # Check open positions limit
        if len(portfolio.positions) >= self.config.max_open_positions:
            return False, f"Max positions reached: {len(portfolio.positions)}"

        return True, "Order validated"

    def get_status(self, portfolio: PortfolioState) -> dict[str, Any]:
        """Get comprehensive risk status."""
        return {
            "circuit_breaker": self.circuit_breaker.status,
            "can_trade": self.can_trade(portfolio),
            "portfolio_drawdown_pct": round(portfolio.max_drawdown_pct, 2),
            "daily_pnl": round(portfolio.daily_pnl, 2),
            "open_positions": len(portfolio.positions),
            "consecutive_losses": portfolio.consecutive_losses,
            "trades_today": portfolio.trades_today,
            "limits": {
                "max_drawdown_pct": self.config.max_portfolio_drawdown_pct,
                "max_daily_loss_pct": self.config.max_daily_loss_pct,
                "max_per_trade_loss_pct": self.config.max_per_trade_loss_pct,
                "max_position_pct": self.config.max_position_pct,
                "max_trades_per_hour": self.config.max_trades_per_hour,
                "max_open_positions": self.config.max_open_positions,
            },
        }
