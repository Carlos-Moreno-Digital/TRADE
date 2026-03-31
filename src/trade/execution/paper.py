"""Paper trading engine - Simulates real trading without risking capital."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger

from trade.data.models import Order, Position, PortfolioState, TradeAction


class PaperTradingEngine:
    """Simulated trading engine for paper trading mode."""

    def __init__(self, initial_capital: float = 100000.0):
        self.portfolio = PortfolioState(
            cash=initial_capital,
            total_value=initial_capital,
            peak_value=initial_capital,
        )
        self.order_history: list[Order] = []
        self.trade_log: list[dict[str, Any]] = []

    def execute_order(self, order: Order) -> bool:
        """Execute an order in paper trading mode."""
        if order.price is None or order.price <= 0:
            logger.warning(f"Invalid order price: {order.price}")
            order.status = "rejected"
            return False

        order_value = order.quantity * order.price

        if order.action == TradeAction.BUY:
            return self._execute_buy(order, order_value)
        elif order.action in (TradeAction.SELL, TradeAction.COVER):
            return self._execute_sell(order)
        elif order.action == TradeAction.SHORT:
            return self._execute_short(order, order_value)

        return False

    def _execute_buy(self, order: Order, order_value: float) -> bool:
        """Execute a buy order."""
        if order_value > self.portfolio.cash:
            logger.warning(f"Insufficient cash for buy: ${self.portfolio.cash:.2f} < ${order_value:.2f}")
            order.status = "rejected"
            return False

        self.portfolio.cash -= order_value
        self.portfolio.positions.append(
            Position(
                symbol=order.symbol,
                asset_type=order.asset_type,
                side="long",
                quantity=order.quantity,
                entry_price=order.price,
                current_price=order.price,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
            )
        )

        order.status = "filled"
        order.fill_price = order.price
        order.fill_timestamp = datetime.utcnow()
        self.order_history.append(order)
        self.portfolio.trades_today += 1

        self._log_trade("BUY", order)
        return True

    def _execute_sell(self, order: Order) -> bool:
        """Execute a sell order (close long position)."""
        for i, pos in enumerate(self.portfolio.positions):
            if pos.symbol == order.symbol and pos.side == "long":
                sell_value = order.quantity * order.price
                pnl = (order.price - pos.entry_price) * order.quantity

                self.portfolio.cash += sell_value
                self.portfolio.positions.pop(i)

                if pnl < 0:
                    self.portfolio.consecutive_losses += 1
                else:
                    self.portfolio.consecutive_losses = 0

                self.portfolio.daily_pnl += pnl
                self.portfolio.total_pnl += pnl

                order.status = "filled"
                order.fill_price = order.price
                order.fill_timestamp = datetime.utcnow()
                self.order_history.append(order)
                self.portfolio.trades_today += 1

                self._log_trade("SELL", order, pnl=pnl)
                return True

        logger.warning(f"No long position found for {order.symbol}")
        order.status = "rejected"
        return False

    def _execute_short(self, order: Order, order_value: float) -> bool:
        """Execute a short sell order."""
        # Simplified: require margin equal to position value
        if order_value > self.portfolio.cash:
            order.status = "rejected"
            return False

        self.portfolio.positions.append(
            Position(
                symbol=order.symbol,
                asset_type=order.asset_type,
                side="short",
                quantity=order.quantity,
                entry_price=order.price,
                current_price=order.price,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
            )
        )

        order.status = "filled"
        order.fill_price = order.price
        self.order_history.append(order)
        self.portfolio.trades_today += 1

        self._log_trade("SHORT", order)
        return True

    def update_positions(self, prices: dict[str, float]) -> None:
        """Update all position prices and P&L."""
        for pos in self.portfolio.positions:
            if pos.symbol in prices:
                pos.update_price(prices[pos.symbol])

        # Update total portfolio value
        position_value = sum(
            pos.quantity * pos.current_price for pos in self.portfolio.positions
        )
        self.portfolio.total_value = self.portfolio.cash + position_value
        self.portfolio.update_drawdown()

        # Check stop losses and take profits
        self._check_exit_conditions(prices)

    def _check_exit_conditions(self, prices: dict[str, float]) -> None:
        """Check stop loss and take profit for all positions."""
        positions_to_close = []

        for pos in self.portfolio.positions:
            price = prices.get(pos.symbol, pos.current_price)

            if pos.side == "long":
                if pos.stop_loss and price <= pos.stop_loss:
                    positions_to_close.append((pos, "stop_loss", price))
                elif pos.take_profit and price >= pos.take_profit:
                    positions_to_close.append((pos, "take_profit", price))
            elif pos.side == "short":
                if pos.stop_loss and price >= pos.stop_loss:
                    positions_to_close.append((pos, "stop_loss", price))
                elif pos.take_profit and price <= pos.take_profit:
                    positions_to_close.append((pos, "take_profit", price))

        for pos, reason, price in positions_to_close:
            order = Order(
                symbol=pos.symbol,
                asset_type=pos.asset_type,
                action=TradeAction.SELL if pos.side == "long" else TradeAction.COVER,
                quantity=pos.quantity,
                price=price,
            )
            logger.info(f"Auto-closing {pos.symbol} ({reason}) @ {price:.2f}")
            self._execute_sell(order)

    def _log_trade(self, action: str, order: Order, pnl: float | None = None) -> None:
        """Log trade for history."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "symbol": order.symbol,
            "quantity": order.quantity,
            "price": order.price,
            "stop_loss": order.stop_loss,
            "take_profit": order.take_profit,
            "pnl": pnl,
        }
        self.trade_log.append(entry)
        logger.info(
            f"[PAPER TRADE] {action} {order.quantity:.2f} {order.symbol} "
            f"@ ${order.price:.2f}" + (f" | P&L: ${pnl:.2f}" if pnl else "")
        )

    def get_summary(self) -> dict[str, Any]:
        """Get trading session summary."""
        winning_trades = [t for t in self.trade_log if (t.get("pnl") or 0) > 0]
        losing_trades = [t for t in self.trade_log if (t.get("pnl") or 0) < 0]

        return {
            "total_trades": len(self.order_history),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / max(1, len(winning_trades) + len(losing_trades)),
            "total_pnl": round(self.portfolio.total_pnl, 2),
            "total_pnl_pct": round(self.portfolio.total_pnl_pct, 2),
            "max_drawdown_pct": round(self.portfolio.max_drawdown_pct, 2),
            "portfolio_value": round(self.portfolio.total_value, 2),
            "cash": round(self.portfolio.cash, 2),
            "open_positions": len(self.portfolio.positions),
        }
