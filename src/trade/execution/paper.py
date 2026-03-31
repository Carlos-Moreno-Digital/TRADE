"""Paper trading engine - Simulates real trading with slippage and spreads."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger

from trade.data.models import Order, Position, PortfolioState, TradeAction


# Realistic spread defaults (in price units, not pips)
DEFAULT_SPREADS = {
    # Forex (in price units - e.g., 0.00015 = 1.5 pips for EURUSD)
    "EURUSD": 0.00015, "GBPUSD": 0.00020, "USDJPY": 0.015,
    "USDCHF": 0.00018, "AUDUSD": 0.00018, "NZDUSD": 0.00020,
    "USDCAD": 0.00020, "EURGBP": 0.00020, "EURJPY": 0.020,
    "GBPJPY": 0.025,
    # Indices (in points)
    "US30": 3.0, "NAS100": 2.0, "SPX500": 0.5, "GER40": 2.0,
    # Commodities
    "XAUUSD": 0.30, "XAGUSD": 0.03, "USOIL": 0.05,
    # Crypto
    "BTCUSD": 50.0, "ETHUSD": 2.0,
    # yfinance format
    "EURUSD=X": 0.00015, "GBPUSD=X": 0.00020, "USDJPY=X": 0.015,
    "AUDUSD=X": 0.00018, "BTC-USD": 50.0, "ETH-USD": 2.0,
}

# Default slippage as fraction of spread
DEFAULT_SLIPPAGE_FRACTION = 0.5  # 50% of spread as slippage


class PaperTradingEngine:
    """Simulated trading engine with realistic slippage and spread modeling."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        spreads: dict[str, float] | None = None,
        slippage_fraction: float = DEFAULT_SLIPPAGE_FRACTION,
    ):
        self.portfolio = PortfolioState()
        self.portfolio.initialize(initial_capital)
        self.spreads = spreads or DEFAULT_SPREADS
        self.slippage_fraction = slippage_fraction
        self.order_history: list[Order] = []
        self.trade_log: list[dict[str, Any]] = []

    def get_spread(self, symbol: str) -> float:
        """Get the spread for a symbol."""
        return self.spreads.get(symbol, 0.0002)  # Default 2 pips

    def get_fill_price(self, symbol: str, side: str, target_price: float) -> float:
        """Calculate realistic fill price with spread and slippage.

        BUY fills at ASK (higher): price + spread/2 + slippage
        SELL fills at BID (lower): price - spread/2 - slippage
        """
        spread = self.get_spread(symbol)
        slippage = spread * self.slippage_fraction

        if side == "buy":
            return target_price + spread / 2 + slippage
        else:  # sell
            return target_price - spread / 2 - slippage

    def execute_order(self, order: Order) -> bool:
        """Execute an order with realistic fills."""
        if order.price is None or order.price <= 0:
            logger.warning(f"Invalid order price: {order.price}")
            order.status = "rejected"
            return False

        # Apply slippage to fill price
        side = "buy" if order.action == TradeAction.BUY else "sell"
        fill_price = self.get_fill_price(order.symbol, side, order.price)
        order_value = order.quantity * fill_price

        if order.action == TradeAction.BUY:
            return self._execute_buy(order, order_value, fill_price)
        elif order.action in (TradeAction.SELL, TradeAction.COVER):
            return self._execute_sell(order, fill_price)
        elif order.action == TradeAction.SHORT:
            return self._execute_short(order, order_value, fill_price)

        return False

    def _execute_buy(self, order: Order, order_value: float, fill_price: float) -> bool:
        if order_value > self.portfolio.cash:
            logger.warning(f"Insufficient cash: ${self.portfolio.cash:.2f} < ${order_value:.2f}")
            order.status = "rejected"
            return False

        self.portfolio.cash -= order_value
        self.portfolio.positions.append(
            Position(
                symbol=order.symbol,
                asset_type=order.asset_type,
                side="long",
                quantity=order.quantity,
                entry_price=fill_price,
                current_price=fill_price,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
            )
        )

        order.status = "filled"
        order.fill_price = fill_price
        order.fill_timestamp = datetime.utcnow()
        self.order_history.append(order)

        slippage_cost = abs(fill_price - order.price) * order.quantity
        self._log_trade("BUY", order, fill_price=fill_price, slippage_cost=slippage_cost)
        return True

    def _execute_sell(self, order: Order, fill_price: float) -> bool:
        for i, pos in enumerate(self.portfolio.positions):
            if pos.symbol == order.symbol and pos.side == "long":
                pnl = (fill_price - pos.entry_price) * pos.quantity
                self.portfolio.cash += order.quantity * fill_price

                self.portfolio.positions.pop(i)
                self.portfolio.record_trade(pnl)

                order.status = "filled"
                order.fill_price = fill_price
                order.fill_timestamp = datetime.utcnow()
                self.order_history.append(order)

                slippage_cost = abs(fill_price - order.price) * order.quantity
                self._log_trade("SELL", order, pnl=pnl, fill_price=fill_price, slippage_cost=slippage_cost)
                return True

        logger.warning(f"No long position found for {order.symbol}")
        order.status = "rejected"
        return False

    def _execute_short(self, order: Order, order_value: float, fill_price: float) -> bool:
        if order_value > self.portfolio.cash:
            order.status = "rejected"
            return False

        self.portfolio.positions.append(
            Position(
                symbol=order.symbol,
                asset_type=order.asset_type,
                side="short",
                quantity=order.quantity,
                entry_price=fill_price,
                current_price=fill_price,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
            )
        )

        order.status = "filled"
        order.fill_price = fill_price
        self.order_history.append(order)
        self._log_trade("SHORT", order, fill_price=fill_price)
        return True

    def update_positions(self, prices: dict[str, float]) -> None:
        """Update all position prices and check exit conditions."""
        for pos in self.portfolio.positions:
            if pos.symbol in prices:
                pos.update_price(prices[pos.symbol])

        position_value = sum(
            pos.quantity * pos.current_price for pos in self.portfolio.positions
        )
        self.portfolio.total_value = self.portfolio.cash + position_value
        self.portfolio.update_drawdown()

        self._check_exit_conditions(prices)

    def _check_exit_conditions(self, prices: dict[str, float]) -> None:
        """Check SL/TP with slippage applied to exit fills."""
        positions_to_close = []

        for pos in self.portfolio.positions:
            price = prices.get(pos.symbol, pos.current_price)

            if pos.side == "long":
                if pos.stop_loss and price <= pos.stop_loss:
                    # SL hit - fill with slippage (sell side)
                    exit_price = self.get_fill_price(pos.symbol, "sell", pos.stop_loss)
                    positions_to_close.append((pos, "stop_loss", exit_price))
                elif pos.take_profit and price >= pos.take_profit:
                    exit_price = self.get_fill_price(pos.symbol, "sell", pos.take_profit)
                    positions_to_close.append((pos, "take_profit", exit_price))
            elif pos.side == "short":
                if pos.stop_loss and price >= pos.stop_loss:
                    exit_price = self.get_fill_price(pos.symbol, "buy", pos.stop_loss)
                    positions_to_close.append((pos, "stop_loss", exit_price))
                elif pos.take_profit and price <= pos.take_profit:
                    exit_price = self.get_fill_price(pos.symbol, "buy", pos.take_profit)
                    positions_to_close.append((pos, "take_profit", exit_price))

        for pos, reason, exit_price in positions_to_close:
            order = Order(
                symbol=pos.symbol,
                asset_type=pos.asset_type,
                action=TradeAction.SELL if pos.side == "long" else TradeAction.COVER,
                quantity=pos.quantity,
                price=exit_price,
            )
            logger.info(f"Auto-closing {pos.symbol} ({reason}) @ {exit_price:.5f}")
            self._execute_sell(order, exit_price)

    def _log_trade(
        self, action: str, order: Order, pnl: float | None = None,
        fill_price: float | None = None, slippage_cost: float = 0,
    ) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "symbol": order.symbol,
            "quantity": order.quantity,
            "target_price": order.price,
            "fill_price": fill_price or order.price,
            "slippage_cost": round(slippage_cost, 4),
            "stop_loss": order.stop_loss,
            "take_profit": order.take_profit,
            "pnl": pnl,
        }
        self.trade_log.append(entry)
        logger.info(
            f"[PAPER] {action} {order.quantity:.2f} {order.symbol} "
            f"@ ${fill_price or order.price:.5f} "
            f"(slippage: ${slippage_cost:.4f})"
            + (f" | P&L: ${pnl:.2f}" if pnl else "")
        )

    def get_summary(self) -> dict[str, Any]:
        winning = [t for t in self.trade_log if (t.get("pnl") or 0) > 0]
        losing = [t for t in self.trade_log if (t.get("pnl") or 0) < 0]
        total_slippage = sum(t.get("slippage_cost", 0) for t in self.trade_log)

        return {
            "total_trades": len(self.order_history),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / max(1, len(winning) + len(losing)),
            "total_pnl": round(self.portfolio.total_pnl, 2),
            "total_slippage_cost": round(total_slippage, 2),
            "max_drawdown_pct": round(self.portfolio.max_drawdown_pct, 2),
            "portfolio_value": round(self.portfolio.total_value, 2),
            "cash": round(self.portfolio.cash, 2),
            "open_positions": len(self.portfolio.positions),
        }
