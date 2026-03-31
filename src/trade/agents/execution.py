"""Execution Agent - Handles order creation and execution (paper/live)."""

from __future__ import annotations

import time
import uuid
from typing import Any

from loguru import logger

from trade.agents.base import BaseAgent, AgentSignal
from trade.data.models import (
    AnalysisContext,
    Order,
    Position,
    TradeAction,
    SignalStrength,
)


class ExecutionAgent(BaseAgent):
    """Executes trades based on portfolio manager decisions. Paper trading by default."""

    name = "execution"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.mode = self.config.get("mode", "paper")
        self._paper_orders: list[Order] = []

    def analyze(self, context: AnalysisContext) -> AgentSignal:
        """Execute or simulate the trade decision."""
        start = time.time()

        # Get the portfolio manager's decision (last signal)
        decision = context.signals[-1] if context.signals else None
        if decision is None or decision.action == TradeAction.HOLD:
            signal = self._make_signal(
                context, TradeAction.HOLD, SignalStrength.NEUTRAL, 1.0,
                reasoning="No trade to execute (HOLD)",
            )
            return self._make_output(signal, {"executed": False})

        # Calculate position size
        latest_price = context.metadata.get("latest_close", 0)
        if latest_price <= 0:
            signal = self._make_signal(
                context, TradeAction.HOLD, SignalStrength.NEUTRAL, 1.0,
                reasoning="Cannot execute: no price data",
            )
            return self._make_output(signal, {"executed": False, "reason": "no_price"})

        # Position sizing: risk-based
        portfolio = context.portfolio
        risk_per_trade_pct = self.config.get("risk_per_trade_pct", 1.0)
        max_position_pct = self.config.get("max_position_pct", 5.0)

        # Calculate quantity based on risk
        risk_amount = portfolio.total_value * (risk_per_trade_pct / 100)
        volatility = context.metadata.get("volatility", 2.0)
        atr_estimate = latest_price * (volatility / 100)

        if atr_estimate > 0:
            quantity = risk_amount / atr_estimate
        else:
            quantity = risk_amount / (latest_price * 0.02)  # Default 2% stop

        # Cap at max position size
        max_value = portfolio.total_value * (max_position_pct / 100)
        max_quantity = max_value / latest_price
        quantity = min(quantity, max_quantity)

        # For crypto, allow fractional. For stocks, round to whole shares
        if context.asset_type.value != "crypto":
            quantity = max(1, int(quantity))

        # Calculate stop loss and take profit
        stop_distance = atr_estimate * 1.5
        take_profit_distance = atr_estimate * 3.0

        if decision.action == TradeAction.BUY:
            stop_loss = latest_price - stop_distance
            take_profit = latest_price + take_profit_distance
        elif decision.action == TradeAction.SELL:
            stop_loss = latest_price + stop_distance
            take_profit = latest_price - take_profit_distance
        else:
            stop_loss = None
            take_profit = None

        # Create order
        order = Order(
            id=str(uuid.uuid4())[:8],
            symbol=context.symbol,
            asset_type=context.asset_type,
            action=decision.action,
            quantity=quantity,
            price=latest_price,
            stop_loss=round(stop_loss, 2) if stop_loss else None,
            take_profit=round(take_profit, 2) if take_profit else None,
        )

        # Execute based on mode
        if self.mode == "paper":
            executed = self._paper_execute(order, context)
        else:
            self._logger.error("Live trading not yet implemented!")
            executed = False

        raw_data = {
            "executed": executed,
            "mode": self.mode,
            "order": order.model_dump(),
            "position_value": round(quantity * latest_price, 2),
            "risk_amount": round(risk_amount, 2),
        }

        action = decision.action if executed else TradeAction.HOLD
        signal = self._make_signal(
            context, action, decision.strength, decision.confidence,
            reasoning=f"{'EXECUTED' if executed else 'NOT EXECUTED'} [{self.mode}] "
                     f"{decision.action.value} {quantity:.2f} {context.symbol} @ {latest_price:.2f} | "
                     f"SL: {stop_loss:.2f if stop_loss else 'N/A'} | "
                     f"TP: {take_profit:.2f if take_profit else 'N/A'}",
        )

        exec_time = (time.time() - start) * 1000
        return self._make_output(signal, raw_data, exec_time=exec_time)

    def _paper_execute(self, order: Order, context: AnalysisContext) -> bool:
        """Simulate order execution in paper trading mode."""
        self._logger.info(
            f"[PAPER] Executing: {order.action.value} {order.quantity:.2f} "
            f"{order.symbol} @ {order.price:.2f}"
        )

        order.status = "filled"
        order.fill_price = order.price
        self._paper_orders.append(order)

        # Update portfolio
        portfolio = context.portfolio
        cost = order.quantity * order.price

        if order.action in (TradeAction.BUY,):
            if portfolio.cash >= cost:
                portfolio.cash -= cost
                portfolio.positions.append(
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
                portfolio.trades_today += 1
                self._logger.info(f"[PAPER] Opened LONG {order.symbol}: {order.quantity:.2f} @ {order.price:.2f}")
                return True
            else:
                self._logger.warning(f"[PAPER] Insufficient cash: {portfolio.cash:.2f} < {cost:.2f}")
                order.status = "rejected"
                return False

        elif order.action in (TradeAction.SELL, TradeAction.SHORT):
            # Close existing long position or open short
            for i, pos in enumerate(portfolio.positions):
                if pos.symbol == order.symbol and pos.side == "long":
                    pnl = (order.price - pos.entry_price) * pos.quantity
                    portfolio.cash += pos.quantity * order.price
                    portfolio.positions.pop(i)
                    portfolio.trades_today += 1

                    if pnl < 0:
                        portfolio.consecutive_losses += 1
                    else:
                        portfolio.consecutive_losses = 0

                    self._logger.info(
                        f"[PAPER] Closed LONG {order.symbol}: P&L = {pnl:.2f}"
                    )
                    return True

            self._logger.info(f"[PAPER] No position to close for {order.symbol}")
            return False

        return False
