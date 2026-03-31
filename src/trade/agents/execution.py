"""Execution Agent - Handles order creation with proper risk controls."""

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
    """Executes trades with proper position sizing using ATR and SL/TP validation."""

    name = "execution"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.mode = self.config.get("mode", "paper")
        self._paper_orders: list[Order] = []

    def analyze(self, context: AnalysisContext) -> AgentSignal:
        """Execute or simulate the trade decision."""
        start = time.time()

        # Get the portfolio manager's decision (explicitly find it)
        decision = None
        if context.signals:
            for sig in reversed(context.signals):
                if sig.source_agent == "portfolio_manager":
                    decision = sig
                    break
            if decision is None:
                decision = context.signals[-1]

        if decision is None or decision.action == TradeAction.HOLD:
            signal = self._make_signal(
                context, TradeAction.HOLD, SignalStrength.NEUTRAL, 1.0,
                reasoning="No trade to execute (HOLD)",
            )
            return self._make_output(signal, {"executed": False})

        # Get price data
        latest_price = context.metadata.get("latest_close", 0)
        if latest_price <= 0:
            signal = self._make_signal(
                context, TradeAction.HOLD, SignalStrength.NEUTRAL, 1.0,
                reasoning="Cannot execute: no price data",
            )
            return self._make_output(signal, {"executed": False, "reason": "no_price"})

        # =====================================================================
        # POSITION SIZING using real ATR (not volatility hack)
        # =====================================================================
        portfolio = context.portfolio
        risk_per_trade_pct = self.config.get("risk_per_trade_pct", 0.5)
        max_position_pct = self.config.get("max_position_pct", 5.0)

        risk_amount = portfolio.total_value * (risk_per_trade_pct / 100)

        # Use real ATR from technical agent - NEVER guess
        atr = context.metadata.get("atr")
        if atr is None or atr <= 0:
            self._logger.warning(f"No ATR available for {context.symbol} - REJECTING trade (safety)")
            signal = self._make_signal(
                context, TradeAction.HOLD, SignalStrength.NEUTRAL, 1.0,
                reasoning="No ATR data - cannot size position safely",
            )
            return self._make_output(signal, {"executed": False, "reason": "no_atr"})

        # Stop loss = 1.5x ATR from entry
        sl_multiplier = self.config.get("sl_atr_multiplier", 1.5)
        # Take profit = target R:R * SL distance
        target_rr = self.config.get("target_rr", 2.0)
        min_rr = self.config.get("min_rr", 1.5)

        stop_distance = atr * sl_multiplier
        tp_distance = stop_distance * target_rr

        # Calculate SL/TP based on direction
        if decision.action == TradeAction.BUY:
            stop_loss = latest_price - stop_distance
            take_profit = latest_price + tp_distance
        elif decision.action in (TradeAction.SELL, TradeAction.SHORT):
            stop_loss = latest_price + stop_distance
            take_profit = latest_price - tp_distance
        else:
            stop_loss = None
            take_profit = None

        # =====================================================================
        # SL/TP SANITY VALIDATION
        # =====================================================================
        if stop_loss is not None and take_profit is not None:
            if decision.action == TradeAction.BUY:
                if not (stop_loss < latest_price < take_profit):
                    self._logger.error(
                        f"SL/TP sanity FAIL for BUY: SL={stop_loss:.5f} "
                        f"Entry={latest_price:.5f} TP={take_profit:.5f}"
                    )
                    signal = self._make_signal(
                        context, TradeAction.HOLD, SignalStrength.NEUTRAL, 1.0,
                        reasoning="SL/TP sanity check failed for BUY",
                    )
                    return self._make_output(signal, {"executed": False, "reason": "sl_tp_invalid"})

            elif decision.action in (TradeAction.SELL, TradeAction.SHORT):
                if not (take_profit < latest_price < stop_loss):
                    self._logger.error(
                        f"SL/TP sanity FAIL for SELL: TP={take_profit:.5f} "
                        f"Entry={latest_price:.5f} SL={stop_loss:.5f}"
                    )
                    signal = self._make_signal(
                        context, TradeAction.HOLD, SignalStrength.NEUTRAL, 1.0,
                        reasoning="SL/TP sanity check failed for SELL",
                    )
                    return self._make_output(signal, {"executed": False, "reason": "sl_tp_invalid"})

            # Validate R:R ratio
            risk = abs(latest_price - stop_loss)
            reward = abs(take_profit - latest_price)
            actual_rr = reward / risk if risk > 0 else 0

            if actual_rr < min_rr:
                self._logger.warning(
                    f"R:R ratio {actual_rr:.2f} < minimum {min_rr}. Rejecting trade."
                )
                signal = self._make_signal(
                    context, TradeAction.HOLD, SignalStrength.NEUTRAL, 1.0,
                    reasoning=f"R:R ratio {actual_rr:.2f} below minimum {min_rr}",
                )
                return self._make_output(signal, {"executed": False, "reason": "rr_too_low"})

        # =====================================================================
        # QUANTITY CALCULATION
        # =====================================================================
        if stop_distance > 0:
            quantity = risk_amount / stop_distance
        else:
            quantity = 0

        # Cap at max position size
        max_value = portfolio.total_value * (max_position_pct / 100)
        max_quantity = max_value / latest_price if latest_price > 0 else 0
        quantity = min(quantity, max_quantity)

        # Cap at max lots
        max_lots = self.config.get("max_lots", 5.0)
        quantity = min(quantity, max_lots)

        # Minimum quantity check
        if quantity < 0.01:
            signal = self._make_signal(
                context, TradeAction.HOLD, SignalStrength.NEUTRAL, 1.0,
                reasoning=f"Quantity too small: {quantity:.4f}",
            )
            return self._make_output(signal, {"executed": False, "reason": "qty_too_small"})

        # For non-crypto, round appropriately
        if context.asset_type.value != "crypto":
            quantity = round(quantity, 2)

        # =====================================================================
        # CREATE ORDER
        # =====================================================================
        order = Order(
            id=str(uuid.uuid4())[:8],
            symbol=context.symbol,
            asset_type=context.asset_type,
            action=decision.action,
            quantity=quantity,
            price=latest_price,
            stop_loss=round(stop_loss, 5) if stop_loss else None,
            take_profit=round(take_profit, 5) if take_profit else None,
        )

        # Execute based on mode
        if self.mode == "paper":
            executed = self._paper_execute(order, context)
        else:
            self._logger.error("Live trading must go through TradeLockerBroker!")
            executed = False

        raw_data = {
            "executed": executed,
            "mode": self.mode,
            "order": order.model_dump(),
            "position_value": round(quantity * latest_price, 2),
            "risk_amount": round(risk_amount, 2),
            "atr": round(atr, 5),
            "stop_distance": round(stop_distance, 5),
            "rr_ratio": round(actual_rr, 2) if stop_loss and take_profit else 0,
        }

        action = decision.action if executed else TradeAction.HOLD
        signal = self._make_signal(
            context, action, decision.strength, decision.confidence,
            reasoning=f"{'EXECUTED' if executed else 'NOT EXECUTED'} [{self.mode}] "
                     f"{decision.action.value} {quantity:.2f} {context.symbol} @ {latest_price:.5f} | "
                     f"SL: {f'{stop_loss:.5f}' if stop_loss else 'N/A'} | "
                     f"TP: {f'{take_profit:.5f}' if take_profit else 'N/A'} | "
                     f"R:R: {f'{actual_rr:.2f}' if (stop_loss and take_profit) else 'N/A'}",
        )

        exec_time = (time.time() - start) * 1000
        return self._make_output(signal, raw_data, exec_time=exec_time)

    def _paper_execute(self, order: Order, context: AnalysisContext) -> bool:
        """Simulate order execution in paper trading mode."""
        self._logger.info(
            f"[PAPER] Executing: {order.action.value} {order.quantity:.2f} "
            f"{order.symbol} @ {order.price:.5f}"
        )

        order.status = "filled"
        order.fill_price = order.price
        self._paper_orders.append(order)

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
                self._logger.info(
                    f"[PAPER] Opened LONG {order.symbol}: {order.quantity:.2f} "
                    f"@ {order.price:.5f} | SL: {order.stop_loss} | TP: {order.take_profit}"
                )
                return True
            else:
                self._logger.warning(f"[PAPER] Insufficient cash: {portfolio.cash:.2f} < {cost:.2f}")
                order.status = "rejected"
                return False

        elif order.action in (TradeAction.SELL, TradeAction.SHORT):
            # First try to close existing long position
            for i, pos in enumerate(portfolio.positions):
                if pos.symbol == order.symbol and pos.side == "long":
                    pnl = (order.price - pos.entry_price) * pos.quantity
                    portfolio.cash += pos.quantity * order.price
                    portfolio.positions.pop(i)
                    self._logger.info(f"[PAPER] Closed LONG {order.symbol}: P&L = {pnl:.2f}")
                    return True

            # No long to close - open a new SHORT position (forex allows shorting)
            margin_required = cost * 0.01  # 1% margin for forex (1:100 leverage)
            if portfolio.cash >= margin_required:
                portfolio.positions.append(
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
                self._logger.info(
                    f"[PAPER] Opened SHORT {order.symbol}: {order.quantity:.2f} "
                    f"@ {order.price:.5f} | SL: {order.stop_loss} | TP: {order.take_profit}"
                )
                return True
            else:
                self._logger.warning(f"[PAPER] Insufficient margin for short")
                order.status = "rejected"
                return False

        return False
