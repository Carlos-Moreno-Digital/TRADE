"""TradeLocker execution broker - Live connection to FunderPro via TradeLocker API.

This module connects our trading agent to a real TradeLocker account
(FunderPro prop firm) for live/paper execution.

IMPORTANT: This executes REAL trades on your funded account.
Always test thoroughly in demo before using with real funds.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any

from loguru import logger

from trade.data.models import (
    AssetType,
    Order,
    Position,
    PortfolioState,
    TradeAction,
)
from trade.execution.broker import BaseBroker
from trade.risk.prop_firm import PropFirmRiskEngine, PropFirmConfig


# TradeLocker symbol mapping for FunderPro
SYMBOL_MAP = {
    # Forex
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "USDJPY": "USDJPY",
    "USDCHF": "USDCHF",
    "AUDUSD": "AUDUSD",
    "NZDUSD": "NZDUSD",
    "USDCAD": "USDCAD",
    "EURGBP": "EURGBP",
    "EURJPY": "EURJPY",
    "GBPJPY": "GBPJPY",
    # Indices
    "US30": "US30",
    "NAS100": "NAS100",
    "SPX500": "SPX500",
    "GER40": "GER40",
    # Commodities
    "XAUUSD": "XAUUSD",
    "XAGUSD": "XAGUSD",
    "USOIL": "USOIL",
    # Crypto
    "BTCUSD": "BTCUSD",
    "ETHUSD": "ETHUSD",
    # yfinance format conversions
    "EURUSD=X": "EURUSD",
    "GBPUSD=X": "GBPUSD",
    "USDJPY=X": "USDJPY",
    "AUDUSD=X": "AUDUSD",
    "BTC-USD": "BTCUSD",
    "ETH-USD": "ETHUSD",
    "GC=F": "XAUUSD",
}


class TradeLockerBroker(BaseBroker):
    """Live broker implementation using TradeLocker API.

    Connects to FunderPro (or any TradeLocker broker) for real trade execution.
    Integrates with PropFirmRiskEngine for safety.
    """

    def __init__(
        self,
        environment: str | None = None,
        username: str | None = None,
        password: str | None = None,
        server: str | None = None,
        risk_engine: PropFirmRiskEngine | None = None,
        demo: bool = True,
    ):
        self.environment = environment or os.getenv(
            "TRADELOCKER_ENVIRONMENT",
            "https://demo.tradelocker.com" if demo else "https://live.tradelocker.com",
        )
        self.username = username or os.getenv("TRADELOCKER_USERNAME", "")
        self.password = password or os.getenv("TRADELOCKER_PASSWORD", "")
        self.server = server or os.getenv("TRADELOCKER_SERVER", "FunderPro")
        self.risk_engine = risk_engine
        self.demo = demo

        self._tl = None
        self._instrument_cache: dict[str, int] = {}
        self._connected = False

        logger.info(
            f"TradeLockerBroker initialized: {self.environment} "
            f"(server: {self.server}, demo: {self.demo})"
        )

    def connect(self) -> bool:
        """Establish connection to TradeLocker."""
        try:
            from tradelocker import TLAPI

            if not self.username or not self.password:
                logger.error(
                    "TradeLocker credentials not set. "
                    "Set TRADELOCKER_USERNAME, TRADELOCKER_PASSWORD, "
                    "TRADELOCKER_SERVER environment variables or pass to constructor."
                )
                return False

            self._tl = TLAPI(
                environment=self.environment,
                username=self.username,
                password=self.password,
                server=self.server,
            )

            # Verify connection by getting account state
            state = self._tl.get_account_state()
            if state is not None:
                self._connected = True
                logger.info(f"Connected to TradeLocker: {self.server}")

                # Initialize risk engine with account balance
                if self.risk_engine:
                    balance = self._extract_balance(state)
                    self.risk_engine.initialize(balance)
                    self.risk_engine.start_trading_day(balance)
                    logger.info(f"Risk engine initialized with balance: ${balance:,.2f}")

                return True
            else:
                logger.error("Failed to get account state from TradeLocker")
                return False

        except ImportError:
            logger.error("tradelocker package not installed. Run: pip install tradelocker")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to TradeLocker: {e}")
            return False

    def disconnect(self) -> None:
        """Close the connection."""
        self._tl = None
        self._connected = False
        logger.info("Disconnected from TradeLocker")

    # =========================================================================
    # Account Info
    # =========================================================================

    def get_account_balance(self) -> float:
        """Get current account balance."""
        self._ensure_connected()
        state = self._tl.get_account_state()
        return self._extract_balance(state)

    def get_account_state(self) -> dict[str, Any]:
        """Get full account state including equity, margin, etc."""
        self._ensure_connected()
        state = self._tl.get_account_state()
        return state if isinstance(state, dict) else {"raw": state}

    def get_portfolio_state(self) -> PortfolioState:
        """Get portfolio state compatible with our data models."""
        self._ensure_connected()
        state = self._tl.get_account_state()
        positions = self.get_positions()

        balance = self._extract_balance(state)
        equity = self._extract_equity(state)

        portfolio = PortfolioState(
            cash=balance,
            total_value=equity,
            positions=positions,
            peak_value=max(equity, balance),
        )
        portfolio.update_drawdown()
        return portfolio

    # =========================================================================
    # Instruments
    # =========================================================================

    def get_instrument_id(self, symbol: str) -> int | None:
        """Get TradeLocker instrument ID for a symbol."""
        # Normalize symbol
        tl_symbol = SYMBOL_MAP.get(symbol, symbol)

        if tl_symbol in self._instrument_cache:
            return self._instrument_cache[tl_symbol]

        self._ensure_connected()
        try:
            instrument_id = self._tl.get_instrument_id_from_symbol_name(tl_symbol)
            if instrument_id:
                self._instrument_cache[tl_symbol] = instrument_id
                return instrument_id
            logger.warning(f"Instrument not found: {tl_symbol}")
            return None
        except Exception as e:
            logger.error(f"Failed to get instrument ID for {tl_symbol}: {e}")
            return None

    def get_price(self, symbol: str) -> tuple[float, float] | None:
        """Get current bid/ask price for a symbol.

        Returns:
            Tuple of (bid, ask) or None if failed.
        """
        self._ensure_connected()
        instrument_id = self.get_instrument_id(symbol)
        if instrument_id is None:
            return None

        try:
            bid = self._tl.get_latest_bid_price(instrument_id)
            ask = self._tl.get_latest_asking_price(instrument_id)
            return (float(bid), float(ask))
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return None

    def get_price_history(
        self, symbol: str, resolution: str = "1H", lookback: str = "30D"
    ) -> Any:
        """Get historical price data from TradeLocker."""
        self._ensure_connected()
        instrument_id = self.get_instrument_id(symbol)
        if instrument_id is None:
            return None

        try:
            return self._tl.get_price_history(
                instrument_id,
                resolution=resolution,
                lookback_period=lookback,
            )
        except Exception as e:
            logger.error(f"Failed to get price history for {symbol}: {e}")
            return None

    # =========================================================================
    # Order Management
    # =========================================================================

    def submit_order(self, order: Order) -> Order:
        """Submit an order to TradeLocker.

        Runs through PropFirmRiskEngine validation first.
        """
        self._ensure_connected()

        # Risk check FIRST
        if self.risk_engine:
            portfolio = self.get_portfolio_state()
            can_trade, reason = self.risk_engine.can_open_trade(portfolio)
            if not can_trade:
                logger.warning(f"RISK BLOCKED: {reason}")
                order.status = "rejected"
                return order

            valid, reason, order = self.risk_engine.validate_order(order, portfolio)
            if not valid:
                logger.warning(f"ORDER REJECTED: {reason}")
                order.status = "rejected"
                return order

        # Get instrument ID
        tl_symbol = SYMBOL_MAP.get(order.symbol, order.symbol)
        instrument_id = self.get_instrument_id(tl_symbol)
        if instrument_id is None:
            logger.error(f"Unknown instrument: {order.symbol}")
            order.status = "rejected"
            return order

        # Map our action to TradeLocker side
        side = self._map_side(order.action)
        if side is None:
            order.status = "rejected"
            return order

        # Map order type
        type_ = "market" if order.order_type == "market" else "limit"

        try:
            logger.info(
                f"EXECUTING: {side} {order.quantity} {tl_symbol} @ "
                f"{'market' if type_ == 'market' else order.price} | "
                f"SL: {order.stop_loss} | TP: {order.take_profit}"
            )

            result = self._tl.create_order(
                instrument_id=instrument_id,
                quantity=order.quantity,
                side=side,
                type_=type_,
                price=order.price if type_ != "market" else None,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
            )

            if result:
                order.status = "filled"
                order.id = str(result) if result else ""
                order.fill_timestamp = datetime.utcnow()

                # Get fill price
                prices = self.get_price(order.symbol)
                if prices:
                    order.fill_price = prices[1] if side == "buy" else prices[0]

                logger.info(
                    f"ORDER FILLED: {side} {order.quantity} {tl_symbol} "
                    f"(ID: {order.id})"
                )
            else:
                order.status = "rejected"
                logger.error(f"Order rejected by TradeLocker: {tl_symbol}")

        except Exception as e:
            order.status = "rejected"
            logger.error(f"Order execution failed: {e}")

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        self._ensure_connected()
        try:
            result = self._tl.delete_order(int(order_id))
            logger.info(f"Order cancelled: {order_id}")
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def close_position_by_id(self, position_id: int) -> bool:
        """Close a specific position."""
        self._ensure_connected()
        try:
            result = self._tl.close_position(position_id=position_id)
            logger.info(f"Position closed: {position_id}")
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to close position {position_id}: {e}")
            return False

    def close_all_positions(self) -> bool:
        """Emergency: close all open positions."""
        self._ensure_connected()
        try:
            result = self._tl.close_all_positions()
            logger.critical("ALL POSITIONS CLOSED")
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return False

    def modify_position_sl_tp(
        self, position_id: int, stop_loss: float | None = None, take_profit: float | None = None
    ) -> bool:
        """Modify stop loss and/or take profit of an open position."""
        self._ensure_connected()
        try:
            params = {}
            if stop_loss is not None:
                params["stopLoss"] = stop_loss
            if take_profit is not None:
                params["takeProfit"] = take_profit

            result = self._tl.modify_position(position_id, params)
            logger.info(f"Position {position_id} modified: SL={stop_loss}, TP={take_profit}")
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to modify position {position_id}: {e}")
            return False

    # =========================================================================
    # Positions
    # =========================================================================

    def get_positions(self) -> list[Position]:
        """Get all open positions as our Position model."""
        self._ensure_connected()
        try:
            raw_positions = self._tl.get_all_positions()
            if raw_positions is None:
                return []

            positions = []
            if isinstance(raw_positions, (list, tuple)):
                for pos in raw_positions:
                    try:
                        positions.append(self._convert_position(pos))
                    except Exception as e:
                        logger.warning(f"Failed to convert position: {e}")

            return positions
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_open_orders(self) -> list[dict]:
        """Get all pending orders."""
        self._ensure_connected()
        try:
            orders = self._tl.get_all_orders()
            return orders if isinstance(orders, list) else []
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []

    # =========================================================================
    # Helpers
    # =========================================================================

    def _ensure_connected(self) -> None:
        """Ensure we have an active connection, with auto-reconnect."""
        if self._connected and self._tl is not None:
            return

        # Try reconnecting with exponential backoff
        delays = [2, 5, 15, 30]
        for attempt, delay in enumerate(delays, 1):
            logger.warning(f"Reconnecting to TradeLocker (attempt {attempt}/{len(delays)})...")
            if self.connect():
                logger.info(f"Reconnected on attempt {attempt}")
                return
            if attempt < len(delays):
                logger.warning(f"Reconnect failed, waiting {delay}s...")
                import time
                time.sleep(delay)

        raise ConnectionError(
            f"Failed to connect to TradeLocker after {len(delays)} attempts"
        )

    @staticmethod
    def _map_side(action: TradeAction) -> str | None:
        """Map our TradeAction to TradeLocker side."""
        if action in (TradeAction.BUY,):
            return "buy"
        elif action in (TradeAction.SELL, TradeAction.SHORT):
            return "sell"
        elif action == TradeAction.COVER:
            return "buy"  # Cover a short = buy
        return None

    @staticmethod
    def _extract_balance(state: Any) -> float:
        """Extract balance from account state."""
        if isinstance(state, dict):
            return float(state.get("balance", state.get("Balance", 0)))
        if isinstance(state, (list, tuple)) and len(state) > 0:
            return float(state[0]) if isinstance(state[0], (int, float)) else 0
        return 0.0

    @staticmethod
    def _extract_equity(state: Any) -> float:
        """Extract equity from account state."""
        if isinstance(state, dict):
            return float(state.get("equity", state.get("Equity",
                        state.get("balance", state.get("Balance", 0)))))
        return TradeLockerBroker._extract_balance(state)

    def _convert_position(self, raw: Any) -> Position:
        """Convert TradeLocker position to our Position model."""
        if isinstance(raw, dict):
            symbol = raw.get("symbol", raw.get("instrumentId", "UNKNOWN"))
            side = "long" if raw.get("side", "").lower() == "buy" else "short"
            return Position(
                symbol=str(symbol),
                asset_type=self._detect_asset_type(str(symbol)),
                side=side,
                quantity=float(raw.get("quantity", raw.get("qty", 0))),
                entry_price=float(raw.get("avgPrice", raw.get("openPrice", 0))),
                current_price=float(raw.get("currentPrice", raw.get("avgPrice", 0))),
                stop_loss=raw.get("stopLoss"),
                take_profit=raw.get("takeProfit"),
                unrealized_pnl=float(raw.get("unrealizedPnl", raw.get("pnl", 0))),
            )
        # Handle tuple/list format
        return Position(
            symbol="UNKNOWN",
            asset_type=AssetType.FOREX,
            side="long",
            quantity=0,
            entry_price=0,
        )

    @staticmethod
    def _detect_asset_type(symbol: str) -> AssetType:
        """Detect asset type from symbol."""
        symbol = symbol.upper()
        if symbol in ("XAUUSD", "XAGUSD", "USOIL"):
            return AssetType.FOREX  # Commodities trade like forex
        if symbol in ("US30", "NAS100", "SPX500", "GER40"):
            return AssetType.STOCK  # Indices
        if symbol in ("BTCUSD", "ETHUSD"):
            return AssetType.CRYPTO
        return AssetType.FOREX
