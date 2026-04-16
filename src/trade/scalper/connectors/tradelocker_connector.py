"""TradeLocker connector — live trading via the official Python API.

Wraps the `tradelocker` pip package into the BaseConnector interface.
Requires: pip install tradelocker

Credentials are loaded from environment variables or .env:
  TRADELOCKER_EMAIL=your@email.com
  TRADELOCKER_PASSWORD=yourpassword
  TRADELOCKER_SERVER=servername
  TRADELOCKER_ENV=https://demo.tradelocker.com  (or live URL)
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from trade.scalper.connectors.base import (
    AccountInfo,
    Bar,
    BaseConnector,
    OrderResult,
    OrderStatus,
    OrderType,
    Position,
    Side,
    Tick,
)


def _load_env() -> dict[str, str]:
    env = {}
    for p in [Path(".env"), Path.home() / ".env"]:
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


class TradeLockerConnector(BaseConnector):
    def __init__(
        self,
        email: str | None = None,
        password: str | None = None,
        server: str | None = None,
        environment: str | None = None,
    ):
        env = _load_env()
        self.email = email or os.environ.get("TRADELOCKER_EMAIL") or env.get("TRADELOCKER_EMAIL", "")
        self.password = password or os.environ.get("TRADELOCKER_PASSWORD") or env.get("TRADELOCKER_PASSWORD", "")
        self.server = server or os.environ.get("TRADELOCKER_SERVER") or env.get("TRADELOCKER_SERVER", "")
        self.environment = environment or os.environ.get("TRADELOCKER_ENV") or env.get("TRADELOCKER_ENV", "https://demo.tradelocker.com")
        self._api = None
        self._instrument_cache: dict[str, int] = {}

    def connect(self) -> bool:
        try:
            from tradelocker import TLAPI
            self._api = TLAPI(
                environment=self.environment,
                username=self.email,
                password=self.password,
                server=self.server,
            )
            # Warm up instrument cache
            instruments = self._api.get_all_instruments()
            if instruments is not None and hasattr(instruments, "iterrows"):
                for _, row in instruments.iterrows():
                    name = str(row.get("name", ""))
                    iid = row.get("tradableInstrumentId", 0)
                    if name and iid:
                        self._instrument_cache[name] = int(iid)
            return True
        except Exception as e:
            print(f"TradeLocker connect error: {e}", flush=True)
            return False

    def disconnect(self) -> None:
        self._api = None

    def _get_instrument_id(self, symbol: str) -> int | None:
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]
        if self._api is None:
            return None
        try:
            iid = self._api.get_instrument_id_from_symbol_name(symbol)
            if iid:
                self._instrument_cache[symbol] = int(iid)
                return int(iid)
        except Exception:
            pass
        return None

    def get_account_info(self) -> AccountInfo:
        if self._api is None:
            return AccountInfo(0, 0, 0, 0)
        try:
            # TradeLocker API may vary; adapt field names as needed
            info = self._api.get_account_state() if hasattr(self._api, "get_account_state") else {}
            if isinstance(info, dict):
                return AccountInfo(
                    balance=float(info.get("balance", 0)),
                    equity=float(info.get("equity", 0)),
                    margin_used=float(info.get("usedMargin", 0)),
                    margin_free=float(info.get("freeMargin", 0)),
                )
            return AccountInfo(0, 0, 0, 0)
        except Exception:
            return AccountInfo(0, 0, 0, 0)

    def get_tick(self, symbol: str) -> Tick | None:
        if self._api is None:
            return None
        iid = self._get_instrument_id(symbol)
        if iid is None:
            return None
        try:
            price = self._api.get_latest_asking_price(iid)
            if price is not None:
                ask = float(price)
                # Approximate bid from ask (the API primarily returns ask)
                spread_est = 0.00008 if "USD" in symbol else 0.40
                return Tick(
                    timestamp=datetime.utcnow(),
                    bid=ask - spread_est,
                    ask=ask,
                )
        except Exception:
            pass
        return None

    def get_bars(self, symbol: str, timeframe: str, count: int) -> list[Bar]:
        if self._api is None:
            return []
        iid = self._get_instrument_id(symbol)
        if iid is None:
            return []
        # Map timeframe string to TradeLocker resolution
        tf_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        resolution = tf_map.get(timeframe, 1)
        try:
            data = self._api.get_price_history(
                instrument_id=iid,
                resolution=resolution,
                start_timestamp=int((datetime.utcnow() - timedelta(minutes=resolution * count * 2)).timestamp() * 1000),
                end_timestamp=int(datetime.utcnow().timestamp() * 1000),
                lookback_period=f"{count}",
            )
            if data is None:
                return []
            bars = []
            # data format depends on API version; handle both dict and DataFrame
            if hasattr(data, "iterrows"):
                for _, row in data.iterrows():
                    bars.append(Bar(
                        timestamp=datetime.utcfromtimestamp(float(row.get("t", 0)) / 1000),
                        open=float(row.get("o", 0)),
                        high=float(row.get("h", 0)),
                        low=float(row.get("l", 0)),
                        close=float(row.get("c", 0)),
                        volume=float(row.get("v", 0)),
                    ))
            return bars[-count:]
        except Exception:
            return []

    def get_positions(self) -> list[Position]:
        if self._api is None:
            return []
        try:
            positions = self._api.get_all_positions() if hasattr(self._api, "get_all_positions") else None
            if positions is None:
                return []
            result = []
            if hasattr(positions, "iterrows"):
                for _, row in positions.iterrows():
                    result.append(Position(
                        id=str(row.get("id", "")),
                        symbol=str(row.get("symbol", "")),
                        side=Side.BUY if str(row.get("side", "")).lower() == "buy" else Side.SELL,
                        entry_price=float(row.get("avgPrice", 0)),
                        quantity=float(row.get("qty", 0)),
                        pnl=float(row.get("unrealizedPl", 0)),
                    ))
            return result
        except Exception:
            return []

    def open_position(
        self, symbol: str, side: Side, quantity: float,
        sl: float | None = None, tp: float | None = None,
        order_type: OrderType = OrderType.MARKET,
    ) -> OrderResult:
        if self._api is None:
            return OrderResult("", OrderStatus.REJECTED, message="not connected")
        iid = self._get_instrument_id(symbol)
        if iid is None:
            return OrderResult("", OrderStatus.REJECTED, message=f"unknown symbol {symbol}")
        try:
            order_id = self._api.create_order(
                instrument_id=iid,
                quantity=quantity,
                side=side.value,
                type_=order_type.value,
            )
            if order_id:
                # Modify to add SL/TP if provided
                if sl is not None or tp is not None:
                    try:
                        self._api.set_stop_loss(order_id, sl) if sl and hasattr(self._api, "set_stop_loss") else None
                        self._api.set_take_profit(order_id, tp) if tp and hasattr(self._api, "set_take_profit") else None
                    except Exception:
                        pass
                return OrderResult(str(order_id), OrderStatus.FILLED)
            return OrderResult("", OrderStatus.REJECTED, message="order returned None")
        except Exception as e:
            return OrderResult("", OrderStatus.REJECTED, message=str(e))

    def close_position(self, position_id: str) -> OrderResult:
        if self._api is None:
            return OrderResult("", OrderStatus.REJECTED, message="not connected")
        try:
            result = self._api.close_position(int(position_id))
            if result:
                return OrderResult(position_id, OrderStatus.FILLED)
            return OrderResult(position_id, OrderStatus.REJECTED, message="close returned False")
        except Exception as e:
            return OrderResult(position_id, OrderStatus.REJECTED, message=str(e))

    def close_all_positions(self) -> int:
        positions = self.get_positions()
        closed = 0
        for p in positions:
            r = self.close_position(p.id)
            if r.status == OrderStatus.FILLED:
                closed += 1
        return closed

    def modify_position(self, position_id: str, sl: float | None = None, tp: float | None = None) -> bool:
        if self._api is None:
            return False
        try:
            if sl and hasattr(self._api, "set_stop_loss"):
                self._api.set_stop_loss(int(position_id), sl)
            if tp and hasattr(self._api, "set_take_profit"):
                self._api.set_take_profit(int(position_id), tp)
            return True
        except Exception:
            return False
