"""Paper trading connector — simulates broker execution locally.

Uses yfinance or cached CSV data for price feeds. Fills at market
price with configurable spread and slippage. Tracks P&L, positions,
and account state identically to the live connector so the scalper
engine doesn't know the difference.

Use for development and validation before connecting real money.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

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


class PaperConnector(BaseConnector):
    def __init__(
        self,
        initial_balance: float = 10_000.0,
        spread: dict[str, float] | None = None,
        slippage_pct: float = 0.0001,
        data_dir: Path = Path("data/dukascopy"),
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.spread = spread or {
            "EURUSD": 0.00008,
            "USDJPY": 0.008,
            "XAUUSD": 0.40,
        }
        self.slippage_pct = slippage_pct
        self.data_dir = data_dir
        self._positions: dict[str, Position] = {}
        self._closed_trades: list[dict] = []
        self._bar_cache: dict[str, pd.DataFrame] = {}
        self._bar_index: dict[str, int] = {}
        self._connected = False

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def _load_bars(self, symbol: str) -> pd.DataFrame:
        if symbol in self._bar_cache:
            return self._bar_cache[symbol]
        # Try 1H first, then 5M
        for suffix in ["_1H.csv", "_5M.csv"]:
            path = self.data_dir / f"{symbol}{suffix}"
            if path.exists():
                df = pd.read_csv(path, parse_dates=["timestamp"])
                df = df.set_index("timestamp").sort_index()
                df = df[~df.index.duplicated(keep="first")]
                self._bar_cache[symbol] = df
                self._bar_index[symbol] = 0
                return df
        return pd.DataFrame()

    def advance_bar(self, symbol: str) -> bool:
        """Move to the next bar (for backtesting/paper simulation)."""
        if symbol not in self._bar_index:
            self._load_bars(symbol)
        if symbol in self._bar_index:
            df = self._bar_cache.get(symbol)
            if df is not None and self._bar_index[symbol] < len(df) - 1:
                self._bar_index[symbol] += 1
                return True
        return False

    def set_bar_index(self, symbol: str, index: int) -> None:
        if symbol not in self._bar_cache:
            self._load_bars(symbol)
        self._bar_index[symbol] = max(0, min(index, len(self._bar_cache.get(symbol, [])) - 1))

    def get_account_info(self) -> AccountInfo:
        # Update equity based on open positions
        unrealized = sum(p.pnl for p in self._positions.values())
        self.equity = self.balance + unrealized
        return AccountInfo(
            balance=self.balance,
            equity=self.equity,
            margin_used=0.0,
            margin_free=self.equity,
        )

    def get_tick(self, symbol: str) -> Tick | None:
        df = self._load_bars(symbol)
        if df.empty:
            return None
        idx = self._bar_index.get(symbol, 0)
        if idx >= len(df):
            return None
        row = df.iloc[idx]
        mid = float(row["close"])
        s = self.spread.get(symbol, 0.0001)
        return Tick(
            timestamp=df.index[idx].to_pydatetime() if hasattr(df.index[idx], "to_pydatetime") else datetime.utcnow(),
            bid=mid - s / 2,
            ask=mid + s / 2,
        )

    def get_bars(self, symbol: str, timeframe: str, count: int) -> list[Bar]:
        df = self._load_bars(symbol)
        if df.empty:
            return []
        idx = self._bar_index.get(symbol, 0)
        start = max(0, idx - count + 1)
        end = idx + 1
        bars = []
        for i in range(start, end):
            row = df.iloc[i]
            bars.append(Bar(
                timestamp=df.index[i].to_pydatetime() if hasattr(df.index[i], "to_pydatetime") else datetime.utcnow(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0)),
            ))
        return bars

    def get_positions(self) -> list[Position]:
        # Update PnL for open positions
        for pid, pos in self._positions.items():
            tick = self.get_tick(pos.symbol)
            if tick:
                if pos.side == Side.BUY:
                    pos.pnl = (tick.bid - pos.entry_price) * pos.quantity
                else:
                    pos.pnl = (pos.entry_price - tick.ask) * pos.quantity
        return list(self._positions.values())

    def open_position(
        self, symbol: str, side: Side, quantity: float,
        sl: float | None = None, tp: float | None = None,
        order_type: OrderType = OrderType.MARKET,
    ) -> OrderResult:
        tick = self.get_tick(symbol)
        if tick is None:
            return OrderResult("", OrderStatus.REJECTED, message="no price data")
        # Fill at ask for buy, bid for sell (+ slippage)
        if side == Side.BUY:
            fill = tick.ask * (1 + self.slippage_pct)
        else:
            fill = tick.bid * (1 - self.slippage_pct)
        pid = str(uuid.uuid4())[:8]
        self._positions[pid] = Position(
            id=pid,
            symbol=symbol,
            side=side,
            entry_price=fill,
            quantity=quantity,
            sl=sl,
            tp=tp,
            open_time=tick.timestamp,
        )
        return OrderResult(pid, OrderStatus.FILLED, fill_price=fill)

    def close_position(self, position_id: str) -> OrderResult:
        pos = self._positions.get(position_id)
        if pos is None:
            return OrderResult(position_id, OrderStatus.REJECTED, message="not found")
        tick = self.get_tick(pos.symbol)
        if tick is None:
            return OrderResult(position_id, OrderStatus.REJECTED, message="no price")
        if pos.side == Side.BUY:
            fill = tick.bid * (1 - self.slippage_pct)
            pnl = (fill - pos.entry_price) * pos.quantity
        else:
            fill = tick.ask * (1 + self.slippage_pct)
            pnl = (pos.entry_price - fill) * pos.quantity
        self.balance += pnl
        self._closed_trades.append({
            "id": position_id,
            "symbol": pos.symbol,
            "side": pos.side.value,
            "entry": pos.entry_price,
            "exit": fill,
            "quantity": pos.quantity,
            "pnl": pnl,
            "open_time": pos.open_time,
            "close_time": tick.timestamp,
        })
        del self._positions[position_id]
        return OrderResult(position_id, OrderStatus.FILLED, fill_price=fill)

    def close_all_positions(self) -> int:
        pids = list(self._positions.keys())
        closed = 0
        for pid in pids:
            r = self.close_position(pid)
            if r.status == OrderStatus.FILLED:
                closed += 1
        return closed

    def modify_position(self, position_id: str, sl: float | None = None, tp: float | None = None) -> bool:
        pos = self._positions.get(position_id)
        if pos is None:
            return False
        if sl is not None:
            pos.sl = sl
        if tp is not None:
            pos.tp = tp
        return True

    def check_sl_tp(self) -> list[str]:
        """Check all open positions against current BAR's high/low for
        SL/TP hits (not just the close price). Pessimistic: SL fires
        first if both are within the bar range.
        """
        closed = []
        for pid in list(self._positions.keys()):
            pos = self._positions.get(pid)
            if pos is None:
                continue
            # Use the bar's high and low for intra-bar detection
            df = self._bar_cache.get(pos.symbol)
            idx = self._bar_index.get(pos.symbol, 0)
            if df is None or idx >= len(df):
                continue
            row = df.iloc[idx]
            bar_high = float(row["high"])
            bar_low = float(row["low"])
            bar_close = float(row["close"])

            hit = False
            exit_price = bar_close
            if pos.side == Side.BUY:
                # Pessimistic: SL first
                if pos.sl and bar_low <= pos.sl:
                    hit = True
                    exit_price = pos.sl
                elif pos.tp and bar_high >= pos.tp:
                    hit = True
                    exit_price = pos.tp
            else:
                if pos.sl and bar_high >= pos.sl:
                    hit = True
                    exit_price = pos.sl
                elif pos.tp and bar_low <= pos.tp:
                    hit = True
                    exit_price = pos.tp

            if hit:
                # Compute PnL at the actual SL/TP price, not at close
                if pos.side == Side.BUY:
                    pnl = (exit_price - pos.entry_price) * pos.quantity
                else:
                    pnl = (pos.entry_price - exit_price) * pos.quantity
                self.balance += pnl
                ts = df.index[idx].to_pydatetime() if hasattr(df.index[idx], "to_pydatetime") else datetime.utcnow()
                self._closed_trades.append({
                    "id": pid,
                    "symbol": pos.symbol,
                    "side": pos.side.value,
                    "entry": pos.entry_price,
                    "exit": exit_price,
                    "quantity": pos.quantity,
                    "pnl": pnl,
                    "open_time": pos.open_time,
                    "close_time": ts,
                })
                del self._positions[pid]
                closed.append(pid)
        return closed

    @property
    def trade_history(self) -> list[dict]:
        return list(self._closed_trades)

    @property
    def total_pnl(self) -> float:
        return self.balance - self.initial_balance
