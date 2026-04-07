"""BBMR backtest using NautilusTrader with realistic fill model.

Goal: cross-validate the vanilla Python backtest against an event-driven,
institutional-grade engine. If the two disagree significantly, the
vanilla backtest is lying.

Uses:
- NautilusTrader BacktestEngine (event-driven, bar-by-bar)
- FillModel with slippage and spread
- talib for ADX (not in Nautilus stock)
- Nautilus BollingerBands + AverageTrueRange
"""
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import talib

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.config import LoggingConfig
from nautilus_trader.indicators.volatility import BollingerBands, AverageTrueRange
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, TimeInForce
from nautilus_trader.model.identifiers import Venue, InstrumentId
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

DATA_DIR = Path("data/dukascopy")
ACCOUNT = 10_000
RISK = 0.003


class BBMRConfig(StrategyConfig, frozen=True):
    bar_type: BarType
    instrument_id: str
    bb_period: int = 30
    bb_std: float = 2.0
    adx_max: int = 20
    atr_sl_mult: float = 1.0
    max_holding_bars: int = 12


class BBMRStrategy(Strategy):
    def __init__(self, config: BBMRConfig):
        super().__init__(config)
        self.bb = BollingerBands(config.bb_period, config.bb_std)
        self.atr = AverageTrueRange(14)
        self.instrument = None
        self.position_entry_bar = None
        self.stop_price = None
        self.tp_price = None
        self.position_side = None
        self.bar_count = 0
        self.highs = []
        self.lows = []
        self.closes = []
        self.trades_log = []
        self.n_breach = 0
        self.n_adx_ok = 0
        self.n_entry_tried = 0

    def on_start(self):
        self._inst_id = InstrumentId.from_str(self.config.instrument_id)
        self.instrument = self.cache.instrument(self._inst_id)
        self.register_indicator_for_bars(self.config.bar_type, self.bb)
        self.register_indicator_for_bars(self.config.bar_type, self.atr)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar):
        self.bar_count += 1
        h = float(bar.high)
        l = float(bar.low)
        c = float(bar.close)
        self.highs.append(h)
        self.lows.append(l)
        self.closes.append(c)

        if len(self.closes) < 200:
            return
        if not self.bb.initialized or not self.atr.initialized:
            return

        # Compute ADX on the fly with talib
        highs = np.array(self.highs, dtype=float)
        lows = np.array(self.lows, dtype=float)
        closes = np.array(self.closes, dtype=float)
        adx_arr = talib.ADX(highs, lows, closes, timeperiod=14)
        adx_now = adx_arr[-1]
        if np.isnan(adx_now):
            return

        upper = self.bb.upper
        middle = self.bb.middle
        lower = self.bb.lower
        atr_now = self.atr.value

        # Manage open position
        portfolio = self.portfolio
        net_qty = portfolio.net_position(self._inst_id)
        if net_qty is None:
            net_qty = 0

        if net_qty != 0:
            bars_held = self.bar_count - self.position_entry_bar
            exit_reason = None
            if self.position_side == "buy":
                if l <= self.stop_price:
                    exit_reason = "SL"
                elif h >= self.tp_price:
                    exit_reason = "TP"
                elif bars_held >= self.config.max_holding_bars:
                    exit_reason = "TIME"
            else:
                if h >= self.stop_price:
                    exit_reason = "SL"
                elif l <= self.tp_price:
                    exit_reason = "TP"
                elif bars_held >= self.config.max_holding_bars:
                    exit_reason = "TIME"

            if exit_reason is not None:
                order = self.order_factory.market(
                    instrument_id=self.instrument.id,
                    order_side=OrderSide.SELL if self.position_side == "buy" else OrderSide.BUY,
                    quantity=Quantity.from_int(abs(int(net_qty))),
                    time_in_force=TimeInForce.GTC,
                    reduce_only=True,
                )
                self.submit_order(order)
                self.position_side = None
                self.position_entry_bar = None
            return

        # Entry
        if adx_now >= self.config.adx_max:
            return

        self.n_adx_ok += 1
        if c < lower or c > upper:
            self.n_breach += 1
        side = None
        if c < lower:
            side = "buy"
            sl = c - atr_now * self.config.atr_sl_mult
            tp = middle
            risk_unit = c - sl
        elif c > upper:
            side = "sell"
            sl = c + atr_now * self.config.atr_sl_mult
            tp = middle
            risk_unit = sl - c
        else:
            return

        if risk_unit <= 0:
            return

        # Position sizing: 0.3% of account
        equity_dollars = float(portfolio.net_exposure(self._inst_id).as_decimal()) \
            if portfolio.net_exposure(self._inst_id) else ACCOUNT
        equity_dollars = ACCOUNT  # keep constant for simplicity
        # For JPY pairs risk is in JPY so divide appropriately.
        # We just scale up to meet the 1000-unit minimum trade size.
        raw = (equity_dollars * RISK) / max(risk_unit, 1e-6)
        # Round up to nearest 1000 (micro lot)
        qty_units = max(1000, int(round(raw / 1000) * 1000))
        qty_units = min(qty_units, 5_000_000)

        self.n_entry_tried += 1
        try:
            order = self.order_factory.market(
                instrument_id=self.instrument.id,
                order_side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                quantity=Quantity.from_int(qty_units),
                time_in_force=TimeInForce.GTC,
            )
            self.submit_order(order)
            self.position_side = side
            self.position_entry_bar = self.bar_count
            self.stop_price = sl
            self.tp_price = tp
        except Exception as e:
            self.log.error(f"order err: {e}")

    def on_stop(self):
        self.log.warning(
            f"FINAL: bars={self.bar_count} adx_ok={self.n_adx_ok} "
            f"breach={self.n_breach} entries={self.n_entry_tried}"
        )

    def on_order_rejected(self, event):
        self.log.warning(f"REJECTED: {event.reason}")

    def on_order_denied(self, event):
        self.log.warning(f"DENIED: {event.reason}")

    def on_order_filled(self, event):
        if self.bar_count < 1000:
            self.log.warning(f"FILLED: {event}")


def load_and_wrangle(symbol: str, instrument, bid_bt: BarType, ask_bt: BarType,
                     spread_abs: float):
    path = DATA_DIR / f"{symbol}_1H.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df["volume"] = df.get("volume", 0).fillna(0).clip(lower=1)

    bid_df = df[["open", "high", "low", "close", "volume"]].copy()
    ask_df = bid_df.copy()
    for col in ("open", "high", "low", "close"):
        ask_df[col] = ask_df[col] + spread_abs

    bid_w = BarDataWrangler(bar_type=bid_bt, instrument=instrument)
    ask_w = BarDataWrangler(bar_type=ask_bt, instrument=instrument)
    bid_bars = bid_w.process(bid_df)
    ask_bars = ask_w.process(ask_df)
    return bid_bars, ask_bars


SPREAD_ABS = {"EURUSD": 0.00008, "USDJPY": 0.008}


def run_backtest(symbol_code: str, pair: str):
    venue = Venue("SIM")
    instrument = TestInstrumentProvider.default_fx_ccy(pair)
    bid_bt = BarType.from_str(f"{instrument.id}-1-HOUR-BID-EXTERNAL")
    ask_bt = BarType.from_str(f"{instrument.id}-1-HOUR-ASK-EXTERNAL")

    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id="BBMR-001",
            logging=LoggingConfig(log_level="ERROR"),
        ),
    )

    fill_model = FillModel(
        prob_fill_on_limit=1.0,
        prob_slippage=0.2,
        random_seed=42,
    )

    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(ACCOUNT, USD)],
        fill_model=fill_model,
    )
    engine.add_instrument(instrument)

    bid_bars, ask_bars = load_and_wrangle(
        symbol_code, instrument, bid_bt, ask_bt, SPREAD_ABS[symbol_code]
    )
    print(f"  {symbol_code}: {len(bid_bars)} bid bars, {len(ask_bars)} ask bars")
    engine.add_data(bid_bars)
    engine.add_data(ask_bars)

    strategy = BBMRStrategy(
        BBMRConfig(
            bar_type=bid_bt,
            instrument_id=instrument.id.value,
        )
    )
    engine.add_strategy(strategy)

    engine.run()

    # Report — use account balance delta (base currency = USD) as truth.
    report = engine.trader.generate_account_report(venue)
    if len(report) > 0:
        final_total = float(report["total"].iloc[-1])
    else:
        final_total = ACCOUNT
    pnl_usd = final_total - ACCOUNT

    fills = engine.trader.generate_order_fills_report()
    positions = engine.trader.generate_positions_report()
    n_pos = len(positions)
    if n_pos > 0:
        def parse_money(v):
            try:
                return float(str(v).split()[0])
            except Exception:
                return 0.0
        raw_pnls = positions["realized_pnl"].map(parse_money).astype(float)
        wins = (raw_pnls > 0).sum()
        wr = wins / n_pos * 100
    else:
        wr = 0.0
    print(
        f"  {symbol_code}: final=${final_total:,.2f} "
        f"pnl=${pnl_usd:+,.2f} ({pnl_usd/ACCOUNT*100:+.1f}%) "
        f"trades={n_pos} wr={wr:.1f}%"
    )

    engine.dispose()
    return positions


if __name__ == "__main__":
    for code, pair in [("EURUSD", "EUR/USD"), ("USDJPY", "USD/JPY")]:
        print(f"\n=== {code} ===")
        try:
            run_backtest(code, pair)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  {code}: ERROR {e}")
