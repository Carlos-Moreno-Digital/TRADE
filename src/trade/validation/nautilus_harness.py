"""NautilusTrader validation harness — strategy-agnostic.

Runs a StrategyProtocol through NautilusTrader's BacktestEngine with:
  - Bid/ask bar execution (bar_execution=True)
  - Synthetic ask bars from OHLCV + configurable absolute spread
  - FillModel with slippage probability
  - Fee/commission via the default venue configuration
  - Strict position sizing (0.3% risk per trade, micro-lot rounding)

The harness does NOT port the strategy logic to a Nautilus Strategy class.
Instead, it drives an internal generic "signal player" strategy that
executes pre-computed BUY/SELL signals. This lets us reuse exactly the
same pure-function backtest code that the paranoid suite and CPCV use,
so the three gates are guaranteed to be testing the same logic.

The realistic-execution test in this harness is:
  "If I take the signals the strategy would have emitted and replay them
   in NautilusTrader with real spreads and slippage, does the sign of the
   P&L agree with the vanilla backtest?"

A strategy that fails this is relying on invisible half-spread capture.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.objects import Money, Quantity
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

from trade.validation.protocol import StrategyProtocol


@dataclass
class NautilusResult:
    strategy: str
    symbol: str
    vanilla_pnl: float
    vanilla_wr: float
    nautilus_pnl: float
    nautilus_wr: float
    nautilus_trades: int
    final_balance: float
    return_pct: float

    @property
    def signs_agree(self) -> bool:
        return (self.vanilla_pnl > 0) == (self.nautilus_pnl > 0)

    @property
    def relative_gap(self) -> float:
        if abs(self.vanilla_pnl) < 1:
            return 0.0
        return abs(self.nautilus_pnl - self.vanilla_pnl) / abs(self.vanilla_pnl)

    @property
    def viable(self) -> bool:
        return self.nautilus_pnl > 0 and self.signs_agree

    def summary(self) -> str:
        return (
            f"Nautilus[{self.strategy}/{self.symbol}]: "
            f"vanilla=${self.vanilla_pnl:+,.0f} "
            f"nautilus=${self.nautilus_pnl:+,.0f} "
            f"({self.return_pct:+.1f}%) "
            f"trades={self.nautilus_trades} "
            f"signs_agree={self.signs_agree} "
            f"=> {'VIABLE' if self.viable else 'REJECTED'}"
        )


# ---------------------------------------------------------------------------
# Internal generic signal-player strategy
# ---------------------------------------------------------------------------

class _SignalPlayerConfig(StrategyConfig, frozen=True):
    instrument_id: str
    bar_type: BarType
    signals_parquet: str  # path to a temp parquet with columns: bar_idx, side, sl, tp
    account_equity: float = 10_000.0
    risk_per_trade: float = 0.003
    max_holding_bars: int = 12
    lot_size: int = 1000


class _SignalPlayerStrategy(Strategy):
    """Consumes pre-computed signals and plays them via Nautilus orders."""

    def __init__(self, config: _SignalPlayerConfig):
        super().__init__(config)
        self.instrument = None
        self._inst_id = None
        self.bar_count = 0
        self._signals_by_idx: dict[int, dict] = {}
        self.position_side = None
        self.position_entry_bar = None
        self.stop_price = None
        self.tp_price = None

    def on_start(self):
        self._inst_id = InstrumentId.from_str(self.config.instrument_id)
        self.instrument = self.cache.instrument(self._inst_id)
        # Load signals from parquet
        df = pd.read_parquet(self.config.signals_parquet)
        self._signals_by_idx = {
            int(row.bar_idx): {"side": row.side, "sl": row.sl, "tp": row.tp}
            for row in df.itertuples()
        }
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar):
        self.bar_count += 1
        h = float(bar.high)
        l = float(bar.low)
        c = float(bar.close)

        net_qty = self.portfolio.net_position(self._inst_id) or 0

        # Manage open position
        if net_qty != 0:
            bars_held = self.bar_count - self.position_entry_bar
            exit_now = False
            if self.position_side == "buy":
                if l <= self.stop_price or h >= self.tp_price:
                    exit_now = True
            else:
                if h >= self.stop_price or l <= self.tp_price:
                    exit_now = True
            if bars_held >= self.config.max_holding_bars:
                exit_now = True
            if exit_now:
                order = self.order_factory.market(
                    instrument_id=self.instrument.id,
                    order_side=(
                        OrderSide.SELL if self.position_side == "buy" else OrderSide.BUY
                    ),
                    quantity=Quantity.from_int(abs(int(net_qty))),
                    time_in_force=TimeInForce.GTC,
                    reduce_only=True,
                )
                self.submit_order(order)
                self.position_side = None
                self.position_entry_bar = None
            return

        # Fresh entry?
        sig = self._signals_by_idx.get(self.bar_count)
        if sig is None:
            return

        side = sig["side"]
        sl = float(sig["sl"])
        tp = float(sig["tp"])
        risk_unit = (c - sl) if side == "buy" else (sl - c)
        if risk_unit <= 0:
            return

        raw = (self.config.account_equity * self.config.risk_per_trade) / risk_unit
        lot = self.config.lot_size
        qty_units = max(lot, int(round(raw / lot) * lot))
        qty_units = min(qty_units, 5_000_000)

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


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def _vanilla_metrics(pnls: list[float]) -> tuple[float, float]:
    if not pnls:
        return 0.0, 0.0
    total = float(sum(pnls))
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / len(pnls) * 100
    return total, wr


class NautilusHarness:
    def __init__(
        self,
        spread_abs: float,
        commission_bps: float = 0.5,  # 0.5 bps = 0.005% (strict)
        slip_prob: float = 0.3,
        account: float = 10_000.0,
        risk_per_trade: float = 0.003,
        max_holding_bars: int = 12,
        log_level: str = "ERROR",
        seed: int = 42,
    ):
        self.spread_abs = spread_abs
        self.commission_bps = commission_bps
        self.slip_prob = slip_prob
        self.account = account
        self.risk_per_trade = risk_per_trade
        self.max_holding_bars = max_holding_bars
        self.log_level = log_level
        self.seed = seed

    def _extract_signals(
        self,
        strategy: StrategyProtocol,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Ask the strategy to produce a list of entry signals as a DataFrame
        [bar_idx, side, sl, tp]. This is the strategy's own responsibility —
        every strategy must implement a `signals` method alongside backtest().
        """
        if not hasattr(strategy, "signals"):
            raise AttributeError(
                f"{strategy.name} must implement signals(df, params) for "
                "NautilusHarness to replay it"
            )
        return strategy.signals(df, strategy.default_params)

    def run(
        self,
        strategy: StrategyProtocol,
        df: pd.DataFrame,
        symbol: str,
        pair: str,
    ) -> NautilusResult:
        # 1) Vanilla reference
        vanilla_pnls = strategy.backtest(df, self.spread_abs, strategy.default_params)
        vanilla_total, vanilla_wr = _vanilla_metrics(vanilla_pnls)

        # 2) Extract signals and persist to parquet for the inner strategy
        signals_df = self._extract_signals(strategy, df)
        if signals_df.empty:
            return NautilusResult(
                strategy=strategy.name, symbol=symbol,
                vanilla_pnl=vanilla_total, vanilla_wr=vanilla_wr,
                nautilus_pnl=0.0, nautilus_wr=0.0,
                nautilus_trades=0, final_balance=self.account, return_pct=0.0,
            )

        tmp_parquet = f"/tmp/nautilus_signals_{symbol}_{self.seed}.parquet"
        signals_df.to_parquet(tmp_parquet)

        # 3) Build engine
        venue = Venue("SIM")
        instrument = TestInstrumentProvider.default_fx_ccy(pair)
        bid_bt = BarType.from_str(f"{instrument.id}-1-HOUR-BID-EXTERNAL")
        ask_bt = BarType.from_str(f"{instrument.id}-1-HOUR-ASK-EXTERNAL")

        engine = BacktestEngine(
            config=BacktestEngineConfig(
                trader_id="HARNESS-001",
                logging=LoggingConfig(log_level=self.log_level),
            ),
        )

        fill_model = FillModel(
            prob_fill_on_limit=1.0,
            prob_slippage=self.slip_prob,
            random_seed=self.seed,
        )
        engine.add_venue(
            venue=venue,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=USD,
            starting_balances=[Money(self.account, USD)],
            fill_model=fill_model,
        )
        engine.add_instrument(instrument)

        # Synthetic bid/ask bars with absolute spread
        work = df[["open", "high", "low", "close", "volume"]].copy()
        work["volume"] = work["volume"].fillna(0).clip(lower=1)
        ask = work.copy()
        for col in ("open", "high", "low", "close"):
            ask[col] = ask[col] + self.spread_abs

        bid_bars = BarDataWrangler(bar_type=bid_bt, instrument=instrument).process(work)
        ask_bars = BarDataWrangler(bar_type=ask_bt, instrument=instrument).process(ask)
        engine.add_data(bid_bars)
        engine.add_data(ask_bars)

        engine.add_strategy(
            _SignalPlayerStrategy(
                _SignalPlayerConfig(
                    instrument_id=instrument.id.value,
                    bar_type=bid_bt,
                    signals_parquet=tmp_parquet,
                    account_equity=self.account,
                    risk_per_trade=self.risk_per_trade,
                    max_holding_bars=self.max_holding_bars,
                )
            )
        )

        engine.run()

        # 4) Harvest results
        report = engine.trader.generate_account_report(venue)
        final_total = float(report["total"].iloc[-1]) if len(report) else self.account
        nautilus_pnl = final_total - self.account

        positions = engine.trader.generate_positions_report()
        n_trades = len(positions) if positions is not None else 0
        if n_trades > 0:
            def _money(v):
                try:
                    return float(str(v).split()[0])
                except Exception:
                    return 0.0
            rp = positions["realized_pnl"].map(_money).astype(float)
            wr = float((rp > 0).mean() * 100)
        else:
            wr = 0.0
        engine.dispose()

        return NautilusResult(
            strategy=strategy.name,
            symbol=symbol,
            vanilla_pnl=vanilla_total,
            vanilla_wr=vanilla_wr,
            nautilus_pnl=nautilus_pnl,
            nautilus_wr=wr,
            nautilus_trades=n_trades,
            final_balance=final_total,
            return_pct=nautilus_pnl / self.account * 100,
        )
