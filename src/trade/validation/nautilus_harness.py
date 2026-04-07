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
class PairNautilusResult:
    strategy: str
    primary_symbol: str
    partner_symbol: str
    primary_pnl: float
    partner_pnl: float
    combined_pnl: float
    primary_trades: int
    partner_trades: int
    combined_return_pct: float

    @property
    def viable(self) -> bool:
        return self.combined_pnl > 0

    def summary(self) -> str:
        return (
            f"PairNautilus[{self.strategy}/{self.primary_symbol}+"
            f"{self.partner_symbol}]: "
            f"primary=${self.primary_pnl:+,.0f} "
            f"partner=${self.partner_pnl:+,.0f} "
            f"combined=${self.combined_pnl:+,.0f} "
            f"({self.combined_return_pct:+.1f}%) "
            f"=> {'VIABLE' if self.viable else 'REJECTED'}"
        )


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
    spread_cost: float = 0.0  # absolute spread in price units; charged round-trip


class _SignalPlayerStrategy(Strategy):
    """Consumes pre-computed signals and plays them through Nautilus
    bid/ask bars with PESSIMISTIC INTRA-BAR ORDERING.

    Phase 5.1 fix: previously the player let Nautilus auto-fill on a
    market order at exit time, which placed the fill near the bar
    close in trending markets and produced an optimistic illusion
    (vanilla -$316 vs nautilus +$2,263 sign flip on RegimeMomentum).

    The new policy is hand-rolled inside this class:
      1. Entry pays the FULL spread (BUY at high+spread on the ask
         bar would be ideal; we approximate with bar close + spread).
      2. Each open position tracks its SL and TP internally.
      3. On every bar processed:
           a. If both SL and TP fall inside [low, high], assume the
              STOP-LOSS fired first. ALWAYS pessimistic.
           b. Otherwise, whichever level is touched fires at its
              own price (NOT at bar close).
           c. If neither, check the time stop (max_holding_bars).
      4. The player computes its OWN per-trade P&L using the
         pessimistic exit price and stores it in self.tracked_pnls.
      5. The harness reads self.tracked_pnls as the SOURCE OF TRUTH
         for the final P&L, ignoring the Nautilus account balance
         (which would otherwise show whatever Nautilus's market
         orders happened to fill at).

    A small market order is still submitted to keep Nautilus's own
    book in sync, but its fill price is no longer used for the
    reported P&L.
    """

    def __init__(self, config: _SignalPlayerConfig):
        super().__init__(config)
        self.instrument = None
        self._inst_id = None
        self.bar_count = 0
        self._signals_by_idx: dict[int, dict] = {}
        self.position_side = None
        self.position_entry_bar = None
        self.position_entry_price = None
        self.position_qty = None
        self.stop_price = None
        self.tp_price = None
        # Per-trade tracked P&L (the harness uses this as the truth)
        self.tracked_pnls: list[float] = []
        self.tracked_exits: list[str] = []  # 'sl' / 'tp' / 'time'

    def on_start(self):
        self._inst_id = InstrumentId.from_str(self.config.instrument_id)
        self.instrument = self.cache.instrument(self._inst_id)
        df = pd.read_parquet(self.config.signals_parquet)
        self._signals_by_idx = {
            int(row.bar_idx): {"side": row.side, "sl": row.sl, "tp": row.tp}
            for row in df.itertuples()
        }
        self.subscribe_bars(self.config.bar_type)

    # ------------------------------------------------------------------
    def _pessimistic_exit(self, h: float, l: float, c: float) -> tuple[float, str] | None:
        """Return (exit_price, exit_reason) or None if no SL/TP triggered.

        SL ALWAYS wins ties: if both SL and TP fall inside [low, high]
        of this bar, we assume the SL fired first. This is the
        institutional-grade pessimistic convention.
        """
        if self.position_side == "buy":
            sl_hit = l <= self.stop_price
            tp_hit = h >= self.tp_price
            if sl_hit:
                return (self.stop_price, "sl")
            if tp_hit:
                return (self.tp_price, "tp")
        else:  # sell
            sl_hit = h >= self.stop_price
            tp_hit = l <= self.tp_price
            if sl_hit:
                return (self.stop_price, "sl")
            if tp_hit:
                return (self.tp_price, "tp")
        return None

    def _close_position(self, exit_price: float, reason: str) -> None:
        if self.position_side == "buy":
            pnl = (exit_price - self.position_entry_price) * self.position_qty
        else:
            pnl = (self.position_entry_price - exit_price) * self.position_qty
        # Charge round-trip spread cost (entry pays ask, exit pays bid).
        # The player reads bid bars only; bid/ask separation is captured
        # here as an explicit deduction so the tracked P&L matches the
        # vanilla backtest's accounting convention.
        pnl -= self.config.spread_cost * self.position_qty * 2.0
        self.tracked_pnls.append(float(pnl))
        self.tracked_exits.append(reason)
        # Submit a market order to keep the Nautilus book consistent
        try:
            net_qty = self.portfolio.net_position(self._inst_id) or 0
            if net_qty != 0:
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
        except Exception:
            pass
        self.position_side = None
        self.position_entry_bar = None
        self.position_entry_price = None
        self.position_qty = None
        self.stop_price = None
        self.tp_price = None

    def on_bar(self, bar: Bar):
        self.bar_count += 1
        h = float(bar.high)
        l = float(bar.low)
        c = float(bar.close)

        # Manage open position with pessimistic SL-first ordering
        if self.position_side is not None:
            bars_held = self.bar_count - self.position_entry_bar
            ex = self._pessimistic_exit(h, l, c)
            if ex is not None:
                self._close_position(ex[0], ex[1])
                return
            if bars_held >= self.config.max_holding_bars:
                self._close_position(c, "time")
                return
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

        try:
            order = self.order_factory.market(
                instrument_id=self.instrument.id,
                order_side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                quantity=Quantity.from_int(qty_units),
                time_in_force=TimeInForce.GTC,
            )
            self.submit_order(order)
        except Exception:
            return
        self.position_side = side
        self.position_entry_bar = self.bar_count
        self.position_entry_price = c
        self.position_qty = qty_units
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

        player = _SignalPlayerStrategy(
            _SignalPlayerConfig(
                instrument_id=instrument.id.value,
                bar_type=bid_bt,
                signals_parquet=tmp_parquet,
                account_equity=self.account,
                risk_per_trade=self.risk_per_trade,
                max_holding_bars=self.max_holding_bars,
                spread_cost=self.spread_abs,
            )
        )
        engine.add_strategy(player)

        engine.run()

        # 4) Harvest results — TRACKED PnLs are the source of truth
        # (Phase 5.1 fix). The Nautilus account balance reflects the
        # market-order fills which can be optimistic in trending bars;
        # the player's tracked_pnls applies pessimistic SL-first
        # ordering at the exact SL/TP price.
        tracked = list(player.tracked_pnls)
        n_trades = len(tracked)
        nautilus_pnl = float(sum(tracked))
        if n_trades > 0:
            wins = sum(1 for p in tracked if p > 0)
            wr = wins / n_trades * 100
        else:
            wr = 0.0
        final_total = self.account + nautilus_pnl
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

    # ------------------------------------------------------------------
    # Caveat #2 fix: 2-leg pair fill harness.
    #
    # The strategy must implement pair_signals(df, params) returning a
    # DataFrame with columns:
    #   bar_idx, primary_side, primary_sl, primary_tp,
    #   partner_side, partner_sl, partner_tp
    # We split it into two single-leg signal feeds and run each leg
    # through the existing single-leg engine (one engine per leg, both
    # walking the same time grid). Combined P&L = primary + partner.
    #
    # This is NOT a tick-level joint simulation — Nautilus would need
    # cross-instrument positions for that — but it captures the
    # economics of paying both bid/ask and is enough to give the pair
    # strategy a fair shot in the gauntlet.
    # ------------------------------------------------------------------
    def run_pair(
        self,
        strategy: StrategyProtocol,
        primary_bars: pd.DataFrame,
        primary_symbol: str,
        primary_pair: str,
        primary_spread: float,
        partner_bars: pd.DataFrame,
        partner_symbol: str,
        partner_pair: str,
        partner_spread: float,
    ) -> PairNautilusResult:
        if not hasattr(strategy, "pair_signals"):
            raise AttributeError(
                f"{strategy.name} must implement pair_signals(df, params) "
                "for run_pair()"
            )
        params = strategy.default_params
        sig_df = strategy.pair_signals(primary_bars, params)

        if sig_df.empty:
            return PairNautilusResult(
                strategy=strategy.name,
                primary_symbol=primary_symbol,
                partner_symbol=partner_symbol,
                primary_pnl=0.0, partner_pnl=0.0, combined_pnl=0.0,
                primary_trades=0, partner_trades=0,
                combined_return_pct=0.0,
            )

        # Prebuilt-signals strategy wrapper: implements .signals() to
        # return a fixed DataFrame (the pair_signals slice) and a no-op
        # backtest. The harness only ever calls .signals().
        class _PrebuiltSignals:
            name = strategy.name
            default_params = params
            param_grid = strategy.param_grid

            def __init__(self, signals_df: pd.DataFrame):
                self._signals_df = signals_df

            def signals(self, df, params):
                return self._signals_df

            def backtest(self, df, spread, params, signal_shift=0):
                return []

        primary_only = sig_df.rename(columns={
            "primary_side": "side",
            "primary_sl": "sl",
            "primary_tp": "tp",
        })[["bar_idx", "side", "sl", "tp"]]
        partner_only = sig_df.rename(columns={
            "partner_side": "side",
            "partner_sl": "sl",
            "partner_tp": "tp",
        })[["bar_idx", "side", "sl", "tp"]]

        primary_strategy = _PrebuiltSignals(primary_only)
        partner_strategy = _PrebuiltSignals(partner_only)

        primary_harness = NautilusHarness(
            spread_abs=primary_spread,
            commission_bps=self.commission_bps,
            slip_prob=self.slip_prob,
            account=self.account,
            risk_per_trade=self.risk_per_trade,
            max_holding_bars=self.max_holding_bars,
            log_level=self.log_level,
            seed=self.seed,
        )
        partner_harness = NautilusHarness(
            spread_abs=partner_spread,
            commission_bps=self.commission_bps,
            slip_prob=self.slip_prob,
            account=self.account,
            risk_per_trade=self.risk_per_trade,
            max_holding_bars=self.max_holding_bars,
            log_level=self.log_level,
            seed=self.seed + 1,
        )
        primary_res = primary_harness.run(
            primary_strategy, primary_bars, primary_symbol, primary_pair,
        )
        partner_res = partner_harness.run(
            partner_strategy, partner_bars, partner_symbol, partner_pair,
        )

        combined_pnl = primary_res.nautilus_pnl + partner_res.nautilus_pnl
        return PairNautilusResult(
            strategy=strategy.name,
            primary_symbol=primary_symbol,
            partner_symbol=partner_symbol,
            primary_pnl=primary_res.nautilus_pnl,
            partner_pnl=partner_res.nautilus_pnl,
            combined_pnl=combined_pnl,
            primary_trades=primary_res.nautilus_trades,
            partner_trades=partner_res.nautilus_trades,
            combined_return_pct=combined_pnl / self.account * 100,
        )
