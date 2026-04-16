"""Backtest the scalper on historical 1H data (paper mode).

Uses the existing 1H Dukascopy data to simulate the scalper.
Although the scalper is designed for 1min bars, this validates
the entire architecture (engine, risk manager, position management)
on the data we have. When 5min/1min data is available, just swap
the PaperConnector's data source.

Token rule: summary only.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trade.scalper.connectors.paper_connector import PaperConnector
from trade.scalper.engine import EngineConfig, ScalperEngine
from trade.scalper.risk_manager import RiskConfig
from trade.scalper.strategies.momentum_scalper import MomentumConfig


def main():
    # Adapted momentum config for 1H bars (wider parameters)
    momentum_cfg = MomentumConfig(
        ema_fast=5,
        ema_slow=20,
        rsi_period=7,
        atr_period=14,
        atr_sl_mult=1.0,
        atr_tp_mult=1.5,
        vwap_threshold=0.001,  # wider for 1H
        min_bars_for_signal=30,
        momentum_lookback=3,
        momentum_min_ratio=0.66,
    )

    risk_cfg = RiskConfig(
        initial_balance=10_000.0,
        risk_per_trade_pct=0.5,
        max_concurrent_positions=3,
        max_consecutive_losses=3,
        cooldown_minutes=30,
    )

    engine_cfg = EngineConfig(
        symbols=["EURUSD", "USDJPY", "XAUUSD"],
        poll_interval_seconds=0,  # no sleep in backtest
        bars_lookback=50,
        timeframe="1h",
        momentum_config=momentum_cfg,
        risk_config=risk_cfg,
    )

    connector = PaperConnector(
        initial_balance=10_000.0,
        spread={
            "EURUSD": 0.00008,
            "USDJPY": 0.008,
            "XAUUSD": 0.40,
        },
        slippage_pct=0.0001,
    )

    engine = ScalperEngine(connector=connector, config=engine_cfg)

    # Run backtest on first 20K bars (~3 years of 1H)
    result = engine.run_backtest(max_bars=20000)

    print("\n" + "=" * 50)
    print("SCALPER BACKTEST RESULT")
    print("=" * 50)
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
