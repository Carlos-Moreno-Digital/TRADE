"""Launch the scalper in LIVE mode on TradeLocker.

Prerequisites:
  1. Set in .env:
       TRADELOCKER_EMAIL=your@email.com
       TRADELOCKER_PASSWORD=yourpassword
       TRADELOCKER_SERVER=servername
       TRADELOCKER_ENV=https://demo.tradelocker.com

  2. pip install tradelocker

Run:
    python run_scalper_live.py

Or in screen for persistence:
    screen -dmS scalper bash -c 'cd ~/TRADE && source venv/bin/activate && python -u run_scalper_live.py 2>&1 | tee /tmp/scalper.log'

Monitor:
    tail -f /tmp/scalper.log
    # or
    screen -r scalper
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trade.scalper.connectors.tradelocker_connector import TradeLockerConnector
from trade.scalper.engine import EngineConfig, ScalperEngine
from trade.scalper.risk_manager import RiskConfig
from trade.scalper.strategies.momentum_scalper import MomentumConfig


def main():
    momentum_cfg = MomentumConfig(
        ema_fast=5,
        ema_slow=20,
        rsi_period=7,
        atr_period=14,
        atr_sl_mult=1.0,
        atr_tp_mult=1.2,
        vwap_threshold=0.0002,
        min_bars_for_signal=30,
        momentum_lookback=3,
        momentum_min_ratio=0.66,
    )

    risk_cfg = RiskConfig(
        initial_balance=10_000.0,
        risk_per_trade_pct=0.5,
        max_concurrent_positions=2,
        max_consecutive_losses=3,
        cooldown_minutes=30,
        session_start_utc=7,    # London open
        session_end_utc=17,     # NY close
    )

    engine_cfg = EngineConfig(
        symbols=["EURUSD", "XAUUSD"],
        poll_interval_seconds=60,  # check every 1 minute
        bars_lookback=100,
        timeframe="1m",
        momentum_config=momentum_cfg,
        risk_config=risk_cfg,
    )

    connector = TradeLockerConnector()
    engine = ScalperEngine(connector=connector, config=engine_cfg)

    print("=" * 50, flush=True)
    print("SCALPER LIVE — FunderPro / TradeLocker", flush=True)
    print(f"  Symbols: {engine_cfg.symbols}", flush=True)
    print(f"  Risk: {risk_cfg.risk_per_trade_pct}% per trade", flush=True)
    print(f"  Session: {risk_cfg.session_start_utc}:00-{risk_cfg.session_end_utc}:00 UTC", flush=True)
    print(f"  Kill switches: daily {risk_cfg.kill_daily_loss_pct}% / total {risk_cfg.kill_total_dd_pct}%", flush=True)
    print("=" * 50, flush=True)

    engine.run_live()


if __name__ == "__main__":
    main()
