"""Scalper Engine — the main trading loop.

Polls the connector every `poll_interval` seconds, evaluates
strategies on the latest bars, checks risk, and executes.

Modes:
  LIVE:  connected to TradeLocker or any broker via BaseConnector
  PAPER: uses PaperConnector with historical data (for backtesting
         or paper validation before going live)

The engine is stateless between iterations — all state lives in the
risk manager and the connector. This means a crash and restart
picks up cleanly from the current account state.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from trade.scalper.connectors.base import BaseConnector, OrderStatus, Side
from trade.scalper.risk_manager import RiskConfig, RiskManager
from trade.scalper.strategies.momentum_scalper import MomentumConfig, MomentumScalper


@dataclass
class EngineConfig:
    symbols: list[str] = field(default_factory=lambda: ["EURUSD", "XAUUSD"])
    poll_interval_seconds: int = 60  # check every 60s (= every 1min bar close)
    bars_lookback: int = 100         # how many 1min bars to fetch per evaluation
    timeframe: str = "1m"
    log_dir: Path = Path("data/scalper_logs")
    # Strategy configs
    momentum_config: MomentumConfig = field(default_factory=MomentumConfig)
    # Risk config
    risk_config: RiskConfig = field(default_factory=RiskConfig)


class ScalperEngine:
    def __init__(
        self,
        connector: BaseConnector,
        config: EngineConfig | None = None,
    ):
        self.connector = connector
        self.cfg = config or EngineConfig()
        self.risk = RiskManager(self.cfg.risk_config)
        self.strategy = MomentumScalper(self.cfg.momentum_config)
        self.cfg.log_dir.mkdir(parents=True, exist_ok=True)
        self._running = False
        self._iteration = 0

    def _log(self, msg: str) -> None:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        # Append to daily log file
        log_file = self.cfg.log_dir / f"{datetime.utcnow().strftime('%Y-%m-%d')}.log"
        try:
            with open(log_file, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _check_and_close_sl_tp(self) -> None:
        """For paper connector: manually check SL/TP.
        Live connector handles SL/TP server-side."""
        if hasattr(self.connector, "check_sl_tp"):
            closed = self.connector.check_sl_tp()
            for pid in closed:
                # Find the trade in history to get PnL
                if hasattr(self.connector, "trade_history"):
                    for t in reversed(self.connector.trade_history):
                        if t["id"] == pid:
                            self.risk.record_trade_result(
                                t["pnl"], datetime.utcnow()
                            )
                            emoji = "+" if t["pnl"] >= 0 else ""
                            self._log(
                                f"  CLOSED {t['symbol']} {t['side']} "
                                f"pnl={emoji}${t['pnl']:.2f}"
                            )
                            break

    def _evaluate_symbol(self, symbol: str) -> None:
        """Fetch bars, evaluate strategy, execute if signal + risk allows."""
        now = datetime.utcnow()

        # Get current account state
        account = self.connector.get_account_info()
        positions = self.connector.get_positions()
        n_open = len(positions)

        # Risk check
        risk_decision = self.risk.check_can_trade(
            now, account.balance, account.equity, n_open
        )
        if not risk_decision.allowed:
            return  # silently skip (risk manager logs its own reasons)

        # Fetch bars
        bars = self.connector.get_bars(symbol, self.cfg.timeframe, self.cfg.bars_lookback)
        if len(bars) < self.cfg.momentum_config.min_bars_for_signal:
            return

        # Evaluate strategy
        signal = self.strategy.evaluate(bars)
        if signal is None:
            return

        # Calculate position size
        lots = self.risk.calculate_position_size(
            symbol, signal.entry_price, signal.sl, account.balance
        )
        if lots <= 0:
            return

        # Execute
        result = self.connector.open_position(
            symbol=symbol,
            side=signal.side,
            quantity=lots,
            sl=signal.sl,
            tp=signal.tp,
        )

        if result.status == OrderStatus.FILLED:
            self.risk.last_trade_time = now
            self._log(
                f"  OPEN {symbol} {signal.side.value.upper()} "
                f"lots={lots:.2f} entry={signal.entry_price:.5f} "
                f"sl={signal.sl:.5f} tp={signal.tp:.5f} "
                f"[{signal.reason}]"
            )
        else:
            self._log(
                f"  ORDER REJECTED {symbol}: {result.message}"
            )

    def _iteration_step(self) -> None:
        """One full iteration of the scalper loop."""
        self._iteration += 1
        now = datetime.utcnow()

        # Weekend close check
        if self.risk.should_close_for_weekend(now):
            n = self.connector.close_all_positions()
            if n > 0:
                self._log(f"WEEKEND CLOSE: closed {n} positions")
            return

        # Check SL/TP on paper connector
        self._check_and_close_sl_tp()

        # Evaluate each symbol
        for symbol in self.cfg.symbols:
            try:
                self._evaluate_symbol(symbol)
            except Exception as e:
                self._log(f"  ERROR {symbol}: {e}")

        # Periodic status log (every 30 iterations = every 30 min)
        if self._iteration % 30 == 0:
            account = self.connector.get_account_info()
            status = self.risk.status_summary(account.balance, account.equity)
            self._log(f"STATUS: {json.dumps(status)}")

    def run_live(self) -> None:
        """Main live loop — runs until killed or risk manager kills it."""
        self._log(f"ENGINE START: symbols={self.cfg.symbols} "
                  f"poll={self.cfg.poll_interval_seconds}s")

        if not self.connector.connect():
            self._log("FATAL: connector failed to connect")
            return

        self._running = True
        try:
            while self._running:
                try:
                    self._iteration_step()
                except KeyboardInterrupt:
                    self._log("INTERRUPTED by user")
                    break
                except Exception as e:
                    self._log(f"ITERATION ERROR: {e}")

                if self.risk.killed_total:
                    self._log("KILLED: total DD limit reached. Shutting down.")
                    self.connector.close_all_positions()
                    break

                time.sleep(self.cfg.poll_interval_seconds)
        finally:
            self.connector.close_all_positions()
            self.connector.disconnect()
            self._log("ENGINE STOPPED")
            self._running = False

    def run_backtest(self, max_bars: int = 0) -> dict:
        """Run through historical data bar-by-bar (paper mode only).

        Returns a summary dict with PnL, trades, WR, etc.
        """
        if not hasattr(self.connector, "advance_bar"):
            raise RuntimeError("backtest requires PaperConnector")

        self._log(f"BACKTEST START: symbols={self.cfg.symbols}")
        self.connector.connect()

        # Find the primary symbol's data length
        primary = self.cfg.symbols[0]
        from trade.scalper.connectors.paper_connector import PaperConnector
        pc: PaperConnector = self.connector  # type: ignore
        df = pc._load_bars(primary)
        total_bars = len(df) if not df.empty else 0
        if max_bars > 0:
            total_bars = min(total_bars, max_bars)

        # Start from bar 100 (warmup)
        start = self.cfg.bars_lookback + 10
        for sym in self.cfg.symbols:
            pc.set_bar_index(sym, start)

        for i in range(start, total_bars):
            for sym in self.cfg.symbols:
                pc.set_bar_index(sym, i)
            self._iteration_step()

            # Progress every 10K bars
            if i % 10000 == 0 and i > start:
                account = pc.get_account_info()
                self._log(
                    f"  backtest bar {i:,}/{total_bars:,} "
                    f"balance=${account.balance:,.2f} "
                    f"trades={len(pc.trade_history)}"
                )

        # Final accounting
        pc.close_all_positions()
        account = pc.get_account_info()
        trades = pc.trade_history
        n = len(trades)
        pnls = [t["pnl"] for t in trades]
        total_pnl = sum(pnls)
        wr = sum(1 for p in pnls if p > 0) / n * 100 if n else 0
        import math
        import numpy as np
        arr = np.array(pnls) if pnls else np.array([0.0])
        sharpe = float(arr.mean() / arr.std() * math.sqrt(252 * 8)) if arr.std() > 0 else 0.0

        summary = {
            "bars_processed": total_bars - start,
            "trades": n,
            "pnl": round(total_pnl, 2),
            "wr": round(wr, 1),
            "sharpe": round(sharpe, 2),
            "final_balance": round(account.balance, 2),
            "max_dd_pct": round(
                (self.risk.initial_balance - min(
                    self.risk.initial_balance,
                    account.balance
                )) / self.risk.initial_balance * 100, 2
            ),
        }
        self._log(f"BACKTEST DONE: {json.dumps(summary)}")
        self.connector.disconnect()
        return summary

    def stop(self) -> None:
        self._running = False
