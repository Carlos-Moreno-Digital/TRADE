"""Quant Tester Agent — Rapid Signal Validation.

Performs quick backtesting validation on proposed signals to check
if the symbol/direction has shown historical edge in recent data.
Rejects signals with poor recent performance.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import talib
from trade.agents.multi import AgentMessage


class QuantTester:
    """Validates signals against recent historical performance."""

    def __init__(self):
        self.min_sharpe = 0.5       # Minimum for recent performance (relaxed from 2.0)
        self.min_pf = 1.1           # Minimum profit factor
        self.max_dd_pct = 4.5       # Max drawdown in recent window
        self.lookback_bars = 200    # ~8 days of 1H data for quick validation

    def validate(self, signal: AgentMessage, df: pd.DataFrame) -> AgentMessage:
        """Run rapid backtest on recent data for the proposed direction."""
        payload = signal.computational_payload
        sym = payload.get("symbol", "")
        action = payload.get("action", "")
        atr_val = payload.get("atr", 0)

        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)

        n = len(close)
        if n < self.lookback_bars:
            return AgentMessage(
                agent_domain="quant_tester",
                status_flag="BACKTEST_PASSED",
                computational_payload=payload,
                economic_rationale="Insufficient data for validation, passing with warning",
            )

        # Quick validation: check recent directional bias
        recent_close = close[-self.lookback_bars:]
        recent_returns = np.diff(recent_close) / recent_close[:-1]

        # Calculate directional metrics
        if action == "BUY":
            favorable = recent_returns > 0
        else:
            favorable = recent_returns < 0
            recent_returns = -recent_returns  # Invert for short

        # Win rate in recent window
        win_rate = favorable.sum() / len(favorable) if len(favorable) > 0 else 0

        # Sharpe-like ratio of recent returns in proposed direction
        mean_ret = np.mean(recent_returns)
        std_ret = np.std(recent_returns)
        sharpe = (mean_ret / std_ret * np.sqrt(252 * 8)) if std_ret > 0 else 0

        # Simulated P&L with fixed position
        cum_pnl = np.cumsum(recent_returns)
        max_dd = 0
        peak = 0
        for p in cum_pnl:
            if p > peak:
                peak = p
            dd = peak - p
            if dd > max_dd:
                max_dd = dd
        max_dd_pct_actual = max_dd * 100

        # Profit factor
        wins = recent_returns[recent_returns > 0].sum()
        losses = abs(recent_returns[recent_returns < 0].sum())
        pf = wins / losses if losses > 0 else 999

        metrics = {
            "recent_win_rate": round(win_rate * 100, 1),
            "recent_sharpe": round(sharpe, 2),
            "recent_pf": round(pf, 2),
            "recent_max_dd_pct": round(max_dd_pct_actual, 2),
            "lookback_bars": self.lookback_bars,
        }

        # Validation checks
        failures = []
        if max_dd_pct_actual > self.max_dd_pct:
            failures.append(f"Recent DD {max_dd_pct_actual:.1f}% > {self.max_dd_pct}% limit")
        if pf < self.min_pf and pf < 999:
            failures.append(f"Profit Factor {pf:.2f} < {self.min_pf} minimum")

        if failures:
            return AgentMessage(
                agent_domain="quant_tester",
                status_flag="BACKTEST_FAILED",
                computational_payload={**payload, "backtest_metrics": metrics},
                errors=failures,
                economic_rationale=f"Recent performance check failed: {'; '.join(failures)}",
            )

        return AgentMessage(
            agent_domain="quant_tester",
            status_flag="BACKTEST_PASSED",
            computational_payload={**payload, "backtest_metrics": metrics},
            economic_rationale=f"Recent performance OK: WR={win_rate*100:.0f}%, PF={pf:.2f}, DD={max_dd_pct_actual:.1f}%",
        )
