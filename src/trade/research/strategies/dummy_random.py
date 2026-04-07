"""DummyRandom — a strategy with zero edge, by construction.

Picks BUY/SELL at random at a configurable per-bar probability. Uses a
fixed ATR-based stop/target. Exists ONLY so we can prove the validation
pipeline rejects noise under strict conditions.

If ANY of the validation gates ever bless this, the pipeline is broken.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import talib


class DummyRandom:
    name: str = "DummyRandom"
    # No edge by construction, so it is not authorized in any regime.
    # The validation pipeline rejects it long before this matters, but
    # the field is required by StrategyProtocol.
    supported_regimes: list[int] = []
    default_params: dict[str, Any] = {
        "entry_prob": 0.05,   # ~5% of eligible bars
        "atr_sl_mult": 1.0,
        "atr_tp_mult": 1.0,   # symmetric = expected value ~0 before costs
        "max_bars": 12,
        "seed": 42,
    }
    # Small grid so DSR penalty is not absurd (strategy has no real knobs).
    param_grid: dict[str, list[Any]] = {
        "entry_prob": [0.03, 0.05, 0.08],
        "atr_sl_mult": [0.8, 1.0, 1.2],
        "atr_tp_mult": [0.8, 1.0, 1.2],
        "max_bars": [6, 12, 24],
    }

    account: float = 10_000.0
    risk_per_trade: float = 0.003

    # ------------------------------------------------------------------
    def _walk(
        self,
        df: pd.DataFrame,
        spread: float,
        params: dict[str, Any],
        signal_shift: int = 0,
        collect_signals: bool = False,
    ):
        """Shared inner loop used by both backtest() and signals().

        When collect_signals=True returns a list of dicts with entry info
        for the NautilusHarness replay. When False returns per-trade PnL
        list for paranoid/CPCV.
        """
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        n = len(close)

        atr = talib.ATR(high, low, close, timeperiod=14)

        rng = np.random.default_rng(params.get("seed", 42))
        entry_prob = params["entry_prob"]
        sl_mult = params["atr_sl_mult"]
        tp_mult = params["atr_tp_mult"]
        max_bars = int(params["max_bars"])

        pnls: list[float] = []
        signals: list[dict] = []
        pos = None
        equity = self.account

        start = max(50, -signal_shift + 1)
        end = n - max_bars - max(0, signal_shift)

        for i in range(start, end):
            sig_i = i - signal_shift
            if sig_i < 0 or sig_i >= n:
                continue
            if math.isnan(atr[sig_i]):
                continue

            if pos is None:
                # Random entry
                if rng.random() > entry_prob:
                    continue
                side = "buy" if rng.random() < 0.5 else "sell"
                entry = close[i]
                if side == "buy":
                    sl = entry - atr[sig_i] * sl_mult
                    tp = entry + atr[sig_i] * tp_mult
                    risk = entry - sl
                else:
                    sl = entry + atr[sig_i] * sl_mult
                    tp = entry - atr[sig_i] * tp_mult
                    risk = sl - entry
                if risk <= 0:
                    continue
                qty = (equity * self.risk_per_trade) / risk
                pos = {
                    "side": side, "entry": entry, "idx": i,
                    "sl": sl, "tp": tp, "qty": qty,
                }
                if collect_signals:
                    # bar_idx is 1-based for Nautilus (its first bar is #1)
                    signals.append({
                        "bar_idx": i + 1,
                        "side": side,
                        "sl": float(sl),
                        "tp": float(tp),
                    })
            else:
                bars = i - pos["idx"]
                exit_price = None
                if pos["side"] == "buy":
                    if low[i] <= pos["sl"]:
                        exit_price = pos["sl"]
                    elif high[i] >= pos["tp"]:
                        exit_price = pos["tp"]
                    elif bars >= max_bars:
                        exit_price = close[i]
                    if exit_price is not None:
                        pnl = (exit_price - pos["entry"]) * pos["qty"] - spread * pos["qty"] * 2
                        pnls.append(pnl)
                        equity += pnl
                        pos = None
                else:
                    if high[i] >= pos["sl"]:
                        exit_price = pos["sl"]
                    elif low[i] <= pos["tp"]:
                        exit_price = pos["tp"]
                    elif bars >= max_bars:
                        exit_price = close[i]
                    if exit_price is not None:
                        pnl = (pos["entry"] - exit_price) * pos["qty"] - spread * pos["qty"] * 2
                        pnls.append(pnl)
                        equity += pnl
                        pos = None

        if collect_signals:
            return pd.DataFrame(signals) if signals else pd.DataFrame(
                columns=["bar_idx", "side", "sl", "tp"]
            )
        return pnls

    # ------------------------------------------------------------------
    # StrategyProtocol API
    def backtest(
        self,
        df: pd.DataFrame,
        spread: float,
        params: dict[str, Any],
        signal_shift: int = 0,
    ) -> list[float]:
        return self._walk(df, spread, params, signal_shift, collect_signals=False)

    def signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        return self._walk(df, 0.0, params, 0, collect_signals=True)
