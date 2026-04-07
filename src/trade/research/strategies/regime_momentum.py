"""Regime-conditional Momentum strategy.

The Statistical Jump Model (SJM) is used as an ACTIVE STATE SELECTOR,
not a passive gate. The strategy switches its behaviour bar-by-bar
based on the regime label predicted from the rolling daily features:

  state 0 = Risk-On / calm  -> Donchian breakout (trend following)
  state 1 = Risk-Off / stress -> rolling z-score mean reversion
                                 (catching exhaustion at vol spikes)

Both branches share the same risk envelope:
  - 0.3% account risk per trade
  - ATR-based stop loss
  - ATR-based take profit (asymmetric in favour of trend, symmetric
    in favour of reversion)
  - Time stop: max_holding bars

Why this design:
  - Pure trend-following strategies are crushed in mean-reverting
    regimes (chop). Pure mean-reversion is crushed in trending
    regimes (large directional moves blow stops). The literature
    is consistent (e.g. Kaminski 2014, Hurst-Ooi-Pedersen 2017):
    momentum and mean-reversion are complementary regimes of the
    same return process, and a regime-aware switch can outperform
    either alone if the regime classifier is sharper than chance.
  - Our SJM is fitted independently per symbol on daily data, so
    it has no look-ahead from the 1H feature window of this
    strategy. The mapping from daily regime to 1H bar is forward-
    fill within the day, which is causal by construction.

supported_regimes = [0, 1] because the strategy ITSELF adapts.
The orchestrator's RegimeGate will not block this strategy when
the regime flips.

Causality:
  - At each bar t, the regime label is predicted using the daily
    SJM model on rolling daily features. The features at day d
    use closes [d-W : d] only. The forward-fill from daily to 1H
    is right-edge inclusive: bar t inherits the label of the
    most recent COMPLETED day (the label for day d is broadcast
    starting from the first 1H bar AFTER d 23:00).
  - Donchian and ATR are computed on the 1H bars with strict
    rolling windows; entry decisions at bar t use only data from
    bars [0 : t).
  - The block bootstrap, future-shift, and time-permutation tests
    in the paranoid suite all apply unchanged because the strategy
    is fully self-contained (no external state to permute).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import talib

from trade.research.regimes.features import build_features
from trade.research.regimes.jump_model import StatisticalJumpModel

ACCOUNT = 10_000.0
RISK_PER_TRADE = 0.003
DEFAULT_MODELS_DIR = Path("data/regimes")


def _normalize_symbol(symbol: str) -> str:
    s = symbol.upper().replace("=X", "").replace("^", "")
    aliases = {"JPY": "USDJPY", "CAD": "USDCAD", "GC": "XAUUSD", "GSPC": "SPX"}
    return aliases.get(s, s)


class RegimeMomentum:
    name: str = "RegimeMomentum"
    # Adapts to BOTH regimes (Risk-On = momentum, Risk-Off = mean reversion).
    # The strategy itself dispatches; the RegimeGate need not block it.
    supported_regimes: list[int] = [0, 1]

    default_params: dict[str, Any] = {
        # Momentum branch (state 0)
        "donchian_period": 20,
        "atr_period": 14,
        "atr_sl_mult": 1.5,
        "atr_tp_mult": 3.0,
        "max_holding": 24,
        # Mean-reversion branch (state 1)
        "mr_lookback": 20,
        "mr_z_entry": 2.0,
        "mr_z_exit": 0.5,
    }

    # Tiny grid kept cheap for the smoke run; the institutional
    # script can widen it.
    param_grid: dict[str, list[Any]] = {
        "donchian_period": [15, 20, 30],
        "atr_sl_mult": [1.0, 1.5],
        "atr_tp_mult": [2.0, 3.0],
        "mr_z_entry": [1.5, 2.0],
    }

    def __init__(
        self,
        symbol: str = "EURUSD",
        model_path: Path | None = None,
    ):
        self.symbol = _normalize_symbol(symbol)
        if model_path is None:
            model_path = DEFAULT_MODELS_DIR / f"{self.symbol}.json"
        if not model_path.exists():
            raise FileNotFoundError(
                f"SJM model for {self.symbol} not found at {model_path}. "
                f"Run persist_sjm.py first."
            )
        payload = json.loads(model_path.read_text())
        self.regime_model = StatisticalJumpModel.from_dict(payload)
        self.risk_off_state = int(payload.get("risk_off_state", 1))
        self.risk_on_state = int(payload.get("risk_on_state", 0))

    # ------------------------------------------------------------------
    def _regime_per_bar(self, bars: pd.DataFrame) -> np.ndarray:
        """Predict the SJM regime for every bar in `bars`.

        Causal mapping: resample to daily last-close, build features
        using only past closes, predict the regime sequence, then
        forward-fill from daily to 1H using the LATEST COMPLETED day's
        label (so bar at 2024-01-02 03:00 sees the label of the
        2024-01-01 daily prediction, NOT 2024-01-02).
        """
        daily = bars["close"].resample("1D").last().dropna()
        if len(daily) < 80:  # need warmup for the features
            return np.zeros(len(bars), dtype=int)
        feats = build_features(daily)
        if feats.empty:
            return np.zeros(len(bars), dtype=int)
        labels = self.regime_model.predict(feats)
        daily_labels = pd.Series(labels, index=feats.index)
        # Shift one day forward so each label only applies to the day
        # AFTER it was computed (no intra-day look-ahead).
        daily_labels.index = daily_labels.index + pd.Timedelta(days=1)
        bar_regimes = daily_labels.reindex(bars.index, method="ffill")
        bar_regimes = bar_regimes.fillna(self.risk_on_state).astype(int)
        return bar_regimes.values

    # ------------------------------------------------------------------
    def _walk(
        self,
        df: pd.DataFrame,
        spread: float,
        params: dict[str, Any],
        signal_shift: int = 0,
        collect_signals: bool = False,
    ):
        if not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
            raise ValueError("df missing OHLCV columns")
        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        n = len(close)

        donchian_period = int(params["donchian_period"])
        atr_period = int(params.get("atr_period", self.default_params["atr_period"]))
        atr_sl_mult = float(params["atr_sl_mult"])
        atr_tp_mult = float(params["atr_tp_mult"])
        max_holding = int(params.get("max_holding", self.default_params["max_holding"]))
        mr_lookback = int(params.get("mr_lookback", self.default_params["mr_lookback"]))
        mr_z_entry = float(params["mr_z_entry"])
        mr_z_exit = float(params.get("mr_z_exit", self.default_params["mr_z_exit"]))

        atr = talib.ATR(high, low, close, timeperiod=atr_period)
        # Donchian channels: ROLLING max/min over [t-period : t-1]
        # (use shift(1) so we never read bar t when deciding at bar t).
        s_high = pd.Series(high)
        s_low = pd.Series(low)
        donchian_upper = s_high.rolling(donchian_period).max().shift(1).values
        donchian_lower = s_low.rolling(donchian_period).min().shift(1).values
        # Mean-reversion z-score: also strictly past
        s_close = pd.Series(close)
        mr_mean = s_close.rolling(mr_lookback).mean().shift(1).values
        mr_std = s_close.rolling(mr_lookback).std(ddof=1).shift(1).values

        regimes = self._regime_per_bar(df)

        pnls: list[float] = []
        signals: list[dict] = []
        pos = None
        equity = ACCOUNT

        warmup = max(donchian_period, atr_period, mr_lookback) + 5
        start = max(warmup, -signal_shift + 1)
        end = n - 1 - max(0, signal_shift)

        for i in range(start, end):
            sig_i = i - signal_shift
            if sig_i <= warmup or sig_i >= n:
                continue
            atr_i = atr[sig_i]
            if math.isnan(atr_i) or atr_i <= 0:
                continue
            regime = int(regimes[sig_i])

            if pos is None:
                side: str | None = None
                sl = tp = None

                if regime == self.risk_on_state:
                    # Donchian breakout (momentum)
                    upper = donchian_upper[sig_i]
                    lower = donchian_lower[sig_i]
                    if math.isnan(upper) or math.isnan(lower):
                        continue
                    if close[sig_i] > upper:
                        side = "buy"
                        sl = close[i] - atr_i * atr_sl_mult
                        tp = close[i] + atr_i * atr_tp_mult
                    elif close[sig_i] < lower:
                        side = "sell"
                        sl = close[i] + atr_i * atr_sl_mult
                        tp = close[i] - atr_i * atr_tp_mult
                else:
                    # Z-score mean reversion (Risk-Off)
                    mu = mr_mean[sig_i]
                    sigma = mr_std[sig_i]
                    if math.isnan(mu) or math.isnan(sigma) or sigma <= 0:
                        continue
                    z = (close[sig_i] - mu) / sigma
                    if z < -mr_z_entry:
                        side = "buy"
                        sl = close[i] - atr_i * atr_sl_mult
                        tp = close[i] + atr_i * mr_z_exit  # tighter MR target
                    elif z > mr_z_entry:
                        side = "sell"
                        sl = close[i] + atr_i * atr_sl_mult
                        tp = close[i] - atr_i * mr_z_exit

                if side is None:
                    continue
                entry = close[i]
                risk = abs(entry - sl)
                if risk <= 0:
                    continue
                qty = (equity * RISK_PER_TRADE) / risk
                pos = {
                    "side": side, "entry": entry, "idx": i,
                    "sl": sl, "tp": tp, "qty": qty, "regime": regime,
                }
                if collect_signals:
                    signals.append({
                        "bar_idx": i + 1,
                        "side": side,
                        "sl": float(sl),
                        "tp": float(tp),
                    })
            else:
                bars_held = i - pos["idx"]
                exit_price = None
                if pos["side"] == "buy":
                    if low[i] <= pos["sl"]:
                        exit_price = pos["sl"]
                    elif high[i] >= pos["tp"]:
                        exit_price = pos["tp"]
                    elif bars_held >= max_holding:
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
                    elif bars_held >= max_holding:
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
    # ------------------------------------------------------------------
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
