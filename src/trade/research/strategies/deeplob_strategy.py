"""DeepLOB strategy adapter — wraps a trained model as a StrategyProtocol.

Used by the validation gauntlet (paranoid + CPCV + Nautilus harness).
The model itself is just a 3-class logit predictor over an LOBFrame
window; this adapter turns those logits into discrete BUY/SELL signals
via a confidence threshold and an ATR-based risk envelope, then walks
the resulting positions bar-by-bar to produce per-trade P&L exactly
like every other strategy in the project.

CRITICAL: every prediction is causal by construction.
  - The synthetic LOB tensor at row t depends only on bars[0:t]
    (enforced by the LOBFrame paranoid suite, 11/11).
  - The window fed to the model at row t is rows [t-W+1:t+1] of that
    causal tensor. No future info ever enters the prediction.
  - Position management uses bar t's high/low for SL/TP fills, which
    is the standard event-driven backtest convention (the order is
    placed at bar t open / signal-bar close, then exits within the
    same bar's range).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import talib
import torch

from trade.research.lobframe import (
    calibrated_config,
    column_index,
    synthesize_lob,
)
from trade.research.lobframe.dataset import LOBDatasetConfig
from trade.research.lobframe.model import DeepLOB

ACCOUNT = 10_000.0
RISK_PER_TRADE = 0.003


class DeepLOBStrategy:
    name: str = "DeepLOBStrategy"
    # Will be set after CPCV in the gauntlet identifies the regimes
    # in which the model has signal. For now: empty -> not authorized
    # in any regime (the RegimeGate would block it in production until
    # the gauntlet validates it).
    supported_regimes: list[int] = []

    default_params: dict[str, Any] = {
        "confidence_threshold": 0.45,
        "atr_sl_mult": 1.0,
        "atr_tp_mult": 2.0,
        "max_holding_bars": 12,
    }
    # Tiny grid: keep CPCV cheap during the smoke run.
    param_grid: dict[str, list[Any]] = {
        "confidence_threshold": [0.40, 0.45, 0.50],
        "atr_sl_mult": [1.0, 1.5],
        "atr_tp_mult": [1.5, 2.0],
        "max_holding_bars": [8, 12],
    }

    def __init__(
        self,
        model: DeepLOB,
        symbol: str = "EURUSD",
        window: int = 50,
    ):
        self.model = model.eval()
        self.symbol = symbol
        self.window = window
        self._lob_cfg = calibrated_config(symbol)
        self._ds_cfg = LOBDatasetConfig(window=window, horizon=5, tau=1e-4)
        self._ask1 = column_index(1, "ask_p")
        self._bid1 = column_index(1, "bid_p")

    # ------------------------------------------------------------------
    # Internal — predict logits for every valid t in one batch
    # ------------------------------------------------------------------
    def _logits_for_bars(self, bars: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Returns (logits, t_indices). logits[i] is the prediction for
        bar t_indices[i]. Both arrays have the same length and exclude
        bars before warmup.
        """
        tensor = synthesize_lob(bars, self._lob_cfg)
        T = tensor.shape[0]
        W = self.window
        if T <= W:
            return np.empty((0, 3), dtype=float), np.empty(0, dtype=int)

        # Build all valid windows: window ends at t in [W-1, T-1]
        starts = np.arange(W - 1, T)
        windows = np.empty((len(starts), W, tensor.shape[1]), dtype=np.float32)
        for i, t in enumerate(starts):
            win = tensor[t - W + 1 : t + 1]
            mu = win.mean(axis=0, keepdims=True)
            sigma = win.std(axis=0, keepdims=True) + 1e-8
            windows[i] = ((win - mu) / sigma).astype(np.float32)

        # Batch inference
        x = torch.from_numpy(windows)
        with torch.no_grad():
            logits = self.model(x).cpu().numpy()
        return logits, starts

    # ------------------------------------------------------------------
    # StrategyProtocol API
    # ------------------------------------------------------------------
    def _walk(
        self,
        bars: pd.DataFrame,
        spread: float,
        params: dict[str, Any],
        signal_shift: int = 0,
        collect_signals: bool = False,
    ):
        if not {"open", "high", "low", "close", "volume"}.issubset(bars.columns):
            raise ValueError("bars missing OHLCV columns")
        close = bars["close"].to_numpy(dtype=float)
        high = bars["high"].to_numpy(dtype=float)
        low = bars["low"].to_numpy(dtype=float)
        n = len(close)

        atr = talib.ATR(high, low, close, timeperiod=14)
        logits, t_indices = self._logits_for_bars(bars)
        if logits.size == 0:
            return [] if not collect_signals else pd.DataFrame(
                columns=["bar_idx", "side", "sl", "tp"]
            )

        # Softmax confidences
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        prob_at = {int(t): probs[i] for i, t in enumerate(t_indices)}

        threshold = params["confidence_threshold"]
        sl_mult = params["atr_sl_mult"]
        tp_mult = params["atr_tp_mult"]
        max_bars = int(params["max_holding_bars"])

        pnls: list[float] = []
        signals: list[dict] = []
        pos = None
        equity = ACCOUNT

        start = max(self.window, -signal_shift + 1)
        end = n - max_bars - max(0, signal_shift)

        for i in range(start, end):
            sig_i = i - signal_shift
            if sig_i < 0 or sig_i >= n:
                continue
            if math.isnan(atr[sig_i]) or atr[sig_i] <= 0:
                continue

            if pos is None:
                p = prob_at.get(sig_i)
                if p is None:
                    continue
                p_down, p_flat, p_up = float(p[0]), float(p[1]), float(p[2])
                side: str | None = None
                if p_up >= threshold and p_up > p_down:
                    side = "buy"
                elif p_down >= threshold and p_down > p_up:
                    side = "sell"
                if side is None:
                    continue
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
                qty = (equity * RISK_PER_TRADE) / risk
                pos = {
                    "side": side, "entry": entry, "idx": i,
                    "sl": sl, "tp": tp, "qty": qty,
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
                    elif bars_held >= max_bars:
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
                    elif bars_held >= max_bars:
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
