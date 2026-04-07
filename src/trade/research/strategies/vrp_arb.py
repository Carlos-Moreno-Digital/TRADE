"""Volatility Risk Premium Arbitrage strategy.

The Volatility Risk Premium (VRP) is the well-documented gap between
implied volatility (what the market is willing to pay for protection)
and realized volatility (what the price actually does). On average,
markets overpay for protection, so a systematic short-volatility
trade earns a small but persistent premium.

We don't have options data, so we approximate the IV proxy with a
hand-rolled GARCH(1,1) volatility forecast (cheap, deterministic, no
extra dependency) and compare it against a short-window realized
volatility (rolling std of log returns):

    sigma_garch_t      ~  the "expected" vol after the recent shocks
    sigma_realized_t   ~  the actual short-term vol of the last W bars
    vrp_ratio_t        =  sigma_garch_t / sigma_realized_t

Interpretation:
  vrp_ratio_t >> 1   ->  GARCH thinks vol should be high, but the
                         price isn't moving that much. This is the
                         "scared but quiet" regime where mean reversion
                         pays.
  vrp_ratio_t ~  1   ->  no mispricing, no signal.
  vrp_ratio_t <  1   ->  vol is exploding faster than GARCH predicted.
                         DO NOT mean-revert into a vol expansion;
                         the strategy stays flat.

Trade rule:
  Only enter when vrp_ratio_t >= vrp_threshold (default 1.20) AND
  the close has stretched at least z_entry standard deviations from
  its rolling mean.
    z_t < -z_entry   ->  buy (oversold extreme + scared market)
    z_t > +z_entry   ->  sell (overbought extreme + scared market)
  Exits:
    |z_t| <= z_exit          (mean-reverted)
    bars_held >= max_holding (time stop)
    SL on entry +/- atr_sl_mult * ATR (catastrophic stop)

Why GARCH(1,1) with frozen parameters:
  - Fitting GARCH with maximum likelihood adds the `arch` package
    and ~50 ms per fit on a 1500-bar window. We don't need it: the
    standard FX values (omega ~ 1e-7, alpha ~ 0.10, beta ~ 0.85)
    are well documented and stable across the EURUSD/USDJPY/GBPUSD
    universe (Engle 2002, Andersen et al. 2003).
  - Frozen parameters guarantee determinism under time-permutation,
    which is what the paranoid suite needs.
  - The strategy is sensitive to the RATIO not the ABSOLUTE level,
    so the GARCH calibration matters less than the recursive
    structure.

Regime gating:
  supported_regimes = [0, 1] — VRP harvest is consistently
  profitable in the calm Risk-On regime (vol overpricing is small
  but reliable) and CAN be profitable in Risk-Off too (vol blow-ups
  often overshoot the "scared" expectation). The strategy is
  enabled in both, and the local SL/timeout cuts losses if the
  regime turns hostile.

Causality:
  - sigma_garch[t] is built from r_{t-1} and sigma_garch[t-1].
    Recursive over bars [0:t]; never reads bar t.
  - sigma_realized[t] is the rolling std of log returns over
    [t-W:t-1]. Strict shift(1).
  - The z-score window mean and std are also shift(1) so the
    decision at bar t depends only on data from bars [0:t).
  - The forward_impact and time_permutation tests in the paranoid
    suite both apply unchanged.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from trade.research.indicators import ATR

ACCOUNT = 10_000.0
RISK_PER_TRADE = 0.003


# Frozen GARCH(1,1) parameters from the FX literature.
# Persistence alpha + beta = 0.95 < 1 (stationary), and beta >> alpha
# (vol clusters more than it reacts to single shocks). Suitable for
# 1H FX bars across all majors without per-symbol re-fitting.
GARCH_OMEGA = 1.0e-7
GARCH_ALPHA = 0.10
GARCH_BETA = 0.85


def _garch11_sigma_series(close: np.ndarray) -> np.ndarray:
    """Causal one-step-ahead GARCH(1,1) volatility forecast.

    Returns an array of length len(close) where element t is the
    forecast for sigma_t computed using only data from bars [0:t).
    The first value is seeded with the unconditional sigma so the
    series is finite from index 0.

    sigma2_t = omega + alpha * r_{t-1}^2 + beta * sigma2_{t-1}
    """
    n = len(close)
    out = np.empty(n, dtype=float)
    if n == 0:
        return out
    log_rets = np.zeros(n, dtype=float)
    log_rets[1:] = np.log(close[1:] / close[:-1])
    # Unconditional sigma2 = omega / (1 - alpha - beta)
    persistence = GARCH_ALPHA + GARCH_BETA
    if persistence < 1.0:
        sigma2_unc = GARCH_OMEGA / (1.0 - persistence)
    else:
        sigma2_unc = 1e-8
    sigma2 = sigma2_unc
    for t in range(n):
        out[t] = math.sqrt(max(sigma2, 1e-16))
        # Update for the NEXT step using the return that just happened
        sigma2 = GARCH_OMEGA + GARCH_ALPHA * (log_rets[t] ** 2) + GARCH_BETA * sigma2
    return out


class VRPArb:
    name: str = "VRPArb"
    # Active in BOTH regimes — the local SL handles regime hostility.
    supported_regimes: list[int] = [0, 1]

    default_params: dict[str, Any] = {
        "rv_lookback": 20,        # bars for realized vol
        "z_lookback": 20,         # bars for z-score window
        "vrp_threshold": 1.20,    # GARCH/RV ratio gate
        "z_entry": 2.0,
        "z_exit": 0.5,
        "atr_period": 14,
        "atr_sl_mult": 2.0,
        "max_holding": 24,
    }

    # Tiny grid kept cheap for the smoke run; can be widened on the VPS.
    param_grid: dict[str, list[Any]] = {
        "rv_lookback": [15, 20, 30],
        "vrp_threshold": [1.10, 1.20, 1.40],
        "z_entry": [1.5, 2.0],
        "max_holding": [12, 24],
    }

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

        rv_lookback = int(params["rv_lookback"])
        z_lookback = int(params.get("z_lookback", self.default_params["z_lookback"]))
        vrp_threshold = float(params["vrp_threshold"])
        z_entry = float(params["z_entry"])
        z_exit = float(params.get("z_exit", self.default_params["z_exit"]))
        atr_period = int(params.get("atr_period", self.default_params["atr_period"]))
        atr_sl_mult = float(params.get("atr_sl_mult", self.default_params["atr_sl_mult"]))
        max_holding = int(params.get("max_holding", self.default_params["max_holding"]))

        # Indicators (all causal — pure numpy/pandas, no ta-lib)
        atr = ATR(high, low, close, period=atr_period)
        sigma_garch = _garch11_sigma_series(close)

        s_close = pd.Series(close)
        log_rets = np.log(s_close / s_close.shift(1)).fillna(0.0)
        sigma_realized = log_rets.rolling(rv_lookback).std(ddof=1).shift(1).to_numpy()

        roll_mean = s_close.rolling(z_lookback).mean().shift(1).to_numpy()
        roll_std = s_close.rolling(z_lookback).std(ddof=1).shift(1).to_numpy()

        warmup = max(rv_lookback, z_lookback, atr_period) + 5
        start = max(warmup, -signal_shift + 1)
        end = n - 1 - max(0, signal_shift)

        pnls: list[float] = []
        signals: list[dict] = []
        pos = None
        equity = ACCOUNT

        for i in range(start, end):
            sig_i = i - signal_shift
            if sig_i <= warmup or sig_i >= n:
                continue
            atr_i = atr[sig_i]
            if math.isnan(atr_i) or atr_i <= 0:
                continue
            sigma_g = sigma_garch[sig_i]
            sigma_r = sigma_realized[sig_i]
            mu = roll_mean[sig_i]
            sd = roll_std[sig_i]
            if (
                math.isnan(sigma_g) or math.isnan(sigma_r) or sigma_r <= 0
                or math.isnan(mu) or math.isnan(sd) or sd <= 0
            ):
                continue
            vrp_ratio = sigma_g / sigma_r
            z = (close[sig_i] - mu) / sd

            if pos is None:
                if vrp_ratio < vrp_threshold:
                    continue
                side: str | None = None
                if z < -z_entry:
                    side = "buy"
                elif z > z_entry:
                    side = "sell"
                if side is None:
                    continue
                entry = close[i]
                if side == "buy":
                    sl = entry - atr_i * atr_sl_mult
                    tp = mu  # mean-reversion target = rolling mean
                    risk = entry - sl
                else:
                    sl = entry + atr_i * atr_sl_mult
                    tp = mu
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
                    elif bars_held >= max_holding:
                        exit_price = close[i]
                    elif abs(z) <= z_exit:
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
                    elif abs(z) <= z_exit:
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
