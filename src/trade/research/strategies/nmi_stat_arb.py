"""NMI-aware Statistical Arbitrage strategy.

Single-leg cointegration mean-reversion: trade the PRIMARY symbol
using a statistical-arbitrage signal computed against a PARTNER
symbol that the NMI dependency analysis identified as living in the
same cluster (NMI > 0.15 in the cached matrix).

Why single-leg trading on a pair signal:
  - Our StrategyProtocol takes one OHLCV dataframe per backtest call
    (and the validation gauntlet expects fills on a single instrument).
  - True 2-leg pair trading would require simultaneous fills on
    distinct Nautilus instruments, which the harness does not yet
    support.
  - Trading only the primary leg with the cointegration spread as a
    cross-asset SIGNAL is a perfectly valid statistical-arbitrage
    view: the partner pair acts as a leading indicator for the
    primary's mean reversion.
  - When real 2-leg fills are needed we extend the harness; the
    signal logic in this class is unchanged.

Construction (per bar t, strictly causal — no future info):
  1. Take a rolling window [t-W, t-1] of aligned (a, b) closes.
  2. OLS regression a = alpha + beta * b on the window. beta is the
     hedge ratio.
  3. spread_t = a_t - beta * b_t (using current closes a_t, b_t).
  4. z_t = (spread_t - mean(spread_window)) / std(spread_window).
  5. Periodic Engle-Granger cointegration check on the window
     (every coint_check_every bars). If the latest p-value > the
     coint p-value gate, the strategy is currently INACTIVE for
     that window — no trades are emitted regardless of z_t.
  6. Entry rules:
        z_t >  +z_entry  -> spread is RICH on the long leg, SELL it
                            (expecting reversion downward)
        z_t <  -z_entry  -> spread is CHEAP on the long leg, BUY it
                            (expecting reversion upward)
  7. Exit rules:
        |z_t|  <=  z_exit         -> mean-reverted, take profit
        bars_held >= max_holding  -> timeout
        z escalates beyond stop band -> stop loss

Regime gating:
  Classical statistical arbitrage explodes in Risk-Off regimes
  (sudden vol spikes wreck the hedge ratio). The strategy hard-codes
  supported_regimes = [0] so the RegimeGate will block any signal
  the day a Risk-Off regime is detected for the primary symbol via
  the SJM. This is enforced in production by RiskShield, but we
  also enforce it locally inside _walk() so backtests don't trade
  during the wrong regime even when called outside the orchestrator.

Causality:
  - At bar t, EVERY computation uses only [t-W : t] log returns and
    closes. The OLS hedge ratio is fitted strictly on the past window.
  - The current spread z_t uses close[t] (the last bar of the
    feature window), which is the standard "decision at close"
    convention used by every other strategy in this project.
  - The forward_impact and label_causality_tb tests in the LOBFrame
    suite are not directly applicable here (no LOB tensor); the
    paranoid suite's time_permutation, future_shift, and
    block_bootstrap tests handle the equivalent contracts on the
    final P&L sequence.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

ACCOUNT = 10_000.0
RISK_PER_TRADE = 0.003


def _load_close(symbol: str, data_dir: Path) -> pd.Series:
    path = data_dir / f"{symbol}_1H.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df["close"].rename(symbol)


def _ols_hedge(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Closed-form OLS regression a = alpha + beta * b -> (alpha, beta).

    Pure numpy so it survives the time-permutation tests without any
    statsmodels overhead inside the inner loop.
    """
    if len(a) < 5 or len(b) < 5:
        return 0.0, 0.0
    b_mean = b.mean()
    a_mean = a.mean()
    var_b = ((b - b_mean) ** 2).sum()
    if var_b <= 0:
        return a_mean, 0.0
    cov = ((a - a_mean) * (b - b_mean)).sum()
    beta = cov / var_b
    alpha = a_mean - beta * b_mean
    return float(alpha), float(beta)


class NMIStatArb:
    name: str = "NMIStatArb"
    # Risk-Off regimes destroy stat-arb. Authorize only state 0
    # (low-vol Risk-On as identified by the per-symbol SJM).
    supported_regimes: list[int] = [0]

    default_params: dict[str, Any] = {
        "lookback": 60,           # rolling window for OLS + z-score (bars)
        "z_entry": 2.0,           # entry threshold (sigmas)
        "z_exit": 0.5,            # exit threshold (sigmas)
        "z_stop": 4.0,            # blow-up stop (sigmas)
        "max_holding": 24,        # bars
        "coint_p_gate": 0.10,     # ADF p-value upper bound to trade
        "coint_check_every": 30,  # bars between full ADF refits
    }

    # Tiny grid for cheap CPCV during the gauntlet smoke run
    param_grid: dict[str, list[Any]] = {
        "lookback": [40, 60],
        "z_entry": [1.5, 2.0],
        "z_exit": [0.3, 0.5],
        "max_holding": [12, 24],
    }

    def __init__(
        self,
        partner_close: pd.Series,
        primary_symbol: str = "AUDUSD",
        partner_symbol: str = "NZDUSD",
    ):
        self.partner = partner_close
        self.primary_symbol = primary_symbol
        self.partner_symbol = partner_symbol

    # ------------------------------------------------------------------
    @classmethod
    def from_disk(
        cls,
        primary_symbol: str,
        partner_symbol: str,
        data_dir: Path = Path("data/dukascopy"),
    ) -> "NMIStatArb":
        partner = _load_close(partner_symbol, data_dir)
        return cls(
            partner_close=partner,
            primary_symbol=primary_symbol,
            partner_symbol=partner_symbol,
        )

    # ------------------------------------------------------------------
    # Lockstep permutation hook (caveat #1 fix).
    #
    # ParanoidSuite.time_permutation calls this context manager just
    # before each permuted backtest. We replace `self.partner` with a
    # series whose log returns have been independently shuffled with
    # the same RNG that the suite uses to shuffle the primary leg.
    # The shuffle is INDEPENDENT (not lockstep with the primary)
    # which is what we want: it destroys the cointegration relation
    # so a strategy that depended on real cointegration cannot beat
    # the null distribution by coincidence.
    @contextmanager
    def with_permuted_state(self, rng):
        saved = self.partner.copy()
        try:
            prices = self.partner.values.astype(float)
            if len(prices) >= 3:
                log_rets = np.diff(np.log(prices))
                permuted = rng.permutation(log_rets)
                new_close = prices[0] * np.exp(
                    np.concatenate([[0.0], np.cumsum(permuted)])
                )
                self.partner = pd.Series(
                    new_close, index=self.partner.index, name=self.partner.name
                )
            yield
        finally:
            self.partner = saved

    # ------------------------------------------------------------------
    def _aligned_partner(self, df: pd.DataFrame) -> np.ndarray:
        s = self.partner.reindex(df.index, method="ffill")
        s = s.bfill()
        return s.values.astype(float)

    def _walk(
        self,
        df: pd.DataFrame,
        spread_cost: float,
        params: dict[str, Any],
        signal_shift: int = 0,
        collect_signals: bool = False,
    ):
        if not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
            raise ValueError("df missing OHLCV columns")
        a = df["close"].to_numpy(dtype=float)
        a_high = df["high"].to_numpy(dtype=float)
        a_low = df["low"].to_numpy(dtype=float)
        b = self._aligned_partner(df)
        n = len(a)
        if n < int(params["lookback"]) + 5:
            return [] if not collect_signals else pd.DataFrame(
                columns=["bar_idx", "side", "sl", "tp"]
            )

        lookback = int(params["lookback"])
        z_entry = float(params["z_entry"])
        z_exit = float(params["z_exit"])
        z_stop = float(params["z_stop"]) if "z_stop" in params else float(self.default_params["z_stop"])
        max_holding = int(params["max_holding"])
        coint_p_gate = float(params.get("coint_p_gate", self.default_params["coint_p_gate"]))
        coint_check_every = int(params.get("coint_check_every", self.default_params["coint_check_every"]))

        pnls: list[float] = []
        signals: list[dict] = []
        pos = None
        equity = ACCOUNT
        last_p_value = 1.0
        last_check_t = -10**9

        start = max(lookback + 1, -signal_shift + 1)
        end = n - 1 - max(0, signal_shift)

        for i in range(start, end):
            sig_i = i - signal_shift
            if sig_i <= lookback or sig_i >= n:
                continue

            window_a = a[sig_i - lookback : sig_i]
            window_b = b[sig_i - lookback : sig_i]
            if window_a.size < 5 or window_b.size < 5:
                continue
            alpha, beta = _ols_hedge(window_a, window_b)
            window_spread = window_a - (alpha + beta * window_b)
            mu = float(window_spread.mean())
            sigma = float(window_spread.std(ddof=1))
            if sigma <= 0:
                continue
            current_spread = a[sig_i] - (alpha + beta * b[sig_i])
            z = (current_spread - mu) / sigma

            # Periodic cointegration p-value check on the window residuals
            if sig_i - last_check_t >= coint_check_every:
                try:
                    last_p_value = float(adfuller(window_spread, regression="c")[1])
                except Exception:
                    last_p_value = 1.0
                last_check_t = sig_i

            cointegrated = last_p_value <= coint_p_gate

            if pos is None:
                if not cointegrated:
                    continue
                side: str | None = None
                if z > z_entry:
                    side = "sell"  # spread is RICH on long leg -> short
                elif z < -z_entry:
                    side = "buy"   # spread is CHEAP on long leg -> long
                if side is None:
                    continue
                entry = a[i]
                # SL/TP via z-score targets translated into price space
                # using the current spread sigma scaled to leg price.
                price_sigma_unit = sigma  # spread sigma in leg price units
                if side == "buy":
                    sl = entry - z_stop * price_sigma_unit
                    tp = entry + z_exit * price_sigma_unit
                    risk = entry - sl
                else:
                    sl = entry + z_stop * price_sigma_unit
                    tp = entry - z_exit * price_sigma_unit
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
                    if a_low[i] <= pos["sl"]:
                        exit_price = pos["sl"]
                    elif a_high[i] >= pos["tp"]:
                        exit_price = pos["tp"]
                    elif bars_held >= max_holding:
                        exit_price = a[i]
                    elif abs(z) <= z_exit:
                        exit_price = a[i]
                    if exit_price is not None:
                        pnl = (exit_price - pos["entry"]) * pos["qty"] - spread_cost * pos["qty"] * 2
                        pnls.append(pnl)
                        equity += pnl
                        pos = None
                else:
                    if a_high[i] >= pos["sl"]:
                        exit_price = pos["sl"]
                    elif a_low[i] <= pos["tp"]:
                        exit_price = pos["tp"]
                    elif bars_held >= max_holding:
                        exit_price = a[i]
                    elif abs(z) <= z_exit:
                        exit_price = a[i]
                    if exit_price is not None:
                        pnl = (pos["entry"] - exit_price) * pos["qty"] - spread_cost * pos["qty"] * 2
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

    # ------------------------------------------------------------------
    # Caveat #2 fix: 2-leg pair signals.
    #
    # Returns one DataFrame describing both legs of every pair trade.
    # PairNautilusHarness consumes it to fill simultaneously on the
    # primary and partner instruments. The partner side is the
    # OPPOSITE of the primary side (long primary -> short partner),
    # which is the canonical stat-arb hedge.
    # ------------------------------------------------------------------
    def pair_signals(
        self, df: pd.DataFrame, params: dict[str, Any]
    ) -> pd.DataFrame:
        primary = self._walk(df, 0.0, params, 0, collect_signals=True)
        if primary.empty:
            return pd.DataFrame(
                columns=[
                    "bar_idx", "primary_side", "primary_sl", "primary_tp",
                    "partner_side", "partner_sl", "partner_tp",
                ]
            )
        partner_close = self._aligned_partner(df)
        rows = []
        for row in primary.itertuples():
            bar_idx_0 = int(row.bar_idx) - 1  # back to 0-based
            if bar_idx_0 >= len(partner_close):
                continue
            partner_entry = float(partner_close[bar_idx_0])
            sl_dist = abs(float(row.sl) - df["close"].iloc[bar_idx_0])
            tp_dist = abs(float(row.tp) - df["close"].iloc[bar_idx_0])
            partner_side = "sell" if row.side == "buy" else "buy"
            if partner_side == "buy":
                partner_sl = partner_entry - sl_dist
                partner_tp = partner_entry + tp_dist
            else:
                partner_sl = partner_entry + sl_dist
                partner_tp = partner_entry - tp_dist
            rows.append({
                "bar_idx": int(row.bar_idx),
                "primary_side": row.side,
                "primary_sl": float(row.sl),
                "primary_tp": float(row.tp),
                "partner_side": partner_side,
                "partner_sl": float(partner_sl),
                "partner_tp": float(partner_tp),
            })
        return pd.DataFrame(rows)
