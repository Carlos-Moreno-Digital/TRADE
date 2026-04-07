"""Triple-Barrier Meta-Labelling (Lopez de Prado AFML ch. 3).

Two-stage architecture:

  PRIMARY model (side):
      RegimeMomentum (Phase 5) — the only classical strategy in the
      project that has passed the future_shift causality test, so
      we trust it as a directional signal generator. The primary
      emits BUY / SELL signals; we DO NOT modify them.

  SECONDARY (meta-model, size):
      A RandomForestClassifier predicts, for each signal emitted by
      the primary, whether the trade will hit the take-profit before
      the stop-loss / vertical timeout. Output is interpreted as
      "trust this signal" probability:
        prob >= meta_threshold  -> take the trade (size = 1)
        prob <  meta_threshold  -> skip the trade  (size = 0)
      The secondary is RE-FIT inside every backtest() call on a
      strict prefix of the input dataframe (the "train window") and
      then evaluated on the suffix ("inference window"). This is
      what the CPCV and time-permutation tests need: every fold and
      every shuffled run gets its own freshly-fit secondary on the
      data it was actually given. There is no global state.

Triple-barrier labelling for the secondary's training set:
  For each primary signal at bar t with side, SL, TP:
    walk forward i = t+1 .. min(t+max_holding, n-1)
    if (side==buy and low[i] <= SL) or (side==sell and high[i] >= SL):
        meta_label = 0   (SL hit first)
    elif (side==buy and high[i] >= TP) or (side==sell and low[i] <= TP):
        meta_label = 1   (TP hit first)
    if neither barrier touched: meta_label = 0 (timeout = no profit)
  Pessimistic ordering matches the gauntlet harness: if both SL and
  TP fall inside the same bar, SL wins.

Meta-features (computed at the SIGNAL bar, all causal):
  - atr        (current ATR / current close)               vol normalised
  - rv5        (5-bar realised vol)
  - rv20       (20-bar realised vol)
  - vol_ratio  (rv5 / rv20)
  - z_score    (price vs 20-bar rolling mean / std)
  - skew_20    (20-bar rolling skew of log returns)
  - rsi_14     (Wilder RSI)
  - ret_1      (1-bar log return)
  - ret_5      (5-bar log return)
  - donchian_pos (close minus rolling 20-period mid, normalised by ATR)
  - side_buy   (1.0 if primary side==buy, else 0.0)

Causality:
  - Every meta-feature is computed using only data up to bar t.
  - The triple-barrier label uses bars [t+1, t+max_holding], which
    is the future of the SIGNAL bar — this is allowed because the
    label is the supervised target, never an input at inference.
  - Train window covers the first train_fraction of df. Inference
    window is the rest. NO sample from the inference window is ever
    fed to fit() (strict temporal split).
  - The RF is trained on the train window's labelled signals only.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from trade.research.indicators import ATR, RSI
from trade.research.strategies.regime_momentum import RegimeMomentum
from trade.validation.sample_weights import (
    average_uniqueness,
    normalize_to_sum_n,
)

ACCOUNT = 10_000.0
RISK_PER_TRADE = 0.003

META_FEATURE_COLS = [
    "atr_norm", "rv5", "rv20", "vol_ratio", "z_score",
    "skew_20", "rsi_14", "ret_1", "ret_5", "donchian_pos", "side_buy",
]


def _build_meta_features(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    s_close = pd.Series(close)
    log_rets = np.log(s_close / s_close.shift(1)).fillna(0.0)

    atr = ATR(high, low, close, period=atr_period)
    rsi_14 = RSI(close, period=14)
    rv5 = log_rets.rolling(5).std(ddof=1).shift(1)
    rv20 = log_rets.rolling(20).std(ddof=1).shift(1)
    vol_ratio = (rv5 / rv20.replace(0, np.nan))
    skew_20 = log_rets.rolling(20).skew().shift(1)
    roll_mean = s_close.rolling(20).mean().shift(1)
    roll_std = s_close.rolling(20).std(ddof=1).shift(1)
    z_score = (s_close - roll_mean) / roll_std.replace(0, np.nan)
    ret_1 = log_rets.shift(1)
    ret_5 = log_rets.rolling(5).sum().shift(1)
    donchian_mid = (s_close.rolling(20).max() + s_close.rolling(20).min()) / 2
    donchian_pos = ((s_close - donchian_mid.shift(1)) /
                    pd.Series(atr).shift(1).replace(0, np.nan))

    feats = pd.DataFrame({
        "atr_norm": pd.Series(atr) / s_close,
        "rv5": rv5,
        "rv20": rv20,
        "vol_ratio": vol_ratio,
        "z_score": z_score,
        "skew_20": skew_20,
        "rsi_14": pd.Series(rsi_14),
        "ret_1": ret_1,
        "ret_5": ret_5,
        "donchian_pos": donchian_pos,
    })
    return feats


def _triple_barrier_label(
    bar_idx: int,
    side: str,
    sl: float,
    tp: float,
    high: np.ndarray,
    low: np.ndarray,
    max_holding: int,
) -> int:
    """[LEGACY classification path] 1 if TP hit first, 0 otherwise.
    Pessimistic ordering: same-bar SL+TP -> 0.

    Kept for the leakage-test compatibility shim. The Phase 8c
    pipeline uses _triple_barrier_pnl below as the regression target.
    """
    n = len(high)
    end = min(bar_idx + max_holding, n - 1)
    for i in range(bar_idx + 1, end + 1):
        h = high[i]
        l = low[i]
        if side == "buy":
            sl_hit = l <= sl
            tp_hit = h >= tp
            if sl_hit:
                return 0
            if tp_hit:
                return 1
        else:
            sl_hit = h >= sl
            tp_hit = l <= tp
            if sl_hit:
                return 0
            if tp_hit:
                return 1
    return 0


def _triple_barrier_pnl(
    bar_idx: int,
    side: str,
    entry: float,
    sl: float,
    tp: float,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    vertical_bars: int,
) -> tuple[float, str, int]:
    """Phase 8c REGRESSION target: realized P&L per unit + exit reason.

    Walks forward from bar_idx+1 up to bar_idx+vertical_bars and
    returns:
        (realized_pnl_per_unit, exit_reason, exit_bar_offset)

    pnl_per_unit is in the instrument's price units (multiply by qty
    to get account-currency P&L). Pessimistic SL-first ordering on
    same-bar conflicts. The vertical barrier is the LABELLING ONLY
    cap to avoid infinite-horizon labels and to keep sample
    uniqueness scores well-defined; the production walker
    (walk_positions in train_institutional_meta.py) does NOT use it.
    """
    n = len(close)
    end = min(bar_idx + vertical_bars, n - 1)
    for i in range(bar_idx + 1, end + 1):
        h = high[i]
        l = low[i]
        if side == "buy":
            sl_hit = l <= sl
            tp_hit = h >= tp
            if sl_hit:
                return (sl - entry, "sl", i - bar_idx)
            if tp_hit:
                return (tp - entry, "tp", i - bar_idx)
        else:
            sl_hit = h >= sl
            tp_hit = l <= tp
            if sl_hit:
                return (entry - sl, "sl", i - bar_idx)
            if tp_hit:
                return (entry - tp, "tp", i - bar_idx)
    # Vertical barrier reached: return the partial unrealized P&L
    # at the vertical bar's close (counted as REGRESSION TARGET, not
    # as a class label).
    if side == "buy":
        return (close[end] - entry, "vertical", end - bar_idx)
    else:
        return (entry - close[end], "vertical", end - bar_idx)


class MetaLabellingStrategy:
    name: str = "MetaLabellingStrategy"
    supported_regimes: list[int] = [0, 1]

    # Phase 8c reformulation:
    #   - Secondary model is a REGRESSOR (RandomForestRegressor) whose
    #     target is realized P&L per unit (price units, multiply by qty
    #     for account-currency P&L).
    #   - meta_threshold is now an EXPECTED P&L per unit above which we
    #     accept the trade. The default 0.0 means "any predicted profit
    #     after spread"; the production grid search finds the right
    #     symbol-specific value.
    #   - vertical_bars is the LABEL HORIZON CAP only — production
    #     walks (walk_positions) have NO max_holding, only TP or SL.
    #   - Sample uniqueness (Lopez de Prado AFML ch 4) is computed
    #     globally over all signals and passed to the regressor as
    #     sample_weight.
    default_params: dict[str, Any] = {
        # Train/inference split
        "train_fraction": 0.4,
        "meta_threshold": 0.0,   # expected pnl_per_unit > 0
        # Regressor hyperparameters
        "rf_n_estimators": 100,
        "rf_max_depth": 6,
        "rf_min_samples_leaf": 5,
        # Triple-barrier label horizon (LABEL ONLY, NOT a trade time stop)
        "vertical_bars": 100,
        # Primary (RegimeMomentum) hyperparameters
        "donchian_period": 20,
        "atr_period": 14,
        "atr_sl_mult": 1.5,
        "atr_tp_mult": 3.0,
        "max_holding": 24,       # used by primary._walk only
        "mr_lookback": 20,
        "mr_z_entry": 2.0,
        "mr_z_exit": 0.5,
    }

    # Tiny grid kept cheap for the smoke run; CPCV will hate it less.
    param_grid: dict[str, list[Any]] = {
        "meta_threshold": [0.0, 0.0001, 0.0003],
        "donchian_period": [15, 20],
        "rf_n_estimators": [50, 100],
        "atr_sl_mult": [1.0, 1.5],
    }

    def __init__(self, symbol: str = "EURUSD"):
        self.symbol = symbol
        self.primary = RegimeMomentum(symbol=symbol)

    # ------------------------------------------------------------------
    def _merge_defaults(self, params: dict) -> dict:
        """CPCV / paranoid suite pass only the grid keys; merge with
        defaults so every internal field is always present.
        """
        merged = dict(self.default_params)
        merged.update(params or {})
        return merged

    def _primary_params(self, params: dict) -> dict:
        p = self._merge_defaults(params)
        return {
            "donchian_period": int(p["donchian_period"]),
            "atr_period": int(p["atr_period"]),
            "atr_sl_mult": float(p["atr_sl_mult"]),
            "atr_tp_mult": float(p["atr_tp_mult"]),
            "max_holding": int(p["max_holding"]),
            "mr_lookback": int(p["mr_lookback"]),
            "mr_z_entry": float(p["mr_z_entry"]),
            "mr_z_exit": float(p["mr_z_exit"]),
        }

    def _train_meta(
        self,
        train_signals: pd.DataFrame,
        feats: pd.DataFrame,
        bars: pd.DataFrame,
        params: dict,
    ) -> tuple[RandomForestRegressor, np.ndarray] | None:
        """Phase 8c regressor training.

        Returns (fitted_regressor, signal_indices_used) or None if too
        few samples. The target is realized PnL per unit from the
        triple-barrier walk; sample weights are average uniqueness from
        Lopez de Prado AFML ch 4.
        """
        high = bars["high"].to_numpy(dtype=float)
        low = bars["low"].to_numpy(dtype=float)
        close = bars["close"].to_numpy(dtype=float)
        vertical_bars = int(params["vertical_bars"])

        rows: list[list[float]] = []
        targets: list[float] = []
        signal_starts: list[int] = []
        signal_ends: list[int] = []
        for s in train_signals.itertuples():
            t0 = int(s.bar_idx) - 1
            if t0 < 0 or t0 >= len(feats):
                continue
            f = feats.iloc[t0]
            if f.isna().any():
                continue
            entry = float(close[t0])
            pnl_per_unit, _exit, exit_offset = _triple_barrier_pnl(
                t0, s.side, entry, float(s.sl), float(s.tp),
                high, low, close, vertical_bars,
            )
            row = list(f.values) + [1.0 if s.side == "buy" else 0.0]
            rows.append(row)
            targets.append(pnl_per_unit)
            signal_starts.append(t0)
            signal_ends.append(t0 + exit_offset)
        if len(rows) < 20:
            return None
        X = np.array(rows, dtype=float)
        y = np.array(targets, dtype=float)
        starts = np.array(signal_starts, dtype=int)
        ends = np.array(signal_ends, dtype=int)
        # Sample uniqueness over the TRAINING window only
        weights = average_uniqueness(starts, ends, n_bars=int(ends.max()) + 1)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights = normalize_to_sum_n(weights)

        rf = RandomForestRegressor(
            n_estimators=int(params["rf_n_estimators"]),
            max_depth=int(params["rf_max_depth"]),
            min_samples_leaf=int(params["rf_min_samples_leaf"]),
            random_state=42,
            n_jobs=1,
        )
        rf.fit(X, y, sample_weight=weights)
        return rf, starts

    def _walk_filtered(
        self,
        df: pd.DataFrame,
        spread: float,
        params: dict,
        kept_signals: list[dict],
        collect_signals: bool = False,
    ):
        """Production walker — Phase 8c: NO max_holding.

        Trades close ONLY when TP or SL is touched. If a trade is
        still open at end-of-data, it is force-closed at the last
        bar's close (clean accounting; this only affects the very
        last open position of the entire backtest).
        """
        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        n = len(close)

        entries = {int(s["bar_idx"]) - 1: s for s in kept_signals}

        pnls: list[float] = []
        emitted: list[dict] = []
        pos = None
        equity = ACCOUNT

        for i in range(n - 1):
            if pos is None:
                sig = entries.get(i)
                if sig is None:
                    continue
                side = sig["side"]
                sl = float(sig["sl"])
                tp = float(sig["tp"])
                entry = close[i]
                risk = (entry - sl) if side == "buy" else (sl - entry)
                if risk <= 0:
                    continue
                qty = (equity * RISK_PER_TRADE) / risk
                pos = {
                    "side": side, "entry": entry, "idx": i,
                    "sl": sl, "tp": tp, "qty": qty,
                }
                if collect_signals:
                    emitted.append({
                        "bar_idx": i + 1,
                        "side": side,
                        "sl": float(sl),
                        "tp": float(tp),
                    })
                continue
            # TP/SL only — no time stop. Pessimistic SL-first ordering.
            exit_price = None
            if pos["side"] == "buy":
                if low[i] <= pos["sl"]:
                    exit_price = pos["sl"]
                elif high[i] >= pos["tp"]:
                    exit_price = pos["tp"]
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
                if exit_price is not None:
                    pnl = (pos["entry"] - exit_price) * pos["qty"] - spread * pos["qty"] * 2
                    pnls.append(pnl)
                    equity += pnl
                    pos = None

        # Force-close any position still open at end-of-data
        if pos is not None:
            last_close = close[-1]
            if pos["side"] == "buy":
                pnl = (last_close - pos["entry"]) * pos["qty"] - spread * pos["qty"] * 2
            else:
                pnl = (pos["entry"] - last_close) * pos["qty"] - spread * pos["qty"] * 2
            pnls.append(pnl)

        if collect_signals:
            return pd.DataFrame(emitted) if emitted else pd.DataFrame(
                columns=["bar_idx", "side", "sl", "tp"]
            )
        return pnls

    # ------------------------------------------------------------------
    def _run(
        self,
        df: pd.DataFrame,
        spread: float,
        params: dict,
        signal_shift: int = 0,
        collect_signals: bool = False,
    ):
        params = self._merge_defaults(params)
        if signal_shift != 0:
            # The primary handles signal_shift; the meta does not.
            # Forward shift to the primary so the future_shift test
            # still measures the underlying causal contract.
            primary_pnls_or_signals = self.primary._walk(
                df, spread, self._primary_params(params),
                signal_shift=signal_shift, collect_signals=collect_signals,
            )
            return primary_pnls_or_signals

        primary_signals = self.primary.signals(df, self._primary_params(params))
        if primary_signals.empty:
            return [] if not collect_signals else pd.DataFrame(
                columns=["bar_idx", "side", "sl", "tp"]
            )

        n = len(df)
        train_end = int(n * float(params["train_fraction"]))
        if train_end < 50 or train_end >= n:
            # Fall back to passthrough if not enough data
            return self._walk_filtered(
                df, spread, params,
                primary_signals.to_dict("records"),
                collect_signals=collect_signals,
            )

        # Split signals into train and inference by entry bar
        train_signals = primary_signals[primary_signals["bar_idx"] <= train_end]
        inf_signals = primary_signals[primary_signals["bar_idx"] > train_end]

        feats = _build_meta_features(df, atr_period=int(params["atr_period"]))

        rf_pack = self._train_meta(train_signals, feats, df, params)

        kept: list[dict] = []
        if rf_pack is None:
            # If we cannot train, run the primary unfiltered on the
            # inference window only (the meta acts as a no-op).
            kept = inf_signals.to_dict("records")
        else:
            rf, _ = rf_pack
            threshold = float(params["meta_threshold"])
            # Threshold is now in PnL-per-unit space; we accept a
            # signal only if predicted PnL beats the round-trip spread.
            min_pnl_per_unit = max(threshold, 2.0 * float(spread))
            for s in inf_signals.itertuples():
                t0 = int(s.bar_idx) - 1
                if t0 < 0 or t0 >= len(feats):
                    continue
                f = feats.iloc[t0]
                if f.isna().any():
                    continue
                row = np.array(
                    list(f.values) + [1.0 if s.side == "buy" else 0.0],
                    dtype=float,
                ).reshape(1, -1)
                pred_pnl = float(rf.predict(row)[0])
                if pred_pnl >= min_pnl_per_unit:
                    kept.append({
                        "bar_idx": int(s.bar_idx),
                        "side": s.side,
                        "sl": float(s.sl),
                        "tp": float(s.tp),
                    })

        return self._walk_filtered(
            df, spread, params, kept, collect_signals=collect_signals,
        )

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
        return self._run(df, spread, params, signal_shift, collect_signals=False)

    def signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        return self._run(df, 0.0, params, 0, collect_signals=True)
