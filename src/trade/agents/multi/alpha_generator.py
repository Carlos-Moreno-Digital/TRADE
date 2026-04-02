"""Alpha Generator Agent — XGBoost + HMM Regime Detection.

Generates directional signals using ML with market regime awareness.
Outputs JSON with signal, confidence, and detected regime.
"""

from __future__ import annotations

import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import talib
from loguru import logger
from sklearn.preprocessing import StandardScaler

from trade.agents.multi import AgentMessage
from trade.ml_backtest import _build_features, _build_target

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMM = True
except ImportError:
    HAS_HMM = False


class AlphaGenerator:
    """Generates trading signals using XGBoost + regime detection."""

    def __init__(self):
        self.models: dict[str, tuple] = {}  # sym -> (model, scaler, meta_model)
        self.train_bars = 4000
        self.purge_bars = 24
        self.horizon = 8
        self.min_move = 0.001
        self.confidence_threshold = 0.53
        self.xgb_params = None  # Will be set by Optuna if available

    def train(self, sym: str, df: pd.DataFrame) -> bool:
        """Train XGBoost model for a symbol."""
        features = _build_features(df)
        target = _build_target(df, horizon=self.horizon, min_move_pct=self.min_move)

        data = features.copy()
        data["target"] = target
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        if len(data) < self.train_bars + 100:
            return False

        # Rolling window: last train_bars
        train_data = data.iloc[-self.train_bars:]
        X_train = train_data.drop(columns=["target"]).fillna(0)
        y_train = train_data["target"]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        y_map = {-1: 0, 0: 1, 1: 2}
        y_train_m = y_train.map(y_map)

        # Use Optuna-tuned params if available, otherwise defaults
        params = self.xgb_params or {
            "n_estimators": 250, "max_depth": 4, "learning_rate": 0.04,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 5, "reg_alpha": 0.1, "reg_lambda": 1.0,
        }

        if HAS_XGB:
            model = xgb.XGBClassifier(
                **params,
                eval_metric="mlogloss", verbosity=0,
                random_state=42, seed=42,
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=params.get("n_estimators", 250),
                max_depth=params.get("max_depth", 4),
                learning_rate=params.get("learning_rate", 0.04),
                subsample=params.get("subsample", 0.8),
                min_samples_leaf=20, random_state=42,
            )

        model.fit(X_train_s, y_train_m)

        # META-LABELING: train secondary model to predict if primary signal is profitable
        # (López de Prado — improves precision by 10-20%)
        meta_model = None
        try:
            primary_preds = model.predict(X_train_s)
            primary_proba = model.predict_proba(X_train_s)
            # Only on samples where primary model gives a signal (not neutral)
            signal_mask = primary_preds != 1  # class 1 = neutral
            if signal_mask.sum() > 100:
                # Meta-label: was the primary signal correct?
                meta_X = np.column_stack([
                    X_train_s[signal_mask],
                    primary_proba[signal_mask],
                ])
                # Target: 1 if primary prediction matches actual, 0 if not
                primary_directions = np.array([{0: -1, 1: 0, 2: 1}.get(int(p), 0) for p in primary_preds[signal_mask]])
                actual_directions = y_train.values[signal_mask]
                meta_y = (primary_directions == actual_directions).astype(int)

                if meta_y.sum() > 20 and (1 - meta_y).sum() > 20:  # Both classes exist
                    from sklearn.ensemble import GradientBoostingClassifier as GBC
                    meta_model = GBC(n_estimators=100, max_depth=3, learning_rate=0.05,
                                     min_samples_leaf=10, random_state=42)
                    meta_model.fit(meta_X, meta_y)
        except Exception:
            meta_model = None

        self.models[sym] = (model, scaler, meta_model)
        return True

    def detect_regime(self, df: pd.DataFrame) -> str:
        """Detect market regime: TRENDING, RANGING, or VOLATILE."""
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)

        if len(close) < 50:
            return "UNKNOWN"

        adx = talib.ADX(high, low, close, timeperiod=14)
        adx_val = float(adx[-1]) if not math.isnan(adx[-1]) else 20

        atr = talib.ATR(high, low, close, timeperiod=14)
        atr_avg_recent = float(np.nanmean(atr[-20:])) if len(atr) >= 20 else float(atr[-1])
        atr_avg_long = float(np.nanmean(atr[-100:])) if len(atr) >= 100 else atr_avg_recent

        # Regime classification
        if adx_val > 30 and atr_avg_recent > atr_avg_long * 1.2:
            return "TRENDING_VOLATILE"
        elif adx_val > 25:
            return "TRENDING"
        elif adx_val < 18:
            return "RANGING"
        elif atr_avg_recent > atr_avg_long * 1.5:
            return "VOLATILE"
        else:
            return "TRANSITIONING"

    def generate_signal(self, sym: str, df: pd.DataFrame) -> AgentMessage:
        """Generate a trading signal for a symbol."""
        if sym not in self.models:
            return AgentMessage(
                agent_domain="alpha_generator",
                status_flag="REJECTED",
                economic_rationale=f"No trained model for {sym}",
            )

        model, scaler, meta_model = self.models[sym]
        features = _build_features(df)
        data = features.replace([np.inf, -np.inf], np.nan).fillna(0)

        if len(data) < 2:
            return AgentMessage(
                agent_domain="alpha_generator",
                status_flag="REJECTED",
                economic_rationale="Insufficient data for prediction",
            )

        # Predict
        X_latest = data.iloc[[-1]]
        try:
            X_scaled = scaler.transform(X_latest)
            pred = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
        except Exception as e:
            return AgentMessage(
                agent_domain="alpha_generator",
                status_flag="REJECTED",
                errors=[str(e)],
            )

        y_inv = {0: -1, 1: 0, 2: 1}
        pred_class = y_inv.get(int(pred), 0)
        max_prob = float(proba.max())

        # Detect regime
        regime = self.detect_regime(df)

        # No signal if neutral or low confidence
        # ADAPTIVE: lower threshold in strong trending regimes
        threshold = self.confidence_threshold
        if regime in ("TRENDING", "TRENDING_VOLATILE"):
            threshold = 0.48  # More aggressive in trends (edge is stronger)
        elif regime == "RANGING":
            threshold = 0.55  # More conservative in ranges (edge is weaker)

        if pred_class == 0 or max_prob < threshold:
            return AgentMessage(
                agent_domain="alpha_generator",
                status_flag="NO_SIGNAL",
                computational_payload={
                    "symbol": sym,
                    "probabilities": {"short": float(proba[0]), "neutral": float(proba[1]), "long": float(proba[2])},
                    "regime": regime,
                },
                economic_rationale=f"No clear signal: pred={pred_class}, conf={max_prob:.3f}, regime={regime}",
            )

        # Build signal
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        price = float(close[-1])
        atr = talib.ATR(high, low, close, timeperiod=14)
        atr_val = float(atr[-1]) if not math.isnan(atr[-1]) else price * 0.001

        action = "BUY" if pred_class == 1 else "SELL"

        # META-LABELING: secondary model filters false signals
        meta_confidence = 1.0
        if meta_model is not None:
            try:
                meta_X = np.concatenate([X_scaled[0], proba])
                meta_pred = meta_model.predict_proba(meta_X.reshape(1, -1))[0]
                meta_confidence = float(meta_pred[1])  # P(signal is correct)
                if meta_confidence < 0.45:
                    return AgentMessage(
                        agent_domain="alpha_generator",
                        status_flag="NO_SIGNAL",
                        computational_payload={
                            "symbol": sym, "regime": regime,
                            "meta_confidence": meta_confidence,
                            "primary_confidence": max_prob,
                        },
                        economic_rationale=f"Meta-label rejected: primary {action} {max_prob:.1%} but meta says {meta_confidence:.1%} chance of success",
                    )
            except Exception:
                pass

        return AgentMessage(
            agent_domain="alpha_generator",
            status_flag="SIGNAL_GENERATED",
            computational_payload={
                "symbol": sym,
                "action": action,
                "entry_price": price,
                "confidence": max_prob,
                "regime": regime,
                "atr": atr_val,
                "probabilities": {"short": float(proba[0]), "neutral": float(proba[1]), "long": float(proba[2])},
                "horizon_hours": self.horizon,
            },
            economic_rationale=f"XGBoost predicts {action} with {max_prob:.1%} confidence in {regime} regime",
        )
