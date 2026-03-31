"""Technical analysis using TA-Lib indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib
from loguru import logger

from trade.config import TechnicalConfig, TechnicalIndicatorConfig


class TechnicalAnalyzer:
    """Computes technical indicators on OHLCV data using TA-Lib."""

    def __init__(self, config: TechnicalConfig | None = None):
        self.config = config or TechnicalConfig()

    def compute_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all configured indicators on a DataFrame."""
        if df.empty:
            return df

        df = df.copy()
        for indicator_config in self.config.indicators:
            try:
                df = self._compute_indicator(df, indicator_config)
            except Exception as e:
                logger.warning(f"Failed to compute {indicator_config.name}: {e}")

        return df

    def _compute_indicator(
        self, df: pd.DataFrame, config: TechnicalIndicatorConfig
    ) -> pd.DataFrame:
        """Compute a single indicator."""
        name = config.name.upper()
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float) if "volume" in df.columns else None

        if name == "RSI":
            period = config.period or 14
            df[f"rsi_{period}"] = talib.RSI(close, timeperiod=period)

        elif name == "MACD":
            fast = config.fast or 12
            slow = config.slow or 26
            signal = config.signal or 9
            macd, macd_signal, macd_hist = talib.MACD(
                close, fastperiod=fast, slowperiod=slow, signalperiod=signal
            )
            df["macd"] = macd
            df["macd_signal"] = macd_signal
            df["macd_hist"] = macd_hist

        elif name == "BBANDS":
            period = config.period or 20
            std = config.std_dev or 2.0
            upper, middle, lower = talib.BBANDS(
                close, timeperiod=period, nbdevup=std, nbdevdn=std
            )
            df["bb_upper"] = upper
            df["bb_middle"] = middle
            df["bb_lower"] = lower

        elif name == "ATR":
            period = config.period or 14
            df[f"atr_{period}"] = talib.ATR(high, low, close, timeperiod=period)

        elif name == "EMA":
            periods = config.periods or [9, 21, 50, 200]
            for period in periods:
                df[f"ema_{period}"] = talib.EMA(close, timeperiod=period)

        elif name == "SMA":
            periods = config.periods or [20, 50, 200]
            for period in periods:
                df[f"sma_{period}"] = talib.SMA(close, timeperiod=period)

        elif name == "VWAP":
            if volume is not None:
                # TA-Lib doesn't have VWAP, compute manually
                typical_price = (high + low + close) / 3
                cum_vol = np.cumsum(volume)
                cum_tp_vol = np.cumsum(typical_price * volume)
                with np.errstate(divide="ignore", invalid="ignore"):
                    vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, 0)
                df["vwap"] = vwap

        elif name == "STOCH":
            k = config.k_period or 14
            d = config.d_period or 3
            slowk, slowd = talib.STOCH(
                high, low, close, fastk_period=k, slowk_period=d, slowd_period=d
            )
            df["stoch_k"] = slowk
            df["stoch_d"] = slowd

        elif name == "ADX":
            period = config.period or 14
            df[f"adx_{period}"] = talib.ADX(high, low, close, timeperiod=period)

        elif name == "OBV":
            if volume is not None:
                df["obv"] = talib.OBV(close, volume)

        else:
            logger.warning(f"Unknown indicator: {name}")

        return df

    def get_signal_summary(self, df: pd.DataFrame) -> dict:
        """Generate a summary of technical signals from the latest data point."""
        if df.empty:
            return {"trend": "unknown", "signals": []}

        latest = df.iloc[-1]
        signals = []

        # RSI signals
        for col in df.columns:
            if col.startswith("rsi_"):
                val = latest.get(col)
                if pd.notna(val):
                    if val > 70:
                        signals.append({"indicator": col, "signal": "overbought", "value": float(val)})
                    elif val < 30:
                        signals.append({"indicator": col, "signal": "oversold", "value": float(val)})

        # EMA trend
        price = float(latest.get("close", 0))
        for col in df.columns:
            if col.startswith("ema_"):
                val = latest.get(col)
                if pd.notna(val) and price > 0:
                    position = "above" if price > val else "below"
                    signals.append({"indicator": col, "signal": f"price_{position}", "value": float(val)})

        # MACD
        macd_hist = latest.get("macd_hist")
        if pd.notna(macd_hist):
            direction = "bullish" if macd_hist > 0 else "bearish"
            signals.append({"indicator": "macd_histogram", "signal": direction, "value": float(macd_hist)})

        # Bollinger Bands
        bb_upper = latest.get("bb_upper")
        bb_lower = latest.get("bb_lower")
        if pd.notna(bb_upper) and pd.notna(bb_lower) and price > 0:
            if price > bb_upper:
                signals.append({"indicator": "bbands", "signal": "above_upper", "value": price})
            elif price < bb_lower:
                signals.append({"indicator": "bbands", "signal": "below_lower", "value": price})

        # Stochastic
        stoch_k = latest.get("stoch_k")
        if pd.notna(stoch_k):
            if stoch_k > 80:
                signals.append({"indicator": "stoch", "signal": "overbought", "value": float(stoch_k)})
            elif stoch_k < 20:
                signals.append({"indicator": "stoch", "signal": "oversold", "value": float(stoch_k)})

        # Determine overall trend
        bullish = sum(1 for s in signals if s["signal"] in ("oversold", "price_above", "bullish", "below_lower"))
        bearish = sum(1 for s in signals if s["signal"] in ("overbought", "price_below", "bearish", "above_upper"))

        trend = "bullish" if bullish > bearish else "bearish" if bearish > bullish else "neutral"

        return {"trend": trend, "signals": signals, "bullish_count": bullish, "bearish_count": bearish}
