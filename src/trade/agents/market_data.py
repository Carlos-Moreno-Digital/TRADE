"""Market Data Agent - Fetches multi-timeframe data for analysis."""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from trade.agents.base import BaseAgent, AgentSignal
from trade.data.models import AnalysisContext, TradeAction, SignalStrength
from trade.data.providers import MarketDataProvider


class MarketDataAgent(BaseAgent):
    """Fetches market data across multiple timeframes."""

    name = "market_data"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.provider = MarketDataProvider(
            provider=self.config.get("provider", "yfinance")
        )

    def analyze(self, context: AnalysisContext) -> AgentSignal:
        """Fetch multi-timeframe market data and assess conditions."""
        start = time.time()
        self._logger.info(f"Analyzing market data for {context.symbol}")

        # =====================================================================
        # MULTI-TIMEFRAME DATA FETCHING
        # Daily = trend context, 1H = entry timing, 15min = precision
        # =====================================================================

        # Daily data (trend context - 3 months)
        df_daily = self.provider.get_historical(
            context.symbol,
            period="3mo",
            interval="1d",
        )

        # 1-Hour data (entry timing - 1 month)
        df_1h = self.provider.get_historical(
            context.symbol,
            period="1mo",
            interval="1h",
        )

        # Use best available data as primary
        if not df_1h.empty:
            df = df_1h  # Prefer intraday for fresher data
            self._logger.info(f"Using 1H data ({len(df_1h)} candles) for {context.symbol}")
        elif not df_daily.empty:
            df = df_daily
            self._logger.info(f"Using daily data ({len(df_daily)} candles) for {context.symbol}")
        else:
            signal = self._make_signal(
                context, TradeAction.HOLD, SignalStrength.NEUTRAL, 0.0,
                reasoning="No market data available",
            )
            return self._make_output(signal, notes="No data")

        # Basic price analysis from primary timeframe
        latest_close = float(df["close"].iloc[-1])
        prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else latest_close
        change_pct = ((latest_close - prev_close) / prev_close * 100) if prev_close > 0 else 0

        # Volume analysis
        avg_volume = float(df["volume"].mean()) if "volume" in df.columns else 0
        latest_volume = float(df["volume"].iloc[-1]) if "volume" in df.columns else 0
        volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 1.0

        # Daily trend (from daily data)
        daily_trend = "neutral"
        change_20d = 0.0
        if not df_daily.empty and len(df_daily) >= 20:
            price_20d_ago = float(df_daily["close"].iloc[-20])
            change_20d = (latest_close - price_20d_ago) / price_20d_ago * 100
            if change_20d > 2:
                daily_trend = "bullish"
            elif change_20d < -2:
                daily_trend = "bearish"

        # Intraday momentum (from 1H data - last 24 candles = 1 day)
        intraday_trend = "neutral"
        change_24h = 0.0
        if not df_1h.empty and len(df_1h) >= 24:
            price_24h_ago = float(df_1h["close"].iloc[-24])
            change_24h = (latest_close - price_24h_ago) / price_24h_ago * 100
            if change_24h > 0.3:
                intraday_trend = "bullish"
            elif change_24h < -0.3:
                intraday_trend = "bearish"

        # Volatility (from 1H returns for more recent reading)
        if not df_1h.empty and len(df_1h) >= 20:
            returns = df_1h["close"].pct_change().dropna()
            volatility = float(returns.tail(20).std() * 100)
        elif not df_daily.empty and len(df_daily) >= 20:
            returns = df_daily["close"].pct_change().dropna()
            volatility = float(returns.tail(20).std() * 100)
        else:
            volatility = 0.0

        # =====================================================================
        # TREND ALIGNMENT CHECK
        # Both timeframes agreeing = higher confidence
        # =====================================================================
        trend_aligned = daily_trend == intraday_trend and daily_trend != "neutral"

        if trend_aligned:
            if daily_trend == "bullish":
                action = TradeAction.BUY
                strength = SignalStrength.BUY
                confidence = min(0.75, 0.5 + abs(change_24h) / 5)
            else:
                action = TradeAction.SELL
                strength = SignalStrength.SELL
                confidence = min(0.75, 0.5 + abs(change_24h) / 5)
        elif daily_trend != "neutral" and intraday_trend == "neutral":
            # Daily has direction, intraday is flat - weak signal
            if daily_trend == "bullish":
                action = TradeAction.BUY
                strength = SignalStrength.WEAK_BUY
                confidence = 0.35
            else:
                action = TradeAction.SELL
                strength = SignalStrength.WEAK_SELL
                confidence = 0.35
        elif daily_trend != intraday_trend and intraday_trend != "neutral" and daily_trend != "neutral":
            # Timeframes DISAGREE - stay out
            action = TradeAction.HOLD
            strength = SignalStrength.NEUTRAL
            confidence = 0.2  # Low confidence when disagreeing
        else:
            action = TradeAction.HOLD
            strength = SignalStrength.NEUTRAL
            confidence = 0.3

        raw_data = {
            "latest_close": latest_close,
            "prev_close": prev_close,
            "change_pct": round(change_pct, 4),
            "change_20d_pct": round(change_20d, 4),
            "change_24h_pct": round(change_24h, 4),
            "daily_trend": daily_trend,
            "intraday_trend": intraday_trend,
            "trend_aligned": trend_aligned,
            "avg_volume": avg_volume,
            "latest_volume": latest_volume,
            "volume_ratio": round(volume_ratio, 4),
            "volatility": round(volatility, 4),
            "data_points_daily": len(df_daily),
            "data_points_1h": len(df_1h),
        }

        # Store in context for other agents
        context.metadata["historical_df"] = df  # Primary (1H or daily)
        context.metadata["historical_df_daily"] = df_daily if not df_daily.empty else None
        context.metadata["historical_df_1h"] = df_1h if not df_1h.empty else None
        context.metadata["latest_close"] = latest_close
        context.metadata["volatility"] = volatility
        context.metadata["daily_trend"] = daily_trend
        context.metadata["intraday_trend"] = intraday_trend
        context.metadata["trend_aligned"] = trend_aligned

        signal = self._make_signal(
            context, action, strength, confidence,
            reasoning=f"Price: {latest_close:.5f} | Daily: {daily_trend} ({change_20d:+.1f}%) | "
                     f"Intraday: {intraday_trend} ({change_24h:+.2f}%) | "
                     f"Aligned: {'YES' if trend_aligned else 'NO'} | "
                     f"Vol: {volatility:.2f}%",
        )

        exec_time = (time.time() - start) * 1000
        return self._make_output(signal, raw_data, exec_time=exec_time)
