"""Market Data Agent - Fetches and prepares market data for analysis."""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from trade.agents.base import BaseAgent, AgentSignal
from trade.data.models import AnalysisContext, TradeAction, SignalStrength
from trade.data.providers import MarketDataProvider


class MarketDataAgent(BaseAgent):
    """Fetches market data and provides initial market assessment."""

    name = "market_data"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.provider = MarketDataProvider(
            provider=self.config.get("provider", "yfinance")
        )

    def analyze(self, context: AnalysisContext) -> AgentSignal:
        """Fetch market data and assess basic market conditions."""
        start = time.time()
        self._logger.info(f"Analyzing market data for {context.symbol}")

        # Fetch historical data
        df = self.provider.get_historical(
            context.symbol,
            period=self.config.get("period", "3mo"),
            interval=self.config.get("interval", "1d"),
        )

        if df.empty:
            signal = self._make_signal(
                context, TradeAction.HOLD, SignalStrength.NEUTRAL, 0.0,
                reasoning="No market data available",
            )
            return self._make_output(signal, notes="No data")

        # Basic price analysis
        latest_close = float(df["close"].iloc[-1])
        prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else latest_close
        change_pct = ((latest_close - prev_close) / prev_close * 100) if prev_close > 0 else 0

        # Volume analysis
        avg_volume = float(df["volume"].mean()) if "volume" in df.columns else 0
        latest_volume = float(df["volume"].iloc[-1]) if "volume" in df.columns else 0
        volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 1.0

        # 20-day price change
        if len(df) >= 20:
            price_20d_ago = float(df["close"].iloc[-20])
            change_20d = (latest_close - price_20d_ago) / price_20d_ago * 100
        else:
            change_20d = 0.0

        # Volatility (20-day standard deviation of returns)
        if len(df) >= 20:
            returns = df["close"].pct_change().dropna()
            volatility = float(returns.tail(20).std() * 100)
        else:
            volatility = 0.0

        # Determine signal
        if change_20d > 5 and volume_ratio > 1.2:
            action = TradeAction.BUY
            strength = SignalStrength.BUY
            confidence = min(0.7, 0.5 + abs(change_20d) / 100)
        elif change_20d < -5 and volume_ratio > 1.2:
            action = TradeAction.SELL
            strength = SignalStrength.SELL
            confidence = min(0.7, 0.5 + abs(change_20d) / 100)
        else:
            action = TradeAction.HOLD
            strength = SignalStrength.NEUTRAL
            confidence = 0.3

        raw_data = {
            "latest_close": latest_close,
            "prev_close": prev_close,
            "change_pct": round(change_pct, 4),
            "change_20d_pct": round(change_20d, 4),
            "avg_volume": avg_volume,
            "latest_volume": latest_volume,
            "volume_ratio": round(volume_ratio, 4),
            "volatility_20d": round(volatility, 4),
            "data_points": len(df),
            "historical_df": df,  # Pass to other agents
        }

        # Store historical data in context metadata for other agents
        context.metadata["historical_df"] = df
        context.metadata["latest_close"] = latest_close
        context.metadata["volatility"] = volatility

        signal = self._make_signal(
            context, action, strength, confidence,
            reasoning=f"Price: {latest_close:.2f}, 20d change: {change_20d:.1f}%, "
                     f"Volume ratio: {volume_ratio:.2f}, Volatility: {volatility:.2f}%",
        )

        exec_time = (time.time() - start) * 1000
        return self._make_output(signal, raw_data, exec_time=exec_time)
