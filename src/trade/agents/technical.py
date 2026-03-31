"""Technical Analysis Agent - Computes and interprets technical indicators."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
from loguru import logger

from trade.agents.base import BaseAgent, AgentSignal
from trade.analysis.technical import TechnicalAnalyzer
from trade.config import TechnicalConfig
from trade.data.models import AnalysisContext, TradeAction, SignalStrength


class TechnicalAgent(BaseAgent):
    """Analyzes market data using technical indicators."""

    name = "technical"

    def __init__(self, config: dict[str, Any] | None = None, technical_config: TechnicalConfig | None = None):
        super().__init__(config)
        self.analyzer = TechnicalAnalyzer(technical_config)

    def analyze(self, context: AnalysisContext) -> AgentSignal:
        """Run technical analysis on historical data."""
        start = time.time()
        self._logger.info(f"Running technical analysis for {context.symbol}")

        df = context.metadata.get("historical_df")
        if df is None or df.empty:
            signal = self._make_signal(
                context, TradeAction.HOLD, SignalStrength.NEUTRAL, 0.0,
                reasoning="No historical data available for technical analysis",
            )
            return self._make_output(signal, notes="No data")

        # Compute all indicators
        df_with_indicators = self.analyzer.compute_all_indicators(df)

        # Get signal summary
        summary = self.analyzer.get_signal_summary(df_with_indicators)
        trend = summary["trend"]
        bullish_count = summary.get("bullish_count", 0)
        bearish_count = summary.get("bearish_count", 0)
        total_signals = bullish_count + bearish_count

        # Determine action based on indicator consensus
        if total_signals == 0:
            action = TradeAction.HOLD
            strength = SignalStrength.NEUTRAL
            confidence = 0.2
        elif bullish_count > bearish_count:
            ratio = bullish_count / total_signals
            if ratio > 0.75:
                action = TradeAction.BUY
                strength = SignalStrength.STRONG_BUY
                confidence = min(0.85, ratio)
            elif ratio > 0.6:
                action = TradeAction.BUY
                strength = SignalStrength.BUY
                confidence = min(0.7, ratio)
            else:
                action = TradeAction.BUY
                strength = SignalStrength.WEAK_BUY
                confidence = min(0.55, ratio)
        elif bearish_count > bullish_count:
            ratio = bearish_count / total_signals
            if ratio > 0.75:
                action = TradeAction.SELL
                strength = SignalStrength.STRONG_SELL
                confidence = min(0.85, ratio)
            elif ratio > 0.6:
                action = TradeAction.SELL
                strength = SignalStrength.SELL
                confidence = min(0.7, ratio)
            else:
                action = TradeAction.SELL
                strength = SignalStrength.WEAK_SELL
                confidence = min(0.55, ratio)
        else:
            action = TradeAction.HOLD
            strength = SignalStrength.NEUTRAL
            confidence = 0.3

        raw_data = {
            "trend": trend,
            "bullish_signals": bullish_count,
            "bearish_signals": bearish_count,
            "total_signals": total_signals,
            "indicator_summary": summary["signals"],
        }

        # Store in context for other agents
        context.metadata["technical_trend"] = trend
        context.metadata["technical_signals"] = summary["signals"]

        signal = self._make_signal(
            context, action, strength, confidence,
            reasoning=f"Trend: {trend}, Bullish: {bullish_count}, Bearish: {bearish_count}",
            metadata=raw_data,
        )

        exec_time = (time.time() - start) * 1000
        return self._make_output(signal, raw_data, exec_time=exec_time)
