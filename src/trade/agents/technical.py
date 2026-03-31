"""Technical Analysis Agent - Multi-timeframe indicator analysis."""

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
    """Analyzes market data using technical indicators across multiple timeframes.

    Strategy: Higher timeframe (daily) confirms trend direction,
    lower timeframe (1H) provides entry timing.
    Confluence = higher confidence.
    """

    name = "technical"

    def __init__(self, config: dict[str, Any] | None = None, technical_config: TechnicalConfig | None = None):
        super().__init__(config)
        self.analyzer = TechnicalAnalyzer(technical_config)

    def analyze(self, context: AnalysisContext) -> AgentSignal:
        """Run multi-timeframe technical analysis."""
        start = time.time()
        self._logger.info(f"Running technical analysis for {context.symbol}")

        # Get data from both timeframes
        df_primary = context.metadata.get("historical_df")
        df_daily = context.metadata.get("historical_df_daily")
        df_1h = context.metadata.get("historical_df_1h")

        if df_primary is None or df_primary.empty:
            signal = self._make_signal(
                context, TradeAction.HOLD, SignalStrength.NEUTRAL, 0.0,
                reasoning="No historical data for technical analysis",
            )
            return self._make_output(signal, notes="No data")

        # =====================================================================
        # MULTI-TIMEFRAME ANALYSIS
        # =====================================================================

        # Analyze primary timeframe (1H if available, else daily)
        df_indicators = self.analyzer.compute_all_indicators(df_primary)
        primary_summary = self.analyzer.get_signal_summary(df_indicators)

        # Analyze daily timeframe for trend confirmation
        daily_trend = "neutral"
        daily_summary = None
        if df_daily is not None and not df_daily.empty and len(df_daily) >= 20:
            df_daily_ind = self.analyzer.compute_all_indicators(df_daily)
            daily_summary = self.analyzer.get_signal_summary(df_daily_ind)
            daily_trend = daily_summary["trend"]

            # Store daily ATR for position sizing (more stable than 1H ATR)
            daily_atr_cols = [c for c in df_daily_ind.columns if c.startswith("atr_")]
            if daily_atr_cols:
                atr_val = df_daily_ind[daily_atr_cols[0]].iloc[-1]
                if pd.notna(atr_val) and atr_val > 0:
                    context.metadata["atr"] = float(atr_val)

        # Primary timeframe signals
        trend = primary_summary["trend"]
        bullish_count = primary_summary.get("bullish_count", 0)
        bearish_count = primary_summary.get("bearish_count", 0)
        total_signals = bullish_count + bearish_count

        # Store ATR from primary if daily not available
        if "atr" not in context.metadata:
            atr_cols = [c for c in df_indicators.columns if c.startswith("atr_")]
            if atr_cols:
                atr_val = df_indicators[atr_cols[0]].iloc[-1]
                if pd.notna(atr_val) and atr_val > 0:
                    context.metadata["atr"] = float(atr_val)

        # =====================================================================
        # CONFLUENCE SCORING
        # Daily trend agrees with 1H signals = boost
        # Daily trend disagrees = reduce or block
        # =====================================================================

        # Base signal from primary timeframe
        if total_signals == 0:
            action = TradeAction.HOLD
            strength = SignalStrength.NEUTRAL
            confidence = 0.2
        elif bullish_count > bearish_count:
            ratio = bullish_count / total_signals
            action = TradeAction.BUY
            if ratio > 0.75:
                strength = SignalStrength.STRONG_BUY
                confidence = min(0.85, ratio)
            elif ratio > 0.6:
                strength = SignalStrength.BUY
                confidence = min(0.7, ratio)
            else:
                strength = SignalStrength.WEAK_BUY
                confidence = min(0.55, ratio)
        elif bearish_count > bullish_count:
            ratio = bearish_count / total_signals
            action = TradeAction.SELL
            if ratio > 0.75:
                strength = SignalStrength.STRONG_SELL
                confidence = min(0.85, ratio)
            elif ratio > 0.6:
                strength = SignalStrength.SELL
                confidence = min(0.7, ratio)
            else:
                strength = SignalStrength.WEAK_SELL
                confidence = min(0.55, ratio)
        else:
            action = TradeAction.HOLD
            strength = SignalStrength.NEUTRAL
            confidence = 0.3

        # Apply multi-timeframe confluence adjustment
        confluence = "none"
        if daily_trend != "neutral" and action != TradeAction.HOLD:
            if (daily_trend == "bullish" and action == TradeAction.BUY) or \
               (daily_trend == "bearish" and action == TradeAction.SELL):
                # CONFLUENCE: both timeframes agree
                confluence = "aligned"
                confidence = min(0.90, confidence * 1.15)  # +15% boost
            elif (daily_trend == "bullish" and action == TradeAction.SELL) or \
                 (daily_trend == "bearish" and action == TradeAction.BUY):
                # CONFLICT: trading against daily trend - dangerous
                confluence = "conflict"
                confidence *= 0.5  # -50% penalty (counter-trend is risky)
                # Downgrade strength
                if action == TradeAction.BUY:
                    strength = SignalStrength.WEAK_BUY
                else:
                    strength = SignalStrength.WEAK_SELL

        raw_data = {
            "trend_primary": trend,
            "trend_daily": daily_trend,
            "confluence": confluence,
            "bullish_signals": bullish_count,
            "bearish_signals": bearish_count,
            "total_signals": total_signals,
            "indicator_summary": primary_summary["signals"],
            "daily_summary": daily_summary["signals"] if daily_summary else [],
        }

        # Store in context
        context.metadata["technical_trend"] = trend
        context.metadata["technical_trend_daily"] = daily_trend
        context.metadata["technical_confluence"] = confluence
        context.metadata["technical_signals"] = primary_summary["signals"]

        signal = self._make_signal(
            context, action, strength, confidence,
            reasoning=f"1H: {trend} (B:{bullish_count}/S:{bearish_count}) | "
                     f"Daily: {daily_trend} | "
                     f"Confluence: {confluence} | "
                     f"Conf: {confidence:.2f}",
            metadata=raw_data,
        )

        exec_time = (time.time() - start) * 1000
        return self._make_output(signal, raw_data, exec_time=exec_time)
