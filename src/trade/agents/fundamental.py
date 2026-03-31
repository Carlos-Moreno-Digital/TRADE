"""Fundamental Analysis Agent - Analyzes company financials and valuation."""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from trade.agents.base import BaseAgent, AgentSignal
from trade.data.models import AnalysisContext, AssetType, TradeAction, SignalStrength
from trade.data.providers import MarketDataProvider


class FundamentalAgent(BaseAgent):
    """Analyzes fundamental data for stocks."""

    name = "fundamental"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.provider = MarketDataProvider()

    def analyze(self, context: AnalysisContext) -> AgentSignal:
        """Analyze fundamental data for a symbol."""
        start = time.time()
        self._logger.info(f"Analyzing fundamentals for {context.symbol}")

        # Fundamentals only apply to stocks
        if context.asset_type != AssetType.STOCK:
            signal = self._make_signal(
                context, TradeAction.HOLD, SignalStrength.NEUTRAL, 0.0,
                reasoning=f"Fundamental analysis not applicable for {context.asset_type.value}",
            )
            return self._make_output(signal, notes="Non-stock asset")

        fundamentals = self.provider.get_fundamentals(context.symbol)
        if not fundamentals:
            signal = self._make_signal(
                context, TradeAction.HOLD, SignalStrength.NEUTRAL, 0.0,
                reasoning="No fundamental data available",
            )
            return self._make_output(signal, notes="No data")

        # Score different fundamental factors
        scores = []

        # P/E ratio analysis
        pe = fundamentals.get("pe_ratio")
        if pe is not None and pe > 0:
            if pe < 15:
                scores.append(("pe_ratio", 1.0, "Undervalued by P/E"))
            elif pe < 25:
                scores.append(("pe_ratio", 0.5, "Fair P/E"))
            elif pe < 40:
                scores.append(("pe_ratio", 0.0, "Expensive by P/E"))
            else:
                scores.append(("pe_ratio", -0.5, "Very expensive by P/E"))

        # Profit margin
        margin = fundamentals.get("profit_margin")
        if margin is not None:
            if margin > 0.2:
                scores.append(("profit_margin", 1.0, "High profit margin"))
            elif margin > 0.1:
                scores.append(("profit_margin", 0.5, "Decent margin"))
            elif margin > 0:
                scores.append(("profit_margin", 0.0, "Low margin"))
            else:
                scores.append(("profit_margin", -0.5, "Negative margin"))

        # ROE
        roe = fundamentals.get("roe")
        if roe is not None:
            if roe > 0.2:
                scores.append(("roe", 1.0, "Excellent ROE"))
            elif roe > 0.1:
                scores.append(("roe", 0.5, "Good ROE"))
            else:
                scores.append(("roe", 0.0, "Low ROE"))

        # Debt to Equity
        de = fundamentals.get("debt_to_equity")
        if de is not None:
            if de < 50:
                scores.append(("debt_equity", 0.5, "Low debt"))
            elif de < 100:
                scores.append(("debt_equity", 0.0, "Moderate debt"))
            else:
                scores.append(("debt_equity", -0.5, "High debt"))

        # Calculate overall score
        if scores:
            avg_score = sum(s[1] for s in scores) / len(scores)
        else:
            avg_score = 0.0

        # Map score to signal
        if avg_score > 0.4:
            action = TradeAction.BUY
            strength = SignalStrength.BUY if avg_score > 0.6 else SignalStrength.WEAK_BUY
        elif avg_score < -0.2:
            action = TradeAction.SELL
            strength = SignalStrength.SELL if avg_score < -0.4 else SignalStrength.WEAK_SELL
        else:
            action = TradeAction.HOLD
            strength = SignalStrength.NEUTRAL

        confidence = min(0.7, 0.3 + abs(avg_score) * 0.5)

        raw_data = {
            "fundamentals": fundamentals,
            "factor_scores": [(s[0], s[1], s[2]) for s in scores],
            "overall_score": round(avg_score, 4),
        }

        signal = self._make_signal(
            context, action, strength, confidence,
            reasoning=f"Fundamental score: {avg_score:.2f} | " +
                     " | ".join(f"{s[0]}: {s[2]}" for s in scores[:3]),
        )

        exec_time = (time.time() - start) * 1000
        return self._make_output(signal, raw_data, exec_time=exec_time)
