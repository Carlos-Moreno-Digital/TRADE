"""Portfolio Manager Agent - Makes final trading decisions based on all signals."""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from trade.agents.base import BaseAgent, AgentSignal
from trade.config import AgentWeightsConfig
from trade.data.models import AnalysisContext, Signal, TradeAction, SignalStrength


# Map SignalStrength to numeric score
STRENGTH_SCORES = {
    SignalStrength.STRONG_BUY: 1.0,
    SignalStrength.BUY: 0.6,
    SignalStrength.WEAK_BUY: 0.3,
    SignalStrength.NEUTRAL: 0.0,
    SignalStrength.WEAK_SELL: -0.3,
    SignalStrength.SELL: -0.6,
    SignalStrength.STRONG_SELL: -1.0,
}


class PortfolioManagerAgent(BaseAgent):
    """Aggregates all agent signals and makes final trading decisions."""

    name = "portfolio_manager"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        weights: AgentWeightsConfig | None = None,
    ):
        super().__init__(config)
        self.weights = weights or AgentWeightsConfig()

    def analyze(self, context: AnalysisContext) -> AgentSignal:
        """Aggregate signals from all agents and decide final action."""
        start = time.time()
        self._logger.info(f"Portfolio manager deciding for {context.symbol}")

        # =====================================================================
        # ABSOLUTE RISK VETO - Cannot be overridden by any agent combination
        # This is the #1 most important safety mechanism
        # =====================================================================
        if context.metadata.get("risk_veto", False):
            risk_flags = context.metadata.get("risk_flags", [])
            self._logger.warning(
                f"ABSOLUTE RISK VETO for {context.symbol} - "
                f"trade BLOCKED. Flags: {risk_flags}"
            )
            signal = self._make_signal(
                context, TradeAction.HOLD, SignalStrength.NEUTRAL, 1.0,
                reasoning=f"ABSOLUTE RISK VETO - {'; '.join(risk_flags) if risk_flags else 'risk limits breached'}",
            )
            return self._make_output(signal, {"decision": "veto", "risk_flags": risk_flags}, exec_time=0)

        # Check risk score even if not full veto - high risk = reduce confidence
        risk_score = context.metadata.get("risk_score", 0.0)
        if risk_score >= 0.5:
            self._logger.warning(
                f"High risk score ({risk_score:.2f}) for {context.symbol} - forcing HOLD"
            )
            signal = self._make_signal(
                context, TradeAction.HOLD, SignalStrength.NEUTRAL, 0.8,
                reasoning=f"Risk score {risk_score:.2f} too high for trading",
            )
            return self._make_output(signal, {"decision": "high_risk"}, exec_time=0)

        # Collect signals from prior agents (EXCLUDE risk_manager - it vetoes, not votes)
        agent_signals = [s for s in context.signals if s.source_agent != "risk_manager"]
        if not agent_signals:
            signal = self._make_signal(
                context, TradeAction.HOLD, SignalStrength.NEUTRAL, 0.1,
                reasoning="No agent signals available",
            )
            return self._make_output(signal, notes="No signals")

        # Weight mapping for agent names (risk_manager excluded from voting)
        weight_map = {
            "market_data": self.weights.market_data,
            "technical": self.weights.technical,
            "sentiment": self.weights.sentiment,
            "fundamental": self.weights.fundamental,
        }

        # Calculate weighted score
        weighted_sum = 0.0
        total_weight = 0.0
        signal_details = []

        for sig in agent_signals:
            agent_name = sig.source_agent
            weight = weight_map.get(agent_name, 0.1)
            score = STRENGTH_SCORES.get(sig.strength, 0.0)
            weighted_score = score * weight * sig.confidence

            weighted_sum += weighted_score
            total_weight += weight

            signal_details.append({
                "agent": agent_name,
                "action": sig.action.value,
                "strength": sig.strength.value,
                "confidence": sig.confidence,
                "weight": weight,
                "weighted_score": round(weighted_score, 4),
            })

        # Normalize
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Apply kill zone bonus / non-optimal penalty to CONFIDENCE, not score
        # P1 FIX #6: Avoid non-linear bias on final_score
        kill_zone = context.metadata.get("kill_zone")
        confidence_modifier = 1.0
        if kill_zone:
            confidence_modifier = 1.1  # +10% confidence in kill zones
        elif not context.metadata.get("optimal_time", True):
            confidence_modifier = 0.7  # -30% confidence outside optimal times

        # Decision thresholds - balanced: not too aggressive, not too conservative
        min_confidence = self.config.get("min_confidence", 0.30)

        if final_score > 0.35:
            action = TradeAction.BUY
            if final_score > 0.65:
                strength = SignalStrength.STRONG_BUY
            elif final_score > 0.45:
                strength = SignalStrength.BUY
            else:
                strength = SignalStrength.WEAK_BUY
        elif final_score < -0.35:
            action = TradeAction.SELL
            if final_score < -0.65:
                strength = SignalStrength.STRONG_SELL
            elif final_score < -0.5:
                strength = SignalStrength.SELL
            else:
                strength = SignalStrength.WEAK_SELL
        else:
            action = TradeAction.HOLD
            strength = SignalStrength.NEUTRAL

        confidence = min(0.9, abs(final_score)) * confidence_modifier

        # Reject if confidence too low
        if confidence < min_confidence:
            action = TradeAction.HOLD
            strength = SignalStrength.NEUTRAL

        raw_data = {
            "final_score": round(final_score, 4),
            "signal_details": signal_details,
            "total_signals": len(agent_signals),
            "confidence": round(confidence, 4),
        }

        signal = self._make_signal(
            context, action, strength, confidence,
            reasoning=f"Weighted score: {final_score:.3f} from {len(agent_signals)} agents | "
                     f"Decision: {action.value} ({strength.value})",
        )

        exec_time = (time.time() - start) * 1000
        return self._make_output(signal, raw_data, exec_time=exec_time)
