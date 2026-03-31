"""Risk Manager Agent - Evaluates and controls risk for proposed trades."""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from trade.agents.base import BaseAgent, AgentSignal
from trade.config import RiskConfig
from trade.data.models import AnalysisContext, TradeAction, SignalStrength


class RiskManagerAgent(BaseAgent):
    """Evaluates risk and can veto trades that violate risk parameters."""

    name = "risk_manager"

    def __init__(self, config: dict[str, Any] | None = None, risk_config: RiskConfig | None = None):
        super().__init__(config)
        self.risk_config = risk_config or RiskConfig()

    def analyze(self, context: AnalysisContext) -> AgentSignal:
        """Evaluate risk for the current trading context."""
        start = time.time()
        self._logger.info(f"Evaluating risk for {context.symbol}")

        portfolio = context.portfolio
        risk_flags = []
        risk_score = 0.0  # 0 = no risk, 1 = maximum risk

        # Check 1: Portfolio drawdown
        if portfolio.max_drawdown_pct >= self.risk_config.max_portfolio_drawdown_pct:
            risk_flags.append(
                f"CRITICAL: Drawdown {portfolio.max_drawdown_pct:.1f}% >= "
                f"limit {self.risk_config.max_portfolio_drawdown_pct}%"
            )
            risk_score = 1.0

        # Check 2: Daily loss limit
        if portfolio.total_value > 0:
            daily_loss_pct = abs(portfolio.daily_pnl / portfolio.total_value * 100) if portfolio.daily_pnl < 0 else 0
            if daily_loss_pct >= self.risk_config.max_daily_loss_pct:
                risk_flags.append(
                    f"CRITICAL: Daily loss {daily_loss_pct:.1f}% >= "
                    f"limit {self.risk_config.max_daily_loss_pct}%"
                )
                risk_score = max(risk_score, 0.9)

        # Check 3: Consecutive losses cooldown
        if portfolio.consecutive_losses >= self.risk_config.consecutive_loss_threshold:
            risk_flags.append(
                f"WARNING: {portfolio.consecutive_losses} consecutive losses >= "
                f"threshold {self.risk_config.consecutive_loss_threshold}"
            )
            risk_score = max(risk_score, 0.8)

        # Check 4: Max trades per hour
        if portfolio.trades_today >= self.risk_config.max_trades_per_hour:
            risk_flags.append(
                f"WARNING: {portfolio.trades_today} trades today >= "
                f"hourly limit {self.risk_config.max_trades_per_hour}"
            )
            risk_score = max(risk_score, 0.7)

        # Check 5: Max open positions
        if len(portfolio.positions) >= self.risk_config.max_open_positions:
            risk_flags.append(
                f"WARNING: {len(portfolio.positions)} positions >= "
                f"limit {self.risk_config.max_open_positions}"
            )
            risk_score = max(risk_score, 0.6)

        # Check 6: Position concentration
        for pos in portfolio.positions:
            if pos.symbol == context.symbol:
                pos_value = abs(pos.quantity * pos.current_price)
                pos_pct = pos_value / portfolio.total_value * 100 if portfolio.total_value > 0 else 0
                if pos_pct >= self.risk_config.max_position_pct:
                    risk_flags.append(
                        f"WARNING: Position in {context.symbol} is {pos_pct:.1f}% "
                        f"of portfolio >= limit {self.risk_config.max_position_pct}%"
                    )
                    risk_score = max(risk_score, 0.6)

        # Check 7: Volatility assessment
        volatility = context.metadata.get("volatility", 0)
        if volatility > 5.0:  # High volatility
            risk_flags.append(f"CAUTION: High volatility ({volatility:.1f}%)")
            risk_score = max(risk_score, 0.4)

        # Determine signal based on risk score
        if risk_score >= 0.8:
            # VETO - too risky, force HOLD
            action = TradeAction.HOLD
            strength = SignalStrength.STRONG_SELL  # Strong signal to NOT trade
            confidence = 0.95
            reasoning = "RISK VETO: " + "; ".join(risk_flags)
        elif risk_score >= 0.5:
            action = TradeAction.HOLD
            strength = SignalStrength.NEUTRAL
            confidence = 0.7
            reasoning = "High risk: " + "; ".join(risk_flags)
        elif risk_score >= 0.3:
            # Allow trading but with caution
            action = TradeAction.HOLD  # Neutral - let other agents decide
            strength = SignalStrength.NEUTRAL
            confidence = 0.5
            reasoning = "Moderate risk: " + "; ".join(risk_flags) if risk_flags else "Moderate risk levels"
        else:
            # Low risk - green light
            action = TradeAction.HOLD  # Neutral positive - doesn't block
            strength = SignalStrength.NEUTRAL
            confidence = 0.3
            reasoning = "Risk levels acceptable"

        raw_data = {
            "risk_score": round(risk_score, 4),
            "risk_flags": risk_flags,
            "portfolio_drawdown_pct": round(portfolio.max_drawdown_pct, 4),
            "open_positions": len(portfolio.positions),
            "consecutive_losses": portfolio.consecutive_losses,
            "trades_today": portfolio.trades_today,
            "is_veto": risk_score >= 0.8,
        }

        # Store risk assessment in context
        context.metadata["risk_score"] = risk_score
        context.metadata["risk_veto"] = risk_score >= 0.8
        context.metadata["risk_flags"] = risk_flags

        signal = self._make_signal(
            context, action, strength, confidence,
            reasoning=reasoning,
            metadata={"risk_score": risk_score, "is_veto": risk_score >= 0.8},
        )

        exec_time = (time.time() - start) * 1000
        return self._make_output(signal, raw_data, exec_time=exec_time)
