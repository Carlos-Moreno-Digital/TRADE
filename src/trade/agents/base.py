"""Base agent class that all trading agents inherit from."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from trade.data.models import AnalysisContext, Signal, TradeAction, SignalStrength


class AgentSignal(BaseModel):
    """Output from an agent's analysis."""

    agent_name: str
    signal: Signal
    raw_data: dict[str, Any] = Field(default_factory=dict)
    analysis_notes: str = ""
    execution_time_ms: float = 0.0


class BaseAgent(ABC):
    """Base class for all trading agents."""

    name: str = "base_agent"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._logger = logger.bind(agent=self.name)

    @abstractmethod
    def analyze(self, context: AnalysisContext) -> AgentSignal:
        """Analyze the market context and return a signal.

        Args:
            context: Current market context with symbol, portfolio, and prior signals.

        Returns:
            AgentSignal with the agent's recommendation.
        """
        ...

    def _make_signal(
        self,
        context: AnalysisContext,
        action: TradeAction,
        strength: SignalStrength = SignalStrength.NEUTRAL,
        confidence: float = 0.5,
        reasoning: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Signal:
        """Helper to create a Signal from the analysis context."""
        return Signal(
            symbol=context.symbol,
            asset_type=context.asset_type,
            action=action,
            strength=strength,
            confidence=confidence,
            source_agent=self.name,
            reasoning=reasoning,
            metadata=metadata or {},
        )

    def _make_output(
        self,
        signal: Signal,
        raw_data: dict[str, Any] | None = None,
        notes: str = "",
        exec_time: float = 0.0,
    ) -> AgentSignal:
        """Helper to create an AgentSignal."""
        return AgentSignal(
            agent_name=self.name,
            signal=signal,
            raw_data=raw_data or {},
            analysis_notes=notes,
            execution_time_ms=exec_time,
        )
