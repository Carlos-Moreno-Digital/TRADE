"""Multi-Agent Trading System — JSON Communication Protocol.

Standard message format for inter-agent communication.
All agents MUST use AgentMessage for inputs and outputs.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
from datetime import datetime
import json


@dataclass
class AgentMessage:
    """Standard JSON message for inter-agent communication."""
    agent_domain: str
    status_flag: str
    computational_payload: dict = field(default_factory=dict)
    economic_rationale: str = ""
    timestamp: str = ""
    errors: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_json(cls, data: str | dict) -> "AgentMessage":
        if isinstance(data, str):
            data = json.loads(data)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def is_approved(self) -> bool:
        return self.status_flag in ("SIGNAL_GENERATED", "RISK_VALIDATED",
                                     "BACKTEST_PASSED", "COMPLIANT", "APPROVED")

    def is_rejected(self) -> bool:
        return self.status_flag in ("REJECTED", "FAILED", "BLOCKED")
