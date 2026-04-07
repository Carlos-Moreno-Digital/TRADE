"""Orchestrator Agent — Pipeline Coordination & Consensus.

Coordinates the flow: Alpha Engine → Risk Shield → Compliance → Execute
Implements 3-phase consensus protocol.
Compliance has ABSOLUTE VETO.

HISTORY:
- XGBoost ML Alpha Generator: FAILED on 16yr Dukascopy (-$26K, 4/17 years).
- BBMR Engine: FAILED paranoid validation + CPCV + NautilusTrader realistic
  fills (EURUSD -78.7%, USDJPY -3.8%, WFE 0.06-0.29, time-permutation p>0.15).
  See VERDICT_BBMR.md. Deleted 2026-04-07.

The alpha slot is intentionally empty until Phase 3 (LOBFrame + Statistical
Jump Model + NMI dependency matrix) produces a validated strategy that
survives the full paranoid + CPCV + NautilusTrader gauntlet.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger
from rich.console import Console

from trade.agents.multi import AgentMessage
from trade.agents.multi.risk_shield import RiskShield
from trade.agents.multi.quant_tester import QuantTester
from trade.agents.multi.compliance import ComplianceAgent

console = Console()


class Orchestrator:
    """Coordinates multi-agent trading pipeline.

    Flow: Alpha Engine → Risk Shield → Compliance → Execute
    3 phases: Collection, Argumentation, Resolution.
    """

    def __init__(self):
        self.alpha = None  # Alpha engine slot — empty until Phase 3 validates one
        self.risk = RiskShield()
        self.quant = QuantTester()
        self.compliance = ComplianceAgent()
        self.pipeline_log: list[dict] = []

    def train_models(self, symbols: list[str], data: dict[str, pd.DataFrame]) -> dict[str, bool]:
        """No-op until a validated alpha engine is wired in (Phase 3)."""
        return {sym: False for sym in symbols}

    def evaluate(self, sym: str, df: pd.DataFrame, account_state: dict) -> AgentMessage:
        """Run the full pipeline for a symbol.

        Returns final decision: APPROVED or REJECTED with full audit trail.
        """
        if self.alpha is None:
            raise NotImplementedError(
                "No alpha engine registered. BBMR was deleted 2026-04-07 after "
                "failing paranoid validation + CPCV + NautilusTrader realistic "
                "fills. Phase 3 (LOBFrame + Statistical Jump Model + NMI) is "
                "expected to produce the next candidate. See VERDICT_BBMR.md."
            )

        pipeline_entry = {
            "symbol": sym,
            "timestamp": datetime.utcnow().isoformat(),
            "phases": [],
        }

        # ============================================
        # PHASE 1: COLLECTION — alpha engine checks conditions
        # ============================================
        alpha_msg = self.alpha.generate_signal(sym, df)
        pipeline_entry["phases"].append({
            "agent": "alpha_engine",
            "status": alpha_msg.status_flag,
            "rationale": alpha_msg.economic_rationale,
        })

        if not alpha_msg.is_approved():
            pipeline_entry["final"] = "NO_SIGNAL"
            self.pipeline_log.append(pipeline_entry)
            return alpha_msg

        # ============================================
        # PHASE 2: ARGUMENTATION — Risk Shield validates
        # ============================================

        # Risk Shield validates position sizing and limits
        risk_msg = self.risk.validate_and_size(alpha_msg, account_state)
        pipeline_entry["phases"].append({
            "agent": "risk_shield",
            "status": risk_msg.status_flag,
            "rationale": risk_msg.economic_rationale,
            "errors": risk_msg.errors,
        })

        if risk_msg.is_rejected():
            pipeline_entry["final"] = "BLOCKED_BY_RISK"
            self.pipeline_log.append(pipeline_entry)
            return risk_msg

        # Skip Quant Tester — BBMR is already validated on 16yr data
        # The Quant Tester added noise without value

        # PHASE 3: RESOLUTION — Compliance has VETO
        # ============================================
        compliance_msg = self.compliance.validate(risk_msg, account_state)
        pipeline_entry["phases"].append({
            "agent": "compliance",
            "status": compliance_msg.status_flag,
            "rationale": compliance_msg.economic_rationale,
            "errors": compliance_msg.errors,
        })

        if compliance_msg.is_rejected():
            # COMPLIANCE VETO IS ABSOLUTE — no override possible
            pipeline_entry["final"] = "VETOED_BY_COMPLIANCE"
            self.pipeline_log.append(pipeline_entry)
            logger.warning(f"COMPLIANCE VETO: {sym} — {compliance_msg.errors}")
            return compliance_msg

        # === ALL AGENTS AGREE: TRADE APPROVED ===
        payload = compliance_msg.computational_payload
        final_msg = AgentMessage(
            agent_domain="orchestrator",
            status_flag="APPROVED",
            computational_payload=payload,
            economic_rationale=(
                f"BBMR {payload.get('action')} {sym}: "
                f"{payload.get('regime', 'RANGING')} | "
                f"R:R={payload.get('rr_ratio', 0):.2f} | "
                f"ADX={payload.get('adx', 0):.1f} | "
                f"Compliance PASS"
            ),
        )

        pipeline_entry["final"] = "APPROVED"
        self.pipeline_log.append(pipeline_entry)

        return final_msg

    def get_diagnostic(self, sym: str, df: pd.DataFrame) -> str:
        """Get diagnostic string showing what BBMR sees."""
        alpha_msg = self.alpha.generate_signal(sym, df)
        payload = alpha_msg.computational_payload

        if alpha_msg.status_flag == "NO_SIGNAL":
            reason = payload.get("reason", "no touch")
            adx = payload.get("adx", "?")
            close = payload.get("close", 0)
            upper = payload.get("upper_bb", 0)
            lower = payload.get("lower_bb", 0)
            if close and upper and lower:
                return (f"{sym}: ADX={adx} | "
                        f"C={close:.5f} [{lower:.5f} — {upper:.5f}] | "
                        f"{reason}")
            return f"{sym}: NO_SIGNAL ({reason}, ADX={adx})"

        if alpha_msg.status_flag == "SIGNAL_GENERATED":
            return (f"{sym}: BBMR {payload.get('action')} @ "
                    f"{payload.get('entry_price', 0):.5f} | "
                    f"R:R={payload.get('rr_ratio', 0):.2f}")

        return f"{sym}: {alpha_msg.status_flag}"
