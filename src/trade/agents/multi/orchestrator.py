"""Orchestrator Agent — Pipeline Coordination & Consensus.

Coordinates the flow: Alpha → Risk Shield → Quant Tester → Compliance → Execute
Implements 3-phase consensus protocol.
Compliance has ABSOLUTE VETO.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger
from rich.console import Console

from trade.agents.multi import AgentMessage
from trade.agents.multi.alpha_generator import AlphaGenerator
from trade.agents.multi.risk_shield import RiskShield
from trade.agents.multi.quant_tester import QuantTester
from trade.agents.multi.compliance import ComplianceAgent

console = Console()


class Orchestrator:
    """Coordinates multi-agent trading pipeline.

    Flow: Alpha Generator → Risk Shield → Quant Tester → Compliance → Execute
    3 phases: Collection, Argumentation, Resolution.
    """

    def __init__(self):
        self.alpha = AlphaGenerator()
        self.risk = RiskShield()
        self.quant = QuantTester()
        self.compliance = ComplianceAgent()
        self.pipeline_log: list[dict] = []

    def train_models(self, symbols: list[str], data: dict[str, pd.DataFrame]) -> dict[str, bool]:
        """Train ML models for all symbols."""
        results = {}
        for sym in symbols:
            df = data.get(sym)
            if df is None:
                results[sym] = False
                continue
            results[sym] = self.alpha.train(sym, df)
        return results

    def evaluate(self, sym: str, df: pd.DataFrame, account_state: dict) -> AgentMessage:
        """Run the full pipeline for a symbol.

        Returns final decision: APPROVED or REJECTED with full audit trail.
        """
        pipeline_entry = {
            "symbol": sym,
            "timestamp": datetime.utcnow().isoformat(),
            "phases": [],
        }

        # ============================================
        # PHASE 1: COLLECTION — Alpha generates signal
        # ============================================
        alpha_msg = self.alpha.generate_signal(sym, df)
        pipeline_entry["phases"].append({
            "agent": "alpha_generator",
            "status": alpha_msg.status_flag,
            "rationale": alpha_msg.economic_rationale,
        })

        if not alpha_msg.is_approved():
            pipeline_entry["final"] = "NO_SIGNAL"
            self.pipeline_log.append(pipeline_entry)
            return alpha_msg

        # ============================================
        # PHASE 2: ARGUMENTATION — Risk + Quant validate
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

        # Quant Tester validates recent performance
        quant_msg = self.quant.validate(risk_msg, df)
        pipeline_entry["phases"].append({
            "agent": "quant_tester",
            "status": quant_msg.status_flag,
            "rationale": quant_msg.economic_rationale,
            "errors": quant_msg.errors,
        })

        if quant_msg.is_rejected():
            pipeline_entry["final"] = "FAILED_BACKTEST"
            self.pipeline_log.append(pipeline_entry)
            return quant_msg

        # ============================================
        # PHASE 3: RESOLUTION — Compliance has VETO
        # ============================================
        compliance_msg = self.compliance.validate(quant_msg, account_state)
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
        final_msg = AgentMessage(
            agent_domain="orchestrator",
            status_flag="APPROVED",
            computational_payload=compliance_msg.computational_payload,
            economic_rationale=(
                f"UNANIMOUS: {sym} {compliance_msg.computational_payload.get('action')} approved. "
                f"Alpha ({alpha_msg.computational_payload.get('confidence', 0):.1%}) → "
                f"Risk (R:R={compliance_msg.computational_payload.get('rr_ratio', 0):.2f}) → "
                f"Quant (PF={compliance_msg.computational_payload.get('backtest_metrics', {}).get('recent_pf', 'N/A')}) → "
                f"Compliance (PASS)"
            ),
        )

        pipeline_entry["final"] = "APPROVED"
        self.pipeline_log.append(pipeline_entry)

        return final_msg

    def get_diagnostic(self, sym: str, df: pd.DataFrame) -> str:
        """Get diagnostic string showing what each agent thinks."""
        alpha_msg = self.alpha.generate_signal(sym, df)
        payload = alpha_msg.computational_payload

        if alpha_msg.status_flag == "NO_SIGNAL":
            probs = payload.get("probabilities", {})
            regime = payload.get("regime", "?")
            best = max(probs, key=probs.get) if probs else "?"
            best_val = probs.get(best, 0)
            return (f"{sym}: {best.upper()} ({best_val:.1%}) | "
                    f"S:{probs.get('short', 0):.1%} N:{probs.get('neutral', 0):.1%} "
                    f"L:{probs.get('long', 0):.1%} | {regime}")

        return f"{sym}: {alpha_msg.status_flag} {payload.get('action', '?')} ({payload.get('confidence', 0):.1%})"
