"""Compliance Agent — FunderPro Rule Enforcement.

ABSOLUTE VETO authority. No trade proceeds without compliance approval.
Validates every signal against FunderPro Classic $10K regulations.
"""

from __future__ import annotations

from datetime import datetime, timezone

from trade.agents.multi import AgentMessage


class ComplianceAgent:
    """Enforces FunderPro Classic $10K compliance rules.

    This agent has VETO power. If ANY rule is violated, the trade is REJECTED.
    No other agent can override a compliance rejection.
    """

    def validate(self, signal: AgentMessage, account_state: dict) -> AgentMessage:
        """Validate a risk-adjusted signal against all FunderPro rules."""
        payload = signal.computational_payload
        violations = []

        # === RULE 1: SL MUST EXIST ===
        if not payload.get("sl_price") or payload["sl_price"] <= 0:
            violations.append("VIOLATION: No Stop Loss. Every trade MUST have a SL.")

        # === RULE 2: R:R MINIMUM ===
        # BBMR has ~1:1 R:R (TP at middle band) but 53.4% WR validated on 16yr.
        # For BBMR, require minimum 0.8 R:R (still positive expectancy at 53% WR).
        rr = payload.get("rr_ratio", 0)
        strategy = payload.get("strategy", "ML")
        min_rr = 0.8 if strategy == "BBMR" else 1.5
        if rr < min_rr:
            violations.append(f"VIOLATION: R:R ratio {rr:.2f} < {min_rr} minimum required.")

        # === RULE 3: MAX RISK PER TRADE 0.75% ===
        risk_pct = payload.get("risk_pct", 0)
        if risk_pct > 0.76:  # Small tolerance for rounding
            violations.append(f"VIOLATION: Risk {risk_pct:.2f}% > 0.75% max per trade.")

        # === RULE 4: NO WEEKEND HOLDING ===
        now = datetime.now(timezone.utc)
        # Friday 21:30 UTC ≈ 16:30 EST (close before weekend)
        if now.weekday() == 4 and now.hour >= 21 and now.minute >= 30:
            violations.append("VIOLATION: Cannot open trades after Friday 16:30 EST.")
        if now.weekday() in (5, 6):  # Saturday, Sunday
            violations.append("VIOLATION: No trading on weekends.")

        # === RULE 5: MAX 20% MARGIN PER ASSET CLASS ===
        margin = payload.get("margin_required", 0)
        balance = account_state.get("balance", 10000)
        asset_class = payload.get("asset_class", "forex")
        current_class_margin = sum(
            p.get("margin", 0) for p in account_state.get("open_positions", [])
            if p.get("asset_class") == asset_class
        )
        if balance > 0 and (current_class_margin + margin) / balance > 0.20:
            violations.append(f"VIOLATION: {asset_class} margin {(current_class_margin + margin)/balance*100:.1f}% > 20% limit.")

        # === RULE 6: NO PROHIBITED STRATEGIES ===
        # Check for martingale patterns (increasing lot after loss)
        recent_trades = account_state.get("recent_trades", [])
        if len(recent_trades) >= 2:
            last_qty = recent_trades[-1].get("quantity", 0)
            prev_qty = recent_trades[-2].get("quantity", 0)
            last_pnl = recent_trades[-1].get("pnl", 0)
            if last_pnl < 0 and payload.get("quantity", 0) > last_qty * 1.5:
                violations.append("VIOLATION: Martingale pattern detected (lot increase after loss).")

        # === RULE 7: MAX OPEN TRADES ===
        if account_state.get("open_trades", 0) >= 2:
            violations.append(f"VIOLATION: Max 2 open trades. Current: {account_state['open_trades']}.")

        # === RULE 8: MAX DAILY TRADES ===
        if account_state.get("trades_today", 0) >= 4:
            violations.append(f"VIOLATION: Max 4 trades/day. Today: {account_state['trades_today']}.")

        # === RULE 9: DAILY LOSS LIMIT ===
        daily_pnl = account_state.get("daily_pnl", 0)
        if daily_pnl < 0 and abs(daily_pnl) / 10000 * 100 >= 5.0:
            violations.append(f"VIOLATION: Daily loss {abs(daily_pnl)/10000*100:.1f}% >= 5% firm limit.")

        # === RULE 10: TOTAL DRAWDOWN LIMIT ===
        total_pnl = account_state.get("total_pnl", 0)
        if total_pnl < 0 and abs(total_pnl) / 10000 * 100 >= 10.0:
            violations.append(f"VIOLATION: Total DD {abs(total_pnl)/10000*100:.1f}% >= 10% firm limit.")

        # === RULE 11: CORRECT LEVERAGE ===
        asset_class = payload.get("asset_class", "forex")
        leverage_limits = {"forex": 100, "metals": 30, "indices": 30, "crypto": 2, "stocks": 5}
        max_lev = leverage_limits.get(asset_class, 30)
        used_lev = payload.get("leverage", 100)
        if used_lev > max_lev:
            violations.append(f"VIOLATION: Leverage {used_lev}x > {max_lev}x max for {asset_class}.")

        # === VERDICT ===
        if violations:
            return AgentMessage(
                agent_domain="compliance",
                status_flag="REJECTED",
                computational_payload=payload,
                errors=violations,
                economic_rationale="Trade REJECTED by Compliance. " + " | ".join(violations),
            )

        return AgentMessage(
            agent_domain="compliance",
            status_flag="COMPLIANT",
            computational_payload=payload,
            economic_rationale=f"All 11 FunderPro rules passed. Trade approved for {payload.get('symbol')} {payload.get('action')}.",
        )
