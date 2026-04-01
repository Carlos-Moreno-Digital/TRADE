"""Risk Shield Agent — Asymmetric Risk Control.

Refines raw signals with defensive risk management layers:
- Position sizing (0.5-1% max per trade)
- Chandelier Exit (ATR-based trailing SL)
- Kill-switch at 3% daily loss
- Margin exposure validation (20% max per asset class)
"""

from __future__ import annotations

import math
from datetime import datetime

from trade.agents.multi import AgentMessage

# FunderPro Classic $10K limits
ACCOUNT_SIZE = 10000.0
MAX_RISK_PER_TRADE_PCT = 0.75  # 0.75% = $75
MIN_RISK_PER_TRADE_PCT = 0.50  # 0.50% = $50
KILL_SWITCH_DAILY_PCT = 3.0    # Hard stop at 3% (firm: 5%)
KILL_SWITCH_TOTAL_PCT = 7.0    # Hard stop at 7% (firm: 10%)
SL_ATR_MULT = 1.5              # SL at 1.5x ATR
TP_ATR_MULT = 2.5              # TP at 2.5x ATR → R:R = 1.67
MAX_MARGIN_PER_CLASS = 0.20    # 20% max margin on single asset class
MAX_OPEN_TRADES = 2
MAX_TRADES_PER_DAY = 4
CONSEC_LOSS_COOLDOWN_SEC = 3600  # 60 min cooldown after 2 losses

# Leverage per asset class
LEVERAGE = {"forex": 100, "metals": 30, "indices": 30, "crypto": 2}


class RiskShield:
    """Refines signals with prop firm risk management."""

    def validate_and_size(self, signal: AgentMessage, account_state: dict) -> AgentMessage:
        """Apply risk controls to a raw signal from Alpha Generator.

        account_state: {
            "balance": float, "daily_pnl": float, "total_pnl": float,
            "open_trades": int, "trades_today": int, "consec_losses": bool,
            "open_positions": [{"symbol": str, "asset_class": str, "margin": float}],
            "last_loss_time": str | None,
        }
        """
        payload = signal.computational_payload
        errors = []

        balance = account_state.get("balance", ACCOUNT_SIZE)
        daily_pnl = account_state.get("daily_pnl", 0)
        total_pnl = account_state.get("total_pnl", 0)

        # === KILL SWITCH: Daily loss ===
        if daily_pnl < 0 and abs(daily_pnl) / ACCOUNT_SIZE * 100 >= KILL_SWITCH_DAILY_PCT:
            return AgentMessage(
                agent_domain="risk_shield",
                status_flag="BLOCKED",
                errors=[f"Daily loss kill-switch: {abs(daily_pnl)/ACCOUNT_SIZE*100:.1f}% >= {KILL_SWITCH_DAILY_PCT}%"],
                economic_rationale="Trading halted to protect daily loss limit",
            )

        # === KILL SWITCH: Total drawdown ===
        if total_pnl < 0 and abs(total_pnl) / ACCOUNT_SIZE * 100 >= KILL_SWITCH_TOTAL_PCT:
            return AgentMessage(
                agent_domain="risk_shield",
                status_flag="BLOCKED",
                errors=[f"Total DD kill-switch: {abs(total_pnl)/ACCOUNT_SIZE*100:.1f}% >= {KILL_SWITCH_TOTAL_PCT}%"],
                economic_rationale="Trading halted to protect total drawdown limit",
            )

        # === CONSECUTIVE LOSS COOLDOWN ===
        if account_state.get("consec_losses"):
            last_loss = account_state.get("last_loss_time")
            if last_loss:
                try:
                    elapsed = (datetime.utcnow() - datetime.fromisoformat(last_loss)).total_seconds()
                    if elapsed < CONSEC_LOSS_COOLDOWN_SEC:
                        remaining = int((CONSEC_LOSS_COOLDOWN_SEC - elapsed) / 60)
                        return AgentMessage(
                            agent_domain="risk_shield",
                            status_flag="BLOCKED",
                            errors=[f"Cooldown: {remaining}min remaining after 2 consecutive losses"],
                            economic_rationale="60-minute cooldown period active",
                        )
                except Exception:
                    pass

        # === MAX OPEN TRADES ===
        if account_state.get("open_trades", 0) >= MAX_OPEN_TRADES:
            return AgentMessage(
                agent_domain="risk_shield",
                status_flag="BLOCKED",
                errors=[f"Max open trades reached: {account_state['open_trades']}/{MAX_OPEN_TRADES}"],
            )

        # === MAX TRADES PER DAY ===
        if account_state.get("trades_today", 0) >= MAX_TRADES_PER_DAY:
            return AgentMessage(
                agent_domain="risk_shield",
                status_flag="BLOCKED",
                errors=[f"Max daily trades reached: {account_state['trades_today']}/{MAX_TRADES_PER_DAY}"],
            )

        # === POSITION SIZING ===
        price = payload.get("entry_price", 0)
        atr = payload.get("atr", 0)
        confidence = payload.get("confidence", 0.5)

        if price <= 0 or atr <= 0:
            return AgentMessage(
                agent_domain="risk_shield",
                status_flag="REJECTED",
                errors=["Invalid price or ATR"],
            )

        # SL/TP calculation
        sl_dist = atr * SL_ATR_MULT
        tp_dist = atr * TP_ATR_MULT

        if payload["action"] == "BUY":
            sl_price = price - sl_dist
            tp_price = price + tp_dist
        else:
            sl_price = price + sl_dist
            tp_price = price - tp_dist

        # R:R check (minimum 1.5:1 per FunderPro)
        rr = tp_dist / sl_dist if sl_dist > 0 else 0
        if rr < 1.5:
            return AgentMessage(
                agent_domain="risk_shield",
                status_flag="REJECTED",
                errors=[f"R:R ratio {rr:.2f} < 1.5 minimum"],
            )

        # Risk amount: scale between 0.5% and 0.75% based on confidence
        risk_pct = MIN_RISK_PER_TRADE_PCT + (confidence - 0.53) * (MAX_RISK_PER_TRADE_PCT - MIN_RISK_PER_TRADE_PCT) / 0.47
        risk_pct = max(MIN_RISK_PER_TRADE_PCT, min(MAX_RISK_PER_TRADE_PCT, risk_pct))
        risk_amt = balance * risk_pct / 100

        # Lot sizing
        qty = risk_amt / sl_dist if sl_dist > 0 else 0

        # Leverage cap
        asset_class = _detect_asset_class(payload["symbol"])
        max_leverage = LEVERAGE.get(asset_class, 30)
        max_qty = balance * max_leverage / price if price > 0 else 0
        qty = min(qty, max_qty)

        if qty <= 0:
            return AgentMessage(
                agent_domain="risk_shield",
                status_flag="REJECTED",
                errors=["Calculated quantity is zero"],
            )

        # === MARGIN EXPOSURE CHECK (20% rule) ===
        margin_required = price * qty / max_leverage
        current_class_margin = sum(
            p.get("margin", 0) for p in account_state.get("open_positions", [])
            if p.get("asset_class") == asset_class
        )
        total_margin_limit = balance * MAX_MARGIN_PER_CLASS
        if current_class_margin + margin_required > total_margin_limit:
            return AgentMessage(
                agent_domain="risk_shield",
                status_flag="BLOCKED",
                errors=[f"20% margin limit: {asset_class} exposure ${current_class_margin + margin_required:.0f} > ${total_margin_limit:.0f}"],
            )

        # === BUILD VALIDATED SIGNAL ===
        spread = _get_spread(payload["symbol"])
        cost = spread * qty * 2  # Round trip

        return AgentMessage(
            agent_domain="risk_shield",
            status_flag="RISK_VALIDATED",
            computational_payload={
                **payload,
                "sl_price": round(sl_price, 5),
                "tp_price": round(tp_price, 5),
                "sl_distance": round(sl_dist, 5),
                "tp_distance": round(tp_dist, 5),
                "rr_ratio": round(rr, 2),
                "quantity": round(qty, 4),
                "risk_pct": round(risk_pct, 3),
                "risk_amount": round(risk_amt, 2),
                "margin_required": round(margin_required, 2),
                "spread_cost": round(cost, 2),
                "asset_class": asset_class,
                "leverage": max_leverage,
            },
            economic_rationale=f"Risk validated: {risk_pct:.2f}% risk, R:R={rr:.2f}, SL={sl_price:.4f}, TP={tp_price:.4f}",
        )


def _detect_asset_class(symbol: str) -> str:
    if symbol.endswith("=X"):
        return "forex"
    elif symbol in ("GC=F", "SI=F"):
        return "metals"
    elif symbol.endswith("-USD"):
        return "crypto"
    elif symbol.startswith("^"):
        return "indices"
    return "forex"


def _get_spread(symbol: str) -> float:
    spreads = {
        "GBPNZD=X": 0.00030, "GC=F": 0.40, "AUDNZD=X": 0.00020,
        "GBPCHF=X": 0.00020, "USDCAD=X": 0.00012, "USDCHF=X": 0.00010,
        "EURJPY=X": 0.012, "EURUSD=X": 0.00008, "GBPUSD=X": 0.00010,
        "USDJPY=X": 0.008,
    }
    return spreads.get(symbol, 0.0002)
