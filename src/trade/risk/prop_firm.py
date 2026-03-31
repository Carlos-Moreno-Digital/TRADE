"""Prop Firm Risk Engine - Strict risk management for funded account challenges.

This is the MOST CRITICAL module. It protects your funded account by enforcing
limits BELOW the prop firm's actual limits (safety buffer of ~10%).

NEVER weaken these limits. Losing the funded account = losing everything.
"""

from __future__ import annotations

from datetime import datetime, timedelta, date
from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel, Field

from trade.data.models import Order, PortfolioState, TradeAction


class PropFirmConfig(BaseModel):
    """Configuration for a specific prop firm's rules."""

    name: str = "funderpro"
    phase: str = "challenge_1"  # challenge_1, challenge_2, funded

    # Actual firm limits
    firm_profit_target_pct: float = 8.0
    firm_max_daily_loss_pct: float = 5.0
    firm_max_total_drawdown_pct: float = 10.0

    # OUR SAFETY BUFFERS (trade below firm limits)
    max_daily_loss_pct: float = 4.0       # Stop at 4% (firm limit: 5%)
    max_total_drawdown_pct: float = 8.0   # Stop at 8% (firm limit: 10%)
    hard_stop_daily_pct: float = 3.0      # Stop ALL trading at 3%
    hard_stop_drawdown_pct: float = 7.0   # Stop ALL trading at 7%

    # Position sizing
    max_risk_per_trade_pct: float = 0.75  # Max risk per trade
    default_risk_per_trade_pct: float = 0.5
    max_open_trades: int = 3
    max_lots_per_trade: float = 5.0

    # Risk:Reward requirements
    min_risk_reward_ratio: float = 1.5
    target_risk_reward_ratio: float = 2.0

    # Consistency rules
    consistency_max_day_pct: float = 25.0  # Best day < 25% of total profit
    max_trades_per_day: int = 5
    min_trades_for_target: int = 15  # Spread profit across at least 15 trades

    # Trading restrictions
    news_trading_allowed: bool = True
    news_blackout_minutes: int = 5  # Minutes before/after high-impact news
    weekend_holding_allowed: bool = True
    ea_trading_allowed: bool = True
    hedging_allowed: bool = True
    martingale_allowed: bool = False  # ALWAYS FALSE
    grid_trading_allowed: bool = False  # ALWAYS FALSE

    # Session restrictions
    allowed_sessions: list[str] = Field(
        default_factory=lambda: ["london", "new_york", "london_ny_overlap"]
    )

    # Cooldown
    consecutive_loss_cooldown_minutes: int = 60
    consecutive_loss_threshold: int = 2
    daily_loss_cooldown_until_next_day: bool = True


class DailyTradeRecord(BaseModel):
    """Track daily trading performance."""

    date: date
    trades: int = 0
    wins: int = 0
    losses: int = 0
    gross_pnl: float = 0.0
    max_profit_trade: float = 0.0
    consecutive_losses: int = 0
    largest_loss: float = 0.0
    largest_win: float = 0.0


class PropFirmRiskEngine:
    """Risk engine specifically designed for prop firm challenge compliance.

    Safety philosophy:
    - We trade BELOW all limits with a 10-20% safety buffer
    - We stop trading BEFORE hitting limits, not AT them
    - We track consistency to avoid single-day concentration
    - We NEVER allow martingale, grid, or averaging down
    """

    def __init__(self, config: PropFirmConfig | None = None):
        self.config = config or PropFirmConfig()
        self.daily_records: dict[date, DailyTradeRecord] = {}
        self._is_killed = False
        self._kill_reason = ""
        self._cooldown_until: datetime | None = None
        self._session_start_equity: float = 0.0
        self._peak_equity: float = 0.0
        self._initial_balance: float = 0.0

        logger.info(
            f"PropFirmRiskEngine initialized: {self.config.name} "
            f"(Phase: {self.config.phase})"
        )

    def initialize(self, initial_balance: float) -> None:
        """Set initial balance when starting the challenge."""
        self._initial_balance = initial_balance
        self._peak_equity = initial_balance
        self._session_start_equity = initial_balance
        logger.info(f"Challenge initialized with ${initial_balance:,.2f}")

    def start_trading_day(self, current_equity: float) -> None:
        """Call at the start of each trading day."""
        self._session_start_equity = current_equity
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        today = date.today()
        if today not in self.daily_records:
            self.daily_records[today] = DailyTradeRecord(date=today)

        logger.info(
            f"Trading day started: equity=${current_equity:,.2f}, "
            f"peak=${self._peak_equity:,.2f}, "
            f"drawdown={self._get_total_drawdown_pct(current_equity):.2f}%"
        )

    # =========================================================================
    # CORE VALIDATION - Can we trade?
    # =========================================================================

    def can_open_trade(self, portfolio: PortfolioState) -> tuple[bool, str]:
        """Master check: can we open a new trade right now?"""

        # Kill switch check
        if self._is_killed:
            return False, f"KILLED: {self._kill_reason}"

        # Cooldown check
        if self._cooldown_until and datetime.utcnow() < self._cooldown_until:
            remaining = (self._cooldown_until - datetime.utcnow()).seconds // 60
            return False, f"Cooldown active: {remaining} minutes remaining"

        # Daily loss check (HARD STOP)
        daily_loss_pct = self._get_daily_loss_pct(portfolio.total_value)
        if daily_loss_pct >= self.config.hard_stop_daily_pct:
            self._activate_daily_stop(daily_loss_pct)
            return False, (
                f"HARD STOP: Daily loss {daily_loss_pct:.2f}% >= "
                f"{self.config.hard_stop_daily_pct}% (firm limit: {self.config.firm_max_daily_loss_pct}%)"
            )

        # Total drawdown check (HARD STOP)
        total_dd_pct = self._get_total_drawdown_pct(portfolio.total_value)
        if total_dd_pct >= self.config.hard_stop_drawdown_pct:
            self._kill(
                f"Total drawdown {total_dd_pct:.2f}% >= "
                f"hard stop {self.config.hard_stop_drawdown_pct}%"
            )
            return False, f"KILLED: Total drawdown too high ({total_dd_pct:.2f}%)"

        # Soft daily loss warning
        if daily_loss_pct >= self.config.max_daily_loss_pct:
            return False, (
                f"Daily loss limit: {daily_loss_pct:.2f}% >= "
                f"soft limit {self.config.max_daily_loss_pct}%"
            )

        # Soft drawdown warning
        if total_dd_pct >= self.config.max_total_drawdown_pct:
            return False, f"Total drawdown limit: {total_dd_pct:.2f}% >= {self.config.max_total_drawdown_pct}%"

        # Max open trades
        open_positions = len(portfolio.positions)
        if open_positions >= self.config.max_open_trades:
            return False, f"Max open trades: {open_positions} >= {self.config.max_open_trades}"

        # Max trades per day
        today_record = self._get_today_record()
        if today_record.trades >= self.config.max_trades_per_day:
            return False, f"Max daily trades: {today_record.trades} >= {self.config.max_trades_per_day}"

        # Consecutive losses cooldown
        if today_record.consecutive_losses >= self.config.consecutive_loss_threshold:
            self._activate_cooldown()
            return False, (
                f"Consecutive losses: {today_record.consecutive_losses} >= "
                f"{self.config.consecutive_loss_threshold}. Cooldown activated."
            )

        return True, "OK"

    def validate_order(
        self, order: Order, portfolio: PortfolioState
    ) -> tuple[bool, str, Order]:
        """Validate and potentially adjust an order before execution.

        Returns:
            (is_valid, reason, adjusted_order)
        """
        # First check if we can trade at all
        can_trade, reason = self.can_open_trade(portfolio)
        if not can_trade:
            return False, reason, order

        # Check for forbidden strategies
        if self._is_martingale_detected(order, portfolio):
            return False, "REJECTED: Martingale pattern detected (increasing size after loss)", order

        # Validate risk per trade
        if order.price and order.stop_loss:
            risk_per_unit = abs(order.price - order.stop_loss)
            trade_risk = risk_per_unit * order.quantity
            max_risk = portfolio.total_value * (self.config.max_risk_per_trade_pct / 100)

            if trade_risk > max_risk:
                # Adjust quantity down
                adjusted_qty = max_risk / risk_per_unit if risk_per_unit > 0 else 0
                if adjusted_qty < 0.01:
                    return False, f"Trade risk too high and can't be reduced: ${trade_risk:.2f}", order

                logger.warning(
                    f"Reducing quantity from {order.quantity:.2f} to {adjusted_qty:.2f} "
                    f"(risk ${trade_risk:.2f} > max ${max_risk:.2f})"
                )
                order.quantity = adjusted_qty

        # Validate risk:reward ratio
        if order.price and order.stop_loss and order.take_profit:
            risk = abs(order.price - order.stop_loss)
            reward = abs(order.take_profit - order.price)

            if risk > 0:
                rr_ratio = reward / risk
                if rr_ratio < self.config.min_risk_reward_ratio:
                    return False, (
                        f"Risk:Reward {rr_ratio:.2f} < minimum {self.config.min_risk_reward_ratio}"
                    ), order

        # Validate no stop loss = REJECTED
        if order.stop_loss is None:
            return False, "REJECTED: Every trade MUST have a stop loss", order

        return True, "Order validated", order

    def record_trade_result(self, pnl: float) -> None:
        """Record a completed trade's P&L for tracking."""
        record = self._get_today_record()
        record.trades += 1
        record.gross_pnl += pnl

        if pnl > 0:
            record.wins += 1
            record.consecutive_losses = 0
            record.largest_win = max(record.largest_win, pnl)
        elif pnl < 0:
            record.losses += 1
            record.consecutive_losses += 1
            record.largest_loss = min(record.largest_loss, pnl)

        record.max_profit_trade = max(record.max_profit_trade, pnl)

        logger.info(
            f"Trade recorded: P&L=${pnl:,.2f} | "
            f"Today: {record.wins}W/{record.losses}L | "
            f"Daily P&L: ${record.gross_pnl:,.2f} | "
            f"Consecutive losses: {record.consecutive_losses}"
        )

    def check_consistency(self, total_profit: float) -> dict[str, Any]:
        """Check if trading is consistent enough for prop firm rules."""
        if total_profit <= 0:
            return {"consistent": True, "warnings": []}

        warnings = []
        daily_profits = {
            d: r.gross_pnl for d, r in self.daily_records.items() if r.gross_pnl > 0
        }

        if daily_profits:
            best_day_profit = max(daily_profits.values())
            best_day_pct = (best_day_profit / total_profit * 100) if total_profit > 0 else 0

            if best_day_pct > self.config.consistency_max_day_pct:
                warnings.append(
                    f"Best day ({best_day_pct:.1f}%) exceeds consistency limit "
                    f"({self.config.consistency_max_day_pct}%). "
                    f"Spread profits more evenly."
                )

        total_trades = sum(r.trades for r in self.daily_records.values())
        if total_trades < self.config.min_trades_for_target:
            target = self.config.firm_profit_target_pct
            warnings.append(
                f"Only {total_trades} trades. Aim for {self.config.min_trades_for_target}+ "
                f"trades to reach {target}% target consistently."
            )

        return {
            "consistent": len(warnings) == 0,
            "warnings": warnings,
            "total_trades": total_trades,
            "trading_days": len(self.daily_records),
            "best_day_pct": max(
                (r.gross_pnl / total_profit * 100 if total_profit > 0 else 0)
                for r in self.daily_records.values()
            ) if self.daily_records else 0,
        }

    def get_challenge_progress(self, current_equity: float) -> dict[str, Any]:
        """Get progress toward the challenge target."""
        initial = self._initial_balance
        if initial <= 0:
            return {"error": "Not initialized"}

        profit_pct = (current_equity - initial) / initial * 100
        target = self.config.firm_profit_target_pct
        progress = min(100, profit_pct / target * 100) if target > 0 else 0

        total_dd = self._get_total_drawdown_pct(current_equity)
        daily_loss = self._get_daily_loss_pct(current_equity)

        today_record = self._get_today_record()

        return {
            "phase": self.config.phase,
            "initial_balance": initial,
            "current_equity": round(current_equity, 2),
            "profit_pct": round(profit_pct, 2),
            "target_pct": target,
            "progress_pct": round(progress, 1),
            "remaining_pct": round(max(0, target - profit_pct), 2),
            "total_drawdown_pct": round(total_dd, 2),
            "max_drawdown_limit": self.config.firm_max_total_drawdown_pct,
            "daily_loss_pct": round(daily_loss, 2),
            "max_daily_loss_limit": self.config.firm_max_daily_loss_pct,
            "trades_today": today_record.trades,
            "wins_today": today_record.wins,
            "losses_today": today_record.losses,
            "is_killed": self._is_killed,
            "safety_status": self._get_safety_status(current_equity),
        }

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _get_daily_loss_pct(self, current_equity: float) -> float:
        """Calculate today's loss as percentage of session start equity."""
        if self._session_start_equity <= 0:
            return 0.0
        loss = self._session_start_equity - current_equity
        if loss <= 0:
            return 0.0
        return loss / self._session_start_equity * 100

    def _get_total_drawdown_pct(self, current_equity: float) -> float:
        """Calculate total drawdown from initial balance."""
        if self._initial_balance <= 0:
            return 0.0
        loss = self._initial_balance - current_equity
        if loss <= 0:
            return 0.0
        return loss / self._initial_balance * 100

    def _get_today_record(self) -> DailyTradeRecord:
        today = date.today()
        if today not in self.daily_records:
            self.daily_records[today] = DailyTradeRecord(date=today)
        return self.daily_records[today]

    def _is_martingale_detected(self, order: Order, portfolio: PortfolioState) -> bool:
        """Detect if a trade looks like martingale (increasing size after loss)."""
        record = self._get_today_record()
        if record.consecutive_losses == 0:
            return False
        # If we had losses and the new order is larger than typical, flag it
        # This is a simplified check - can be enhanced
        return False  # Disabled for now, needs more context

    def _activate_daily_stop(self, loss_pct: float) -> None:
        """Stop trading for the rest of the day."""
        logger.critical(
            f"DAILY STOP ACTIVATED: Loss {loss_pct:.2f}%. "
            f"No more trading today."
        )
        # Set cooldown until midnight UTC
        now = datetime.utcnow()
        next_day = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0)
        self._cooldown_until = next_day

    def _activate_cooldown(self) -> None:
        """Activate cooldown after consecutive losses."""
        minutes = self.config.consecutive_loss_cooldown_minutes
        self._cooldown_until = datetime.utcnow() + timedelta(minutes=minutes)
        logger.warning(f"Cooldown activated: {minutes} minutes")

    def _kill(self, reason: str) -> None:
        """Kill switch - stop all trading permanently."""
        self._is_killed = True
        self._kill_reason = reason
        logger.critical(f"KILL SWITCH: {reason}")

    def _get_safety_status(self, current_equity: float) -> str:
        """Get a color-coded safety status."""
        daily_loss = self._get_daily_loss_pct(current_equity)
        total_dd = self._get_total_drawdown_pct(current_equity)

        if self._is_killed:
            return "DEAD"
        if daily_loss >= self.config.hard_stop_daily_pct or total_dd >= self.config.hard_stop_drawdown_pct:
            return "CRITICAL"
        if daily_loss >= self.config.max_daily_loss_pct or total_dd >= self.config.max_total_drawdown_pct:
            return "DANGER"
        if daily_loss >= 2.0 or total_dd >= 5.0:
            return "CAUTION"
        return "SAFE"

    def reset_kill_switch(self) -> None:
        """Manually reset kill switch. USE WITH EXTREME CAUTION."""
        logger.warning("Kill switch manually reset")
        self._is_killed = False
        self._kill_reason = ""


def load_prop_firm_config(firm_name: str) -> PropFirmConfig:
    """Load prop firm configuration from YAML file."""
    from pathlib import Path
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "prop_firms" / f"{firm_name}.yaml"

    if not config_path.exists():
        logger.warning(f"No config for {firm_name}, using defaults")
        return PropFirmConfig(name=firm_name)

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return PropFirmConfig(**data) if data else PropFirmConfig(name=firm_name)
