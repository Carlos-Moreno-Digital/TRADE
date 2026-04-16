"""FunderPro-compliant risk manager.

Hard-codes FunderPro Classic $10K rules. Every order goes through
this before reaching the connector. If ANY rule would be violated,
the order is BLOCKED and the reason is logged.

Rules enforced:
  1. Max total drawdown: 10% of initial balance ($1,000 on $10K)
  2. Max daily loss: 5% of balance at day start ($500 on $10K)
     Day resets at 22:00 UTC (5pm EST = FunderPro server time)
  3. Max concurrent positions: configurable (default 3)
  4. Max consecutive losses before cooldown: 3 → pause 30 min
  5. Weekend close: ALL positions must close by Friday 21:50 UTC
  6. Max position size: configurable per symbol
  7. Min time between trades: 30 seconds (anti-flooding)

Kill switches:
  - Daily loss >= 3% → stop trading for the day (NOT 5%, we leave
    a 2% buffer because positions can move against us AFTER the
    check)
  - Total DD >= 8% → stop ALL trading permanently (2% buffer from
    the 10% hard limit)
  - 3 consecutive losses → 30 minute cooldown
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


@dataclass
class RiskDecision:
    allowed: bool
    reason: str
    risk_used_pct: float = 0.0  # current DD as % of initial
    daily_loss_pct: float = 0.0  # today's loss as % of day-start balance


@dataclass
class RiskConfig:
    initial_balance: float = 10_000.0
    # FunderPro hard limits
    max_total_dd_pct: float = 10.0   # will KILL at 8% (2% buffer)
    max_daily_loss_pct: float = 5.0  # will KILL at 3% (2% buffer)
    # Operational limits
    kill_total_dd_pct: float = 8.0   # softer kill before hard limit
    kill_daily_loss_pct: float = 3.0 # softer kill before hard limit
    max_concurrent_positions: int = 3
    max_consecutive_losses: int = 3
    cooldown_minutes: int = 30
    min_seconds_between_trades: int = 30
    # Position sizing
    risk_per_trade_pct: float = 0.5  # 0.5% of balance per trade
    max_lots: dict[str, float] = field(default_factory=lambda: {
        "EURUSD": 1.0,    # 1 lot = 100K units
        "USDJPY": 1.0,
        "XAUUSD": 0.5,    # gold is more volatile
    })
    # Session control
    session_start_utc: int = 7    # 07:00 UTC = London open
    session_end_utc: int = 17     # 17:00 UTC = NY close
    friday_close_utc: int = 21    # Close all by Friday 21:00 UTC
    friday_close_minute: int = 50 # 21:50 UTC to be safe


class RiskManager:
    def __init__(self, config: RiskConfig | None = None):
        self.cfg = config or RiskConfig()
        self.initial_balance = self.cfg.initial_balance
        self.day_start_balance = self.cfg.initial_balance
        self.day_start_date: str = ""
        self.consecutive_losses = 0
        self.last_trade_time: datetime | None = None
        self.cooldown_until: datetime | None = None
        self.killed_daily = False
        self.killed_total = False
        self.trades_today = 0

    def _update_day(self, now: datetime, current_balance: float) -> None:
        """Reset daily counters at 22:00 UTC (FunderPro day boundary)."""
        # FunderPro day resets at 5pm EST = 22:00 UTC
        day_key = now.strftime("%Y-%m-%d")
        if now.hour >= 22:
            day_key = (now + timedelta(days=1)).strftime("%Y-%m-%d")
        if day_key != self.day_start_date:
            self.day_start_date = day_key
            self.day_start_balance = current_balance
            self.killed_daily = False
            self.trades_today = 0

    def check_can_trade(
        self,
        now: datetime,
        current_balance: float,
        current_equity: float,
        n_open_positions: int,
    ) -> RiskDecision:
        """Check ALL risk rules before allowing a new trade."""
        self._update_day(now, current_balance)

        # 1. Total DD kill switch
        total_dd_pct = (self.initial_balance - current_equity) / self.initial_balance * 100
        if total_dd_pct >= self.cfg.kill_total_dd_pct:
            self.killed_total = True
            return RiskDecision(
                False,
                f"TOTAL DD KILL: {total_dd_pct:.1f}% >= {self.cfg.kill_total_dd_pct}%",
                risk_used_pct=total_dd_pct,
            )
        if self.killed_total:
            return RiskDecision(False, "PERMANENTLY KILLED (total DD)")

        # 2. Daily loss kill switch
        daily_loss = self.day_start_balance - current_equity
        daily_loss_pct = daily_loss / self.day_start_balance * 100 if self.day_start_balance > 0 else 0
        if daily_loss_pct >= self.cfg.kill_daily_loss_pct:
            self.killed_daily = True
            return RiskDecision(
                False,
                f"DAILY KILL: {daily_loss_pct:.1f}% >= {self.cfg.kill_daily_loss_pct}%",
                daily_loss_pct=daily_loss_pct,
            )
        if self.killed_daily:
            return RiskDecision(False, "DAILY KILLED (wait for 22:00 UTC reset)")

        # 3. Cooldown after consecutive losses
        if self.cooldown_until and now < self.cooldown_until:
            remaining = (self.cooldown_until - now).total_seconds() / 60
            return RiskDecision(
                False,
                f"COOLDOWN: {remaining:.0f}min remaining after {self.cfg.max_consecutive_losses} consecutive losses",
            )

        # 4. Max concurrent positions
        if n_open_positions >= self.cfg.max_concurrent_positions:
            return RiskDecision(
                False,
                f"MAX POSITIONS: {n_open_positions}/{self.cfg.max_concurrent_positions}",
            )

        # 5. Min time between trades (anti-flood)
        if self.last_trade_time:
            elapsed = (now - self.last_trade_time).total_seconds()
            if elapsed < self.cfg.min_seconds_between_trades:
                return RiskDecision(
                    False,
                    f"ANTI-FLOOD: {elapsed:.0f}s < {self.cfg.min_seconds_between_trades}s min",
                )

        # 6. Session hours check
        if now.weekday() < 5:  # Monday-Friday
            if now.hour < self.cfg.session_start_utc or now.hour >= self.cfg.session_end_utc:
                return RiskDecision(False, f"OUTSIDE SESSION: {now.hour}:00 UTC")
        else:
            return RiskDecision(False, "WEEKEND: no trading")

        # 7. Friday pre-close (no new positions after Friday 21:00 UTC)
        if now.weekday() == 4 and now.hour >= self.cfg.friday_close_utc:
            return RiskDecision(False, "FRIDAY CLOSE: no new positions")

        return RiskDecision(
            True, "OK",
            risk_used_pct=total_dd_pct,
            daily_loss_pct=daily_loss_pct,
        )

    def should_close_for_weekend(self, now: datetime) -> bool:
        """Returns True if it's Friday after the close time."""
        return (now.weekday() == 4
                and (now.hour > self.cfg.friday_close_utc
                     or (now.hour == self.cfg.friday_close_utc
                         and now.minute >= self.cfg.friday_close_minute)))

    def record_trade_result(self, pnl: float, now: datetime) -> None:
        """Update internal state after a trade closes."""
        self.last_trade_time = now
        self.trades_today += 1
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.cfg.max_consecutive_losses:
                self.cooldown_until = now + timedelta(minutes=self.cfg.cooldown_minutes)
        else:
            self.consecutive_losses = 0
            self.cooldown_until = None

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        sl_price: float,
        current_balance: float,
    ) -> float:
        """Risk-based lot sizing. Returns lot size (1 lot = 100K units for FX)."""
        risk_amount = current_balance * self.cfg.risk_per_trade_pct / 100
        sl_distance = abs(entry_price - sl_price)
        if sl_distance <= 0:
            return 0.0
        # For FX: 1 lot = 100K units, pip value depends on pair
        # For XAUUSD: 1 lot = 100 oz
        if "XAU" in symbol:
            lot_value = 100  # 100 oz per lot
        else:
            lot_value = 100_000  # 100K units per lot
        units = risk_amount / sl_distance
        lots = units / lot_value
        # Cap at max lots
        max_lot = self.cfg.max_lots.get(symbol, 1.0)
        lots = min(lots, max_lot)
        # Floor to 0.01 lots (micro lot)
        lots = max(round(lots, 2), 0.01)
        return lots

    def status_summary(self, current_balance: float, current_equity: float) -> dict:
        """One-line status for logging."""
        total_dd = (self.initial_balance - current_equity) / self.initial_balance * 100
        daily_loss = (self.day_start_balance - current_equity) / max(self.day_start_balance, 1) * 100
        return {
            "balance": round(current_balance, 2),
            "equity": round(current_equity, 2),
            "total_dd_pct": round(total_dd, 2),
            "daily_loss_pct": round(daily_loss, 2),
            "consec_losses": self.consecutive_losses,
            "trades_today": self.trades_today,
            "killed_daily": self.killed_daily,
            "killed_total": self.killed_total,
        }
