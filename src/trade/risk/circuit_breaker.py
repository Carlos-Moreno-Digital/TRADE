"""Circuit breaker - Emergency stop for autonomous trading."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from loguru import logger

from trade.config import RiskConfig
from trade.data.models import PortfolioState


class CircuitBreaker:
    """Safety mechanism that halts trading when risk limits are breached.

    This is the MOST CRITICAL safety component. When triggered,
    ALL trading activity stops until manually reset or cooldown expires.
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self._is_tripped = False
        self._trip_reason = ""
        self._trip_time: datetime | None = None
        self._cooldown_until: datetime | None = None
        self._trip_history: list[dict[str, Any]] = []

    @property
    def is_active(self) -> bool:
        """Check if circuit breaker is currently blocking trades."""
        if self._cooldown_until and datetime.utcnow() < self._cooldown_until:
            return True
        if self._is_tripped:
            return True
        return False

    @property
    def status(self) -> dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "is_active": self.is_active,
            "is_tripped": self._is_tripped,
            "reason": self._trip_reason,
            "trip_time": self._trip_time.isoformat() if self._trip_time else None,
            "cooldown_until": self._cooldown_until.isoformat() if self._cooldown_until else None,
            "trip_count": len(self._trip_history),
        }

    def check(self, portfolio: PortfolioState) -> bool:
        """Check portfolio against risk limits. Returns True if trading is allowed."""
        if self._is_tripped:
            logger.warning(f"Circuit breaker ACTIVE: {self._trip_reason}")
            return False

        # Check cooldown
        if self._cooldown_until and datetime.utcnow() < self._cooldown_until:
            remaining = (self._cooldown_until - datetime.utcnow()).seconds
            logger.warning(f"Cooldown active: {remaining}s remaining")
            return False

        # Reset cooldown if expired
        if self._cooldown_until and datetime.utcnow() >= self._cooldown_until:
            self._cooldown_until = None

        # Check 1: Max drawdown
        if portfolio.max_drawdown_pct >= self.config.max_portfolio_drawdown_pct:
            self._trip(
                f"Max drawdown breached: {portfolio.max_drawdown_pct:.1f}% >= "
                f"{self.config.max_portfolio_drawdown_pct}%",
                permanent=True,
            )
            return False

        # Check 2: Daily loss
        if portfolio.total_value > 0:
            daily_loss_pct = abs(min(0, portfolio.daily_pnl) / portfolio.total_value * 100)
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                self._trip(
                    f"Daily loss limit: {daily_loss_pct:.1f}% >= {self.config.max_daily_loss_pct}%",
                    permanent=False,
                )
                # Set cooldown until next day
                now = datetime.utcnow()
                next_day = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0)
                self._cooldown_until = next_day
                return False

        # Check 3: Consecutive losses
        if portfolio.consecutive_losses >= self.config.consecutive_loss_threshold:
            hours = self.config.consecutive_loss_cooldown_hours
            self._cooldown_until = datetime.utcnow() + timedelta(hours=hours)
            logger.warning(
                f"Consecutive loss cooldown: {portfolio.consecutive_losses} losses, "
                f"cooling down for {hours}h"
            )
            return False

        # Check 4: Trades per hour
        if portfolio.trades_today >= self.config.max_trades_per_hour:
            logger.warning(f"Trade limit reached: {portfolio.trades_today} trades today")
            return False

        return True

    def _trip(self, reason: str, permanent: bool = False) -> None:
        """Trip the circuit breaker."""
        self._is_tripped = permanent
        self._trip_reason = reason
        self._trip_time = datetime.utcnow()
        self._trip_history.append({
            "reason": reason,
            "time": self._trip_time.isoformat(),
            "permanent": permanent,
        })
        logger.critical(f"CIRCUIT BREAKER TRIPPED: {reason}")

    def reset(self) -> None:
        """Manually reset the circuit breaker. Use with caution."""
        logger.warning("Circuit breaker manually reset")
        self._is_tripped = False
        self._trip_reason = ""
        self._cooldown_until = None

    def kill_switch(self) -> None:
        """Emergency kill switch - permanently stops all trading."""
        self._trip("KILL SWITCH ACTIVATED - Manual reset required", permanent=True)
