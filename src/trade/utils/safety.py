"""Safety utilities: rate limiting and protection mechanisms."""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from loguru import logger


class RateLimiter:
    """Rate limiter for API calls and trading operations."""

    def __init__(self, max_calls: int = 60, period_seconds: int = 60):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self._calls: list[float] = []

    def can_proceed(self) -> bool:
        """Check if we can make another call within the rate limit."""
        now = time.time()
        cutoff = now - self.period_seconds

        # Remove old calls
        self._calls = [t for t in self._calls if t > cutoff]

        return len(self._calls) < self.max_calls

    def record_call(self) -> None:
        """Record a call."""
        self._calls.append(time.time())

    def wait_if_needed(self) -> None:
        """Block until we can proceed."""
        while not self.can_proceed():
            sleep_time = 1.0
            logger.debug(f"Rate limited, waiting {sleep_time}s")
            time.sleep(sleep_time)
        self.record_call()


class TradingSafetyGuard:
    """Global safety guard for the trading system."""

    def __init__(self):
        self._kill_switch_active = False
        self._api_limiter = RateLimiter(max_calls=60, period_seconds=60)
        self._trade_limiter = RateLimiter(max_calls=10, period_seconds=3600)
        self._error_counts: dict[str, int] = defaultdict(int)
        self._max_errors = 5

    @property
    def is_killed(self) -> bool:
        return self._kill_switch_active

    def kill(self, reason: str = "Manual kill switch") -> None:
        """Activate kill switch - stops all trading."""
        self._kill_switch_active = True
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")

    def revive(self) -> None:
        """Deactivate kill switch."""
        self._kill_switch_active = False
        logger.warning("Kill switch deactivated")

    def can_make_api_call(self) -> bool:
        """Check if API call is allowed."""
        if self._kill_switch_active:
            return False
        return self._api_limiter.can_proceed()

    def can_trade(self) -> bool:
        """Check if trading is allowed."""
        if self._kill_switch_active:
            return False
        return self._trade_limiter.can_proceed()

    def record_error(self, category: str) -> None:
        """Record an error. Auto-kills if too many errors in a category."""
        self._error_counts[category] += 1
        if self._error_counts[category] >= self._max_errors:
            self.kill(f"Too many errors in {category}: {self._error_counts[category]}")

    def reset_errors(self, category: str | None = None) -> None:
        """Reset error counts."""
        if category:
            self._error_counts[category] = 0
        else:
            self._error_counts.clear()
