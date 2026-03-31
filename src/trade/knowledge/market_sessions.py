"""Market sessions, kill zones, and optimal trading times.

The forex market operates 24/5 but volatility and opportunity vary dramatically
by session. Kill zones are the high-probability windows within each session.
"""

from __future__ import annotations

from datetime import datetime, time, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel


class Session(str, Enum):
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    LONDON_NY_OVERLAP = "london_ny_overlap"
    LONDON_CLOSE = "london_close"


class SessionInfo(BaseModel):
    name: Session
    start: time  # UTC
    end: time    # UTC
    volatility: str  # low, medium, high, highest
    best_pairs: list[str]
    description: str


# All times in UTC
SESSIONS: dict[Session, SessionInfo] = {
    Session.ASIAN: SessionInfo(
        name=Session.ASIAN,
        start=time(0, 0),
        end=time(9, 0),
        volatility="low",
        best_pairs=["USDJPY", "AUDUSD", "NZDUSD", "EURJPY", "AUDJPY"],
        description="Tokyo/Sydney session. Range-bound, lower spreads on JPY/AUD pairs.",
    ),
    Session.LONDON: SessionInfo(
        name=Session.LONDON,
        start=time(7, 0),
        end=time(16, 0),
        volatility="high",
        best_pairs=["EURUSD", "GBPUSD", "EURGBP", "GBPJPY", "XAUUSD"],
        description="Highest volume session. Major moves start here. EUR and GBP pairs dominate.",
    ),
    Session.NEW_YORK: SessionInfo(
        name=Session.NEW_YORK,
        start=time(12, 0),
        end=time(21, 0),
        volatility="high",
        best_pairs=["EURUSD", "GBPUSD", "USDCAD", "US30", "NAS100", "XAUUSD"],
        description="US data releases. Strong USD moves. Best for indices.",
    ),
    Session.LONDON_NY_OVERLAP: SessionInfo(
        name=Session.LONDON_NY_OVERLAP,
        start=time(12, 0),
        end=time(16, 0),
        volatility="highest",
        best_pairs=["EURUSD", "GBPUSD", "XAUUSD", "US30", "NAS100"],
        description="PEAK liquidity and volatility. Best window for trading. Most volume globally.",
    ),
    Session.LONDON_CLOSE: SessionInfo(
        name=Session.LONDON_CLOSE,
        start=time(15, 0),
        end=time(16, 0),
        volatility="medium",
        best_pairs=["EURUSD", "GBPUSD"],
        description="Institutional position squaring. Can see reversals.",
    ),
}


class KillZone(BaseModel):
    """High-probability trading window within a session."""

    name: str
    session: Session
    start: time  # UTC
    end: time    # UTC
    description: str
    probability_boost: float  # Multiplier for confidence (1.0 = no boost)


KILL_ZONES: list[KillZone] = [
    KillZone(
        name="Asian Kill Zone",
        session=Session.ASIAN,
        start=time(0, 0),
        end=time(4, 0),
        description="ICT Asian KZ: Accumulation phase. Look for range setups. "
                    "Institutional positioning before London.",
        probability_boost=1.1,
    ),
    KillZone(
        name="London Open Kill Zone",
        session=Session.LONDON,
        start=time(7, 0),
        end=time(10, 0),
        description="ICT London KZ: HIGHEST probability. Institutions break Asian ranges. "
                    "Look for liquidity sweeps of Asian highs/lows followed by displacement.",
        probability_boost=1.3,
    ),
    KillZone(
        name="New York Open Kill Zone",
        session=Session.NEW_YORK,
        start=time(12, 0),
        end=time(15, 0),
        description="ICT NY KZ: Second-best probability. US economic data releases. "
                    "Look for continuation of London move or reversal.",
        probability_boost=1.25,
    ),
    KillZone(
        name="London Close Kill Zone",
        session=Session.LONDON_CLOSE,
        start=time(15, 0),
        end=time(16, 0),
        description="ICT London Close KZ: Position squaring. Counter-trend setups. "
                    "Lower probability but good for quick scalps.",
        probability_boost=1.05,
    ),
]


def get_current_session(utc_now: datetime | None = None) -> SessionInfo | None:
    """Get the current active session based on UTC time."""
    if utc_now is None:
        utc_now = datetime.now(timezone.utc)

    current_time = utc_now.time()

    # Check weekend
    if utc_now.weekday() >= 5:  # Saturday=5, Sunday=6
        return None

    for session_info in SESSIONS.values():
        start = session_info.start
        end = session_info.end
        if start <= end:
            if start <= current_time <= end:
                return session_info
        else:  # Wraps midnight
            if current_time >= start or current_time <= end:
                return session_info

    return None


def get_active_kill_zone(utc_now: datetime | None = None) -> KillZone | None:
    """Get the current active kill zone, if any."""
    if utc_now is None:
        utc_now = datetime.now(timezone.utc)

    current_time = utc_now.time()

    if utc_now.weekday() >= 5:
        return None

    for kz in KILL_ZONES:
        if kz.start <= current_time <= kz.end:
            return kz

    return None


def is_optimal_trading_time(utc_now: datetime | None = None) -> tuple[bool, str]:
    """Check if now is a good time to trade."""
    if utc_now is None:
        utc_now = datetime.now(timezone.utc)

    if utc_now.weekday() >= 5:
        return False, "Weekend - markets closed"

    kz = get_active_kill_zone(utc_now)
    if kz:
        return True, f"Active kill zone: {kz.name} ({kz.description})"

    session = get_current_session(utc_now)
    if session and session.volatility in ("high", "highest"):
        return True, f"Active high-volatility session: {session.name.value}"

    if session:
        return False, f"Low probability: {session.name.value} session (volatility: {session.volatility})"

    return False, "No active session"
