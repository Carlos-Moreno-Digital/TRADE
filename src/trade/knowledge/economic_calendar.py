"""Economic calendar events and their market impact.

High-impact events cause the biggest moves and are the most dangerous
for prop firm traders (can blow through stop losses with slippage).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventImpact(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EventType(BaseModel):
    """Definition of an economic event type."""

    name: str
    impact: EventImpact
    description: str
    affected_currencies: list[str]
    affected_assets: list[str] = Field(default_factory=list)
    avg_pip_move: dict[str, int] = Field(default_factory=dict)  # pair -> pips
    typical_duration_minutes: int = 30
    trading_advice: str = ""


# High-impact events ranked by market-moving potential
HIGH_IMPACT_EVENTS: list[EventType] = [
    EventType(
        name="Non-Farm Payrolls (NFP)",
        impact=EventImpact.HIGH,
        description="US employment report. THE most market-moving event monthly. "
                    "Released first Friday of each month at 13:30 UTC.",
        affected_currencies=["USD"],
        affected_assets=["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US30", "NAS100"],
        avg_pip_move={"EURUSD": 80, "GBPUSD": 100, "XAUUSD": 300},
        typical_duration_minutes=60,
        trading_advice="AVOID trading 5 min before/after. Or trade the post-NFP trend "
                      "15+ minutes after release once direction is clear.",
    ),
    EventType(
        name="FOMC Interest Rate Decision",
        impact=EventImpact.HIGH,
        description="Federal Reserve rate decision + statement. Moves ALL markets. "
                    "Released 8 times per year at 19:00 UTC. Press conference at 19:30.",
        affected_currencies=["USD"],
        affected_assets=["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US30", "NAS100", "BTCUSD"],
        avg_pip_move={"EURUSD": 100, "XAUUSD": 400, "US30": 500},
        typical_duration_minutes=120,
        trading_advice="CLOSE all positions before. The whipsaw can destroy accounts. "
                      "Trade the trend AFTER the press conference.",
    ),
    EventType(
        name="CPI (Consumer Price Index)",
        impact=EventImpact.HIGH,
        description="Inflation data. Affects rate expectations. Released monthly ~13:30 UTC. "
                    "Higher than expected = USD bullish (rates stay high). Lower = USD bearish.",
        affected_currencies=["USD"],
        affected_assets=["EURUSD", "GBPUSD", "XAUUSD", "US30", "NAS100"],
        avg_pip_move={"EURUSD": 60, "XAUUSD": 250},
        typical_duration_minutes=45,
        trading_advice="Second most dangerous event. Close or reduce before. "
                      "Trade the reaction after initial move settles (15+ min).",
    ),
    EventType(
        name="ECB Interest Rate Decision",
        impact=EventImpact.HIGH,
        description="European Central Bank rate decision. Major EUR mover. "
                    "Press conference 30 minutes after decision.",
        affected_currencies=["EUR"],
        affected_assets=["EURUSD", "EURGBP", "EURJPY", "DAX"],
        avg_pip_move={"EURUSD": 80, "EURGBP": 50},
        typical_duration_minutes=90,
        trading_advice="Affects all EUR pairs. Wait for press conference clarity.",
    ),
    EventType(
        name="BOE Interest Rate Decision",
        impact=EventImpact.HIGH,
        description="Bank of England rate decision. GBP mover.",
        affected_currencies=["GBP"],
        affected_assets=["GBPUSD", "EURGBP", "GBPJPY"],
        avg_pip_move={"GBPUSD": 80},
        typical_duration_minutes=60,
        trading_advice="Caution with GBP pairs. BOE often surprises.",
    ),
    EventType(
        name="BOJ Interest Rate Decision",
        impact=EventImpact.HIGH,
        description="Bank of Japan rate decision. Major JPY and carry trade impact.",
        affected_currencies=["JPY"],
        affected_assets=["USDJPY", "EURJPY", "GBPJPY"],
        avg_pip_move={"USDJPY": 100},
        typical_duration_minutes=60,
        trading_advice="BOJ surprises cause extreme JPY moves. Reduce JPY exposure.",
    ),
]

MEDIUM_IMPACT_EVENTS: list[EventType] = [
    EventType(
        name="GDP (Gross Domestic Product)",
        impact=EventImpact.MEDIUM,
        description="Quarterly economic growth. Preliminary release most impactful.",
        affected_currencies=["USD", "EUR", "GBP"],
        avg_pip_move={"EURUSD": 40},
        trading_advice="Trade with caution. Usually confirms existing trend.",
    ),
    EventType(
        name="PMI (Purchasing Managers Index)",
        impact=EventImpact.MEDIUM,
        description="Leading economic indicator. Above 50 = expansion, below 50 = contraction.",
        affected_currencies=["USD", "EUR", "GBP"],
        avg_pip_move={"EURUSD": 30},
        trading_advice="Flash PMI more impactful than final. Trade breakout.",
    ),
    EventType(
        name="Retail Sales",
        impact=EventImpact.MEDIUM,
        description="Consumer spending data. Indicates economic health.",
        affected_currencies=["USD", "EUR", "GBP"],
        avg_pip_move={"EURUSD": 35},
        trading_advice="Core retail sales (ex-auto) more important than headline.",
    ),
    EventType(
        name="Employment/Claims Data",
        impact=EventImpact.MEDIUM,
        description="Weekly jobless claims and employment change data.",
        affected_currencies=["USD"],
        avg_pip_move={"EURUSD": 20},
        trading_advice="Usually lower impact unless extreme deviation from forecast.",
    ),
]


# Interpretation rules
HAWKISH_DOVISH = {
    "hawkish": {
        "meaning": "Central bank favors higher rates to fight inflation",
        "currency_impact": "BULLISH for the currency (higher rates attract capital)",
        "stock_impact": "BEARISH for stocks (higher borrowing costs)",
        "gold_impact": "BEARISH for gold (opportunity cost of holding non-yielding asset)",
        "signals": [
            "raising rates", "inflation concerns", "strong labor market",
            "reducing QE", "tapering", "tightening",
        ],
    },
    "dovish": {
        "meaning": "Central bank favors lower rates to stimulate economy",
        "currency_impact": "BEARISH for the currency (lower rates reduce capital inflow)",
        "stock_impact": "BULLISH for stocks (cheaper borrowing)",
        "gold_impact": "BULLISH for gold (lower opportunity cost, inflation hedge)",
        "signals": [
            "cutting rates", "growth concerns", "easing",
            "QE", "stimulus", "accommodation",
        ],
    },
}


def get_event_by_name(name: str) -> EventType | None:
    """Find an event type by name (case-insensitive partial match)."""
    name_lower = name.lower()
    for event in HIGH_IMPACT_EVENTS + MEDIUM_IMPACT_EVENTS:
        if name_lower in event.name.lower():
            return event
    return None


def get_affected_pairs(currency: str) -> list[str]:
    """Get all pairs affected by a currency's events."""
    pairs = set()
    for event in HIGH_IMPACT_EVENTS + MEDIUM_IMPACT_EVENTS:
        if currency in event.affected_currencies:
            pairs.update(event.affected_assets)
    return sorted(pairs)


def should_avoid_trading(event_name: str, minutes_until: int, blackout_minutes: int = 5) -> bool:
    """Check if we should avoid trading due to upcoming event."""
    event = get_event_by_name(event_name)
    if event is None:
        return False
    if event.impact == EventImpact.HIGH and abs(minutes_until) <= blackout_minutes:
        return True
    return False
