"""Smart Money Concepts (SMC) / ICT methodology knowledge base.

Inner Circle Trader (ICT) concepts used by institutional traders:
- Order Blocks: Where institutions placed their orders
- Fair Value Gaps: Imbalances that price tends to fill
- Liquidity: Where stop losses cluster (targets for institutions)
- Market Structure: Break of Structure (BOS), Change of Character (CHoCH)
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MarketStructure(str, Enum):
    BULLISH = "bullish"      # Higher highs and higher lows
    BEARISH = "bearish"      # Lower highs and lower lows
    RANGING = "ranging"      # No clear direction
    BOS_BULLISH = "bos_bullish"   # Break of Structure to upside
    BOS_BEARISH = "bos_bearish"   # Break of Structure to downside
    CHOCH_BULLISH = "choch_bullish"  # Change of Character to bullish
    CHOCH_BEARISH = "choch_bearish"  # Change of Character to bearish


class ZoneType(str, Enum):
    ORDER_BLOCK = "order_block"
    FAIR_VALUE_GAP = "fair_value_gap"
    LIQUIDITY = "liquidity"
    SUPPORT = "support"
    RESISTANCE = "resistance"
    PREMIUM = "premium"      # Above 50% Fib (expensive)
    DISCOUNT = "discount"    # Below 50% Fib (cheap)


class SmartMoneyZone(BaseModel):
    """A zone identified by Smart Money analysis."""

    zone_type: ZoneType
    direction: str  # bullish or bearish
    price_high: float
    price_low: float
    strength: float = 0.5  # 0.0 to 1.0
    description: str = ""
    mitigated: bool = False  # Has price already returned to this zone?
    timestamp_index: int = -1  # Index in the DataFrame where zone was created


# ICT/SMC Trading Rules Knowledge Base
ICT_TRADING_RULES: dict[str, Any] = {
    "order_blocks": {
        "definition": (
            "An Order Block (OB) is the last bearish candle before a bullish impulse "
            "(bullish OB) or the last bullish candle before a bearish impulse (bearish OB). "
            "It represents where institutional orders were placed."
        ),
        "how_to_trade": [
            "Wait for price to return to the OB zone (pullback)",
            "Look for a reaction (rejection candle) at the OB",
            "Enter in the direction of the original impulse",
            "Stop loss: beyond the OB (wick of the OB candle)",
            "Take profit: next liquidity target or opposing OB",
        ],
        "validity": [
            "OB must have caused a Break of Structure (BOS)",
            "The impulse move from the OB must be strong (displacement)",
            "OB should create a Fair Value Gap",
            "Higher timeframe OB > Lower timeframe OB",
        ],
        "invalidation": "Price closes through the entire OB = invalidated",
    },

    "fair_value_gaps": {
        "definition": (
            "A Fair Value Gap (FVG) is a 3-candle pattern where candle 1's high is "
            "below candle 3's low (bullish FVG) or candle 1's low is above candle 3's high "
            "(bearish FVG). It represents an imbalance that price tends to fill."
        ),
        "how_to_trade": [
            "Identify FVG on higher timeframe (H4, Daily)",
            "Wait for price to retrace into the FVG",
            "Enter when price reaches the 50% level of the FVG",
            "Stop loss: beyond the FVG",
            "Take profit: next structure point",
        ],
        "key_rules": [
            "FVGs act as magnets for price",
            "Unfilled FVGs above = price likely to reach up",
            "Unfilled FVGs below = price likely to reach down",
            "Consequent encroachment: 50% of FVG is key level",
        ],
    },

    "liquidity_concepts": {
        "definition": (
            "Liquidity = where stop losses cluster. Institutions need liquidity to fill "
            "their large orders. They move price to stop loss clusters to get fills."
        ),
        "types": {
            "buy_side_liquidity": (
                "Stop losses of short sellers above recent highs. "
                "Price sweeps above highs to trigger these stops."
            ),
            "sell_side_liquidity": (
                "Stop losses of long positions below recent lows. "
                "Price sweeps below lows to trigger these stops."
            ),
            "equal_highs": "Multiple touches at same level = liquidity magnet above",
            "equal_lows": "Multiple touches at same level = liquidity magnet below",
        },
        "how_to_trade": [
            "Identify where stops are clustered (above highs, below lows)",
            "Wait for a sweep (wick through the level)",
            "After the sweep, look for reversal",
            "Enter after displacement in opposite direction",
            "Target: opposite liquidity pool",
        ],
    },

    "market_structure": {
        "bullish_structure": "Higher highs (HH) and higher lows (HL). Buy at HL.",
        "bearish_structure": "Lower highs (LH) and lower lows (LL). Sell at LH.",
        "break_of_structure": (
            "BOS = price breaks a key swing point IN the direction of the trend. "
            "Bullish BOS: price breaks above a recent high. "
            "Bearish BOS: price breaks below a recent low."
        ),
        "change_of_character": (
            "CHoCH = FIRST break against the prevailing trend. "
            "Signals potential trend reversal. "
            "Bullish CHoCH: in a downtrend, price breaks above a recent lower high. "
            "Bearish CHoCH: in an uptrend, price breaks below a recent higher low."
        ),
    },

    "premium_discount": {
        "definition": (
            "Divide the current range (swing high to swing low) using Fibonacci. "
            "Above 50% = Premium zone (expensive, look to sell). "
            "Below 50% = Discount zone (cheap, look to buy). "
            "The 50% level is the equilibrium."
        ),
        "rules": [
            "In a bullish trend: buy in discount zone (below 50%)",
            "In a bearish trend: sell in premium zone (above 50%)",
            "Optimal Trade Entry (OTE): 62%-79% Fibonacci retracement",
            "Combine with Order Block in discount/premium for best entries",
        ],
    },

    "displacement": {
        "definition": (
            "A strong, aggressive price move with large candle bodies and "
            "minimal wicks. Indicates institutional commitment to a direction. "
            "Displacement creates Fair Value Gaps."
        ),
        "significance": (
            "Displacement after a liquidity sweep confirms the move. "
            "No displacement = the sweep might be a false signal."
        ),
    },

    "optimal_trade_entry": {
        "definition": (
            "OTE is the 62% to 79% Fibonacci retracement zone within a "
            "displacement leg. This is where institutions re-enter after "
            "the initial move."
        ),
        "steps": [
            "1. Identify displacement move (strong candles, FVG created)",
            "2. Draw Fibonacci from swing low to swing high (bullish) or vice versa",
            "3. Wait for pullback to 62%-79% zone",
            "4. Look for Order Block + FVG confluence in OTE zone",
            "5. Enter with stop below the displacement origin",
            "6. Target: -27% to -62% Fibonacci extension",
        ],
    },
}


# Time and Price Theory (ICT)
TIME_PRICE_THEORY: dict[str, str] = {
    "power_of_3": (
        "Every trading session has 3 phases: "
        "1) Accumulation (Asian session: sideways, building positions), "
        "2) Manipulation (London open: fake breakout to grab liquidity), "
        "3) Distribution (Actual move in the real direction). "
        "Wait for manipulation (liquidity grab) before entering."
    ),
    "judas_swing": (
        "The initial false move at session open designed to trap traders. "
        "London often moves opposite to the day's real direction first, "
        "grabs stops, then reverses. Don't chase the first move."
    ),
    "weekly_profiles": {
        "monday": "Accumulation day. Range often set. Avoid or small positions.",
        "tuesday": "Key reversal day. Often the week's high or low is made.",
        "wednesday": "Manipulation day. Mid-week reversal common.",
        "thursday": "Distribution/continuation. Best trending day.",
        "friday": "Profit taking. Close positions before weekend. Reduced size.",
    },
    "daily_bias": (
        "Determine bullish or bearish bias BEFORE the session. "
        "Use higher timeframe structure (Daily/H4) + displacement direction. "
        "Only take trades aligned with your daily bias."
    ),
}


def get_concept_description(concept: str) -> str | None:
    """Get description of an ICT/SMC concept."""
    if concept in ICT_TRADING_RULES:
        rule = ICT_TRADING_RULES[concept]
        if isinstance(rule, dict) and "definition" in rule:
            return rule["definition"]
    if concept in TIME_PRICE_THEORY:
        val = TIME_PRICE_THEORY[concept]
        return val if isinstance(val, str) else str(val)
    return None
