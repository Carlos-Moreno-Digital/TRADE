"""Algorithmic Smart Money Concepts (SMC/ICT) detection.

Detects institutional price action patterns:
- Order Blocks (OB): Where institutions placed large orders
- Fair Value Gaps (FVG): Price imbalances that tend to get filled
- Liquidity Levels: Where stop losses cluster
- Market Structure: Break of Structure (BOS) and Change of Character (CHoCH)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


def detect_order_blocks(
    df: pd.DataFrame,
    lookback: int = 50,
    min_impulse_pct: float = 0.3,
) -> list[dict[str, Any]]:
    """Detect Order Blocks - last opposing candle before a strong impulse move.

    Bullish OB: Last bearish candle before a strong bullish move
    Bearish OB: Last bullish candle before a strong bearish move

    Args:
        df: DataFrame with open, high, low, close columns
        lookback: How many candles to look back
        min_impulse_pct: Minimum impulse move size as % of price

    Returns:
        List of detected order blocks
    """
    if len(df) < 10:
        return []

    order_blocks = []
    close = df["close"].values
    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values
    n = min(len(df), lookback)

    for i in range(3, n):
        idx = len(df) - n + i

        # Check for strong impulse move (candle i)
        body_size = abs(close[idx] - open_[idx])
        candle_range = high[idx] - low[idx]
        if candle_range == 0:
            continue

        body_ratio = body_size / candle_range
        impulse_pct = body_size / close[idx] * 100

        # Need strong candle (large body relative to range) and minimum size
        if body_ratio < 0.6 or impulse_pct < min_impulse_pct:
            continue

        # Bullish impulse: close > open significantly
        if close[idx] > open_[idx]:
            # Look for last bearish candle before this impulse
            for j in range(idx - 1, max(idx - 4, 0), -1):
                if close[j] < open_[j]:  # Bearish candle
                    order_blocks.append({
                        "type": "bullish_ob",
                        "high": float(open_[j]),  # OB zone top
                        "low": float(close[j]),   # OB zone bottom
                        "index": j,
                        "strength": round(impulse_pct, 3),
                        "mitigated": _is_mitigated(df, j, "bullish"),
                    })
                    break

        # Bearish impulse: close < open significantly
        elif close[idx] < open_[idx]:
            for j in range(idx - 1, max(idx - 4, 0), -1):
                if close[j] > open_[j]:  # Bullish candle
                    order_blocks.append({
                        "type": "bearish_ob",
                        "high": float(close[j]),  # OB zone top
                        "low": float(open_[j]),    # OB zone bottom
                        "index": j,
                        "strength": round(impulse_pct, 3),
                        "mitigated": _is_mitigated(df, j, "bearish"),
                    })
                    break

    return order_blocks


def detect_fair_value_gaps(
    df: pd.DataFrame,
    lookback: int = 50,
    min_gap_pct: float = 0.05,
) -> list[dict[str, Any]]:
    """Detect Fair Value Gaps (FVG) - 3-candle imbalance patterns.

    Bullish FVG: Candle 1 high < Candle 3 low (gap up)
    Bearish FVG: Candle 1 low > Candle 3 high (gap down)

    Args:
        df: DataFrame with high, low columns
        lookback: How many candles to analyze
        min_gap_pct: Minimum gap size as % of price

    Returns:
        List of detected FVGs
    """
    if len(df) < 5:
        return []

    fvgs = []
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = min(len(df), lookback)

    for i in range(2, n):
        idx = len(df) - n + i
        if idx < 2:
            continue

        # Bullish FVG: candle[i-2] high < candle[i] low
        if low[idx] > high[idx - 2]:
            gap_size = low[idx] - high[idx - 2]
            gap_pct = gap_size / close[idx] * 100

            if gap_pct >= min_gap_pct:
                midpoint = (low[idx] + high[idx - 2]) / 2
                fvgs.append({
                    "type": "bullish_fvg",
                    "high": float(low[idx]),       # Top of gap
                    "low": float(high[idx - 2]),   # Bottom of gap
                    "midpoint": float(midpoint),   # 50% level (key)
                    "index": idx,
                    "gap_pct": round(gap_pct, 4),
                    "filled": _is_fvg_filled(df, idx, high[idx - 2], low[idx], "bullish"),
                })

        # Bearish FVG: candle[i-2] low > candle[i] high
        if high[idx] < low[idx - 2]:
            gap_size = low[idx - 2] - high[idx]
            gap_pct = gap_size / close[idx] * 100

            if gap_pct >= min_gap_pct:
                midpoint = (low[idx - 2] + high[idx]) / 2
                fvgs.append({
                    "type": "bearish_fvg",
                    "high": float(low[idx - 2]),  # Top of gap
                    "low": float(high[idx]),      # Bottom of gap
                    "midpoint": float(midpoint),
                    "index": idx,
                    "gap_pct": round(gap_pct, 4),
                    "filled": _is_fvg_filled(df, idx, high[idx], low[idx - 2], "bearish"),
                })

    return fvgs


def detect_liquidity_levels(
    df: pd.DataFrame,
    lookback: int = 50,
    touch_threshold: float = 0.001,
) -> list[dict[str, Any]]:
    """Detect liquidity levels - equal highs/lows where stops cluster.

    Equal highs = buy-side liquidity (short stops above)
    Equal lows = sell-side liquidity (long stops below)
    """
    if len(df) < 10:
        return []

    levels = []
    high = df["high"].values
    low = df["low"].values
    n = min(len(df), lookback)

    # Find equal highs (buy-side liquidity)
    for i in range(n - 1):
        idx_i = len(df) - n + i
        for j in range(i + 1, min(i + 15, n)):
            idx_j = len(df) - n + j
            price_diff = abs(high[idx_i] - high[idx_j])
            avg_price = (high[idx_i] + high[idx_j]) / 2

            if avg_price > 0 and price_diff / avg_price < touch_threshold:
                levels.append({
                    "type": "buy_side_liquidity",
                    "level": float(avg_price),
                    "touches": 2,
                    "description": f"Equal highs at {avg_price:.5f} - stops above",
                })
                break

    # Find equal lows (sell-side liquidity)
    for i in range(n - 1):
        idx_i = len(df) - n + i
        for j in range(i + 1, min(i + 15, n)):
            idx_j = len(df) - n + j
            price_diff = abs(low[idx_i] - low[idx_j])
            avg_price = (low[idx_i] + low[idx_j]) / 2

            if avg_price > 0 and price_diff / avg_price < touch_threshold:
                levels.append({
                    "type": "sell_side_liquidity",
                    "level": float(avg_price),
                    "touches": 2,
                    "description": f"Equal lows at {avg_price:.5f} - stops below",
                })
                break

    return levels


def detect_market_structure(
    df: pd.DataFrame,
    lookback: int = 50,
) -> dict[str, Any]:
    """Detect market structure: swing highs/lows, BOS, CHoCH.

    Returns current market structure analysis.
    """
    if len(df) < 20:
        return {"structure": "unknown", "swings": [], "events": []}

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = min(len(df), lookback)

    # Find swing highs and lows (simple pivot detection)
    swings = []
    for i in range(2, n - 2):
        idx = len(df) - n + i

        # Swing high: higher than 2 candles on each side
        if high[idx] > high[idx - 1] and high[idx] > high[idx - 2] and \
           high[idx] > high[idx + 1] and high[idx] > high[idx + 2]:
            swings.append({"type": "high", "price": float(high[idx]), "index": idx})

        # Swing low: lower than 2 candles on each side
        if low[idx] < low[idx - 1] and low[idx] < low[idx - 2] and \
           low[idx] < low[idx + 1] and low[idx] < low[idx + 2]:
            swings.append({"type": "low", "price": float(low[idx]), "index": idx})

    if len(swings) < 4:
        return {"structure": "unknown", "swings": swings, "events": []}

    # Determine structure from last 4 swings
    recent = swings[-4:]
    highs = [s["price"] for s in recent if s["type"] == "high"]
    lows = [s["price"] for s in recent if s["type"] == "low"]

    structure = "ranging"
    if len(highs) >= 2 and len(lows) >= 2:
        hh = highs[-1] > highs[-2] if len(highs) >= 2 else False
        hl = lows[-1] > lows[-2] if len(lows) >= 2 else False
        lh = highs[-1] < highs[-2] if len(highs) >= 2 else False
        ll = lows[-1] < lows[-2] if len(lows) >= 2 else False

        if hh and hl:
            structure = "bullish"  # Higher highs + higher lows
        elif lh and ll:
            structure = "bearish"  # Lower highs + lower lows

    # Detect BOS/CHoCH
    events = []
    current_price = float(close[-1])

    if structure == "bullish" and lows:
        last_hl = lows[-1]
        if current_price < last_hl:
            events.append({
                "type": "choch_bearish",
                "description": f"Price {current_price:.5f} broke below higher low {last_hl:.5f}",
                "significance": "high",
            })

    elif structure == "bearish" and highs:
        last_lh = highs[-1]
        if current_price > last_lh:
            events.append({
                "type": "choch_bullish",
                "description": f"Price {current_price:.5f} broke above lower high {last_lh:.5f}",
                "significance": "high",
            })

    return {
        "structure": structure,
        "swings": swings[-6:],  # Last 6 swings
        "events": events,
        "swing_highs": highs,
        "swing_lows": lows,
    }


def get_smc_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Run complete Smart Money analysis on a DataFrame."""
    if df is None or df.empty or len(df) < 10:
        return {"available": False}

    try:
        order_blocks = detect_order_blocks(df)
        fvgs = detect_fair_value_gaps(df)
        liquidity = detect_liquidity_levels(df)
        structure = detect_market_structure(df)

        # Filter to unmitigated/unfilled zones (active zones)
        active_obs = [ob for ob in order_blocks if not ob.get("mitigated", True)]
        active_fvgs = [fvg for fvg in fvgs if not fvg.get("filled", True)]

        return {
            "available": True,
            "market_structure": structure["structure"],
            "structure_events": structure["events"],
            "order_blocks": active_obs[-5:],  # Last 5 active OBs
            "fair_value_gaps": active_fvgs[-5:],  # Last 5 unfilled FVGs
            "liquidity_levels": liquidity[-5:],
            "total_obs": len(order_blocks),
            "active_obs": len(active_obs),
            "total_fvgs": len(fvgs),
            "active_fvgs": len(active_fvgs),
        }
    except Exception as e:
        logger.warning(f"SMC analysis failed: {e}")
        return {"available": False, "error": str(e)}


# =========================================================================
# Helper functions
# =========================================================================

def _is_mitigated(df: pd.DataFrame, ob_index: int, ob_type: str) -> bool:
    """Check if an order block has been mitigated (price returned to it)."""
    if ob_index >= len(df) - 1:
        return False

    close = df["close"].values
    open_ = df["open"].values

    ob_low = min(close[ob_index], open_[ob_index])
    ob_high = max(close[ob_index], open_[ob_index])

    # Check if price has returned to the OB zone after it was created
    for i in range(ob_index + 2, len(df)):
        low_i = df["low"].values[i]
        high_i = df["high"].values[i]

        if ob_type == "bullish" and low_i <= ob_high:
            return True
        if ob_type == "bearish" and high_i >= ob_low:
            return True

    return False


def _is_fvg_filled(
    df: pd.DataFrame, fvg_index: int, gap_low: float, gap_high: float, fvg_type: str
) -> bool:
    """Check if a FVG has been filled (price returned through it)."""
    for i in range(fvg_index + 1, len(df)):
        if fvg_type == "bullish" and df["low"].values[i] <= gap_low:
            return True
        if fvg_type == "bearish" and df["high"].values[i] >= gap_high:
            return True
    return False
