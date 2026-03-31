"""Position sizing methods for risk-adjusted trade sizing."""

from __future__ import annotations

import math

from loguru import logger


def fixed_fractional(
    portfolio_value: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
) -> float:
    """Fixed fractional position sizing.

    Args:
        portfolio_value: Total portfolio value.
        risk_pct: Percentage of portfolio to risk per trade (e.g., 1.0 for 1%).
        entry_price: Expected entry price.
        stop_loss: Stop loss price.

    Returns:
        Number of shares/units to buy.
    """
    risk_amount = portfolio_value * (risk_pct / 100)
    price_risk = abs(entry_price - stop_loss)

    if price_risk <= 0:
        logger.warning("Stop loss equals entry price, returning 0 quantity")
        return 0.0

    quantity = risk_amount / price_risk
    return max(0, quantity)


def kelly_criterion(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.5,
) -> float:
    """Kelly criterion for optimal position sizing.

    Args:
        win_rate: Historical win rate (0.0 to 1.0).
        avg_win: Average winning trade amount.
        avg_loss: Average losing trade amount (positive number).
        fraction: Fraction of Kelly to use (0.5 = half-Kelly, safer).

    Returns:
        Fraction of portfolio to allocate (0.0 to 1.0).
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0

    win_loss_ratio = avg_win / avg_loss
    kelly = win_rate - ((1 - win_rate) / win_loss_ratio)

    # Apply fractional Kelly (e.g., half-Kelly for safety)
    kelly *= fraction

    # Clamp to reasonable range
    return max(0.0, min(kelly, 0.25))  # Never more than 25%


def volatility_adjusted(
    portfolio_value: float,
    risk_pct: float,
    entry_price: float,
    atr: float,
    atr_multiplier: float = 2.0,
) -> float:
    """Volatility-adjusted position sizing using ATR.

    Args:
        portfolio_value: Total portfolio value.
        risk_pct: Percentage of portfolio to risk.
        entry_price: Expected entry price.
        atr: Average True Range value.
        atr_multiplier: Multiplier for ATR to set stop distance.

    Returns:
        Number of shares/units.
    """
    risk_amount = portfolio_value * (risk_pct / 100)
    stop_distance = atr * atr_multiplier

    if stop_distance <= 0:
        return 0.0

    quantity = risk_amount / stop_distance
    return max(0, quantity)
