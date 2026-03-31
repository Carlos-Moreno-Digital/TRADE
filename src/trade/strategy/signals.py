"""Signal aggregation and filtering utilities."""

from __future__ import annotations

from trade.data.models import Signal, TradeAction, SignalStrength


STRENGTH_NUMERIC = {
    SignalStrength.STRONG_BUY: 3,
    SignalStrength.BUY: 2,
    SignalStrength.WEAK_BUY: 1,
    SignalStrength.NEUTRAL: 0,
    SignalStrength.WEAK_SELL: -1,
    SignalStrength.SELL: -2,
    SignalStrength.STRONG_SELL: -3,
}


def aggregate_signals(signals: list[Signal], min_confidence: float = 0.3) -> Signal | None:
    """Aggregate multiple signals into a single consensus signal.

    Args:
        signals: List of signals from different sources.
        min_confidence: Minimum confidence threshold.

    Returns:
        Aggregated signal or None if no consensus.
    """
    if not signals:
        return None

    # Filter by confidence
    valid = [s for s in signals if s.confidence >= min_confidence]
    if not valid:
        return None

    # Calculate weighted average score
    total_weight = sum(s.confidence for s in valid)
    weighted_score = sum(
        STRENGTH_NUMERIC[s.strength] * s.confidence for s in valid
    ) / total_weight

    # Map score back to action and strength
    if weighted_score > 1.5:
        action, strength = TradeAction.BUY, SignalStrength.STRONG_BUY
    elif weighted_score > 0.5:
        action, strength = TradeAction.BUY, SignalStrength.BUY
    elif weighted_score > 0.1:
        action, strength = TradeAction.BUY, SignalStrength.WEAK_BUY
    elif weighted_score < -1.5:
        action, strength = TradeAction.SELL, SignalStrength.STRONG_SELL
    elif weighted_score < -0.5:
        action, strength = TradeAction.SELL, SignalStrength.SELL
    elif weighted_score < -0.1:
        action, strength = TradeAction.SELL, SignalStrength.WEAK_SELL
    else:
        action, strength = TradeAction.HOLD, SignalStrength.NEUTRAL

    avg_confidence = total_weight / len(valid)
    reference = valid[0]

    return Signal(
        symbol=reference.symbol,
        asset_type=reference.asset_type,
        action=action,
        strength=strength,
        confidence=min(0.95, avg_confidence),
        source_agent="signal_aggregator",
        reasoning=f"Consensus from {len(valid)} signals: score={weighted_score:.2f}",
    )
