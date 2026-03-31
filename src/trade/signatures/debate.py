"""DSPy Signatures for Bull/Bear debate agents."""

from __future__ import annotations

import dspy


class BullArgument(dspy.Signature):
    """Build the STRONGEST possible case for BUYING this instrument.
    You are a conviction-driven bull analyst. Use every piece of evidence to support buying.
    Be specific with price levels, indicators, and Smart Money concepts."""

    symbol: str = dspy.InputField(desc="Trading symbol")
    technical_summary: str = dspy.InputField(
        desc="Technical analysis results: indicators, patterns, SMC zones"
    )
    sentiment_summary: str = dspy.InputField(
        desc="Sentiment analysis: news, economic events, market regime"
    )
    fundamental_summary: str = dspy.InputField(
        desc="Fundamental factors: rates, economic data, central bank stance"
    )
    market_context: str = dspy.InputField(
        desc="Current session, kill zone, correlations, recent price action"
    )

    argument: str = dspy.OutputField(
        desc="Detailed bullish argument with specific evidence"
    )
    key_catalysts: str = dspy.OutputField(
        desc="Top 3 reasons to buy NOW"
    )
    entry_price: str = dspy.OutputField(desc="Optimal long entry price/zone")
    stop_loss: str = dspy.OutputField(desc="Where to place stop loss and why")
    take_profit: str = dspy.OutputField(desc="Price targets with reasoning")
    risk_reward_ratio: str = dspy.OutputField(desc="Calculated R:R ratio")
    conviction: float = dspy.OutputField(
        desc="Bull conviction score 0.0 to 1.0 (be honest, don't inflate)"
    )


class BearArgument(dspy.Signature):
    """Build the STRONGEST possible case for SELLING this instrument.
    You are a conviction-driven bear analyst. Use every piece of evidence to support selling.
    Be specific with price levels, indicators, and Smart Money concepts."""

    symbol: str = dspy.InputField(desc="Trading symbol")
    technical_summary: str = dspy.InputField(
        desc="Technical analysis results: indicators, patterns, SMC zones"
    )
    sentiment_summary: str = dspy.InputField(
        desc="Sentiment analysis: news, economic events, market regime"
    )
    fundamental_summary: str = dspy.InputField(
        desc="Fundamental factors: rates, economic data, central bank stance"
    )
    market_context: str = dspy.InputField(
        desc="Current session, kill zone, correlations, recent price action"
    )

    argument: str = dspy.OutputField(
        desc="Detailed bearish argument with specific evidence"
    )
    key_catalysts: str = dspy.OutputField(
        desc="Top 3 reasons to sell NOW"
    )
    entry_price: str = dspy.OutputField(desc="Optimal short entry price/zone")
    stop_loss: str = dspy.OutputField(desc="Where to place stop loss and why")
    take_profit: str = dspy.OutputField(desc="Price targets with reasoning")
    risk_reward_ratio: str = dspy.OutputField(desc="Calculated R:R ratio")
    conviction: float = dspy.OutputField(
        desc="Bear conviction score 0.0 to 1.0 (be honest, don't inflate)"
    )


class JudgeVerdict(dspy.Signature):
    """You are the final decision maker. Evaluate the bull and bear cases objectively.
    Your PRIMARY concern is RISK MANAGEMENT for a prop firm challenge.
    Better to miss a trade than to lose money. Capital preservation is #1.

    Rules:
    - Only trade if one side has significantly stronger evidence
    - Require minimum 1.5:1 risk-reward ratio
    - If both sides are close, verdict is HOLD
    - Consider the prop firm drawdown limits in your decision
    - Factor in the current session and whether it's a kill zone"""

    bull_case: str = dspy.InputField(desc="Complete bullish argument")
    bull_conviction: float = dspy.InputField(desc="Bull conviction 0-1")
    bear_case: str = dspy.InputField(desc="Complete bearish argument")
    bear_conviction: float = dspy.InputField(desc="Bear conviction 0-1")
    prop_firm_status: str = dspy.InputField(
        desc="Current drawdown %, daily P&L, trades today, challenge progress"
    )
    session_info: str = dspy.InputField(
        desc="Current session, kill zone, time of day"
    )

    verdict: str = dspy.OutputField(desc="BUY, SELL, or HOLD")
    confidence: float = dspy.OutputField(desc="0.0 to 1.0")
    position_size_pct: float = dspy.OutputField(
        desc="Position size as % of account (0.25 to 0.75)"
    )
    entry_price: str = dspy.OutputField(desc="Exact entry price or zone")
    stop_loss: str = dspy.OutputField(desc="Exact stop loss price")
    take_profit: str = dspy.OutputField(desc="Take profit price(s)")
    risk_reward: str = dspy.OutputField(desc="Calculated risk:reward ratio")
    reasoning: str = dspy.OutputField(
        desc="Why this verdict, addressing both bull and bear arguments"
    )
