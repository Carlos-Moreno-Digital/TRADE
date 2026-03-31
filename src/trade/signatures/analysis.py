"""DSPy Signatures for analysis agents."""

from __future__ import annotations

import dspy


class TechnicalAnalysis(dspy.Signature):
    """Analyze technical indicators and Smart Money Concepts to produce a trading signal.
    You are an expert technical analyst with deep knowledge of ICT/SMC methodology."""

    symbol: str = dspy.InputField(desc="Trading symbol (e.g., EURUSD, XAUUSD)")
    timeframe: str = dspy.InputField(desc="Chart timeframe (e.g., H1, H4, D1)")
    indicators_json: str = dspy.InputField(
        desc="JSON with RSI, MACD, Bollinger Bands, EMA, ATR values"
    )
    market_structure: str = dspy.InputField(
        desc="Current structure: trend direction, BOS/CHoCH levels, swing points"
    )
    smart_money_zones: str = dspy.InputField(
        desc="Order blocks, Fair Value Gaps, liquidity levels identified"
    )
    current_session: str = dspy.InputField(
        desc="Current market session and kill zone status"
    )

    signal: str = dspy.OutputField(desc="BUY, SELL, or HOLD")
    confidence: float = dspy.OutputField(desc="Confidence score from 0.0 to 1.0")
    entry_zone: str = dspy.OutputField(desc="Optimal entry price zone")
    stop_loss_level: str = dspy.OutputField(desc="Recommended stop loss level with reasoning")
    take_profit_level: str = dspy.OutputField(desc="Recommended take profit level")
    reasoning: str = dspy.OutputField(desc="Detailed technical reasoning")


class SentimentAnalysis(dspy.Signature):
    """Analyze financial news and market sentiment for a trading instrument.
    Consider geopolitical events, central bank policies, and market psychology."""

    symbol: str = dspy.InputField(desc="Trading symbol")
    news_headlines: str = dspy.InputField(desc="Recent news headlines and summaries")
    economic_events: str = dspy.InputField(
        desc="Upcoming economic events that could affect this pair"
    )
    market_regime: str = dspy.InputField(
        desc="Current risk-on/risk-off environment"
    )

    sentiment: str = dspy.OutputField(desc="BULLISH, BEARISH, or NEUTRAL")
    confidence: float = dspy.OutputField(desc="0.0 to 1.0")
    key_drivers: str = dspy.OutputField(desc="Main factors driving sentiment")
    risks: str = dspy.OutputField(desc="Key risks that could reverse sentiment")
    reasoning: str = dspy.OutputField(desc="Detailed sentiment analysis")


class FundamentalAnalysis(dspy.Signature):
    """Analyze fundamental factors affecting a currency pair or instrument."""

    symbol: str = dspy.InputField(desc="Trading symbol")
    fundamentals_json: str = dspy.InputField(
        desc="Economic data: interest rates, GDP, inflation, employment for relevant countries"
    )
    central_bank_stance: str = dspy.InputField(
        desc="Current monetary policy stance (hawkish/dovish) for relevant central banks"
    )

    signal: str = dspy.OutputField(desc="BUY, SELL, or HOLD")
    confidence: float = dspy.OutputField(desc="0.0 to 1.0")
    reasoning: str = dspy.OutputField(desc="Fundamental analysis reasoning")
