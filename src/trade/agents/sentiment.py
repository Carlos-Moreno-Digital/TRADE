"""Sentiment Agent - Analyzes news and social media sentiment using LLM."""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from trade.agents.base import BaseAgent, AgentSignal
from trade.analysis.sentiment import SentimentAnalyzer
from trade.data.models import AnalysisContext, TradeAction, SignalStrength
from trade.data.providers import MarketDataProvider


class SentimentAgent(BaseAgent):
    """Analyzes market sentiment from news and social media."""

    name = "sentiment"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        engine = self.config.get("engine", "basic")
        self.analyzer = SentimentAnalyzer(engine=engine)
        self.provider = MarketDataProvider()

    def analyze(self, context: AnalysisContext) -> AgentSignal:
        """Analyze sentiment for the given symbol."""
        start = time.time()
        self._logger.info(f"Analyzing sentiment for {context.symbol}")

        # Fetch news
        news_items = self.provider.get_news(
            context.symbol,
            limit=self.config.get("max_articles", 10),
        )

        if not news_items:
            signal = self._make_signal(
                context, TradeAction.HOLD, SignalStrength.NEUTRAL, 0.1,
                reasoning="No news articles found for sentiment analysis",
            )
            return self._make_output(signal, notes="No news data")

        # Analyze sentiment
        result = self.analyzer.analyze_news(news_items)
        sentiment = result["sentiment"]
        score = result["score"]
        confidence = result["confidence"]

        # Map sentiment to trading action
        if sentiment == "bullish" and score > 0.3:
            action = TradeAction.BUY
            strength = SignalStrength.BUY if score > 0.5 else SignalStrength.WEAK_BUY
        elif sentiment == "bearish" and score < -0.3:
            action = TradeAction.SELL
            strength = SignalStrength.SELL if score < -0.5 else SignalStrength.WEAK_SELL
        else:
            action = TradeAction.HOLD
            strength = SignalStrength.NEUTRAL

        raw_data = {
            "sentiment": sentiment,
            "score": score,
            "confidence": confidence,
            "articles_analyzed": result["analyzed"],
            "headlines": [n.title for n in news_items[:5]],
        }

        # Store in context
        context.metadata["sentiment"] = sentiment
        context.metadata["sentiment_score"] = score

        signal = self._make_signal(
            context, action, strength, confidence * 0.8,  # Slightly lower confidence for sentiment
            reasoning=f"Sentiment: {sentiment} (score: {score:.3f}), "
                     f"based on {result['analyzed']} articles",
        )

        exec_time = (time.time() - start) * 1000
        return self._make_output(signal, raw_data, exec_time=exec_time)
