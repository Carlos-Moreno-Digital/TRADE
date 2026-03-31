"""Sentiment analysis for financial news and social media."""

from __future__ import annotations

from loguru import logger

from trade.data.models import NewsItem


class SentimentAnalyzer:
    """Analyzes sentiment from news articles and text.

    Supports multiple engines:
    - 'basic': Simple keyword-based sentiment (no external deps)
    - 'llm': Uses LLM (Claude/GPT) via DSPy for nuanced analysis
    - 'finbert': Uses FinBERT model (requires transformers + torch)
    """

    # Keyword lists for basic sentiment
    BULLISH_KEYWORDS = {
        "surge", "soar", "rally", "gain", "profit", "bullish", "upgrade",
        "beat", "record", "high", "growth", "strong", "positive", "boom",
        "outperform", "buy", "breakout", "momentum", "recovery", "optimistic",
    }
    BEARISH_KEYWORDS = {
        "crash", "plunge", "drop", "fall", "loss", "bearish", "downgrade",
        "miss", "low", "decline", "weak", "negative", "recession", "sell",
        "underperform", "risk", "fear", "crisis", "warning", "pessimistic",
    }

    def __init__(self, engine: str = "basic"):
        self.engine = engine
        self._finbert_pipeline = None

    def analyze_news(self, news_items: list[NewsItem]) -> dict:
        """Analyze sentiment from a list of news items."""
        if not news_items:
            return {"sentiment": "neutral", "score": 0.0, "confidence": 0.0, "analyzed": 0}

        scores = []
        for item in news_items:
            text = f"{item.title} {item.summary or ''}"
            score = self._analyze_text(text)
            scores.append(score)
            item.sentiment_score = score

        avg_score = sum(scores) / len(scores) if scores else 0.0
        confidence = min(len(scores) / 5.0, 1.0)  # More articles = higher confidence

        if avg_score > 0.2:
            sentiment = "bullish"
        elif avg_score < -0.2:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "score": round(avg_score, 4),
            "confidence": round(confidence, 4),
            "analyzed": len(scores),
            "individual_scores": scores,
        }

    def _analyze_text(self, text: str) -> float:
        """Analyze sentiment of a single text. Returns -1.0 to 1.0."""
        if self.engine == "basic":
            return self._basic_sentiment(text)
        elif self.engine == "finbert":
            return self._finbert_sentiment(text)
        elif self.engine == "llm":
            return self._llm_sentiment(text)
        else:
            logger.warning(f"Unknown sentiment engine: {self.engine}, falling back to basic")
            return self._basic_sentiment(text)

    def _basic_sentiment(self, text: str) -> float:
        """Simple keyword-based sentiment analysis."""
        words = set(text.lower().split())
        bullish = len(words & self.BULLISH_KEYWORDS)
        bearish = len(words & self.BEARISH_KEYWORDS)
        total = bullish + bearish

        if total == 0:
            return 0.0
        return (bullish - bearish) / total

    def _finbert_sentiment(self, text: str) -> float:
        """FinBERT-based sentiment analysis."""
        try:
            if self._finbert_pipeline is None:
                from transformers import pipeline
                self._finbert_pipeline = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                )

            result = self._finbert_pipeline(text[:512])[0]
            label = result["label"].lower()
            score = result["score"]

            if label == "positive":
                return score
            elif label == "negative":
                return -score
            return 0.0
        except ImportError:
            logger.warning("transformers not installed. Install with: pip install trade[sentiment]")
            return self._basic_sentiment(text)
        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            return self._basic_sentiment(text)

    def _llm_sentiment(self, text: str) -> float:
        """LLM-based sentiment analysis using DSPy."""
        try:
            import dspy

            class SentimentPredictor(dspy.Signature):
                """Analyze the financial sentiment of the given text."""
                text: str = dspy.InputField(desc="Financial news or social media text")
                sentiment_score: float = dspy.OutputField(
                    desc="Sentiment score from -1.0 (very bearish) to 1.0 (very bullish)"
                )

            predictor = dspy.Predict(SentimentPredictor)
            result = predictor(text=text[:1000])
            score = float(result.sentiment_score)
            return max(-1.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"LLM sentiment failed: {e}, falling back to basic")
            return self._basic_sentiment(text)
