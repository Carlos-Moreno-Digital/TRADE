"""Sentiment Agent — NLP-based news and social media scanner.

Emulates the "AlphaSense / PitchBook" terminal by scraping free
sources for forex-relevant sentiment. Returns a scalar sentiment
score in [-1, +1] that can be used as a meta-feature for any
strategy's secondary model.

Sources (all free, no API key required by default):
  - Forex Factory RSS (economic calendar + impact ratings)
  - Investing.com RSS (market news headlines)

The agent performs:
  1. Fetch latest headlines from configured RSS feeds
  2. Simple keyword-based sentiment scoring (no transformer needed;
     when the user installs transformers on the VPS, a FinBERT path
     is available but NOT required)
  3. Returns a SentimentSnapshot dataclass with the aggregate score,
     headline count, and the most impactful headline text

Causality: the agent ONLY reads already-published headlines. There
is no look-ahead; the snapshot timestamp is the wall-clock time of
the fetch, not the publication time of the newest headline.

Integration point: the meta-labelling pipeline calls
  agent.get_current_sentiment() and includes the score as an
  additional meta-feature alongside ATR, RSI, etc.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime

import requests

# Keyword lexicons (finance-specific, not general sentiment).
# Weighted by impact: high-impact words get ±2, medium ±1, low ±0.5.
_BULLISH = {
    # High impact
    "rate cut": 2, "dovish": 2, "stimulus": 2, "beat expectations": 2,
    "strong gdp": 2, "record high": 1.5, "hawkish boj": 1.5,
    # Medium
    "growth": 1, "employment gains": 1, "surplus": 1, "upgrade": 1,
    "recovery": 1, "optimism": 1, "rally": 1,
    # Low
    "stable": 0.5, "steady": 0.5,
}
_BEARISH = {
    # High impact
    "rate hike": 2, "hawkish": 2, "recession": 2, "miss expectations": 2,
    "weak gdp": 2, "crisis": 2, "default": 2, "sanctions": 1.5,
    # Medium
    "contraction": 1, "unemployment": 1, "deficit": 1, "downgrade": 1,
    "sell-off": 1, "fear": 1, "collapse": 1, "war": 1.5,
    # Low
    "volatile": 0.5, "uncertainty": 0.5, "slowdown": 0.5,
}

# Free RSS feeds (no auth required)
DEFAULT_FEEDS = [
    "https://www.forexfactory.com/rss",
    "https://www.investing.com/rss/news_14.rss",  # forex news
]


@dataclass
class SentimentSnapshot:
    score: float              # [-1, +1] aggregate
    n_headlines: int
    top_headline: str = ""
    fetch_time: str = ""
    source: str = "keyword"   # 'keyword' or 'finbert'
    raw_scores: list[float] = field(default_factory=list)


def _fetch_rss(url: str, timeout: int = 10) -> list[str]:
    """Fetch headlines from an RSS feed. Returns list of title strings."""
    try:
        r = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 TRADE-Bot/1.0"
        })
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
        titles = []
        for item in root.iter("item"):
            title = item.find("title")
            if title is not None and title.text:
                titles.append(title.text.strip())
        return titles
    except Exception:
        return []


def _keyword_score(headline: str) -> float:
    """Score a single headline using the keyword lexicon. Returns [-1,+1]."""
    h = headline.lower()
    bull = sum(w for kw, w in _BULLISH.items() if kw in h)
    bear = sum(w for kw, w in _BEARISH.items() if kw in h)
    total = bull + bear
    if total == 0:
        return 0.0
    return max(-1.0, min(1.0, (bull - bear) / total))


class SentimentAgent:
    """Stateless sentiment scanner. Call get_current_sentiment() to
    fetch, score, and aggregate the latest headlines.
    """

    def __init__(
        self,
        feeds: list[str] | None = None,
        use_finbert: bool = False,
    ):
        self.feeds = feeds or DEFAULT_FEEDS
        self.use_finbert = use_finbert
        self._finbert_pipeline = None

    def _score_headlines(self, headlines: list[str]) -> list[float]:
        if self.use_finbert:
            return self._finbert_score(headlines)
        return [_keyword_score(h) for h in headlines]

    def _finbert_score(self, headlines: list[str]) -> list[float]:
        """Optional FinBERT path. Only used when `transformers` is
        installed and use_finbert=True. NOT required for production.
        """
        if self._finbert_pipeline is None:
            try:
                from transformers import pipeline
                self._finbert_pipeline = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    truncation=True,
                    max_length=128,
                )
            except ImportError:
                return [_keyword_score(h) for h in headlines]
        scores = []
        for h in headlines:
            try:
                result = self._finbert_pipeline(h[:512])[0]
                label = result["label"].lower()
                conf = float(result["score"])
                if label == "positive":
                    scores.append(conf)
                elif label == "negative":
                    scores.append(-conf)
                else:
                    scores.append(0.0)
            except Exception:
                scores.append(0.0)
        return scores

    def get_current_sentiment(self) -> SentimentSnapshot:
        """Fetch and aggregate sentiment from all configured feeds."""
        all_headlines: list[str] = []
        for feed_url in self.feeds:
            all_headlines.extend(_fetch_rss(feed_url))
        if not all_headlines:
            return SentimentSnapshot(
                score=0.0, n_headlines=0,
                fetch_time=datetime.utcnow().isoformat(),
            )
        scores = self._score_headlines(all_headlines)
        avg = float(sum(scores) / len(scores)) if scores else 0.0
        # Find most impactful headline
        if scores:
            max_idx = max(range(len(scores)), key=lambda i: abs(scores[i]))
            top = all_headlines[max_idx]
        else:
            top = ""
        return SentimentSnapshot(
            score=max(-1.0, min(1.0, avg)),
            n_headlines=len(all_headlines),
            top_headline=top[:200],
            fetch_time=datetime.utcnow().isoformat(),
            source="finbert" if self.use_finbert else "keyword",
            raw_scores=scores[:20],  # keep only first 20 for diagnostics
        )

    def score_text(self, text: str) -> float:
        """Score a single text string. Useful for backtesting with
        historical news archives."""
        scores = self._score_headlines([text])
        return scores[0] if scores else 0.0
