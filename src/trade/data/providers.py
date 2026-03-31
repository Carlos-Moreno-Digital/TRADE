"""Market data providers - unified interface for yfinance, OpenBB, etc."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from loguru import logger

from trade.data.models import AssetType, NewsItem, OHLCV, Quote


class MarketDataProvider:
    """Unified market data provider supporting stocks, crypto, and forex."""

    def __init__(self, provider: str = "yfinance"):
        self.provider = provider
        self._yf = None

    def _get_yfinance(self):
        if self._yf is None:
            import yfinance as yf
            self._yf = yf
        return self._yf

    def get_historical(
        self,
        symbol: str,
        period: str = "3mo",
        interval: str = "1d",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data."""
        logger.info(f"Fetching historical data for {symbol} ({period}, {interval})")
        yf = self._get_yfinance()
        ticker = yf.Ticker(symbol)

        kwargs = {"interval": interval}
        if start and end:
            kwargs["start"] = start
            kwargs["end"] = end
        else:
            kwargs["period"] = period

        df = ticker.history(**kwargs)
        if df.empty:
            logger.warning(f"No historical data returned for {symbol}")
            return df

        # Normalize column names
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df.index.name = "timestamp"
        return df

    def get_realtime_quote(self, symbol: str) -> Quote | None:
        """Get real-time quote for a symbol."""
        logger.debug(f"Fetching quote for {symbol}")
        yf = self._get_yfinance()
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info

        asset_type = self._detect_asset_type(symbol)

        try:
            return Quote(
                symbol=symbol,
                asset_type=asset_type,
                price=float(info.last_price),
                bid=getattr(info, "bid", None),
                ask=getattr(info, "ask", None),
                volume=getattr(info, "last_volume", None),
                change_pct=getattr(info, "day_change", None),
            )
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return None

    def get_fundamentals(self, symbol: str) -> dict:
        """Get fundamental data for a stock."""
        logger.info(f"Fetching fundamentals for {symbol}")
        yf = self._get_yfinance()
        ticker = yf.Ticker(symbol)

        try:
            info = ticker.info
            return {
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "pb_ratio": info.get("priceToBook"),
                "market_cap": info.get("marketCap"),
                "revenue": info.get("totalRevenue"),
                "profit_margin": info.get("profitMargins"),
                "debt_to_equity": info.get("debtToEquity"),
                "roe": info.get("returnOnEquity"),
                "eps": info.get("trailingEps"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
            }
        except Exception as e:
            logger.error(f"Failed to get fundamentals for {symbol}: {e}")
            return {}

    def get_news(self, symbol: str, limit: int = 10) -> list[NewsItem]:
        """Get recent news for a symbol."""
        logger.info(f"Fetching news for {symbol}")
        yf = self._get_yfinance()
        ticker = yf.Ticker(symbol)

        news_items = []
        try:
            for article in (ticker.news or [])[:limit]:
                news_items.append(
                    NewsItem(
                        title=article.get("title", ""),
                        source=article.get("publisher", "unknown"),
                        published=datetime.fromtimestamp(article["providerPublishTime"])
                        if "providerPublishTime" in article
                        else None,
                        url=article.get("link"),
                        summary=article.get("summary"),
                    )
                )
        except Exception as e:
            logger.error(f"Failed to get news for {symbol}: {e}")

        return news_items

    def get_multiple_historical(
        self, symbols: list[str], period: str = "3mo", interval: str = "1d"
    ) -> dict[str, pd.DataFrame]:
        """Fetch historical data for multiple symbols."""
        results = {}
        for symbol in symbols:
            try:
                df = self.get_historical(symbol, period=period, interval=interval)
                if not df.empty:
                    results[symbol] = df
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
        return results

    @staticmethod
    def _detect_asset_type(symbol: str) -> AssetType:
        """Detect asset type from symbol format."""
        if symbol.endswith("=X"):
            return AssetType.FOREX
        if symbol.endswith("-USD") or symbol.endswith("-USDT"):
            return AssetType.CRYPTO
        return AssetType.STOCK
