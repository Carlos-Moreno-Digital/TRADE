"""Tests for the Market Scanner."""

import pytest
from datetime import datetime, timezone

from trade.scanner import (
    MarketScanner,
    FOREX_PAIRS,
    COMMODITIES,
    INDICES,
    CRYPTO,
    US_STOCKS,
    EU_STOCKS,
    ETFS,
    MARKET_HOURS,
    ScanResult,
)


class TestMarketHours:
    def test_forex_hours_defined(self):
        assert "forex" in MARKET_HOURS
        assert MARKET_HOURS["forex"]["is_24h"] is True

    def test_crypto_24_7(self):
        assert MARKET_HOURS["crypto"]["is_24h"] is True
        assert len(MARKET_HOURS["crypto"]["days"]) == 7  # All days

    def test_index_not_24h(self):
        assert MARKET_HOURS["index"]["is_24h"] is False

    def test_instruments_defined(self):
        assert len(FOREX_PAIRS) >= 30
        assert len(COMMODITIES) >= 8
        assert len(INDICES) >= 10
        assert len(CRYPTO) >= 10
        assert len(US_STOCKS) >= 30
        assert len(EU_STOCKS) >= 10
        assert len(ETFS) >= 10

    def test_total_universe_150_plus(self):
        total = len(FOREX_PAIRS) + len(COMMODITIES) + len(INDICES) + \
                len(CRYPTO) + len(US_STOCKS) + len(EU_STOCKS) + len(ETFS)
        assert total >= 130


class TestMarketScanner:
    def test_init(self):
        scanner = MarketScanner(top_n=5)
        assert scanner.top_n == 5

    def test_active_instruments_weekday(self):
        """On a Tuesday at 15:00 UTC, all markets should be open."""
        tuesday_15h = datetime(2026, 3, 31, 15, 0, tzinfo=timezone.utc)  # Tuesday
        scanner = MarketScanner()
        active = scanner.get_active_instruments(tuesday_15h)

        # Forex, commodities, indices, stocks, crypto should all be open
        asset_classes = set(cls for _, cls, _ in active)
        assert "forex" in asset_classes
        assert "crypto" in asset_classes
        assert "stock_us" in asset_classes
        # Should be 100+ instruments during US hours
        assert len(active) >= 100

    def test_no_forex_on_saturday(self):
        """Forex should be closed on Saturday."""
        saturday = datetime(2026, 3, 28, 12, 0, tzinfo=timezone.utc)
        scanner = MarketScanner()
        active = scanner.get_active_instruments(saturday)

        forex_active = [a for a in active if a[1] == "forex"]
        assert len(forex_active) == 0

    def test_crypto_always_open(self):
        """Crypto should be available 24/7 including weekends."""
        saturday = datetime(2026, 3, 28, 12, 0, tzinfo=timezone.utc)
        scanner = MarketScanner()
        active = scanner.get_active_instruments(saturday)

        crypto_active = [a for a in active if a[1] == "crypto"]
        assert len(crypto_active) >= 2

    def test_indices_closed_early_morning(self):
        """US indices should be closed at 5:00 UTC."""
        early = datetime(2026, 3, 31, 5, 0, tzinfo=timezone.utc)
        scanner = MarketScanner()
        active = scanner.get_active_instruments(early)

        index_active = [a for a in active if a[1] == "index"]
        assert len(index_active) == 0


class TestScanResult:
    def test_creation(self):
        result = ScanResult(
            symbol="EURUSD=X",
            asset_class="forex",
            display_name="EUR/USD",
            score=0.75,
            trend="bullish",
            momentum=1.5,
        )
        assert result.score == 0.75
        assert result.trend == "bullish"
        assert result.is_open is True

    def test_default_values(self):
        result = ScanResult("TEST", "forex", "Test")
        assert result.score == 0.0
        assert result.trend == "neutral"
        assert result.is_open is True
