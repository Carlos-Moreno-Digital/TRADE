"""Tests for the SQLite cache store."""

import pytest
import tempfile
from pathlib import Path

import pandas as pd

from trade.cache.store import CacheStore


@pytest.fixture
def cache(tmp_path):
    db_path = tmp_path / "test_cache.db"
    store = CacheStore(db_path)
    yield store
    store.close()


class TestCacheStore:
    def test_init_creates_db(self, cache):
        assert cache.db_path.exists()

    def test_set_and_get(self, cache):
        cache.set_cached("test_key", {"value": 42}, ttl_hours=1.0)
        result = cache.get_cached("test_key")
        assert result is not None
        assert result["value"] == 42

    def test_get_expired(self, cache):
        cache.set_cached("expired_key", {"value": 1}, ttl_hours=-1.0)  # Already expired
        result = cache.get_cached("expired_key")
        assert result is None

    def test_get_missing(self, cache):
        result = cache.get_cached("nonexistent")
        assert result is None

    def test_cache_overwrite(self, cache):
        cache.set_cached("key", {"v": 1})
        cache.set_cached("key", {"v": 2})
        result = cache.get_cached("key")
        assert result["v"] == 2

    def test_cache_dataframe(self, cache):
        df = pd.DataFrame({"close": [100, 101, 102], "volume": [1000, 1100, 1200]})
        cache.set_cached_df("ohlcv_test", df, ttl_hours=1.0)

        result = cache.get_cached_df("ohlcv_test")
        assert result is not None
        assert len(result) == 3
        assert "close" in result.columns

    def test_clear_expired(self, cache):
        cache.set_cached("fresh", {"v": 1}, ttl_hours=1.0)
        cache.set_cached("stale", {"v": 2}, ttl_hours=-1.0)
        removed = cache.clear_expired()
        assert removed >= 1
        assert cache.get_cached("fresh") is not None

    def test_log_decision(self, cache):
        decision_id = cache.log_decision(
            symbol="EURUSD",
            action="BUY",
            confidence=0.82,
            bull_argument="Strong uptrend",
            bear_argument="Weak resistance",
            judge_verdict="BUY with 0.5% risk",
            risk_score=0.3,
            executed=True,
            entry_price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100,
        )
        assert decision_id > 0

    def test_get_recent_decisions(self, cache):
        cache.log_decision(symbol="EURUSD", action="BUY", confidence=0.8)
        cache.log_decision(symbol="GBPUSD", action="SELL", confidence=0.6)

        decisions = cache.get_recent_decisions(limit=10)
        assert len(decisions) == 2

    def test_update_pnl(self, cache):
        did = cache.log_decision(symbol="EURUSD", action="BUY", confidence=0.8)
        cache.update_decision_pnl(did, 150.0)

        decisions = cache.get_recent_decisions(limit=1)
        assert decisions[0]["pnl"] == 150.0

    def test_portfolio_snapshot(self, cache):
        cache.save_portfolio_snapshot(
            cash=9500.0,
            total_value=10200.0,
            daily_pnl=200.0,
            drawdown_pct=0.0,
        )
        # Just verify no error

    def test_trade_history(self, cache):
        tid = cache.log_trade(
            symbol="EURUSD",
            action="BUY",
            quantity=0.1,
            entry_price=1.1000,
            exit_price=1.1050,
            pnl=50.0,
            session="london",
            strategy="ict_ob",
        )
        assert tid > 0

    def test_trade_stats_empty(self, cache):
        stats = cache.get_trade_stats()
        assert stats["total_trades"] == 0

    def test_trade_stats_with_data(self, cache):
        cache.log_trade("EURUSD", "BUY", 0.1, pnl=100.0)
        cache.log_trade("GBPUSD", "SELL", 0.1, pnl=-50.0)
        cache.log_trade("XAUUSD", "BUY", 0.1, pnl=200.0)

        stats = cache.get_trade_stats()
        assert stats["total_trades"] == 3
        assert stats["wins"] == 2
        assert stats["losses"] == 1
        assert stats["total_pnl"] == 250.0
        assert stats["win_rate"] == pytest.approx(0.6667, abs=0.01)
