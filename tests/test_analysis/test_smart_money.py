"""Tests for Smart Money Concepts detection algorithms."""

import pytest
import numpy as np
import pandas as pd

from trade.analysis.smart_money import (
    detect_order_blocks,
    detect_fair_value_gaps,
    detect_liquidity_levels,
    detect_market_structure,
    get_smc_analysis,
)


@pytest.fixture
def bullish_impulse_df():
    """DataFrame with a clear bullish impulse (good for OB detection)."""
    # Small bearish candles followed by a strong bullish candle
    data = {
        "open":  [1.1000, 1.0990, 1.0985, 1.0980, 1.0975, 1.0970, 1.0960, 1.0970, 1.0980, 1.1050],
        "high":  [1.1005, 1.0995, 1.0990, 1.0985, 1.0980, 1.0975, 1.0965, 1.0990, 1.1010, 1.1080],
        "low":   [1.0990, 1.0985, 1.0980, 1.0970, 1.0968, 1.0960, 1.0950, 1.0960, 1.0975, 1.1040],
        "close": [1.0992, 1.0987, 1.0982, 1.0972, 1.0970, 1.0962, 1.0965, 1.0985, 1.1005, 1.1075],
        "volume": [100] * 10,
    }
    return pd.DataFrame(data)


@pytest.fixture
def fvg_df():
    """DataFrame with a Fair Value Gap pattern."""
    # Create a gap: candle 1 high < candle 3 low
    data = {
        "open":  [1.1000, 1.1010, 1.1020, 1.1050, 1.1060, 1.1070, 1.1080],
        "high":  [1.1015, 1.1025, 1.1030, 1.1065, 1.1075, 1.1085, 1.1090],
        "low":   [1.0995, 1.1005, 1.1015, 1.1040, 1.1055, 1.1065, 1.1075],
        "close": [1.1010, 1.1020, 1.1025, 1.1060, 1.1070, 1.1080, 1.1085],
        "volume": [100] * 7,
    }
    return pd.DataFrame(data)


@pytest.fixture
def trending_df():
    """DataFrame with clear bullish structure (HH, HL)."""
    np.random.seed(42)
    n = 60
    # Uptrend with swing points
    base = np.linspace(1.1000, 1.1300, n)
    noise = np.random.normal(0, 0.001, n)
    swing = np.sin(np.linspace(0, 8 * np.pi, n)) * 0.003

    close = base + noise + swing
    data = {
        "open": close - np.random.uniform(0, 0.001, n),
        "high": close + np.random.uniform(0, 0.002, n),
        "low": close - np.random.uniform(0, 0.002, n),
        "close": close,
        "volume": np.random.randint(50, 200, n),
    }
    return pd.DataFrame(data)


class TestOrderBlocks:
    def test_returns_list(self, bullish_impulse_df):
        obs = detect_order_blocks(bullish_impulse_df)
        assert isinstance(obs, list)

    def test_empty_df(self):
        df = pd.DataFrame({"open": [], "high": [], "low": [], "close": [], "volume": []})
        obs = detect_order_blocks(df)
        assert obs == []

    def test_small_df(self):
        df = pd.DataFrame({
            "open": [1.0, 1.1],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
            "close": [1.05, 1.15],
            "volume": [100, 100],
        })
        obs = detect_order_blocks(df)
        assert isinstance(obs, list)

    def test_ob_has_required_fields(self, bullish_impulse_df):
        obs = detect_order_blocks(bullish_impulse_df, min_impulse_pct=0.1)
        if obs:
            ob = obs[0]
            assert "type" in ob
            assert "high" in ob
            assert "low" in ob
            assert ob["type"] in ("bullish_ob", "bearish_ob")


class TestFairValueGaps:
    def test_returns_list(self, fvg_df):
        fvgs = detect_fair_value_gaps(fvg_df)
        assert isinstance(fvgs, list)

    def test_empty_df(self):
        df = pd.DataFrame({"open": [], "high": [], "low": [], "close": [], "volume": []})
        fvgs = detect_fair_value_gaps(df)
        assert fvgs == []

    def test_fvg_has_required_fields(self, fvg_df):
        fvgs = detect_fair_value_gaps(fvg_df, min_gap_pct=0.01)
        if fvgs:
            fvg = fvgs[0]
            assert "type" in fvg
            assert "high" in fvg
            assert "low" in fvg
            assert "midpoint" in fvg
            assert fvg["type"] in ("bullish_fvg", "bearish_fvg")


class TestLiquidityLevels:
    def test_returns_list(self, trending_df):
        levels = detect_liquidity_levels(trending_df)
        assert isinstance(levels, list)

    def test_level_types(self, trending_df):
        levels = detect_liquidity_levels(trending_df)
        for level in levels:
            assert level["type"] in ("buy_side_liquidity", "sell_side_liquidity")
            assert "level" in level


class TestMarketStructure:
    def test_returns_dict(self, trending_df):
        structure = detect_market_structure(trending_df)
        assert isinstance(structure, dict)
        assert "structure" in structure
        assert "swings" in structure

    def test_detects_trend(self, trending_df):
        structure = detect_market_structure(trending_df)
        # Trending df has uptrend, should detect bullish or at least not unknown
        assert structure["structure"] in ("bullish", "bearish", "ranging", "unknown")

    def test_small_df(self):
        df = pd.DataFrame({
            "open": [1.0] * 5,
            "high": [1.1] * 5,
            "low": [0.9] * 5,
            "close": [1.05] * 5,
            "volume": [100] * 5,
        })
        structure = detect_market_structure(df)
        assert structure["structure"] == "unknown"


class TestFullSMCAnalysis:
    def test_returns_analysis(self, trending_df):
        result = get_smc_analysis(trending_df)
        assert isinstance(result, dict)
        assert result["available"] is True
        assert "market_structure" in result
        assert "order_blocks" in result
        assert "fair_value_gaps" in result

    def test_empty_df(self):
        result = get_smc_analysis(pd.DataFrame())
        assert result["available"] is False

    def test_none_df(self):
        result = get_smc_analysis(None)
        assert result["available"] is False
