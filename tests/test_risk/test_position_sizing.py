"""Tests for position sizing methods."""

from trade.risk.position_sizing import fixed_fractional, kelly_criterion, volatility_adjusted


class TestFixedFractional:
    def test_basic_calculation(self):
        quantity = fixed_fractional(
            portfolio_value=100000,
            risk_pct=1.0,
            entry_price=100.0,
            stop_loss=95.0,
        )
        # Risk $1000, stop distance $5 -> 200 shares
        assert quantity == 200.0

    def test_zero_stop_distance(self):
        quantity = fixed_fractional(
            portfolio_value=100000,
            risk_pct=1.0,
            entry_price=100.0,
            stop_loss=100.0,
        )
        assert quantity == 0.0

    def test_small_risk(self):
        quantity = fixed_fractional(
            portfolio_value=10000,
            risk_pct=0.5,
            entry_price=50.0,
            stop_loss=48.0,
        )
        # Risk $50, stop distance $2 -> 25 shares
        assert quantity == 25.0


class TestKellyCriterion:
    def test_positive_edge(self):
        fraction = kelly_criterion(
            win_rate=0.6,
            avg_win=100,
            avg_loss=80,
        )
        assert fraction > 0
        assert fraction <= 0.25  # Capped at 25%

    def test_no_edge(self):
        fraction = kelly_criterion(
            win_rate=0.5,
            avg_win=100,
            avg_loss=100,
        )
        assert fraction == 0.0

    def test_negative_edge(self):
        fraction = kelly_criterion(
            win_rate=0.3,
            avg_win=100,
            avg_loss=100,
        )
        assert fraction == 0.0

    def test_half_kelly(self):
        full = kelly_criterion(win_rate=0.6, avg_win=100, avg_loss=80, fraction=1.0)
        half = kelly_criterion(win_rate=0.6, avg_win=100, avg_loss=80, fraction=0.5)
        # Half-Kelly should be less than full Kelly
        assert half < full
        assert half > 0


class TestVolatilityAdjusted:
    def test_basic_calculation(self):
        quantity = volatility_adjusted(
            portfolio_value=100000,
            risk_pct=1.0,
            entry_price=100.0,
            atr=5.0,
            atr_multiplier=2.0,
        )
        # Risk $1000, stop distance 5*2=10 -> 100 shares
        assert quantity == 100.0

    def test_zero_atr(self):
        quantity = volatility_adjusted(
            portfolio_value=100000,
            risk_pct=1.0,
            entry_price=100.0,
            atr=0.0,
        )
        assert quantity == 0.0
