"""Tests for configuration loading."""

from trade.config import (
    load_config,
    TradingConfig,
    RiskConfig,
    TechnicalConfig,
    AgentWeightsConfig,
)


class TestTradingConfig:
    def test_default_config(self):
        config = TradingConfig()
        assert config.mode == "paper"
        assert config.is_paper is True

    def test_risk_defaults(self):
        config = TradingConfig()
        assert config.risk.max_portfolio_drawdown_pct == 10.0
        assert config.risk.max_daily_loss_pct == 3.0

    def test_agent_weights(self):
        config = TradingConfig()
        weights = config.agent_weights
        total = (
            weights.technical + weights.fundamental +
            weights.sentiment + weights.risk + weights.market_data
        )
        assert abs(total - 1.0) < 0.01  # Weights should sum to ~1.0

    def test_load_default_config(self):
        config = load_config()
        assert config.mode == "paper"
        assert len(config.markets.stocks.watchlist) > 0
        assert len(config.markets.crypto.watchlist) > 0

    def test_load_nonexistent_config(self):
        config = load_config("/nonexistent/path.yaml")
        assert config.mode == "paper"  # Falls back to defaults


class TestRiskConfig:
    def test_sensible_limits(self):
        config = RiskConfig()
        assert config.max_portfolio_drawdown_pct > 0
        assert config.max_daily_loss_pct > 0
        assert config.max_per_trade_loss_pct > 0
        assert config.max_daily_loss_pct < config.max_portfolio_drawdown_pct
        assert config.max_per_trade_loss_pct < config.max_daily_loss_pct
