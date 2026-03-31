"""Configuration management for the trading system."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


class RiskConfig(BaseModel):
    max_portfolio_drawdown_pct: float = 10.0
    max_daily_loss_pct: float = 3.0
    max_per_trade_loss_pct: float = 2.0
    max_position_pct: float = 5.0
    max_trades_per_hour: int = 10
    consecutive_loss_cooldown_hours: float = 1.0
    consecutive_loss_threshold: int = 3
    max_open_positions: int = 10
    min_risk_reward_ratio: float = 2.0


class TechnicalIndicatorConfig(BaseModel):
    name: str
    period: int | None = None
    periods: list[int] | None = None
    fast: int | None = None
    slow: int | None = None
    signal: int | None = None
    std_dev: float | None = None
    overbought: float | None = None
    oversold: float | None = None
    k_period: int | None = None
    d_period: int | None = None


class TechnicalConfig(BaseModel):
    indicators: list[TechnicalIndicatorConfig] = Field(default_factory=list)
    timeframes: list[str] = Field(default_factory=lambda: ["1d", "4h", "1h"])


class SentimentConfig(BaseModel):
    enabled: bool = True
    engine: str = "llm"
    sources: list[str] = Field(default_factory=lambda: ["news"])
    min_articles: int = 3
    weight_in_decision: float = 0.2


class MarketConfig(BaseModel):
    enabled: bool = True
    provider: str = "yfinance"
    watchlist: list[str] = Field(default_factory=list)
    trading_hours: dict[str, Any] | None = None


class MarketsConfig(BaseModel):
    stocks: MarketConfig = Field(default_factory=MarketConfig)
    crypto: MarketConfig = Field(default_factory=MarketConfig)
    forex: MarketConfig = Field(default_factory=MarketConfig)


class PortfolioConfig(BaseModel):
    initial_capital: float = 100000.0
    rebalance_frequency: str = "weekly"
    diversification: dict[str, Any] = Field(default_factory=dict)


class AgentWeightsConfig(BaseModel):
    technical: float = 0.30
    fundamental: float = 0.20
    sentiment: float = 0.15
    risk: float = 0.25
    market_data: float = 0.10


class AutonomousConfig(BaseModel):
    enabled: bool = True
    scan_interval_seconds: int = 300
    max_concurrent_analyses: int = 5
    require_confirmation_for_live: bool = True


class LoggingConfig(BaseModel):
    console: bool = True
    file: bool = True
    file_path: str = "logs/trade.log"
    trade_log_path: str = "logs/trades.log"
    rotation: str = "10 MB"
    retention: str = "30 days"


class TradingConfig(BaseModel):
    """Main configuration container."""

    general: dict[str, Any] = Field(default_factory=lambda: {"mode": "paper"})
    autonomous: AutonomousConfig = Field(default_factory=AutonomousConfig)
    markets: MarketsConfig = Field(default_factory=MarketsConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    technical: TechnicalConfig = Field(default_factory=TechnicalConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    agent_weights: AgentWeightsConfig = Field(default_factory=AgentWeightsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @property
    def mode(self) -> str:
        return self.general.get("mode", "paper")

    @property
    def is_paper(self) -> bool:
        return self.mode == "paper"


class EnvSettings(BaseSettings):
    """Environment variables."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    trade_mode: str = "paper"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openbb_token: str | None = None
    alpaca_api_key: str | None = None
    alpaca_secret_key: str | None = None
    binance_api_key: str | None = None
    binance_secret_key: str | None = None
    news_api_key: str | None = None
    log_level: str = "INFO"


def load_config(config_path: str | Path | None = None) -> TradingConfig:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = CONFIG_DIR / "default.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        return TradingConfig()

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return TradingConfig(**raw) if raw else TradingConfig()


def load_strategy(strategy_name: str) -> dict[str, Any]:
    """Load a strategy configuration from YAML."""
    strategy_path = CONFIG_DIR / "strategies" / f"{strategy_name}.yaml"
    if not strategy_path.exists():
        raise FileNotFoundError(f"Strategy not found: {strategy_path}")

    with open(strategy_path) as f:
        return yaml.safe_load(f)
