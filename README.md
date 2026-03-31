# TRADE - Autonomous Multi-Agent Trading System

An AI-powered autonomous trading system that uses multiple specialized agents to analyze markets, generate signals, manage risk, and execute trades across stocks, crypto, and forex.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   ORCHESTRATOR                       │
├──────────┬──────────┬──────────┬──────────┬─────────┤
│ Market   │Technical │Sentiment │Fundament.│  Risk   │
│  Data    │ Analysis │ (LLM)    │ Analysis │ Manager │
│  Agent   │  Agent   │  Agent   │  Agent   │  Agent  │
├──────────┴──────────┴──────────┴──────────┴─────────┤
│              PORTFOLIO MANAGER AGENT                  │
├─────────────────────────────────────────────────────┤
│              EXECUTION AGENT (Paper/Live)             │
└─────────────────────────────────────────────────────┘
```

### Agents

| Agent | Role | Engine |
|-------|------|--------|
| **Market Data** | Fetches OHLCV, quotes, news | yfinance / OpenBB |
| **Technical** | RSI, MACD, Bollinger, EMA, ATR... | pandas-ta (algorithmic) |
| **Sentiment** | News & social media analysis | LLM (Claude/GPT) / FinBERT |
| **Fundamental** | P/E, margins, ROE, debt analysis | Algorithmic |
| **Risk Manager** | Circuit breakers, position limits | Algorithmic (strict rules) |
| **Portfolio Manager** | Weighted signal aggregation | Algorithmic |
| **Execution** | Order creation, paper/live trading | Paper engine / Broker API |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Analyze a stock
python -m trade.main --symbol AAPL

# Analyze crypto
python -m trade.main --symbol BTC-USD

# Analyze full watchlist
python -m trade.main --watchlist

# Run autonomous mode (paper trading)
python -m trade.main --autonomous

# Output as JSON
python -m trade.main --symbol TSLA --json
```

## Configuration

Edit `config/default.yaml` to customize:
- Watchlists (stocks, crypto, forex)
- Risk parameters (drawdown limits, position sizing)
- Technical indicators
- Sentiment analysis engine
- Agent weights for decision making

## Safety

- **Paper trading is the default** - live trading requires explicit configuration
- **Circuit breaker**: Auto-stops on max drawdown (10%), daily loss (3%), or consecutive losses
- **Kill switch**: Emergency stop for all trading activity
- **Rate limiting**: Prevents API abuse and overtrading
- **All decisions are logged** with full reasoning trails

## License

MIT
