# TRADE - Autonomous Multi-Agent Trading System

## Project Overview
Multi-agent AI trading system supporting stocks, crypto, and forex. Uses hybrid approach: LLMs for sentiment/news analysis + classical algorithms for technical indicators and execution.

## Architecture
- `src/trade/agents/` - 7 specialized agents orchestrated by a central orchestrator
- `src/trade/data/` - Market data providers (yfinance, OpenBB)
- `src/trade/analysis/` - Technical indicators (pandas-ta) and sentiment (LLM-based)
- `src/trade/risk/` - Circuit breakers, position sizing, risk management
- `src/trade/execution/` - Paper trading (default) and broker interfaces
- `src/trade/strategy/` - Signal generation and strategy definitions
- `config/` - YAML configuration files

## Commands
- Install: `pip install -e ".[dev]"`
- Run: `python -m trade.main --symbol AAPL --mode paper`
- Tests: `pytest tests/`
- Lint: `ruff check src/`
- Format: `ruff format src/`

## Safety Rules
- PAPER TRADING IS THE DEFAULT. Never change to live without explicit user confirmation.
- Circuit breakers are mandatory - never disable them.
- All agent decisions must be logged.
- Max drawdown: 10%, Max daily loss: 3%, Max per-trade loss: 2%.
- Always validate data before making trading decisions.

## Adding a New Agent
1. Create new file in `src/trade/agents/`
2. Inherit from `BaseAgent` in `agents/base.py`
3. Implement `analyze(context: AnalysisContext) -> AgentSignal`
4. Register in `agents/__init__.py`
5. Add to orchestrator pipeline in `agents/orchestrator.py`

## Configuration
- `config/default.yaml` - Global settings
- `config/strategies/*.yaml` - Strategy definitions
- `.env` - API keys and secrets (never commit!)
