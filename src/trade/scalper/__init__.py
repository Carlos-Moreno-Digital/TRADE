"""FunderPro Scalper — high-frequency automated trading system.

Architecture:
  ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
  │  Connector   │────▶│  Scalper      │────▶│  Risk       │
  │  (TradeLocker│     │  Engine       │     │  Manager    │
  │   or Paper)  │◀────│  (Strategies) │◀────│  (FunderPro)│
  └─────────────┘     └──────────────┘     └─────────────┘
         │                    │                     │
         ▼                    ▼                     ▼
  ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
  │  Data Feed   │     │  Signal       │     │  Session    │
  │  (1min OHLCV │     │  Generator    │     │  Controller │
  │   + ticks)   │     │  (5-15/day)   │     │  (hours,    │
  └─────────────┘     └──────────────┘     │   weekend)  │
                                            └─────────────┘

FunderPro rules baked into risk manager:
  - Max DD: 10% of initial balance
  - Daily loss: 5% of balance (reset 5pm EST / 22:00 UTC)
  - Close all positions before weekend
  - No latency arbitrage
  - Min 4 trading days to pass
"""
