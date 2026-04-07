"""LOBFrame research package — synthetic LOB synthesis + Nautilus bridge.

References:
  - Briola, A. (2024). LOBFrame: An open-source framework for processing
    limit order book data.
  - Cont, R. & Stoikov, S. (2014). Order book dynamics in liquid markets.
"""
from trade.research.lobframe.cont_stoikov import (
    ContStoikovConfig,
    synthesize_lob,
    synthesize_snapshots,
)
from trade.research.lobframe.data_schema import (
    LEVELS,
    LOBSnapshot,
    N_FEATURES,
    ask_price_columns,
    ask_volume_columns,
    best_ask,
    best_bid,
    bid_price_columns,
    bid_volume_columns,
    column_index,
    column_names,
    mid_price,
    spread,
)
from trade.research.lobframe.leakage_tests import ALL_TESTS, run_all
from trade.research.lobframe.nautilus_bridge import LOBFrameBridge

__all__ = [
    "ContStoikovConfig",
    "synthesize_lob",
    "synthesize_snapshots",
    "LEVELS",
    "N_FEATURES",
    "LOBSnapshot",
    "column_index",
    "column_names",
    "ask_price_columns",
    "ask_volume_columns",
    "bid_price_columns",
    "bid_volume_columns",
    "best_ask",
    "best_bid",
    "mid_price",
    "spread",
    "ALL_TESTS",
    "run_all",
    "LOBFrameBridge",
]
