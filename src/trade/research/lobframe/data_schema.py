"""LOBFrame tensor schema (Briola 2024 / LOBSTER convention).

Each LOB observation is a flat row of `4 * LEVELS` floats. For LEVELS=10
this is 40 features per timestamp, layout:

  [ ask_p_1, ask_v_1, bid_p_1, bid_v_1,
    ask_p_2, ask_v_2, bid_p_2, bid_v_2,
    ...
    ask_p_10, ask_v_10, bid_p_10, bid_v_10 ]

Why ask first: the LOBSTER format puts asks first by historical
convention. We follow it so any model trained on LOBSTER tapes can
ingest our tensors unchanged.

Invariants every LOBFrame row must satisfy (enforced by the leakage
tests):
  - ask_p_1 > bid_p_1                       (positive spread)
  - ask_p_k strictly increasing in k        (asks ascend)
  - bid_p_k strictly decreasing in k        (bids descend)
  - all volumes > 0
  - no NaNs
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

LEVELS: int = 10
N_FEATURES: int = 4 * LEVELS  # 40 for 10 levels
ASK_PRICE = 0
ASK_VOLUME = 1
BID_PRICE = 2
BID_VOLUME = 3


def column_index(level: int, kind: str) -> int:
    """Return the flat-row index for a (level, kind) pair.

    level is 1-based (1..LEVELS); kind in {'ask_p','ask_v','bid_p','bid_v'}.
    """
    if not 1 <= level <= LEVELS:
        raise ValueError(f"level out of range: {level}")
    base = (level - 1) * 4
    offsets = {
        "ask_p": ASK_PRICE,
        "ask_v": ASK_VOLUME,
        "bid_p": BID_PRICE,
        "bid_v": BID_VOLUME,
    }
    if kind not in offsets:
        raise ValueError(f"unknown kind {kind}")
    return base + offsets[kind]


def column_names(levels: int = LEVELS) -> List[str]:
    out: List[str] = []
    for k in range(1, levels + 1):
        out += [f"ask_p_{k}", f"ask_v_{k}", f"bid_p_{k}", f"bid_v_{k}"]
    return out


@dataclass
class LOBSnapshot:
    """A single point-in-time order book in LOBSTER layout.

    Use as_row() to flatten into the schema-conformant float array.
    """

    timestamp_ns: int
    ask_prices: np.ndarray  # shape (LEVELS,)
    ask_volumes: np.ndarray
    bid_prices: np.ndarray
    bid_volumes: np.ndarray

    def __post_init__(self) -> None:
        for arr_name in ("ask_prices", "ask_volumes", "bid_prices", "bid_volumes"):
            arr = getattr(self, arr_name)
            if arr.shape[0] != LEVELS:
                raise ValueError(
                    f"{arr_name} has shape {arr.shape}, expected ({LEVELS},)"
                )

    def as_row(self) -> np.ndarray:
        row = np.empty(N_FEATURES, dtype=float)
        for k in range(LEVELS):
            row[k * 4 + ASK_PRICE] = self.ask_prices[k]
            row[k * 4 + ASK_VOLUME] = self.ask_volumes[k]
            row[k * 4 + BID_PRICE] = self.bid_prices[k]
            row[k * 4 + BID_VOLUME] = self.bid_volumes[k]
        return row


def best_bid(row: np.ndarray) -> float:
    return float(row[column_index(1, "bid_p")])


def best_ask(row: np.ndarray) -> float:
    return float(row[column_index(1, "ask_p")])


def mid_price(row: np.ndarray) -> float:
    return 0.5 * (best_bid(row) + best_ask(row))


def spread(row: np.ndarray) -> float:
    return best_ask(row) - best_bid(row)


def ask_price_columns() -> List[int]:
    return [column_index(k, "ask_p") for k in range(1, LEVELS + 1)]


def bid_price_columns() -> List[int]:
    return [column_index(k, "bid_p") for k in range(1, LEVELS + 1)]


def ask_volume_columns() -> List[int]:
    return [column_index(k, "ask_v") for k in range(1, LEVELS + 1)]


def bid_volume_columns() -> List[int]:
    return [column_index(k, "bid_v") for k in range(1, LEVELS + 1)]
