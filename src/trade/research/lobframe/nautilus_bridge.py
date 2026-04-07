"""LOBFrame ↔ NautilusTrader bridge.

Translates between Nautilus order book events and the flat LOBFrame
tensor schema. Two ingestion paths:

  1. Synthetic snapshots (current research path):
       LOBFrameBridge.from_snapshots([LOBSnapshot, ...]) -> np.ndarray

  2. Real Nautilus order book (when we have L2/L3 data):
       LOBFrameBridge.from_nautilus_book(book) -> LOBSnapshot
     where `book` is a nautilus_trader.model.book.OrderBook instance.

The bridge intentionally does NOT depend on a live Nautilus engine —
it only knows how to read the OrderBook public API (`bids()` /
`asks()` returning sorted level lists). This means it works against
historical Parquet caches via Nautilus's own loaders without
needing the BacktestEngine to be running.
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np

from trade.research.lobframe.data_schema import (
    LEVELS,
    N_FEATURES,
    LOBSnapshot,
    column_index,
)


class LOBFrameBridge:
    """Stateless converter between LOBSnapshot and LOBFrame tensor rows."""

    def __init__(self, levels: int = LEVELS):
        self.levels = levels
        self.n_features = 4 * levels

    # ------------------------------------------------------------------
    # Synthetic / pre-built snapshots
    # ------------------------------------------------------------------
    def from_snapshots(self, snapshots: Iterable[LOBSnapshot]) -> np.ndarray:
        rows: List[np.ndarray] = []
        for snap in snapshots:
            rows.append(snap.as_row())
        if not rows:
            return np.empty((0, self.n_features), dtype=float)
        return np.vstack(rows)

    # ------------------------------------------------------------------
    # Real Nautilus order book
    # ------------------------------------------------------------------
    def from_nautilus_book(self, book) -> LOBSnapshot:  # noqa: ANN001
        """Extract a LOBSnapshot from a Nautilus OrderBook instance.

        We pull `book.bids()` and `book.asks()`. Each returns a list of
        BookLevel objects with `.price` and `.size` attributes ordered
        from best to worst. We pad with the worst-known price/zero
        volume if the book has fewer than `self.levels` levels.
        """
        bids = list(book.bids())[: self.levels]
        asks = list(book.asks())[: self.levels]
        if not bids or not asks:
            raise ValueError("Empty book on at least one side")

        ask_p = np.zeros(self.levels, dtype=float)
        ask_v = np.zeros(self.levels, dtype=float)
        bid_p = np.zeros(self.levels, dtype=float)
        bid_v = np.zeros(self.levels, dtype=float)

        for k in range(self.levels):
            if k < len(asks):
                ask_p[k] = float(asks[k].price)
                ask_v[k] = float(asks[k].size)
            else:
                # Pad: extend with last known price + zero volume
                ask_p[k] = ask_p[k - 1]
                ask_v[k] = 1e-6
            if k < len(bids):
                bid_p[k] = float(bids[k].price)
                bid_v[k] = float(bids[k].size)
            else:
                bid_p[k] = bid_p[k - 1]
                bid_v[k] = 1e-6

        ts_ns = int(getattr(book, "ts_event", 0) or 0)
        return LOBSnapshot(
            timestamp_ns=ts_ns,
            ask_prices=ask_p,
            ask_volumes=ask_v,
            bid_prices=bid_p,
            bid_volumes=bid_v,
        )

    def from_nautilus_books(self, books: Iterable) -> np.ndarray:
        return self.from_snapshots(self.from_nautilus_book(b) for b in books)
