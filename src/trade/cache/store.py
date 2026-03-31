"""SQLite-based cache for market data and decision logging.

Purposes:
1. Reduce API calls (don't hammer yfinance/OpenBB during development)
2. Audit trail: log every decision for post-analysis
3. Portfolio snapshots for drawdown tracking
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


DEFAULT_DB_PATH = Path("cache/trade_cache.db")

# TTL defaults in hours
DEFAULT_TTLS = {
    "ohlcv_daily": 1.0,
    "ohlcv_intraday": 0.083,  # 5 minutes
    "fundamentals": 24.0,
    "news": 0.5,  # 30 minutes
    "calendar": 6.0,
}


class CacheStore:
    """SQLite cache for market data and audit logging."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS data_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key TEXT UNIQUE NOT NULL,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_cache_key ON data_cache(cache_key);
            CREATE INDEX IF NOT EXISTS idx_expires ON data_cache(expires_at);

            CREATE TABLE IF NOT EXISTS decisions_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                asset_type TEXT,
                action TEXT NOT NULL,
                strength TEXT,
                confidence REAL,
                bull_argument TEXT,
                bear_argument TEXT,
                judge_verdict TEXT,
                risk_score REAL,
                prop_firm_check TEXT,
                executed INTEGER DEFAULT 0,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                quantity REAL,
                pnl REAL DEFAULT NULL,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                cash REAL NOT NULL,
                total_value REAL NOT NULL,
                daily_pnl REAL DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                drawdown_pct REAL DEFAULT 0,
                peak_value REAL,
                open_positions INTEGER DEFAULT 0,
                trades_today INTEGER DEFAULT 0,
                positions_json TEXT
            );

            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                pnl REAL,
                pnl_pct REAL,
                hold_duration_minutes INTEGER,
                session TEXT,
                strategy TEXT,
                notes TEXT
            );
        """)
        conn.commit()
        logger.debug(f"Cache database initialized at {self.db_path}")

    # =========================================================================
    # Data Cache (OHLCV, news, fundamentals)
    # =========================================================================

    def get_cached(self, key: str) -> Any | None:
        """Get a cached value if not expired."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT data, expires_at FROM data_cache WHERE cache_key = ?",
            (key,)
        ).fetchone()

        if row is None:
            return None

        expires = datetime.fromisoformat(row["expires_at"])
        if datetime.utcnow() > expires:
            conn.execute("DELETE FROM data_cache WHERE cache_key = ?", (key,))
            conn.commit()
            return None

        return json.loads(row["data"])

    def set_cached(self, key: str, data: Any, ttl_hours: float = 1.0) -> None:
        """Cache a value with TTL."""
        conn = self._get_conn()
        expires = datetime.utcnow() + timedelta(hours=ttl_hours)
        serialized = json.dumps(data, default=str)

        conn.execute(
            """INSERT OR REPLACE INTO data_cache (cache_key, data, expires_at)
               VALUES (?, ?, ?)""",
            (key, serialized, expires.isoformat()),
        )
        conn.commit()

    def get_cached_df(self, key: str) -> pd.DataFrame | None:
        """Get a cached DataFrame."""
        data = self.get_cached(key)
        if data is None:
            return None
        return pd.DataFrame(data)

    def set_cached_df(self, key: str, df: pd.DataFrame, ttl_hours: float = 1.0) -> None:
        """Cache a DataFrame."""
        data = df.reset_index().to_dict(orient="records")
        self.set_cached(key, data, ttl_hours)

    def clear_expired(self) -> int:
        """Remove all expired cache entries. Returns count of removed entries."""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM data_cache WHERE expires_at < ?",
            (datetime.utcnow().isoformat(),),
        )
        conn.commit()
        return cursor.rowcount

    # =========================================================================
    # Decision Logging (Audit Trail)
    # =========================================================================

    def log_decision(
        self,
        symbol: str,
        action: str,
        confidence: float,
        asset_type: str = "",
        strength: str = "",
        bull_argument: str = "",
        bear_argument: str = "",
        judge_verdict: str = "",
        risk_score: float = 0.0,
        prop_firm_check: str = "",
        executed: bool = False,
        entry_price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        quantity: float | None = None,
        notes: str = "",
    ) -> int:
        """Log a trading decision for audit trail."""
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO decisions_log
               (symbol, asset_type, action, strength, confidence, bull_argument,
                bear_argument, judge_verdict, risk_score, prop_firm_check,
                executed, entry_price, stop_loss, take_profit, quantity, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (symbol, asset_type, action, strength, confidence, bull_argument,
             bear_argument, judge_verdict, risk_score, prop_firm_check,
             int(executed), entry_price, stop_loss, take_profit, quantity, notes),
        )
        conn.commit()
        return cursor.lastrowid

    def update_decision_pnl(self, decision_id: int, pnl: float) -> None:
        """Update the P&L of a completed trade."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE decisions_log SET pnl = ? WHERE id = ?",
            (pnl, decision_id),
        )
        conn.commit()

    # =========================================================================
    # Portfolio Snapshots
    # =========================================================================

    def save_portfolio_snapshot(
        self,
        cash: float,
        total_value: float,
        daily_pnl: float = 0.0,
        total_pnl: float = 0.0,
        drawdown_pct: float = 0.0,
        peak_value: float = 0.0,
        open_positions: int = 0,
        trades_today: int = 0,
        positions_json: str = "[]",
    ) -> None:
        """Save a portfolio snapshot."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO portfolio_snapshots
               (cash, total_value, daily_pnl, total_pnl, drawdown_pct,
                peak_value, open_positions, trades_today, positions_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (cash, total_value, daily_pnl, total_pnl, drawdown_pct,
             peak_value, open_positions, trades_today, positions_json),
        )
        conn.commit()

    # =========================================================================
    # Trade History
    # =========================================================================

    def log_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        entry_price: float | None = None,
        exit_price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        pnl: float | None = None,
        pnl_pct: float | None = None,
        hold_duration_minutes: int | None = None,
        session: str = "",
        strategy: str = "",
        notes: str = "",
    ) -> int:
        """Log a completed trade."""
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO trade_history
               (symbol, action, quantity, entry_price, exit_price, stop_loss,
                take_profit, pnl, pnl_pct, hold_duration_minutes, session,
                strategy, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (symbol, action, quantity, entry_price, exit_price, stop_loss,
             take_profit, pnl, pnl_pct, hold_duration_minutes, session,
             strategy, notes),
        )
        conn.commit()
        return cursor.lastrowid

    # =========================================================================
    # Analytics Queries
    # =========================================================================

    def get_trade_stats(self, days: int = 30) -> dict[str, Any]:
        """Get trading statistics for the last N days."""
        conn = self._get_conn()
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        rows = conn.execute(
            "SELECT * FROM trade_history WHERE timestamp > ?", (cutoff,)
        ).fetchall()

        if not rows:
            return {"total_trades": 0, "message": "No trades in period"}

        trades = [dict(r) for r in rows]
        pnls = [t["pnl"] for t in trades if t["pnl"] is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        return {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / max(1, len(wins) + len(losses)),
            "total_pnl": sum(pnls),
            "avg_win": sum(wins) / max(1, len(wins)),
            "avg_loss": sum(losses) / max(1, len(losses)) if losses else 0,
            "profit_factor": abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf"),
            "best_trade": max(pnls) if pnls else 0,
            "worst_trade": min(pnls) if pnls else 0,
        }

    def get_recent_decisions(self, limit: int = 20) -> list[dict]:
        """Get recent trading decisions."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM decisions_log ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
