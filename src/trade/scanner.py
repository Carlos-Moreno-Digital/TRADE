"""Intelligent Market Scanner - Screens 30+ instruments, deep analyzes the best.

Two-phase approach:
1. QUICK SCAN: Check all instruments for basic setup (price trend, momentum, volume)
   Takes ~1-2 seconds per instrument using cached data
2. DEEP ANALYSIS: Run full agent pipeline only on the top 3-5 candidates
   Takes ~5-10 seconds per instrument (indicators, sentiment, SMC, risk)

This prevents wasting API calls and compute on instruments with no setup.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from rich.console import Console
from rich.table import Table

from trade.data.providers import MarketDataProvider
from trade.knowledge.market_sessions import get_current_session, Session

console = Console()


# =========================================================================
# FULL INSTRUMENT UNIVERSE
# =========================================================================

FOREX_PAIRS = [
    # Majors
    ("EURUSD=X", "forex", "EUR/USD"),
    ("GBPUSD=X", "forex", "GBP/USD"),
    ("USDJPY=X", "forex", "USD/JPY"),
    ("USDCHF=X", "forex", "USD/CHF"),
    ("AUDUSD=X", "forex", "AUD/USD"),
    ("NZDUSD=X", "forex", "NZD/USD"),
    ("USDCAD=X", "forex", "USD/CAD"),
    # Crosses
    ("EURGBP=X", "forex", "EUR/GBP"),
    ("EURJPY=X", "forex", "EUR/JPY"),
    ("GBPJPY=X", "forex", "GBP/JPY"),
    ("AUDJPY=X", "forex", "AUD/JPY"),
    ("EURAUD=X", "forex", "EUR/AUD"),
    ("GBPAUD=X", "forex", "GBP/AUD"),
    ("EURNZD=X", "forex", "EUR/NZD"),
    ("GBPNZD=X", "forex", "GBP/NZD"),
    ("CADCHF=X", "forex", "CAD/CHF"),
]

COMMODITIES = [
    ("GC=F", "commodity", "Gold"),
    ("SI=F", "commodity", "Silver"),
    ("CL=F", "commodity", "Oil WTI"),
]

INDICES = [
    ("^DJI", "index", "Dow Jones"),
    ("^IXIC", "index", "Nasdaq"),
    ("^GSPC", "index", "S&P 500"),
    ("^GDAXI", "index", "DAX"),
    ("^FTSE", "index", "FTSE 100"),
]

CRYPTO = [
    ("BTC-USD", "crypto", "Bitcoin"),
    ("ETH-USD", "crypto", "Ethereum"),
    ("SOL-USD", "crypto", "Solana"),
]

# Market hours (UTC) - None means 24 hours
MARKET_HOURS: dict[str, dict[str, Any]] = {
    "forex": {
        "open_utc": (22, 0),   # Sunday 22:00 UTC
        "close_utc": (22, 0),  # Friday 22:00 UTC
        "days": [0, 1, 2, 3, 4],  # Mon-Fri (Sun evening open handled separately)
        "is_24h": True,
        "note": "24/5 - closed weekends",
    },
    "commodity": {
        "open_utc": (22, 0),
        "close_utc": (21, 0),
        "days": [0, 1, 2, 3, 4],
        "is_24h": True,  # Nearly 24h with short breaks
        "note": "Nearly 24/5 with 1h daily break",
    },
    "index": {
        "open_utc": (14, 30),  # US market open
        "close_utc": (21, 0),  # US market close
        "days": [0, 1, 2, 3, 4],
        "is_24h": False,
        "note": "US hours 14:30-21:00 UTC (futures trade longer)",
    },
    "crypto": {
        "open_utc": (0, 0),
        "close_utc": (23, 59),
        "days": [0, 1, 2, 3, 4, 5, 6],  # 24/7
        "is_24h": True,
        "note": "24/7 - never closes",
    },
}


class ScanResult:
    """Result of a quick scan for one instrument."""

    def __init__(
        self,
        symbol: str,
        asset_class: str,
        display_name: str,
        score: float = 0.0,
        trend: str = "neutral",
        momentum: float = 0.0,
        volume_ratio: float = 1.0,
        is_open: bool = True,
        reason: str = "",
    ):
        self.symbol = symbol
        self.asset_class = asset_class
        self.display_name = display_name
        self.score = score
        self.trend = trend
        self.momentum = momentum
        self.volume_ratio = volume_ratio
        self.is_open = is_open
        self.reason = reason


class MarketScanner:
    """Scans the full instrument universe and selects top candidates."""

    def __init__(self, provider: MarketDataProvider | None = None, top_n: int = 3):
        self.provider = provider or MarketDataProvider()
        self.top_n = top_n
        self._scan_cache: dict[str, ScanResult] = {}

    def get_active_instruments(self, now_utc: datetime | None = None) -> list[tuple[str, str, str]]:
        """Get instruments that are currently tradeable based on market hours."""
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)

        weekday = now_utc.weekday()
        hour = now_utc.hour

        active = []

        for instruments, asset_class in [
            (FOREX_PAIRS, "forex"),
            (COMMODITIES, "commodity"),
            (INDICES, "index"),
            (CRYPTO, "crypto"),
        ]:
            hours = MARKET_HOURS[asset_class]

            # Check day
            if weekday not in hours["days"]:
                continue

            # Check hours (for non-24h markets)
            if not hours["is_24h"]:
                open_h = hours["open_utc"][0]
                close_h = hours["close_utc"][0]
                if not (open_h <= hour < close_h):
                    continue

            # Weekend check for forex (closed Sat 22:00 - Sun 22:00 UTC)
            if asset_class == "forex" and weekday >= 5:
                if weekday == 5:  # Saturday - closed
                    continue
                if weekday == 6 and hour < 22:  # Sunday before 22:00 - closed
                    continue

            for sym, cls, name in instruments:
                active.append((sym, cls, name))

        return active

    def quick_scan(
        self,
        instruments: list[tuple[str, str, str]] | None = None,
        now_utc: datetime | None = None,
    ) -> list[ScanResult]:
        """Phase 1: Quick scan all instruments for basic setup quality.

        Uses 1-day change + 5-day change + volume to score each instrument.
        ~1-2 seconds per instrument (uses cache).
        """
        if instruments is None:
            instruments = self.get_active_instruments(now_utc)

        if not instruments:
            logger.info("No markets currently open")
            return []

        logger.info(f"Quick scanning {len(instruments)} instruments...")
        results: list[ScanResult] = []

        for symbol, asset_class, display_name in instruments:
            try:
                result = self._scan_one(symbol, asset_class, display_name)
                results.append(result)
            except Exception as e:
                logger.debug(f"Scan failed for {symbol}: {e}")

        # Sort by absolute score (strongest signal first, direction doesn't matter)
        results.sort(key=lambda r: abs(r.score), reverse=True)

        return results

    def get_top_candidates(
        self,
        instruments: list[tuple[str, str, str]] | None = None,
        now_utc: datetime | None = None,
    ) -> list[ScanResult]:
        """Get the top N candidates for deep analysis."""
        all_results = self.quick_scan(instruments, now_utc)

        # Filter: must have some directional bias (not flat)
        candidates = [r for r in all_results if abs(r.score) > 0.1 and r.is_open]

        top = candidates[:self.top_n]

        if top:
            logger.info(
                f"Top {len(top)} candidates: " +
                ", ".join(f"{r.display_name} ({r.score:+.2f})" for r in top)
            )
        else:
            logger.info("No strong candidates found in scan")

        return top

    def print_scan_results(self, results: list[ScanResult]) -> None:
        """Print scan results as a rich table."""
        table = Table(title=f"Market Scan ({len(results)} instruments)")
        table.add_column("#", width=3)
        table.add_column("Symbol", style="cyan", width=12)
        table.add_column("Name", width=12)
        table.add_column("Class", width=8)
        table.add_column("Trend", width=8)
        table.add_column("Score", width=8)
        table.add_column("Momentum", width=10)
        table.add_column("Volume", width=8)
        table.add_column("Open", width=5)

        for i, r in enumerate(results[:20], 1):  # Show top 20
            score_color = "green" if r.score > 0.2 else "red" if r.score < -0.2 else "dim"
            trend_color = "green" if "bull" in r.trend else "red" if "bear" in r.trend else "dim"
            open_str = "[green]YES[/green]" if r.is_open else "[red]NO[/red]"

            table.add_row(
                str(i),
                r.symbol,
                r.display_name,
                r.asset_class,
                f"[{trend_color}]{r.trend}[/{trend_color}]",
                f"[{score_color}]{r.score:+.2f}[/{score_color}]",
                f"{r.momentum:+.2f}%",
                f"{r.volume_ratio:.1f}x",
                open_str,
            )

        console.print(table)

    def _scan_one(self, symbol: str, asset_class: str, display_name: str) -> ScanResult:
        """Quick scan a single instrument."""
        try:
            df = self.provider.get_historical(symbol, period="1mo", interval="1d")
        except Exception:
            return ScanResult(symbol, asset_class, display_name, is_open=False, reason="No data")

        if df.empty or len(df) < 5:
            return ScanResult(symbol, asset_class, display_name, score=0, reason="Insufficient data")

        close = df["close"].values
        latest = float(close[-1])
        prev = float(close[-2])

        # 1-day change
        change_1d = (latest - prev) / prev * 100 if prev > 0 else 0

        # 5-day change (momentum)
        if len(df) >= 5:
            price_5d = float(close[-5])
            momentum = (latest - price_5d) / price_5d * 100
        else:
            momentum = change_1d

        # Volume ratio
        if "volume" in df.columns:
            avg_vol = float(df["volume"].tail(20).mean())
            latest_vol = float(df["volume"].iloc[-1])
            volume_ratio = latest_vol / avg_vol if avg_vol > 0 else 1.0
        else:
            volume_ratio = 1.0

        # Trend detection
        if len(df) >= 20:
            sma20 = float(df["close"].tail(20).mean())
            trend = "bullish" if latest > sma20 else "bearish"
        else:
            trend = "bullish" if momentum > 0 else "bearish" if momentum < 0 else "neutral"

        # Score: combines momentum strength + volume confirmation
        # Higher absolute score = better setup (direction doesn't matter)
        vol_bonus = min(1.5, volume_ratio) / 1.5  # 0 to 1.0
        score = momentum * (0.5 + 0.5 * vol_bonus)

        # Normalize to -1 to 1
        score = max(-1.0, min(1.0, score / 5.0))

        return ScanResult(
            symbol=symbol,
            asset_class=asset_class,
            display_name=display_name,
            score=round(score, 3),
            trend=trend,
            momentum=round(momentum, 2),
            volume_ratio=round(volume_ratio, 2),
            is_open=True,
            reason=f"{trend} | mom={momentum:+.2f}% | vol={volume_ratio:.1f}x",
        )
