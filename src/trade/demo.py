"""Demo Learning Mode - Run the bot in observation mode, log everything, learn.

This mode:
- Runs the full analysis pipeline on a schedule
- Does NOT execute any trades (observation only)
- Logs every decision to SQLite for later analysis
- Shows rich terminal output so you can watch the bot "think"
- Generates reports on what the bot WOULD have done

Usage:
    python -m trade.main --demo
    python -m trade.main --demo --interval 15  # Every 15 minutes
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trade.agents.orchestrator import Orchestrator
from trade.cache.store import CacheStore
from trade.config import load_config, TradingConfig
from trade.data.models import AssetType
from trade.knowledge.market_sessions import (
    get_current_session,
    get_active_kill_zone,
    is_optimal_trading_time,
)
from trade.risk.prop_firm import PropFirmRiskEngine, load_prop_firm_config
from trade.utils.logging import setup_logging

console = Console()


class DemoRunner:
    """Runs the trading bot in observation/learning mode."""

    def __init__(
        self,
        config: TradingConfig | None = None,
        prop_firm: str = "funderpro_classic_10k",
        interval_minutes: int = 15,
        symbols: list[str] | None = None,
    ):
        self.config = config or load_config()
        self.interval = interval_minutes * 60
        self.symbols = symbols or ["EURUSD=X", "GBPUSD=X", "XAUUSD=X"]

        # Initialize components
        self.cache = CacheStore()
        prop_config = load_prop_firm_config(prop_firm)
        self.risk_engine = PropFirmRiskEngine(prop_config)

        # Force paper mode
        self.config.general["mode"] = "paper"

        self.orchestrator = Orchestrator(
            config=self.config,
            risk_engine=self.risk_engine,
            cache=self.cache,
        )

        self._cycle_count = 0
        self._total_would_have_traded = 0

    def run(self) -> None:
        """Run the demo loop."""
        console.print(Panel.fit(
            "[bold cyan]TRADE - Demo Learning Mode[/bold cyan]\n"
            f"Symbols: {', '.join(self.symbols)}\n"
            f"Interval: {self.interval // 60} minutes\n"
            f"Mode: OBSERVATION ONLY (no real trades)\n"
            f"Logging to: {self.cache.db_path}",
            title="Demo Mode Active",
            border_style="cyan",
        ))

        while True:
            try:
                self._run_cycle()
            except KeyboardInterrupt:
                console.print("\n[yellow]Demo mode stopped by user.[/yellow]")
                self._print_session_summary()
                break
            except Exception as e:
                logger.error(f"Error in demo cycle: {e}")

            try:
                self._wait_for_next_cycle()
            except KeyboardInterrupt:
                console.print("\n[yellow]Demo mode stopped by user.[/yellow]")
                self._print_session_summary()
                break

    def run_once(self) -> list[dict]:
        """Run a single analysis cycle (useful for testing)."""
        return self._run_cycle()

    def _run_cycle(self) -> list[dict]:
        """Run one analysis cycle."""
        self._cycle_count += 1
        now = datetime.now(timezone.utc)

        console.print(f"\n{'='*60}")
        console.print(
            f"[bold]Cycle #{self._cycle_count}[/bold] | "
            f"{now.strftime('%Y-%m-%d %H:%M UTC')}"
        )

        # Show session info
        session = get_current_session(now)
        kill_zone = get_active_kill_zone(now)
        is_optimal, reason = is_optimal_trading_time(now)

        session_str = session.name.value if session else "CLOSED"
        kz_str = kill_zone.name if kill_zone else "None"

        color = "green" if is_optimal else "yellow" if session else "red"
        console.print(f"[{color}]Session: {session_str} | Kill Zone: {kz_str}[/{color}]")
        console.print(f"[dim]{reason}[/dim]")

        # Run analysis for each symbol
        results = []
        for symbol in self.symbols:
            asset_type = self._detect_type(symbol)
            result = self.orchestrator.analyze_symbol(symbol, asset_type)
            results.append(result)

            decision = result["final_decision"]
            if decision != "hold":
                self._total_would_have_traded += 1

        # Show results
        self._print_cycle_results(results)

        # Save portfolio snapshot
        portfolio = self.orchestrator.portfolio
        self.cache.save_portfolio_snapshot(
            cash=portfolio.cash,
            total_value=portfolio.total_value,
            daily_pnl=portfolio.daily_pnl,
            drawdown_pct=portfolio.max_drawdown_pct,
        )

        return results

    def _print_cycle_results(self, results: list[dict]) -> None:
        """Print results for this cycle."""
        table = Table(title=None, show_header=True, header_style="bold")
        table.add_column("Symbol", style="cyan", width=12)
        table.add_column("Decision", width=10)
        table.add_column("Agents Summary", width=40)
        table.add_column("Time", width=8)

        for result in results:
            decision = result["final_decision"]
            d_style = "green bold" if "buy" in decision else "red bold" if "sell" in decision else "dim"

            # Summarize agent opinions
            agent_summary = []
            for a in result.get("agents", []):
                emoji = "+" if "buy" in a["action"] else "-" if "sell" in a["action"] else "="
                agent_summary.append(f"{a['name'][:4]}:{emoji}")

            skip_reason = result.get("skip_reason", "")
            if skip_reason:
                agent_summary = [f"SKIPPED: {skip_reason[:30]}"]

            table.add_row(
                result["symbol"],
                f"[{d_style}]{decision.upper()}[/{d_style}]",
                " | ".join(agent_summary),
                f"{result['total_time_ms']:.0f}ms",
            )

        console.print(table)

    def _wait_for_next_cycle(self) -> None:
        """Wait for the next cycle with countdown."""
        remaining = self.interval
        while remaining > 0:
            mins = remaining // 60
            secs = remaining % 60
            console.print(
                f"\r[dim]Next cycle in {mins}m {secs}s... "
                f"(Ctrl+C to stop)[/dim]",
                end="",
            )
            time.sleep(min(10, remaining))
            remaining -= 10
        console.print()

    def _print_session_summary(self) -> None:
        """Print summary of the demo session."""
        console.print(f"\n{'='*60}")
        console.print("[bold]Demo Session Summary[/bold]")
        console.print(f"Total cycles: {self._cycle_count}")
        console.print(f"Would-have-traded signals: {self._total_would_have_traded}")

        stats = self.cache.get_trade_stats(days=1)
        decisions = self.cache.get_recent_decisions(limit=50)

        if decisions:
            buy_count = sum(1 for d in decisions if d["action"] == "buy")
            sell_count = sum(1 for d in decisions if d["action"] == "sell")
            hold_count = sum(1 for d in decisions if d["action"] == "hold")

            console.print(f"Decisions: BUY={buy_count} | SELL={sell_count} | HOLD={hold_count}")

        console.print(f"\nFull logs saved to: {self.cache.db_path}")
        console.print("Run [bold]python -m trade.main --report[/bold] to analyze")

    @staticmethod
    def _detect_type(symbol: str) -> AssetType:
        if symbol.endswith("=X"):
            return AssetType.FOREX
        if symbol.endswith("-USD"):
            return AssetType.CRYPTO
        if symbol in ("US30", "NAS100", "SPX500"):
            return AssetType.STOCK
        return AssetType.FOREX
