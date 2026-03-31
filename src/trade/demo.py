"""Demo Learning Mode - 24/7 market scanning with session-aware scheduling.

How it works:
1. Scanner checks which markets are OPEN right now
2. Quick scan 20-30 instruments (1-2 sec each) to find setups
3. Deep analyze top 3 candidates with full agent pipeline
4. Log everything to SQLite
5. Repeat on schedule, adapting to market hours

Usage:
    python -m trade.main --demo                    # Default: scan + analyze every 5 min
    python -m trade.main --demo --interval 1       # Every 1 minute
    python -m trade.main --demo --symbol EURUSD=X  # Only analyze one symbol
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
from trade.scanner import MarketScanner

console = Console()


class DemoRunner:
    """Runs the trading bot in 24/7 observation/learning mode with intelligent scanning."""

    def __init__(
        self,
        config: TradingConfig | None = None,
        prop_firm: str = "funderpro_classic_10k",
        interval_minutes: int = 5,
        symbols: list[str] | None = None,
        top_n: int = 3,
    ):
        self.config = config or load_config()
        self.interval = interval_minutes * 60
        self.fixed_symbols = symbols  # None = use scanner, list = fixed symbols
        self.top_n = top_n

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

        self.scanner = MarketScanner(top_n=top_n)

        self._cycle_count = 0
        self._total_signals = {"buy": 0, "sell": 0, "hold": 0}
        self._total_scanned = 0

    def run(self) -> None:
        """Run the 24/7 demo loop."""
        console.print(Panel.fit(
            "[bold cyan]TRADE - 24/7 Demo Learning Mode[/bold cyan]\n"
            f"Mode: {'Fixed symbols: ' + ', '.join(self.fixed_symbols) if self.fixed_symbols else f'Scanner (top {self.top_n} of 30+ instruments)'}\n"
            f"Interval: {self.interval // 60} minutes\n"
            f"Trading: OBSERVATION ONLY (no real trades)\n"
            f"Logging to: {self.cache.db_path}",
            title="24/7 Scanner Active",
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
        """Run a single cycle (useful for testing)."""
        return self._run_cycle()

    def _run_cycle(self) -> list[dict]:
        """Run one full scan + analysis cycle."""
        self._cycle_count += 1
        now = datetime.now(timezone.utc)

        console.print(f"\n{'='*70}")
        console.print(
            f"[bold]Cycle #{self._cycle_count}[/bold] | "
            f"{now.strftime('%Y-%m-%d %H:%M UTC')} | "
            f"({now.strftime('%H:%M')} hora española = UTC+1/+2)"
        )

        # Show session info
        session = get_current_session(now)
        kill_zone = get_active_kill_zone(now)
        is_optimal, reason = is_optimal_trading_time(now)

        session_str = session.name.value if session else "CLOSED"
        kz_str = kill_zone.name if kill_zone else "None"

        color = "green bold" if kill_zone else "green" if is_optimal else "yellow" if session else "red"
        console.print(f"[{color}]Session: {session_str} | Kill Zone: {kz_str}[/{color}]")

        if self.fixed_symbols:
            # Fixed symbol mode
            return self._analyze_fixed_symbols()
        else:
            # Scanner mode
            return self._scan_and_analyze()

    def _scan_and_analyze(self) -> list[dict]:
        """Phase 1: Scan all markets. Phase 2: Deep analyze top candidates."""
        now = datetime.now(timezone.utc)

        # PHASE 1: Quick scan all open markets
        active_instruments = self.scanner.get_active_instruments(now)
        if not active_instruments:
            console.print("[dim]No markets currently open. Waiting...[/dim]")
            return []

        console.print(f"[dim]Scanning {len(active_instruments)} open instruments...[/dim]")
        scan_results = self.scanner.quick_scan(active_instruments, now)
        self._total_scanned += len(scan_results)

        # Show scan summary
        self.scanner.print_scan_results(scan_results)

        # PHASE 2: Get top candidates for deep analysis
        top_candidates = self.scanner.get_top_candidates(active_instruments, now)

        if not top_candidates:
            console.print("[yellow]No strong setups found. All markets flat or weak.[/yellow]")
            return []

        console.print(f"\n[bold]Deep analyzing top {len(top_candidates)} candidates:[/bold]")

        # Run full agent pipeline on top candidates
        results = []
        for candidate in top_candidates:
            asset_type = self._detect_type(candidate.symbol, candidate.asset_class)
            result = self.orchestrator.analyze_symbol(candidate.symbol, asset_type)
            results.append(result)

            # Track signal counts
            decision = result["final_decision"]
            self._total_signals[decision] = self._total_signals.get(decision, 0) + 1

        # Print results table
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

    def _analyze_fixed_symbols(self) -> list[dict]:
        """Analyze fixed list of symbols (no scanning)."""
        results = []
        for symbol in self.fixed_symbols:
            asset_type = self._detect_type(symbol)
            result = self.orchestrator.analyze_symbol(symbol, asset_type)
            results.append(result)

            decision = result["final_decision"]
            self._total_signals[decision] = self._total_signals.get(decision, 0) + 1

        self._print_cycle_results(results)
        return results

    def _print_cycle_results(self, results: list[dict]) -> None:
        """Print results for this cycle."""
        table = Table(show_header=True, header_style="bold")
        table.add_column("Symbol", style="cyan", width=12)
        table.add_column("Decision", width=10)
        table.add_column("Agents", width=45)
        table.add_column("SMC", width=15)
        table.add_column("Time", width=8)

        for result in results:
            decision = result["final_decision"]
            d_style = "green bold" if "buy" in decision else "red bold" if "sell" in decision else "dim"

            # Agent summary
            agent_summary = []
            for a in result.get("agents", []):
                icon = "+" if "buy" in a["action"] else "-" if "sell" in a["action"] else "="
                agent_summary.append(f"{a['name'][:4]}:{icon}")

            # SMC info from the analysis
            smc_info = ""
            for a in result.get("agents", []):
                if a.get("name") == "smc":
                    smc_info = a.get("reasoning", "")[:15]

            skip_reason = result.get("skip_reason", "")
            if skip_reason:
                agent_summary = [f"SKIP: {skip_reason[:35]}"]

            table.add_row(
                result["symbol"],
                f"[{d_style}]{decision.upper()}[/{d_style}]",
                " | ".join(agent_summary),
                smc_info or "[dim]n/a[/dim]",
                f"{result['total_time_ms']:.0f}ms",
            )

        console.print(table)

        # Show prop firm challenge progress
        if self.risk_engine:
            progress = self.risk_engine.get_challenge_progress(
                self.orchestrator.portfolio.total_value
            )
            status = progress.get("safety_status", "?")
            status_color = {
                "SAFE": "green", "CAUTION": "yellow",
                "DANGER": "red", "CRITICAL": "red bold", "DEAD": "red bold",
            }.get(status, "white")

            console.print(
                f"[bold]Challenge:[/bold] {progress.get('progress_pct', 0):.1f}% "
                f"| Equity: ${progress.get('current_equity', 0):,.2f} "
                f"| DD: {progress.get('total_drawdown_pct', 0):.2f}% "
                f"| Status: [{status_color}]{status}[/{status_color}] "
                f"| Trades today: {progress.get('trades_today', 0)}"
            )

    def _wait_for_next_cycle(self) -> None:
        """Wait for next cycle with countdown."""
        remaining = self.interval
        while remaining > 0:
            mins = remaining // 60
            secs = remaining % 60
            now = datetime.now(timezone.utc)
            console.print(
                f"\r[dim]Next scan in {mins}m {secs}s | "
                f"Signals: BUY={self._total_signals.get('buy', 0)} "
                f"SELL={self._total_signals.get('sell', 0)} "
                f"HOLD={self._total_signals.get('hold', 0)} | "
                f"Scanned: {self._total_scanned} total | "
                f"(Ctrl+C to stop)[/dim]",
                end="",
            )
            time.sleep(min(10, remaining))
            remaining -= 10
        console.print()

    def _print_session_summary(self) -> None:
        """Print end-of-session summary."""
        console.print(f"\n{'='*70}")
        console.print("[bold]Demo Session Summary[/bold]")
        console.print(f"Cycles completed: {self._cycle_count}")
        console.print(f"Instruments scanned: {self._total_scanned}")
        console.print(
            f"Signals: BUY={self._total_signals.get('buy', 0)} | "
            f"SELL={self._total_signals.get('sell', 0)} | "
            f"HOLD={self._total_signals.get('hold', 0)}"
        )

        decisions = self.cache.get_recent_decisions(limit=100)
        if decisions:
            executed = sum(1 for d in decisions if d.get("executed"))
            console.print(f"Would-have-traded: {executed}")

        console.print(f"\nLogs saved to: [bold]{self.cache.db_path}[/bold]")
        console.print("Run [bold]python -m trade.main --report[/bold] to analyze")

    @staticmethod
    def _detect_type(symbol: str, asset_class: str = "") -> AssetType:
        if asset_class == "crypto" or symbol.endswith("-USD"):
            return AssetType.CRYPTO
        if asset_class == "index" or symbol.startswith("^"):
            return AssetType.STOCK
        return AssetType.FOREX
