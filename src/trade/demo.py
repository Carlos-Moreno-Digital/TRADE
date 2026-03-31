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
            # P2 FIX #12: Check if risk engine killed the account
            if self.risk_engine and self.risk_engine._is_killed:
                console.print(f"\n[red bold]{'='*70}[/red bold]")
                console.print(f"[red bold]ACCOUNT KILLED: {self.risk_engine._kill_reason}[/red bold]")
                console.print(f"[red bold]Bot stopped automatically. Manual reset required.[/red bold]")
                console.print(f"[red bold]{'='*70}[/red bold]")
                self._print_session_summary()
                break

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
        # Reuse scan_results to avoid double-scanning
        top_candidates = self.scanner.get_top_candidates(scan_results=scan_results)

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
        """Print results + full account dashboard."""

        # =====================================================================
        # 1. TRADE DECISIONS TABLE
        # =====================================================================
        table = Table(title="Trade Decisions", show_header=True, header_style="bold")
        table.add_column("Symbol", style="cyan", width=12)
        table.add_column("Decision", width=10)
        table.add_column("Agents", width=40)
        table.add_column("Confidence", width=10)
        table.add_column("Time", width=8)

        for result in results:
            decision = result["final_decision"]
            d_style = "green bold" if "buy" in decision else "red bold" if "sell" in decision else "dim"

            agent_summary = []
            for a in result.get("agents", []):
                icon = "+" if "buy" in a["action"] else "-" if "sell" in a["action"] else "="
                agent_summary.append(f"{a['name'][:4]}:{icon}")

            skip_reason = result.get("skip_reason", "")
            if skip_reason:
                agent_summary = [f"SKIP: {skip_reason[:30]}"]

            # Get portfolio manager confidence
            pm_conf = 0.0
            for a in result.get("agents", []):
                if a.get("name") == "portfolio_manager":
                    pm_conf = a.get("confidence", 0)

            table.add_row(
                result["symbol"],
                f"[{d_style}]{decision.upper()}[/{d_style}]",
                " | ".join(agent_summary),
                f"{pm_conf:.0%}",
                f"{result['total_time_ms']:.0f}ms",
            )

        console.print(table)

        # =====================================================================
        # 2. ACCOUNT DASHBOARD (the $10K panel)
        # =====================================================================
        portfolio = self.orchestrator.portfolio
        prop = self.risk_engine

        # Challenge progress
        progress = prop.get_challenge_progress(portfolio.total_value) if prop else {}
        status = progress.get("safety_status", "?")
        status_color = {
            "SAFE": "green", "CAUTION": "yellow",
            "DANGER": "red", "CRITICAL": "red bold", "DEAD": "red bold",
        }.get(status, "white")

        initial = progress.get("initial_balance", 10000)
        equity = portfolio.total_value
        profit = equity - initial
        profit_pct = (profit / initial * 100) if initial > 0 else 0
        target_pct = progress.get("target_pct", 10)
        target_amount = initial * target_pct / 100
        remaining = target_amount - profit
        progress_pct = progress.get("progress_pct", 0)

        # Build account panel
        lines = []
        lines.append(f"[bold]CUENTA FUNDERPRO $10K CLASSIC[/bold]")
        lines.append(f"")

        # Balance
        profit_color = "green" if profit >= 0 else "red"
        lines.append(f"  Capital inicial:  ${initial:>10,.2f}")
        lines.append(f"  Equity actual:    ${equity:>10,.2f}  [{profit_color}]({profit:+,.2f} / {profit_pct:+.2f}%)[/{profit_color}]")
        lines.append(f"  Cash disponible:  ${portfolio.cash:>10,.2f}")
        lines.append(f"")

        # Progress toward target
        bar_len = 30
        filled = int(bar_len * min(1, progress_pct / 100))
        bar = "█" * filled + "░" * (bar_len - filled)
        lines.append(f"  Objetivo: ${target_amount:,.0f} ({target_pct}%)  |  Progreso: {progress_pct:.1f}%")
        lines.append(f"  [{profit_color}]{bar}[/{profit_color}]  Faltan: ${max(0, remaining):,.2f}")
        lines.append(f"")

        # Risk limits
        daily_pnl = portfolio.daily_pnl
        daily_pnl_color = "green" if daily_pnl >= 0 else "red"
        dd = progress.get("total_drawdown_pct", 0)
        daily_loss_pct = progress.get("daily_loss_pct", 0)

        lines.append(f"  [bold]LIMITES DE RIESGO[/bold]")
        lines.append(f"  P&L del dia:     [{daily_pnl_color}]${daily_pnl:>+10,.2f}[/{daily_pnl_color}]  (limite: -${initial * 0.05:,.0f} = -5%)")
        lines.append(f"  Drawdown total:   {dd:>6.2f}%     (limite: 10% = -${initial * 0.10:,.0f})")
        lines.append(f"  Trades hoy:       {progress.get('trades_today', 0):>6d}      (limite: {prop.config.max_trades_per_day if prop else 5})")
        lines.append(f"  Perdidas seguidas:{portfolio.consecutive_losses:>6d}      (limite: {prop.config.consecutive_loss_threshold if prop else 2})")
        lines.append(f"  Estado:           [{status_color}]{status:>6s}[/{status_color}]")
        lines.append(f"")

        # Open positions
        if portfolio.positions:
            lines.append(f"  [bold]POSICIONES ABIERTAS ({len(portfolio.positions)})[/bold]")
            for pos in portfolio.positions:
                pnl = pos.unrealized_pnl
                pnl_pct = pos.unrealized_pnl_pct
                pnl_color = "green" if pnl >= 0 else "red"
                side_icon = "🔼" if pos.side == "long" else "🔽"
                sl_str = f"SL:{pos.stop_loss:.5f}" if pos.stop_loss else "SL:---"
                tp_str = f"TP:{pos.take_profit:.5f}" if pos.take_profit else "TP:---"
                lines.append(
                    f"  {side_icon} {pos.symbol:<12s} {pos.side.upper():<5s} "
                    f"x{pos.quantity:<6.2f} @ {pos.entry_price:.5f}  "
                    f"[{pnl_color}]P&L: ${pnl:>+8,.2f} ({pnl_pct:>+.1f}%)[/{pnl_color}]  "
                    f"{sl_str}  {tp_str}"
                )
        else:
            lines.append(f"  [dim]Sin posiciones abiertas[/dim]")

        console.print(Panel(
            "\n".join(lines),
            title=f"[bold]Dashboard Cuenta ${initial:,.0f}[/bold]",
            border_style=status_color,
            width=90,
        ))

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
