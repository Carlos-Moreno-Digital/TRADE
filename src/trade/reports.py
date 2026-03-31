"""Report generator - Analyze demo/live trading decisions from SQLite."""

from __future__ import annotations

from collections import Counter
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trade.cache.store import CacheStore

console = Console()


class ReportGenerator:
    """Generates analysis reports from cached trading decisions."""

    def __init__(self, cache: CacheStore | None = None):
        self.cache = cache or CacheStore()

    def print_report(self, days: int = 7) -> None:
        """Print a comprehensive report to the terminal."""
        console.print(Panel.fit(
            f"[bold]Trading Analysis Report - Last {days} days[/bold]",
            border_style="cyan",
        ))

        decisions = self.cache.get_recent_decisions(limit=500)
        trade_stats = self.cache.get_trade_stats(days=days)

        if not decisions:
            console.print("[yellow]No decisions logged yet. Run demo mode first.[/yellow]")
            return

        self._print_decision_summary(decisions)
        self._print_trade_stats(trade_stats)
        self._print_agent_accuracy(decisions)
        self._print_symbol_breakdown(decisions)
        self._print_session_analysis(decisions)
        self._print_risk_analysis(decisions)

    def _print_decision_summary(self, decisions: list[dict]) -> None:
        """Summary of all decisions."""
        actions = Counter(d["action"] for d in decisions)
        total = len(decisions)
        executed = sum(1 for d in decisions if d.get("executed"))

        console.print(f"\n[bold]Decision Summary[/bold] ({total} total)")
        console.print(
            f"  BUY: {actions.get('buy', 0)} | "
            f"SELL: {actions.get('sell', 0)} | "
            f"HOLD: {actions.get('hold', 0)}"
        )
        console.print(f"  Executed: {executed} | Observation only: {total - executed}")

    def _print_trade_stats(self, stats: dict) -> None:
        """P&L and performance stats."""
        if stats.get("total_trades", 0) == 0:
            console.print("\n[dim]No completed trades yet.[/dim]")
            return

        console.print(f"\n[bold]Trade Performance[/bold]")
        table = Table(show_header=False)
        table.add_column("Metric", style="bold")
        table.add_column("Value")

        table.add_row("Total Trades", str(stats["total_trades"]))
        table.add_row("Wins / Losses", f"{stats['wins']} / {stats['losses']}")

        wr = stats.get("win_rate", 0) * 100
        wr_color = "green" if wr >= 50 else "red"
        table.add_row("Win Rate", f"[{wr_color}]{wr:.1f}%[/{wr_color}]")

        pnl = stats.get("total_pnl", 0)
        pnl_color = "green" if pnl >= 0 else "red"
        table.add_row("Total P&L", f"[{pnl_color}]${pnl:,.2f}[/{pnl_color}]")

        table.add_row("Avg Win", f"${stats.get('avg_win', 0):,.2f}")
        table.add_row("Avg Loss", f"${stats.get('avg_loss', 0):,.2f}")

        pf = stats.get("profit_factor", 0)
        pf_str = f"{pf:.2f}" if pf < 999 else "∞"
        table.add_row("Profit Factor", pf_str)

        table.add_row("Best Trade", f"${stats.get('best_trade', 0):,.2f}")
        table.add_row("Worst Trade", f"${stats.get('worst_trade', 0):,.2f}")

        console.print(table)

    def _print_agent_accuracy(self, decisions: list[dict]) -> None:
        """Analyze which agents were most accurate."""
        console.print(f"\n[bold]Risk Engine Analysis[/bold]")

        vetoed = sum(1 for d in decisions if d.get("prop_firm_check") == "veto")
        passed = sum(1 for d in decisions if d.get("prop_firm_check") == "pass")

        console.print(f"  Risk checks passed: {passed}")
        console.print(f"  Risk vetoes (trades blocked): {vetoed}")

        if vetoed > 0:
            veto_pct = vetoed / (vetoed + passed) * 100
            console.print(f"  Veto rate: {veto_pct:.1f}%")

    def _print_symbol_breakdown(self, decisions: list[dict]) -> None:
        """Breakdown by symbol."""
        symbol_counts = Counter(d["symbol"] for d in decisions)

        if not symbol_counts:
            return

        console.print(f"\n[bold]Symbol Breakdown[/bold]")
        table = Table()
        table.add_column("Symbol")
        table.add_column("Decisions")
        table.add_column("Buy")
        table.add_column("Sell")
        table.add_column("Hold")

        for symbol, count in symbol_counts.most_common(10):
            symbol_decisions = [d for d in decisions if d["symbol"] == symbol]
            buys = sum(1 for d in symbol_decisions if d["action"] == "buy")
            sells = sum(1 for d in symbol_decisions if d["action"] == "sell")
            holds = sum(1 for d in symbol_decisions if d["action"] == "hold")
            table.add_row(symbol, str(count), str(buys), str(sells), str(holds))

        console.print(table)

    def _print_session_analysis(self, decisions: list[dict]) -> None:
        """Analyze decisions by session."""
        console.print(f"\n[bold]Session Analysis[/bold]")

        session_data = Counter()
        for d in decisions:
            notes = d.get("notes", "")
            if "Session:" in notes:
                session = notes.split("Session:")[1].split("|")[0].strip()
                session_data[session] += 1

        if session_data:
            for session, count in session_data.most_common():
                console.print(f"  {session}: {count} decisions")

    def _print_risk_analysis(self, decisions: list[dict]) -> None:
        """Risk metric analysis."""
        risk_scores = [d.get("risk_score", 0) for d in decisions if d.get("risk_score")]

        if risk_scores:
            avg_risk = sum(risk_scores) / len(risk_scores)
            max_risk = max(risk_scores)
            console.print(f"\n[bold]Risk Metrics[/bold]")
            console.print(f"  Avg risk score: {avg_risk:.3f}")
            console.print(f"  Max risk score: {max_risk:.3f}")

        confidences = [d.get("confidence", 0) for d in decisions if d.get("confidence")]
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            console.print(f"  Avg confidence: {avg_conf:.3f}")
