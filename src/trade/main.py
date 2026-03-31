"""TRADE - Autonomous Multi-Agent Trading System - Main Entry Point."""

from __future__ import annotations

import argparse
import json
import sys
import time

from loguru import logger
from rich.console import Console

from trade.config import load_config, TradingConfig
from trade.agents.orchestrator import Orchestrator
from trade.cache.store import CacheStore
from trade.data.models import AssetType
from trade.risk.prop_firm import PropFirmRiskEngine, load_prop_firm_config
from trade.utils.logging import setup_logging

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRADE - Autonomous Multi-Agent Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  trade --symbol EURUSD=X                Analyze EUR/USD
  trade --symbol XAUUSD=X --type forex   Analyze Gold
  trade --watchlist                       Analyze all configured symbols
  trade --demo                            Run in demo learning mode
  trade --demo --interval 15              Demo every 15 minutes
  trade --report                          Show analysis report
  trade --autonomous                      Run in autonomous mode
        """,
    )

    parser.add_argument("--symbol", "-s", type=str, help="Symbol to analyze")
    parser.add_argument(
        "--type", "-t", type=str, choices=["stock", "crypto", "forex"],
        default=None, help="Asset type"
    )
    parser.add_argument("--watchlist", "-w", action="store_true", help="Analyze full watchlist")
    parser.add_argument("--autonomous", "-a", action="store_true", help="Run in autonomous mode")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to config file")
    parser.add_argument(
        "--mode", "-m", type=str, choices=["paper", "live"], default=None,
        help="Trading mode (default: paper)"
    )
    parser.add_argument("--backtest", type=str, nargs="?", const="30", help="Run backtest (number of days, default 30)")
    parser.add_argument("--start", type=str, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Demo mode
    parser.add_argument("--demo", action="store_true", help="Run in demo learning mode (no trades)")
    parser.add_argument("--interval", type=int, default=15, help="Demo interval in minutes (default: 15)")
    parser.add_argument(
        "--prop-firm", type=str, default="funderpro_classic_10k",
        help="Prop firm config name (default: funderpro_classic_10k)"
    )

    # Scalp backtest
    parser.add_argument("--scalp", action="store_true", help="Run 5-minute scalping backtest")
    parser.add_argument("--scalp-validate", action="store_true", help="Validate scalp on 10 random dates from 2 years")

    # Scanner
    parser.add_argument("--top-n", type=int, default=3, help="Number of top candidates to deep analyze (default: 3)")

    # Reports
    parser.add_argument("--report", action="store_true", help="Show analysis report")
    parser.add_argument("--report-days", type=int, default=7, help="Report period in days (default: 7)")

    return parser.parse_args()


def run_single_analysis(orchestrator: Orchestrator, symbol: str, asset_type: AssetType | None, as_json: bool) -> None:
    if asset_type is None:
        if symbol.endswith("=X"):
            asset_type = AssetType.FOREX
        elif symbol.endswith("-USD"):
            asset_type = AssetType.CRYPTO
        else:
            asset_type = AssetType.STOCK

    result = orchestrator.analyze_symbol(symbol, asset_type)

    if as_json:
        clean = {k: v for k, v in result.items() if k != "historical_df"}
        print(json.dumps(clean, indent=2, default=str))
    else:
        orchestrator.print_summary([result])


def run_watchlist(orchestrator: Orchestrator, as_json: bool) -> None:
    results = orchestrator.analyze_watchlist()
    if as_json:
        print(json.dumps(results, indent=2, default=str))
    else:
        orchestrator.print_summary(results)


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Load config
    config = load_config(args.config)
    if args.mode:
        config.general["mode"] = args.mode

    setup_logging(config.logging)

    console.print("[bold cyan]TRADE[/bold cyan] - Autonomous Multi-Agent Trading System v0.1.0")
    console.print(f"Mode: [{'green' if config.is_paper else 'red'}]{config.mode}[/{'green' if config.is_paper else 'red'}]")

    # =========================================================================
    # DEMO MODE
    # =========================================================================
    if args.demo:
        from trade.demo import DemoRunner

        symbols = [args.symbol] if args.symbol else None
        runner = DemoRunner(
            config=config,
            prop_firm=args.prop_firm,
            interval_minutes=args.interval,
            symbols=symbols,
            top_n=args.top_n,
        )
        runner.run()
        return

    # =========================================================================
    # SCALP BACKTEST MODE
    # =========================================================================
    if args.scalp_validate:
        from trade.backtest_5min import ScalpBacktester
        bt = ScalpBacktester(account_size=10000)
        result = bt.run_random_samples(n_samples=10)
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        return

    if args.scalp:
        from trade.backtest_5min import ScalpBacktester
        bt = ScalpBacktester(account_size=10000)
        result = bt.run()
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        return

    # =========================================================================
    # REPORT MODE
    # =========================================================================
    if args.report:
        from trade.reports import ReportGenerator

        reporter = ReportGenerator()
        reporter.print_report(days=args.report_days)
        return

    # =========================================================================
    # ANALYSIS MODES
    # =========================================================================

    # Initialize with prop firm risk engine
    cache = CacheStore()
    prop_config = load_prop_firm_config(args.prop_firm)
    risk_engine = PropFirmRiskEngine(prop_config)

    orchestrator = Orchestrator(
        config=config,
        risk_engine=risk_engine,
        cache=cache,
    )

    if args.backtest:
        from trade.backtest import Backtester

        bt = Backtester(prop_firm=args.prop_firm, top_n=args.top_n)
        symbols = [args.symbol] if args.symbol else None
        days = int(args.backtest) if args.backtest.isdigit() else 30
        result = bt.run(days=days, symbols=symbols)

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        return

    elif args.autonomous:
        _run_autonomous(orchestrator, config)

    elif args.watchlist:
        run_watchlist(orchestrator, args.json)

    elif args.symbol:
        asset_type = AssetType(args.type) if args.type else None
        run_single_analysis(orchestrator, args.symbol, asset_type, args.json)

    else:
        console.print("\nNo action specified. Use --help for options.")
        console.print("  Quick start: [bold]python -m trade.main --demo[/bold]")


def _run_autonomous(orchestrator: Orchestrator, config: TradingConfig) -> None:
    """Run in autonomous mode."""
    interval = config.autonomous.scan_interval_seconds

    console.print(f"\n[bold red]{'='*60}[/bold red]")
    console.print("[bold red]  AUTONOMOUS TRADING MODE[/bold red]")
    console.print(f"[bold red]  Mode: {config.mode.upper()}[/bold red]")
    console.print(f"[bold red]{'='*60}[/bold red]\n")

    if config.mode == "live":
        confirm = input("Type 'CONFIRM LIVE TRADING' to proceed: ")
        if confirm != "CONFIRM LIVE TRADING":
            console.print("Aborted.")
            return

    while True:
        try:
            results = orchestrator.analyze_watchlist()
            orchestrator.print_summary(results)
            time.sleep(interval)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped by user[/yellow]")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(30)


if __name__ == "__main__":
    main()
