"""TRADE - Autonomous Multi-Agent Trading System - Main Entry Point."""

from __future__ import annotations

import argparse
import json
import sys
import time

from loguru import logger
from rich.console import Console

from trade.config import load_config, EnvSettings, TradingConfig
from trade.agents.orchestrator import Orchestrator
from trade.data.models import AssetType
from trade.risk.circuit_breaker import CircuitBreaker
from trade.utils.logging import setup_logging

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRADE - Autonomous Multi-Agent Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  trade --symbol AAPL                    Analyze a single stock
  trade --symbol BTC-USD --type crypto   Analyze Bitcoin
  trade --watchlist                      Analyze all configured symbols
  trade --autonomous                     Run in autonomous mode
  trade --backtest AAPL --start 2024-01-01 --end 2024-12-31
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
    parser.add_argument("--backtest", type=str, help="Run backtest for symbol")
    parser.add_argument("--start", type=str, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    return parser.parse_args()


def run_single_analysis(orchestrator: Orchestrator, symbol: str, asset_type: AssetType | None, as_json: bool) -> None:
    """Run analysis on a single symbol."""
    if asset_type is None:
        if symbol.endswith("=X"):
            asset_type = AssetType.FOREX
        elif symbol.endswith("-USD"):
            asset_type = AssetType.CRYPTO
        else:
            asset_type = AssetType.STOCK

    result = orchestrator.analyze_symbol(symbol, asset_type)

    if as_json:
        # Remove non-serializable data
        clean = {k: v for k, v in result.items() if k != "historical_df"}
        print(json.dumps(clean, indent=2, default=str))
    else:
        orchestrator.print_summary([result])


def run_watchlist(orchestrator: Orchestrator, as_json: bool) -> None:
    """Run analysis on the full watchlist."""
    results = orchestrator.analyze_watchlist()

    if as_json:
        print(json.dumps(results, indent=2, default=str))
    else:
        orchestrator.print_summary(results)


def run_autonomous(orchestrator: Orchestrator, config: TradingConfig) -> None:
    """Run in autonomous mode - continuously analyze and trade."""
    interval = config.autonomous.scan_interval_seconds
    circuit_breaker = CircuitBreaker(config.risk)

    console.print(f"\n[bold red]{'='*60}[/bold red]")
    console.print("[bold red]  AUTONOMOUS TRADING MODE ACTIVE[/bold red]")
    console.print(f"[bold red]  Mode: {config.mode.upper()}[/bold red]")
    console.print(f"[bold red]  Scan interval: {interval}s[/bold red]")
    console.print(f"[bold red]{'='*60}[/bold red]\n")

    if config.mode == "live" and config.autonomous.require_confirmation_for_live:
        console.print("[bold red]WARNING: Live trading requires explicit confirmation![/bold red]")
        confirm = input("Type 'CONFIRM LIVE TRADING' to proceed: ")
        if confirm != "CONFIRM LIVE TRADING":
            console.print("Aborted. Switching to paper mode.")
            config.general["mode"] = "paper"

    cycle = 0
    while True:
        cycle += 1
        logger.info(f"=== Autonomous cycle {cycle} ===")

        # Check circuit breaker
        if not circuit_breaker.check(orchestrator.portfolio):
            console.print("[red]Circuit breaker active - trading paused[/red]")
            console.print(f"  Status: {circuit_breaker.status}")
            time.sleep(interval)
            continue

        try:
            results = orchestrator.analyze_watchlist()
            orchestrator.print_summary(results)
        except KeyboardInterrupt:
            console.print("\n[yellow]Autonomous mode stopped by user[/yellow]")
            break
        except Exception as e:
            logger.error(f"Error in autonomous cycle: {e}")

        try:
            logger.info(f"Sleeping {interval}s until next cycle...")
            time.sleep(interval)
        except KeyboardInterrupt:
            console.print("\n[yellow]Autonomous mode stopped by user[/yellow]")
            break


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override mode if specified
    if args.mode:
        config.general["mode"] = args.mode

    # Setup logging
    setup_logging(config.logging)

    console.print("[bold cyan]TRADE[/bold cyan] - Autonomous Multi-Agent Trading System v0.1.0")
    console.print(f"Mode: [{'green' if config.is_paper else 'red'}]{config.mode}[/{'green' if config.is_paper else 'red'}]")

    # Create orchestrator
    orchestrator = Orchestrator(config)

    if args.backtest:
        from trade.backtesting.engine import BacktestEngine
        engine = BacktestEngine(config)
        result = engine.run(
            args.backtest,
            start_date=args.start or "2024-01-01",
            end_date=args.end or "2024-12-31",
        )
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            console.print(f"\nBacktest result: {json.dumps(result['portfolio'], indent=2)}")

    elif args.autonomous:
        run_autonomous(orchestrator, config)

    elif args.watchlist:
        run_watchlist(orchestrator, args.json)

    elif args.symbol:
        asset_type = AssetType(args.type) if args.type else None
        run_single_analysis(orchestrator, args.symbol, asset_type, args.json)

    else:
        # Default: analyze a demo symbol
        console.print("\nNo symbol specified. Running demo analysis on AAPL...")
        run_single_analysis(orchestrator, "AAPL", AssetType.STOCK, args.json)


if __name__ == "__main__":
    main()
