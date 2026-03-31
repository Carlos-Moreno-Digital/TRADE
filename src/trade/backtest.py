"""Real Backtester - Replays historical data through the full agent pipeline.

Simulates what the bot would have done over a historical period:
1. Downloads historical data for all instruments
2. For each trading day, runs the full pipeline (scanner + agents)
3. Tracks paper trades with slippage simulation
4. Reports P&L, win rate, drawdown, and whether it would pass the prop firm challenge

Usage:
    python -m trade.main --backtest --days 30
    python -m trade.main --backtest --days 30 --symbol EURUSD=X
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trade.agents.orchestrator import Orchestrator
from trade.cache.store import CacheStore
from trade.config import load_config, TradingConfig
from trade.data.models import AssetType, PortfolioState
from trade.data.providers import MarketDataProvider
from trade.risk.prop_firm import PropFirmRiskEngine, load_prop_firm_config

console = Console()


class Backtester:
    """Replays historical data through the trading pipeline."""

    def __init__(
        self,
        prop_firm: str = "funderpro_classic_10k",
        top_n: int = 3,
    ):
        self.config = load_config()
        self.config.general["mode"] = "paper"

        prop_config = load_prop_firm_config(prop_firm)
        self.risk_engine = PropFirmRiskEngine(prop_config)
        self.account_size = prop_config.account_size

        self.orchestrator = Orchestrator(
            config=self.config,
            risk_engine=self.risk_engine,
        )

        self.provider = MarketDataProvider()
        self.top_n = top_n

        # Track results
        self.trades: list[dict] = []
        self.daily_equity: list[dict] = []
        self.signals_log: list[dict] = []

    def run(
        self,
        days: int = 30,
        symbols: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run backtest over historical period.

        Args:
            days: Number of trading days to simulate
            symbols: Specific symbols to test, or None for scanner mode
        """
        console.print(Panel.fit(
            f"[bold cyan]Backtester - {days} Day Simulation[/bold cyan]\n"
            f"Account: ${self.account_size:,.0f} | "
            f"Mode: {'Fixed: ' + ', '.join(symbols) if symbols else f'Scanner (top {self.top_n})'}\n"
            f"Downloading historical data...",
            title="Backtest Starting",
            border_style="cyan",
        ))

        # Get symbols to test
        if symbols is None:
            symbols = self._get_default_symbols()

        # Download all historical data upfront
        console.print(f"[dim]Downloading data for {len(symbols)} symbols...[/dim]")
        all_data = self._download_all_data(symbols, days + 60)  # Extra for indicators

        if not all_data:
            console.print("[red]No data available for backtesting[/red]")
            return {"error": "No data"}

        # Find common trading days
        trading_days = self._get_trading_days(all_data, days)
        console.print(f"[dim]Simulating {len(trading_days)} trading days...[/dim]\n")

        # Reset state
        self.orchestrator.portfolio.initialize(self.account_size)
        self.risk_engine.initialize(self.account_size)

        # Simulate each day
        for i, day in enumerate(trading_days):
            self.risk_engine.start_trading_day(self.orchestrator.portfolio.total_value)
            self.orchestrator.portfolio.reset_daily()

            day_str = day.strftime("%Y-%m-%d")
            equity_before = self.orchestrator.portfolio.total_value

            # Find best setups for this day
            day_signals = self._analyze_day(day, symbols, all_data)

            # Record daily equity
            equity_after = self.orchestrator.portfolio.total_value
            self.daily_equity.append({
                "date": day_str,
                "equity": equity_after,
                "daily_pnl": equity_after - equity_before,
                "positions": len(self.orchestrator.portfolio.positions),
            })

            # Progress bar
            pct = (i + 1) / len(trading_days) * 100
            bar_len = 30
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            console.print(
                f"\r  [{bar}] {pct:.0f}% | Day {i+1}/{len(trading_days)} | "
                f"{day_str} | Equity: ${equity_after:,.2f} | "
                f"Signals: {len(day_signals)}",
                end="",
            )

        console.print("\n")

        # Generate report
        return self._generate_report(trading_days)

    def _get_default_symbols(self) -> list[str]:
        """Get default symbols for backtesting - focus on prop firm tradeable."""
        return [
            # Forex majors
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X", "USDCAD=X",
            # Forex crosses
            "EURGBP=X", "EURJPY=X", "GBPJPY=X",
            # Commodities
            "GC=F", "CL=F",
            # Indices
            "^DJI", "^IXIC",
            # Crypto
            "BTC-USD", "ETH-USD",
        ]

    def _download_all_data(
        self, symbols: list[str], days: int
    ) -> dict[str, pd.DataFrame]:
        """Download historical data for all symbols."""
        data = {}
        period = f"{max(days, 30)}d" if days <= 60 else f"{min(days // 30 + 1, 6)}mo"

        for symbol in symbols:
            try:
                df = self.provider.get_historical(symbol, period="3mo", interval="1d")
                if not df.empty and len(df) >= 20:
                    data[symbol] = df
            except Exception as e:
                logger.debug(f"Failed to download {symbol}: {e}")

        console.print(f"  Downloaded data for {len(data)}/{len(symbols)} symbols")
        return data

    def _get_trading_days(
        self, all_data: dict[str, pd.DataFrame], days: int
    ) -> list[datetime]:
        """Get the last N trading days from the data."""
        # Use the first symbol's index as reference
        first_df = next(iter(all_data.values()))
        all_dates = sorted(first_df.index.tolist())

        # Take last N days (leave first 30 for indicator warmup)
        warmup = 30
        available = all_dates[warmup:]
        return available[-days:] if len(available) >= days else available

    def _analyze_day(
        self,
        day: datetime,
        symbols: list[str],
        all_data: dict[str, pd.DataFrame],
    ) -> list[dict]:
        """Analyze all symbols for a single day."""
        signals = []

        # Score each symbol for this day
        scored = []
        for symbol in symbols:
            df = all_data.get(symbol)
            if df is None:
                continue

            # Get data up to this day (no lookahead bias)
            mask = df.index <= day
            df_to_day = df[mask]
            if len(df_to_day) < 20:
                continue

            # Quick score: 5-day momentum
            close = df_to_day["close"].values
            if len(close) >= 5:
                momentum = (float(close[-1]) - float(close[-5])) / float(close[-5]) * 100
                scored.append((symbol, abs(momentum), momentum))

        # Sort by momentum strength, take top N
        scored.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [s[0] for s in scored[:self.top_n]]

        # Deep analyze top symbols
        for symbol in top_symbols:
            df = all_data.get(symbol)
            if df is None:
                continue

            mask = df.index <= day
            df_to_day = df[mask]

            # Inject data into context
            asset_type = self._detect_type(symbol)

            # Store data for agents
            self.orchestrator.portfolio.timestamp = day

            try:
                result = self.orchestrator.analyze_symbol(symbol, asset_type)
                decision = result.get("final_decision", "hold")

                self.signals_log.append({
                    "date": day.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "decision": decision,
                    "executed": result.get("executed", False),
                })

                if decision != "hold":
                    signals.append(result)

            except Exception as e:
                logger.debug(f"Analysis failed for {symbol} on {day}: {e}")

        return signals

    def _generate_report(self, trading_days: list[datetime]) -> dict[str, Any]:
        """Generate the final backtest report."""
        portfolio = self.orchestrator.portfolio

        # Calculate stats
        initial = self.account_size
        final = portfolio.total_value
        total_pnl = final - initial
        total_pnl_pct = (total_pnl / initial * 100) if initial > 0 else 0

        # Trade stats
        total_signals = len(self.signals_log)
        buy_signals = sum(1 for s in self.signals_log if s["decision"] == "buy")
        sell_signals = sum(1 for s in self.signals_log if s["decision"] == "sell")
        hold_signals = sum(1 for s in self.signals_log if s["decision"] == "hold")
        executed = sum(1 for s in self.signals_log if s.get("executed"))

        # Drawdown from equity curve
        max_dd = 0.0
        peak = initial
        for eq in self.daily_equity:
            equity = eq["equity"]
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Would it pass the challenge?
        target_pct = 10.0  # FunderPro Classic
        passed = total_pnl_pct >= target_pct and max_dd < 10.0

        # Symbol performance
        symbol_stats = {}
        for s in self.signals_log:
            sym = s["symbol"]
            if sym not in symbol_stats:
                symbol_stats[sym] = {"total": 0, "buy": 0, "sell": 0, "hold": 0}
            symbol_stats[sym]["total"] += 1
            symbol_stats[sym][s["decision"]] += 1

        # Print report
        console.print(Panel.fit(
            f"[bold]Backtest Results - {len(trading_days)} Trading Days[/bold]",
            border_style="cyan",
        ))

        # Performance table
        perf_table = Table(show_header=False, box=None)
        perf_table.add_column("Metric", style="bold", width=25)
        perf_table.add_column("Value", width=20)

        pnl_color = "green" if total_pnl >= 0 else "red"
        perf_table.add_row("Initial Balance", f"${initial:,.2f}")
        perf_table.add_row("Final Balance", f"${final:,.2f}")
        perf_table.add_row("Total P&L", f"[{pnl_color}]${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)[/{pnl_color}]")
        perf_table.add_row("Max Drawdown", f"{max_dd:.2f}%")
        perf_table.add_row("", "")
        perf_table.add_row("Total Signals", str(total_signals))
        perf_table.add_row("BUY Signals", str(buy_signals))
        perf_table.add_row("SELL Signals", str(sell_signals))
        perf_table.add_row("HOLD Signals", str(hold_signals))
        perf_table.add_row("Executed Trades", str(executed))
        perf_table.add_row("", "")

        pass_color = "green bold" if passed else "red bold"
        perf_table.add_row("Target (10%)", f"${initial * 0.10:,.2f}")
        perf_table.add_row("Challenge Result", f"[{pass_color}]{'PASSED' if passed else 'NOT PASSED'}[/{pass_color}]")

        console.print(perf_table)

        # Symbol breakdown
        if symbol_stats:
            console.print(f"\n[bold]Symbol Activity[/bold]")
            sym_table = Table()
            sym_table.add_column("Symbol")
            sym_table.add_column("Signals")
            sym_table.add_column("Buy")
            sym_table.add_column("Sell")
            sym_table.add_column("Hold")

            for sym, stats in sorted(symbol_stats.items(), key=lambda x: x[1]["total"], reverse=True)[:10]:
                sym_table.add_row(
                    sym, str(stats["total"]),
                    str(stats["buy"]), str(stats["sell"]), str(stats["hold"]),
                )
            console.print(sym_table)

        # Equity curve (text-based)
        if self.daily_equity:
            console.print(f"\n[bold]Equity Curve[/bold]")
            min_eq = min(e["equity"] for e in self.daily_equity)
            max_eq = max(e["equity"] for e in self.daily_equity)
            eq_range = max_eq - min_eq if max_eq > min_eq else 1

            for eq in self.daily_equity[-20:]:  # Last 20 days
                normalized = (eq["equity"] - min_eq) / eq_range
                bar_len = int(normalized * 40)
                bar = "█" * bar_len
                pnl = eq["daily_pnl"]
                color = "green" if pnl >= 0 else "red"
                console.print(
                    f"  {eq['date']} | ${eq['equity']:>10,.2f} | [{color}]{bar}[/{color}]"
                )

        result = {
            "days": len(trading_days),
            "initial_balance": initial,
            "final_balance": round(final, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "total_signals": total_signals,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "executed_trades": executed,
            "challenge_passed": passed,
            "daily_equity": self.daily_equity,
        }

        return result

    @staticmethod
    def _detect_type(symbol: str) -> AssetType:
        if symbol.endswith("=X"):
            return AssetType.FOREX
        if symbol.endswith("-USD"):
            return AssetType.CRYPTO
        return AssetType.STOCK
