"""Prove the validation pipeline rejects a zero-edge strategy.

Feeds DummyRandom (random long/short entries) through:
  1. NautilusHarness (realistic fills, spread, slippage)
  2. CPCV (combinatorial purged cross-validation)
  3. ParanoidSuite (7 tests)

Expected: every single gate must reject it. If any gate accepts it,
the pipeline itself is broken.
"""
from pathlib import Path
import sys

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trade.research.strategies.dummy_random import DummyRandom
from trade.validation.cpcv import CPCV
from trade.validation.nautilus_harness import NautilusHarness
from trade.validation.paranoid import ParanoidSuite

console = Console()
DATA_DIR = Path("data/dukascopy")

SYMBOLS = {
    "EURUSD": {"pair": "EUR/USD", "spread": 0.00008},
    "USDJPY": {"pair": "USD/JPY", "spread": 0.008},
}


def load(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{symbol}_1H.csv", parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def main():
    strategy = DummyRandom()
    n_trials = 1
    for v in strategy.param_grid.values():
        n_trials *= len(v)
    console.print(Panel.fit(
        f"[bold red]PIPELINE SMOKE TEST[/bold red] — {strategy.name}\n"
        f"param_grid: {n_trials} combos\n"
        "Expected: every gate REJECTS. If anything passes, pipeline is broken.",
        title="Dummy rejection test",
        border_style="red",
    ))

    for sym, cfg in SYMBOLS.items():
        df = load(sym)
        console.print(
            f"\n[cyan bold]{sym}[/cyan bold] {len(df):,} bars "
            f"{df.index[0].date()} → {df.index[-1].date()}"
        )

        # ---- Gate 1: Nautilus harness (realistic fills) ----
        console.print("  [dim]Gate 1: NautilusHarness...[/dim]")
        harness = NautilusHarness(
            spread_abs=cfg["spread"],
            commission_bps=0.5,
            slip_prob=0.3,
            account=10_000.0,
            risk_per_trade=0.003,
        )
        nautilus_res = harness.run(strategy, df, sym, cfg["pair"])
        console.print(f"    {nautilus_res.summary()}")

        # ---- Gate 2: CPCV ----
        console.print("  [dim]Gate 2: CPCV (this is slow)...[/dim]")
        cpcv = CPCV(strategy, n_folds=6, n_test_folds=2, embargo_bars=50)
        cpcv_res = cpcv.run(df, cfg["spread"])
        console.print(f"    {cpcv_res.summary()}")

        # ---- Gate 3: Paranoid suite ----
        console.print("  [dim]Gate 3: ParanoidSuite (500 permutations)...[/dim]")
        suite = ParanoidSuite(
            strategy, df, cfg["spread"], n_trials=n_trials, seed=42
        )
        paranoid_res = suite.run_all()
        console.print(f"    {paranoid_res.passed_count}/{len(paranoid_res.per_test)} passed")

        # ---- Report ----
        t = Table(title=f"{sym} — {strategy.name} pipeline gates")
        t.add_column("Gate")
        t.add_column("Metric")
        t.add_column("Result")
        t.add_row(
            "Nautilus (real fills)",
            f"PnL=${nautilus_res.nautilus_pnl:+,.0f}",
            f"[{'green' if nautilus_res.viable else 'red'}]"
            f"{'PASS' if nautilus_res.viable else 'FAIL'}[/]",
        )
        t.add_row(
            "CPCV",
            f"OOS_SR={cpcv_res.mean_oos_sharpe:+.2f} "
            f"WFE={cpcv_res.wfe:.2f} PBO={cpcv_res.pbo:.2f}",
            f"[{'green' if cpcv_res.viable else 'red'}]"
            f"{'PASS' if cpcv_res.viable else 'FAIL'}[/]",
        )
        for test_name, r in paranoid_res.per_test.items():
            mark = "PASS" if r.get("passed") else "FAIL"
            color = "green" if r.get("passed") else "red"
            t.add_row(
                f"Paranoid: {test_name}",
                r.get("verdict", "?"),
                f"[{color}]{mark}[/]",
            )
        console.print(t)

        # ---- Final assertion ----
        viable_anywhere = (
            nautilus_res.viable
            or cpcv_res.viable
            or paranoid_res.all_passed
        )
        if viable_anywhere:
            console.print(
                f"  [bold red]!!! PIPELINE BROKEN: {sym} accepted a "
                f"zero-edge strategy !!![/bold red]"
            )
            sys.exit(1)
        else:
            console.print(
                f"  [bold green]PIPELINE OK: {sym} rejected DummyRandom "
                f"in every gate.[/bold green]"
            )


if __name__ == "__main__":
    main()
