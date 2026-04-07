"""Gauntlet runner for NMIStatArb (Phase 4 in parallel with the
DeepLOB institutional run).

Single command:
  - Loads AUDUSD as primary leg
  - Loads NZDUSD as partner (highest NMI pair = 0.292 from analyze_nmi)
  - Wraps NMIStatArb as a StrategyProtocol
  - Runs Gate A (Nautilus realistic fills) and Gate C (Paranoid suite)
  - Skips Gate B (CPCV) for the smoke run because the partner-load
    coupling is not fold-friendly; the institutional run does it once
    the harness supports cross-asset folds.

Token rule: only summary lines, no dataframes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trade.research.strategies.nmi_stat_arb import NMIStatArb
from trade.validation.nautilus_harness import NautilusHarness
from trade.validation.paranoid import ParanoidSuite

console = Console()
DATA_DIR = Path("data/dukascopy")
PRIMARY = "AUDUSD"
PARTNER = "NZDUSD"
PAIR = "AUD/USD"
SPREAD = 0.0001  # ~1 pip
N_BARS = 1500


def load_primary() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{PRIMARY}_1H.csv", parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df.iloc[:N_BARS].copy()


def main() -> int:
    console.print(Panel.fit(
        f"[bold magenta]NMIStatArb Gauntlet[/bold magenta]\n"
        f"primary={PRIMARY} partner={PARTNER} bars={N_BARS}\n"
        f"NMI({PRIMARY},{PARTNER})=0.292 (top pair from analyze_nmi.py)",
        title="Phase 4 — Stat-Arb v0",
        border_style="magenta",
    ))

    bars = load_primary()
    console.print(f"  primary bars: {bars.shape}")
    strategy = NMIStatArb.from_disk(
        primary_symbol=PRIMARY, partner_symbol=PARTNER, data_dir=DATA_DIR,
    )
    n_combos = 1
    for v in strategy.param_grid.values():
        n_combos *= len(v)
    console.print(
        f"  strategy={strategy.name} supported_regimes={strategy.supported_regimes} "
        f"param_combos={n_combos}"
    )

    # ---- Vanilla sanity ----
    pnls = strategy.backtest(bars, SPREAD, strategy.default_params)
    total = float(np.sum(pnls)) if pnls else 0.0
    wr = (sum(1 for p in pnls if p > 0) / len(pnls) * 100) if pnls else 0.0
    console.print(
        f"\n  [bold]Vanilla sanity[/bold] trades={len(pnls)} pnl=${total:+,.0f} wr={wr:.1f}%"
    )

    # ---- Gate A: NautilusHarness ----
    console.print("\n  [bold]Gate A: NautilusHarness[/bold]")
    harness = NautilusHarness(
        spread_abs=SPREAD,
        commission_bps=0.5,
        slip_prob=0.3,
        account=10_000.0,
        risk_per_trade=0.003,
    )
    nautilus_res = harness.run(strategy, bars, PRIMARY, PAIR)
    console.print(f"    {nautilus_res.summary()}")

    # ---- Gate C: Paranoid suite (cheap settings) ----
    console.print("\n  [bold]Gate C: ParanoidSuite[/bold]")
    suite = ParanoidSuite(
        strategy, bars, SPREAD, n_trials=n_combos, seed=42,
    )
    paranoid = {
        "future_shift": suite.future_shift(),
        "time_permutation": suite.time_permutation(n_permutations=50),
        "block_bootstrap": suite.block_bootstrap(n_bootstrap=200),
        "deflated_sharpe": suite.deflated_sharpe(),
        "minbtl": suite.minbtl(),
        "param_stability": suite.param_stability(),
        "wfe": suite.wfe(n_splits=3),
    }
    passed = sum(int(r.get("passed", False)) for r in paranoid.values())

    # ---- Verdict ----
    console.print("\n")
    t = Table(title=f"NMIStatArb Gauntlet — {PRIMARY}/{PARTNER}")
    t.add_column("Gate")
    t.add_column("Metric")
    t.add_column("Result")

    t.add_row(
        "Nautilus (real fills)",
        f"PnL=${nautilus_res.nautilus_pnl:+,.0f} trades={nautilus_res.nautilus_trades}",
        "[green]PASS[/green]" if nautilus_res.viable else "[red]FAIL[/red]",
    )
    for name, r in paranoid.items():
        ok = r.get("passed", False)
        color = "green" if ok else "red"
        t.add_row(
            f"Paranoid: {name}",
            r.get("verdict", "?"),
            f"[{color}]{'PASS' if ok else 'FAIL'}[/{color}]",
        )
    console.print(t)
    console.print(f"  paranoid total: {passed}/{len(paranoid)} passed")

    fully_viable = nautilus_res.viable and passed == len(paranoid)
    if fully_viable:
        console.print(
            "  [bold green]VIABLE[/bold green] - candidate for Orchestrator.alpha"
        )
    else:
        console.print(
            "  [bold yellow]REJECTED[/bold yellow] - structured verdict, not promoted"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
