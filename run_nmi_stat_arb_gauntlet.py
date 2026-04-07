"""Gauntlet runner for NMIStatArb (Phase 4 with 3-caveat fixes).

Resolved caveats vs the v0 run:

  Caveat 1 (lockstep permutation):
    NMIStatArb now exposes with_permuted_state(rng) as a context
    manager that ParanoidSuite.time_permutation enters before each
    shuffled backtest. The partner pair is shuffled INDEPENDENTLY
    so the cointegration relationship is destroyed and the strategy
    must beat the null without any cross-asset structure.

  Caveat 2 (2-leg fill harness):
    NautilusHarness now has run_pair() that consumes pair_signals()
    and runs each leg through its own _SignalPlayerStrategy on
    distinct instruments. Combined P&L = primary + partner with
    realistic bid/ask on both sides.

  Caveat 3 (cross-asset CPCV):
    NMIStatArb._aligned_partner reindexes self.partner to the input
    df.index, so when CPCV slices the primary into folds the partner
    is sliced in lockstep automatically. We re-enable CPCV in the
    runner to confirm the contract.

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
from trade.validation.cpcv import CPCV
from trade.validation.nautilus_harness import NautilusHarness
from trade.validation.paranoid import ParanoidSuite

console = Console()
DATA_DIR = Path("data/dukascopy")
PRIMARY = "AUDUSD"
PARTNER = "NZDUSD"
PRIMARY_PAIR = "AUD/USD"
PARTNER_PAIR = "NZD/USD"
PRIMARY_SPREAD = 0.0001
PARTNER_SPREAD = 0.0001
N_BARS = 1500


def load_bars(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{symbol}_1H.csv", parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df.iloc[:N_BARS].copy()


def main() -> int:
    console.print(Panel.fit(
        f"[bold magenta]NMIStatArb Gauntlet (3 caveats fixed)[/bold magenta]\n"
        f"primary={PRIMARY} partner={PARTNER} bars={N_BARS}\n"
        f"NMI({PRIMARY},{PARTNER})=0.292",
        title="Phase 4 — Stat-Arb v1",
        border_style="magenta",
    ))

    primary_bars = load_bars(PRIMARY)
    partner_bars = load_bars(PARTNER)
    console.print(
        f"  primary bars: {primary_bars.shape}  partner bars: {partner_bars.shape}"
    )

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
    pnls = strategy.backtest(primary_bars, PRIMARY_SPREAD, strategy.default_params)
    total = float(np.sum(pnls)) if pnls else 0.0
    wr = (sum(1 for p in pnls if p > 0) / len(pnls) * 100) if pnls else 0.0
    console.print(
        f"\n  [bold]Vanilla sanity[/bold] trades={len(pnls)} "
        f"pnl=${total:+,.0f} wr={wr:.1f}%"
    )

    # ---- Gate A: NautilusHarness single-leg (legacy view) ----
    console.print("\n  [bold]Gate A1: NautilusHarness single-leg[/bold]")
    harness = NautilusHarness(
        spread_abs=PRIMARY_SPREAD, commission_bps=0.5, slip_prob=0.3,
        account=10_000.0, risk_per_trade=0.003,
    )
    single_leg_res = harness.run(strategy, primary_bars, PRIMARY, PRIMARY_PAIR)
    console.print(f"    {single_leg_res.summary()}")

    # ---- Gate A2: NautilusHarness 2-leg pair (caveat #2 fix) ----
    console.print("\n  [bold]Gate A2: NautilusHarness 2-leg pair[/bold]")
    pair_res = harness.run_pair(
        strategy=strategy,
        primary_bars=primary_bars,
        primary_symbol=PRIMARY,
        primary_pair=PRIMARY_PAIR,
        primary_spread=PRIMARY_SPREAD,
        partner_bars=partner_bars,
        partner_symbol=PARTNER,
        partner_pair=PARTNER_PAIR,
        partner_spread=PARTNER_SPREAD,
    )
    console.print(f"    {pair_res.summary()}")

    # ---- Gate B: CPCV (caveat #3 verification) ----
    console.print("\n  [bold]Gate B: CPCV[/bold]")
    cpcv = CPCV(strategy, n_folds=4, n_test_folds=1, embargo_bars=20)
    cpcv_res = cpcv.run(primary_bars, PRIMARY_SPREAD)
    console.print(f"    {cpcv_res.summary()}")

    # ---- Gate C: ParanoidSuite with lockstep permutation (caveat #1) ----
    console.print(
        "\n  [bold]Gate C: ParanoidSuite (lockstep permutation enabled)[/bold]"
    )
    suite = ParanoidSuite(
        strategy, primary_bars, PRIMARY_SPREAD, n_trials=n_combos, seed=42,
    )
    paranoid = {
        "future_shift": suite.future_shift(),
        "time_permutation": suite.time_permutation(n_permutations=100),
        "block_bootstrap": suite.block_bootstrap(n_bootstrap=200),
        "deflated_sharpe": suite.deflated_sharpe(),
        "minbtl": suite.minbtl(),
        "param_stability": suite.param_stability(),
        "wfe": suite.wfe(n_splits=3),
    }
    passed = sum(int(r.get("passed", False)) for r in paranoid.values())

    # ---- Verdict ----
    console.print("\n")
    t = Table(title=f"NMIStatArb v1 Gauntlet — {PRIMARY}/{PARTNER}")
    t.add_column("Gate")
    t.add_column("Metric")
    t.add_column("Result")

    t.add_row(
        "Nautilus single-leg",
        f"PnL=${single_leg_res.nautilus_pnl:+,.0f} "
        f"trades={single_leg_res.nautilus_trades}",
        "[green]PASS[/green]" if single_leg_res.viable else "[red]FAIL[/red]",
    )
    t.add_row(
        "Nautilus 2-leg pair",
        f"primary=${pair_res.primary_pnl:+,.0f} "
        f"partner=${pair_res.partner_pnl:+,.0f} "
        f"combined=${pair_res.combined_pnl:+,.0f}",
        "[green]PASS[/green]" if pair_res.viable else "[red]FAIL[/red]",
    )
    t.add_row(
        "CPCV",
        f"OOS_SR={cpcv_res.mean_oos_sharpe:+.2f} WFE={cpcv_res.wfe:.2f} "
        f"PBO={cpcv_res.pbo:.2f}",
        "[green]PASS[/green]" if cpcv_res.viable else "[red]FAIL[/red]",
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

    fully_viable = (
        pair_res.viable
        and cpcv_res.viable
        and passed == len(paranoid)
    )
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
