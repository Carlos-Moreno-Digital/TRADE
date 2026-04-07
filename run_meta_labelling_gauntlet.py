"""Gauntlet runner for MetaLabellingStrategy (Phase 7).

Token rule: only the final verdict table on stdout. The training of
the inner RandomForest happens silently inside backtest() calls.
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

from trade.research.strategies.meta_labelling import MetaLabellingStrategy
from trade.validation.cpcv import CPCV
from trade.validation.nautilus_harness import NautilusHarness
from trade.validation.paranoid import ParanoidSuite

console = Console()
DATA_DIR = Path("data/dukascopy")
SYM = "EURUSD"
PAIR = "EUR/USD"
SPREAD = 8e-5
N_BARS = 1500


def load_bars() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{SYM}_1H.csv", parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df.iloc[:N_BARS].copy()


def main() -> int:
    bars = load_bars()
    strategy = MetaLabellingStrategy(symbol=SYM)
    n_combos = 1
    for v in strategy.param_grid.values():
        n_combos *= len(v)

    console.print(Panel.fit(
        f"[bold magenta]Meta-Labelling Gauntlet[/bold magenta]\n"
        f"primary=RegimeMomentum  meta=RandomForestClassifier\n"
        f"symbol={SYM} bars={N_BARS} combos={n_combos}",
        title="Phase 7 — Triple-Barrier Meta-Labelling",
        border_style="magenta",
    ))

    # Vanilla sanity vs primary alone
    primary_only = strategy.primary.backtest(
        bars, SPREAD, strategy.primary.default_params,
    )
    p_total = float(np.sum(primary_only)) if primary_only else 0.0
    p_wr = (
        sum(1 for p in primary_only if p > 0) / len(primary_only) * 100
        if primary_only else 0.0
    )
    console.print(
        f"\n  [dim]primary alone:[/dim] trades={len(primary_only)} "
        f"pnl=${p_total:+,.0f} wr={p_wr:.1f}%"
    )

    pnls = strategy.backtest(bars, SPREAD, strategy.default_params)
    total = float(np.sum(pnls)) if pnls else 0.0
    wr = (sum(1 for p in pnls if p > 0) / len(pnls) * 100) if pnls else 0.0
    console.print(
        f"  [dim]meta-filtered:[/dim] trades={len(pnls)} "
        f"pnl=${total:+,.0f} wr={wr:.1f}%"
    )
    if primary_only:
        filter_rate = 1.0 - len(pnls) / max(len(primary_only), 1)
        console.print(f"  [dim]meta filter rate: {filter_rate*100:.1f}%[/dim]")

    # Gate A: Nautilus
    harness = NautilusHarness(
        spread_abs=SPREAD, commission_bps=0.5, slip_prob=0.3,
        account=10_000.0, risk_per_trade=0.003,
    )
    nautilus_res = harness.run(strategy, bars, SYM, PAIR)

    # Gate B: CPCV
    cpcv = CPCV(strategy, n_folds=4, n_test_folds=1, embargo_bars=20)
    cpcv_res = cpcv.run(bars, SPREAD)

    # Gate C: Paranoid suite
    suite = ParanoidSuite(
        strategy, bars, SPREAD, n_trials=n_combos, seed=42,
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

    # Verdict
    console.print()
    t = Table(title=f"Meta-Labelling Gauntlet — {SYM}")
    t.add_column("Gate")
    t.add_column("Metric")
    t.add_column("Result")
    t.add_row(
        "Nautilus (real fills)",
        f"PnL=${nautilus_res.nautilus_pnl:+,.0f} trades={nautilus_res.nautilus_trades}",
        "[green]PASS[/green]" if nautilus_res.viable else "[red]FAIL[/red]",
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
        nautilus_res.viable
        and cpcv_res.viable
        and passed == len(paranoid)
    )
    if fully_viable:
        console.print("  [bold green]VIABLE[/bold green]")
    else:
        console.print("  [bold yellow]REJECTED[/bold yellow] - structured verdict")
    return 0


if __name__ == "__main__":
    sys.exit(main())
