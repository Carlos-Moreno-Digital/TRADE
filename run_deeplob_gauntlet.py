"""DeepLOB end-to-end: train -> wrap as strategy -> run validation gauntlet.

Token rule: prints only summary lines (per-epoch loss, gauntlet
verdicts, no tensors / dataframes).

Pipeline:
  1. Load EURUSD bars (yfinance, ~1500 bars to keep it cheap)
  2. Synthesize causal LOB tensor (calibrated cont_stoikov)
  3. Train DeepLOB for a few epochs with temporal split + early stop
  4. Wrap the trained model as DeepLOBStrategy (StrategyProtocol)
  5. Run the gauntlet:
        Gate A: NautilusHarness (realistic fills)
        Gate B: CPCV (combinatorial purged cross-validation)
        Gate C: ParanoidSuite (7 statistical tests)
  6. Emit a structured verdict.

Failure under any gate is acceptable for this smoke run — what
matters is that the gauntlet ACCEPTS the model and emits a verdict.
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

from trade.research.lobframe import calibrated_config, synthesize_lob
from trade.research.lobframe.dataset import LOBDatasetConfig
from trade.research.lobframe.train import TrainConfig, save_model, train_deeplob
from trade.research.strategies.deeplob_strategy import DeepLOBStrategy
from trade.validation.cpcv import CPCV
from trade.validation.nautilus_harness import NautilusHarness
from trade.validation.paranoid import ParanoidSuite

console = Console()
DATA_DIR = Path("data/dukascopy")
SYM = "EURUSD"
PAIR = "EUR/USD"
SPREAD = 0.00008
N_BARS = 1500   # cheap subset

# Smaller window so the smoke run completes in seconds
WINDOW = 50


def load_bars() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{SYM}_1H.csv", parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df.iloc[:N_BARS].copy()


def shrink_param_grid(strategy: DeepLOBStrategy) -> None:
    """Trim the param grid even further so the gauntlet completes in
    a couple of minutes during the smoke run.
    """
    strategy.param_grid = {
        "confidence_threshold": [0.40, 0.45],
        "atr_sl_mult": [1.0],
        "atr_tp_mult": [2.0],
        "max_holding_bars": [12],
    }


def main() -> int:
    console.print(Panel.fit(
        f"[bold magenta]DeepLOB Gauntlet[/bold magenta]\n"
        f"symbol={SYM} bars={N_BARS} window={WINDOW}",
        title="Phase 3 Part 3",
        border_style="magenta",
    ))

    bars = load_bars()
    cfg = calibrated_config(SYM)
    tensor = synthesize_lob(bars, cfg)
    console.print(f"  bars={bars.shape} tensor={tensor.shape}")

    # ---------- Training ----------
    console.print("\n  [bold]Training DeepLOB[/bold]")
    train_cfg = TrainConfig(
        epochs=4,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-4,
        val_fraction=0.3,
        early_stop_patience=2,
    )
    ds_cfg = LOBDatasetConfig(window=WINDOW, horizon=5, tau=1e-4)
    model, summaries = train_deeplob(
        tensor,
        cfg=train_cfg,
        dataset_cfg=ds_cfg,
        log_fn=lambda s: console.print(s),
    )
    if not summaries:
        console.print("  [red]training produced no epochs[/red]")
        return 1
    final = summaries[-1]
    save_model(model, Path("data/models/deeplob_eurusd.pt"))
    console.print(
        f"  best val_loss={min(s.val_loss for s in summaries):.4f} "
        f"final val_acc={final.val_accuracy:.3f}"
    )

    # ---------- Strategy adapter ----------
    strategy = DeepLOBStrategy(model=model, symbol=SYM, window=WINDOW)
    shrink_param_grid(strategy)
    n_combos = 1
    for v in strategy.param_grid.values():
        n_combos *= len(v)
    console.print(
        f"\n  [bold]Adapter wrapped[/bold] strategy={strategy.name} "
        f"param_combos={n_combos}"
    )

    # ---------- Gate A: Nautilus realistic fills ----------
    console.print("\n  [bold]Gate A: NautilusHarness[/bold]")
    harness = NautilusHarness(
        spread_abs=SPREAD,
        commission_bps=0.5,
        slip_prob=0.3,
        account=10_000.0,
        risk_per_trade=0.003,
    )
    nautilus_res = harness.run(strategy, bars, SYM, PAIR)
    console.print(f"    {nautilus_res.summary()}")

    # ---------- Gate B: CPCV ----------
    console.print("\n  [bold]Gate B: CPCV[/bold]")
    cpcv = CPCV(strategy, n_folds=4, n_test_folds=1, embargo_bars=20)
    cpcv_res = cpcv.run(bars, SPREAD)
    console.print(f"    {cpcv_res.summary()}")

    # ---------- Gate C: Paranoid suite (cheap settings) ----------
    console.print("\n  [bold]Gate C: ParanoidSuite[/bold]")
    suite = ParanoidSuite(
        strategy, bars, SPREAD, n_trials=n_combos, seed=42,
    )
    # Direct test calls so we can pass cheap permutation/bootstrap counts
    paranoid_results = {
        "future_shift": suite.future_shift(),
        "time_permutation": suite.time_permutation(n_permutations=50),
        "block_bootstrap": suite.block_bootstrap(n_bootstrap=200),
        "deflated_sharpe": suite.deflated_sharpe(),
        "minbtl": suite.minbtl(),
        "param_stability": suite.param_stability(),
        "wfe": suite.wfe(n_splits=3),
    }
    passed = sum(int(r.get("passed", False)) for r in paranoid_results.values())

    # ---------- Verdict ----------
    console.print("\n")
    t = Table(title="DeepLOB Gauntlet Verdict")
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
    for name, r in paranoid_results.items():
        ok = r.get("passed", False)
        color = "green" if ok else "red"
        t.add_row(
            f"Paranoid: {name}",
            r.get("verdict", "?"),
            f"[{color}]{'PASS' if ok else 'FAIL'}[/{color}]",
        )
    console.print(t)
    console.print(f"  paranoid total: {passed}/{len(paranoid_results)} passed")

    fully_viable = (
        nautilus_res.viable
        and cpcv_res.viable
        and passed == len(paranoid_results)
    )
    if fully_viable:
        console.print(
            "  [bold green]VIABLE[/bold green] - candidate for Orchestrator.alpha"
        )
    else:
        console.print(
            "  [bold yellow]REJECTED[/bold yellow] - gauntlet emitted a structured "
            "verdict, model is NOT promoted"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
