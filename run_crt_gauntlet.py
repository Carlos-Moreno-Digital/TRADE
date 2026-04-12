"""Gauntlet runner for CRT Multi-TF + Sentiment + Telegram integration.

Runs CRT through the full validation pipeline and also exercises:
  - SentimentAgent (fetch + keyword score)
  - TelegramAlert (config check + dry-run message format)

Token rule: only summary lines.
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

from trade.alerts.telegram_bot import TelegramAlert
from trade.data.agents.sentiment_agent import SentimentAgent
from trade.research.strategies.crt_strategy import CRTStrategy
from trade.validation.cpcv import CPCV
from trade.validation.nautilus_harness import NautilusHarness
from trade.validation.paranoid import ParanoidSuite

console = Console()
DATA_DIR = Path("data/dukascopy")
SYM = "EURUSD"
PAIR = "EUR/USD"
SPREAD = 8e-5
N_BARS = 5000  # CRT needs more bars for HTF resampling


def load_bars() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{SYM}_1H.csv", parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df.iloc[:N_BARS].copy()


def main() -> int:
    console.print(Panel.fit(
        f"[bold magenta]CRT + Sentiment + Telegram Gauntlet[/bold magenta]\n"
        f"symbol={SYM} bars={N_BARS}",
        title="Phase 9 — Institutional Integration",
        border_style="magenta",
    ))

    # ---- 1. Sentiment Agent smoke ----
    console.print("\n  [bold]1. SentimentAgent (keyword mode)[/bold]")
    sa = SentimentAgent(use_finbert=False)
    test_headlines = [
        "Fed signals rate cut in September meeting",
        "EUR/USD crashes on weak GDP data",
        "Markets remain stable ahead of NFP",
    ]
    for h in test_headlines:
        s = sa.score_text(h)
        console.print(f"    [{s:+.2f}] {h[:60]}")
    # Live fetch attempt (may fail if no internet or RSS down)
    snap = sa.get_current_sentiment()
    console.print(
        f"    live: score={snap.score:+.2f} headlines={snap.n_headlines} "
        f"source={snap.source}"
    )

    # ---- 2. Telegram Alert check ----
    console.print("\n  [bold]2. TelegramAlert config check[/bold]")
    tg = TelegramAlert()
    console.print(
        f"    configured={tg.is_configured} "
        f"(set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env to enable)"
    )

    # ---- 3. CRT Strategy ----
    bars = load_bars()
    strategy = CRTStrategy()
    n_combos = 1
    for v in strategy.param_grid.values():
        n_combos *= len(v)

    console.print(f"\n  [bold]3. CRT Strategy[/bold] bars={bars.shape} combos={n_combos}")

    pnls = strategy.backtest(bars, SPREAD, strategy.default_params)
    total = float(np.sum(pnls)) if pnls else 0.0
    wr = (sum(1 for p in pnls if p > 0) / len(pnls) * 100) if pnls else 0.0
    console.print(
        f"    vanilla: trades={len(pnls)} pnl=${total:+,.0f} wr={wr:.1f}%"
    )

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
    t = Table(title=f"CRT Gauntlet — {SYM}")
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

    # ---- 4. Telegram summary (only if configured) ----
    if tg.is_configured:
        tg.send_raw(
            f"*CRT GAUNTLET*\n"
            f"trades={len(pnls)} pnl=${total:+,.0f}\n"
            f"paranoid={passed}/{len(paranoid)}\n"
            f"{'VIABLE' if fully_viable else 'REJECTED'}"
        )
        console.print("  [dim]Telegram summary sent[/dim]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
