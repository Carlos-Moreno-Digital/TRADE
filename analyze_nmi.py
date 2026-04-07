"""NMI dependency analysis over the 8-asset validation universe.

Loads every symbol in data/dukascopy/*_1H.csv, inner-joins them on
timestamp, computes the Kraskov-based NMI matrix, applies hierarchical
clustering over (1 - NMI), and prints:
  1. NMI heatmap as a text table
  2. Pearson correlation matrix (for side-by-side comparison)
  3. Absolute delta |NMI - |Pearson|| (where NMI captures non-linear
     dependence that correlation misses)
  4. Linkage merges in dendrogram order
  5. Cluster assignments at k = 2, 3, 4

All output is text/table so it renders cleanly in a terminal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trade.research.dependency import (
    align_returns,
    build_nmi_matrix,
    cluster_assignments,
    hierarchical_clusters,
    linkage_merges_text,
)

console = Console()
DATA_DIR = Path("data/dukascopy")

UNIVERSE = [
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD",
    "USDJPY", "USDCAD",
    "XAUUSD", "SPX",
]


def load_close(symbol: str) -> pd.Series | None:
    path = DATA_DIR / f"{symbol}_1H.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df["close"].rename(symbol)


def color_nmi(v: float) -> str:
    if v >= 0.5:
        return "bright_red"
    if v >= 0.3:
        return "red"
    if v >= 0.15:
        return "yellow"
    if v >= 0.05:
        return "green"
    return "dim"


def fmt_matrix(m: pd.DataFrame, colorize: bool = True, decimals: int = 2) -> Table:
    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("", style="bold", width=8)
    for c in m.columns:
        t.add_column(c, justify="right", width=7)
    for row_name in m.index:
        row = [row_name]
        for col_name in m.columns:
            v = float(m.loc[row_name, col_name])
            txt = f"{v:.{decimals}f}"
            if colorize:
                color = color_nmi(v) if row_name != col_name else "white"
                txt = f"[{color}]{txt}[/{color}]"
            row.append(txt)
        t.add_row(*row)
    return t


def build_tree(linkage_mat: np.ndarray, names: list[str]) -> Tree:
    """rich.Tree representation of the hierarchical structure."""
    n = len(names)
    # Build a lookup: id -> Tree node with accumulated leaf list
    nodes: dict[int, dict] = {
        i: {"label": names[i], "children": [], "leaves": [names[i]], "dist": 0.0}
        for i in range(n)
    }
    for step, row in enumerate(linkage_mat):
        left = int(row[0])
        right = int(row[1])
        dist = float(row[2])
        new_id = n + step
        leaves = nodes[left]["leaves"] + nodes[right]["leaves"]
        nodes[new_id] = {
            "label": f"merge d={dist:.3f} ({len(leaves)} assets)",
            "children": [left, right],
            "leaves": leaves,
            "dist": dist,
        }
    root_id = n + len(linkage_mat) - 1
    root = Tree(
        f"[bold]ROOT[/bold] — {len(names)} assets, max d={nodes[root_id]['dist']:.3f}",
        guide_style="cyan",
    )

    def walk(parent_tree, node_id):
        node = nodes[node_id]
        if not node["children"]:
            parent_tree.add(f"[bold white]{node['label']}[/bold white]")
            return
        sub = parent_tree.add(f"[dim]{node['label']}[/dim]")
        for c in node["children"]:
            walk(sub, c)

    walk(root, root_id)
    return root


def main():
    console.print(Panel.fit(
        "[bold magenta]NMI DEPENDENCY ANALYSIS[/bold magenta]\n"
        "Kraskov k-NN estimator, 8-asset universe, hierarchical clustering over 1-NMI",
        title="Phase 3 Step 2",
        border_style="magenta",
    ))

    # ---- Load and align ----
    series = {}
    for sym in UNIVERSE:
        s = load_close(sym)
        if s is None:
            console.print(f"  [red]missing {sym}[/red]")
            continue
        series[sym] = s
        console.print(f"  [green]loaded[/green] {sym}: {len(s):,} bars")

    rets = align_returns(series, resample="1D")
    console.print(
        f"\n  [bold]Aligned (daily log returns):[/bold] {len(rets):,} common obs, "
        f"{rets.index[0].date()} → {rets.index[-1].date()}"
    )
    console.print(f"  Assets: {list(rets.columns)}")

    # ---- Independence sanity check ----
    # NMI of each series against a shuffled copy of itself: should be ~0.
    from trade.research.dependency.nmi import nmi as _nmi
    rng = np.random.default_rng(7)
    baseline = []
    for col in rets.columns:
        x = rets[col].values
        baseline.append(_nmi(x, rng.permutation(x), k=5))
    mean_noise = float(np.mean(baseline))
    max_noise = float(np.max(baseline))
    console.print(
        f"\n  [dim]Independence baseline (own series shuffled): "
        f"mean NMI={mean_noise:.3f}, max={max_noise:.3f}[/dim]"
    )
    console.print(
        "  [dim]→ any off-diagonal NMI above this noise floor is a real "
        "dependency signal[/dim]"
    )

    # ---- NMI matrix ----
    console.print("\n  [dim]Computing NMI matrix (Kraskov k=5)...[/dim]")
    nmi_m = build_nmi_matrix(rets, k=5)

    console.print()
    console.print(Panel.fit(
        "[bold]NMI Matrix (0=independent, 1=fully dependent)[/bold]\n"
        "red ≥ 0.5  |  orange 0.3-0.5  |  yellow 0.15-0.3  |  green 0.05-0.15  |  dim <0.05",
        border_style="magenta",
    ))
    console.print(fmt_matrix(nmi_m, colorize=True, decimals=2))

    # ---- Pearson for comparison ----
    pearson = rets.corr(method="pearson")
    console.print()
    console.print(Panel.fit("[bold]Pearson correlation (for comparison)[/bold]",
                            border_style="blue"))
    console.print(fmt_matrix(pearson, colorize=False, decimals=2))

    # NOTE on NMI vs Pearson scale: NMI and |Pearson| are NOT directly
    # comparable in absolute scale. For two perfectly correlated Gaussians
    # with rho=0.9, Pearson = 0.9 but NMI ~ 0.58 because NMI measures the
    # fraction of the joint entropy the two series share (an information-
    # theoretic quantity), not linear regression slope. Only the RANKING
    # of pairs by NMI is meaningful for comparison with Pearson rankings.

    # ---- Rank comparison: does NMI rank pairs like Pearson does? ----
    pearson_abs = pearson.abs()
    pairs = []
    for i in range(len(nmi_m)):
        for j in range(i + 1, len(nmi_m)):
            a, b = nmi_m.columns[i], nmi_m.columns[j]
            pairs.append((a, b, float(nmi_m.iloc[i, j]), float(pearson_abs.iloc[i, j])))
    nmi_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
    pear_sorted = sorted(pairs, key=lambda x: x[3], reverse=True)
    nmi_rank = {(a, b): r for r, (a, b, _, _) in enumerate(nmi_sorted)}
    pear_rank = {(a, b): r for r, (a, b, _, _) in enumerate(pear_sorted)}
    rank_diffs = [abs(nmi_rank[(a, b)] - pear_rank[(a, b)]) for a, b, _, _ in pairs]
    console.print(
        f"\n  [dim]NMI vs |Pearson| rank agreement: "
        f"mean rank delta = {float(np.mean(rank_diffs)):.2f} "
        f"(out of {len(pairs)} pairs)[/dim]"
    )

    # ---- Hierarchical clustering ----
    console.print("\n  [dim]Hierarchical clustering (average linkage over 1-NMI)...[/dim]")
    names = list(nmi_m.columns)
    Z = hierarchical_clusters(nmi_m, method="average")
    merges = linkage_merges_text(Z, names)

    console.print()
    t = Table(title="Linkage merges (dendrogram order)", show_header=True)
    t.add_column("Step", width=4)
    t.add_column("Left", width=28)
    t.add_column("Right", width=28)
    t.add_column("Distance", width=10, justify="right")
    t.add_column("Size", width=5, justify="right")
    for step, left, right, dist, size in merges:
        t.add_row(str(step), left, right, f"{dist:.4f}", str(size))
    console.print(t)

    # ---- Cluster assignments for several k ----
    console.print()
    for k in (2, 3, 4):
        panel = Panel.fit(
            "\n".join(
                f"  [bold]{cid}[/bold] → " + ", ".join(assets)
                for cid, assets in cluster_assignments(Z, names, k).items()
            ),
            title=f"Clusters at k={k}",
            border_style="cyan",
        )
        console.print(panel)

    # ---- Rich tree view ----
    console.print()
    console.print(Panel.fit("[bold]Hierarchy tree[/bold]", border_style="green"))
    console.print(build_tree(Z, names))

    # ---- Interpretation aid: top dependencies ----
    console.print()
    t = Table(title="Top 10 pairs by NMI (with Pearson rank for comparison)",
              show_header=True)
    t.add_column("#", width=3)
    t.add_column("A", width=10)
    t.add_column("B", width=10)
    t.add_column("NMI", width=8)
    t.add_column("NMI rank", width=10)
    t.add_column("|Pearson|", width=10)
    t.add_column("Pearson rank", width=14)
    for i, (a, b, v, p) in enumerate(nmi_sorted[:10], 1):
        t.add_row(
            str(i), a, b,
            f"{v:.3f}", f"#{nmi_rank[(a, b)] + 1}",
            f"{p:.3f}", f"#{pear_rank[(a, b)] + 1}",
        )
    console.print(t)


if __name__ == "__main__":
    main()
