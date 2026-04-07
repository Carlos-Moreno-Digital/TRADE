"""Non-linear dependency analysis — NMI matrix + hierarchical clustering.

Replaces Pearson correlation for portfolio construction. Pearson is
linear and collapses in the tails exactly when we need it (crisis
regimes). Normalized Mutual Information captures any statistical
dependence (linear, non-linear, tail) without distributional
assumptions.

References:
  - Kraskov, Stögbauer, Grassberger (2004), "Estimating mutual
    information", Phys. Rev. E 69, 066138.
  - Kozachenko, Leonenko (1987) entropy estimator.
"""
from trade.research.dependency.nmi import (
    kraskov_mi,
    kozachenko_entropy,
    nmi,
)
from trade.research.dependency.matrix import (
    build_nmi_matrix,
    align_returns,
)
from trade.research.dependency.clustering import (
    hierarchical_clusters,
    cluster_assignments,
    linkage_merges_text,
)

__all__ = [
    "kraskov_mi",
    "kozachenko_entropy",
    "nmi",
    "build_nmi_matrix",
    "align_returns",
    "hierarchical_clusters",
    "cluster_assignments",
    "linkage_merges_text",
]
