"""Hierarchical clustering over (1 - NMI) as distance.

HRP-style (Hierarchical Risk Parity) clustering: group assets whose
return series share the most statistical dependency so the portfolio
construction layer can avoid concentration in a "diversified" basket
that is in reality one bet.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def hierarchical_clusters(
    nmi_matrix: pd.DataFrame,
    method: str = "average",
) -> np.ndarray:
    """Return the scipy linkage matrix for (1 - NMI)."""
    dist = 1.0 - nmi_matrix.values
    np.fill_diagonal(dist, 0.0)
    # Numerical safety: symmetrize and clip
    dist = (dist + dist.T) / 2.0
    dist = np.clip(dist, 0.0, 1.0)
    condensed = squareform(dist, checks=False)
    return linkage(condensed, method=method)


def cluster_assignments(
    linkage_mat: np.ndarray,
    names: list[str],
    n_clusters: int,
) -> dict[int, list[str]]:
    """Cut the dendrogram at n_clusters and return {cluster_id: [assets]}."""
    labels = fcluster(linkage_mat, t=n_clusters, criterion="maxclust")
    out: dict[int, list[str]] = {}
    for name, lab in zip(names, labels):
        out.setdefault(int(lab), []).append(name)
    return dict(sorted(out.items()))


def linkage_merges_text(
    linkage_mat: np.ndarray,
    names: list[str],
) -> list[tuple[int, str, str, float, int]]:
    """Decode the linkage matrix into human-readable merges.

    Returns a list of (step, left, right, distance, size) tuples, where
    `left` and `right` can be leaf asset names or synthetic cluster
    labels like '<c5>' for intermediate merges.
    """
    n = len(names)
    labels: dict[int, str] = {i: names[i] for i in range(n)}
    merges = []
    for step, row in enumerate(linkage_mat):
        left_id = int(row[0])
        right_id = int(row[1])
        dist = float(row[2])
        size = int(row[3])
        left = labels[left_id]
        right = labels[right_id]
        new_id = n + step
        labels[new_id] = f"<c{step+1}>"
        merges.append((step + 1, left, right, dist, size))
    return merges
