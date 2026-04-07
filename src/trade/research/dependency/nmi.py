"""Kraskov-Stögbauer-Grassberger MI + Kozachenko-Leonenko entropy.

Hand-rolled implementations, no sklearn black-box. Uses Chebyshev
(L-infinity) distance as in Kraskov 2004 Algorithm 1 so the MI bias
cancels cleanly between the joint and marginal counts.

All returns are in nats (natural log).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma


def _to_col(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1, 1)
    # Tiny gaussian noise to break ties (needed for kNN with discrete/repeated values)
    rng = np.random.default_rng(0)
    arr = arr + 1e-10 * rng.standard_normal(arr.shape)
    return arr


def kraskov_mi(x, y, k: int = 5) -> float:
    """Kraskov Algorithm 1 mutual information estimator (nats).

    MI(X,Y) = psi(k) - <psi(n_x + 1) + psi(n_y + 1)> + psi(N)

    where n_x(i), n_y(i) are the counts of points strictly within
    the Chebyshev distance eps(i)/2 to point i along each axis, and
    eps(i) is twice the distance to the k-th nearest neighbor in the
    joint (X,Y) space.
    """
    x = _to_col(x)
    y = _to_col(y)
    n = len(x)
    if n < k + 2:
        return 0.0
    xy = np.hstack([x, y])
    tree = cKDTree(xy)
    # Distance to k-th neighbor in joint space (p=inf => Chebyshev).
    # query returns self as 0th neighbor, so k+1 gets the k-th true neighbor.
    d_joint, _ = tree.query(xy, k=k + 1, p=np.inf)
    eps = d_joint[:, -1]  # shape (n,)

    tree_x = cKDTree(x)
    tree_y = cKDTree(y)
    # Count of marginal neighbors strictly within eps (exclude self).
    nx = np.empty(n, dtype=np.int64)
    ny = np.empty(n, dtype=np.int64)
    for i in range(n):
        r = max(eps[i] - 1e-12, 0.0)
        nx[i] = len(tree_x.query_ball_point(x[i], r=r, p=np.inf)) - 1
        ny[i] = len(tree_y.query_ball_point(y[i], r=r, p=np.inf)) - 1

    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)
    return float(max(0.0, mi))


def kozachenko_entropy(x, k: int = 5) -> float:
    """Kozachenko-Leonenko differential entropy estimator (nats).

    H(X) = -psi(k) + psi(N) + log(c_d) + (d / N) * sum(log(2 * eps_i))

    For Chebyshev norm in d dimensions the unit ball volume is 2^d,
    so log(c_d) = d * log(2).
    """
    x = _to_col(x)
    n = len(x)
    d = x.shape[1]
    if n < k + 2:
        return 0.0
    tree = cKDTree(x)
    d_self, _ = tree.query(x, k=k + 1, p=np.inf)
    eps = d_self[:, -1]
    eps = np.maximum(eps, 1e-12)
    log_cd = d * np.log(2.0)
    return float(
        -digamma(k) + digamma(n) + log_cd + (d / n) * np.sum(np.log(2.0 * eps))
    )


def nmi(x, y, k: int = 5) -> float:
    """Normalized Mutual Information in [0, 1].

    NMI(X,Y) = MI(X,Y) / sqrt(H(X) * H(Y))

    Clipped to [0, 1] because the estimators are noisy and can
    produce slightly out-of-range values on small samples.
    """
    mi = kraskov_mi(x, y, k=k)
    hx = kozachenko_entropy(x, k=k)
    hy = kozachenko_entropy(y, k=k)
    denom = np.sqrt(max(hx * hy, 1e-12))
    if denom <= 0:
        return 0.0
    return float(np.clip(mi / denom, 0.0, 1.0))
