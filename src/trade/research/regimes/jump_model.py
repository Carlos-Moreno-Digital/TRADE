"""Statistical Jump Model — Nystrup, Lindstrom, Madsen (2020).

Optimizes the joint state-sequence + state-parameter problem

    min over (z, theta)
        sum_t || x_t - mu_{z_t} ||^2  +  lambda * sum_t 1[z_{t-1} != z_t]

via alternating minimization:

  Step A (z | theta): exact DP/Viterbi-style pass
  Step B (theta | z): per-state means

This avoids the non-convex EM dance of HMMs and gives a transparent
"jump cost" parameter lambda. As lambda -> 0 the model collapses to
plain k-means; as lambda -> inf the entire series sits in one regime.

Reference:
  Nystrup, P., Lindstrom, E., Madsen, H. (2020). "Learning hidden
  Markov models with persistent states by penalizing jumps", Expert
  Systems with Applications 150, 113307.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class JumpModelFit:
    states: np.ndarray              # length-T int array of regime labels
    centers: np.ndarray             # (K, d) cluster centers in standardized space
    n_jumps: int                    # number of state transitions
    inertia: float                  # sum of squared distances to assigned center
    objective: float                # full SJM objective (inertia + lambda * jumps)
    lambda_: float
    n_iter: int


class StatisticalJumpModel:
    """Penalized k-means for time series, a la Nystrup et al. 2020."""

    def __init__(
        self,
        n_states: int = 2,
        lambda_: float = 5.0,
        max_iter: int = 50,
        random_state: int = 42,
    ):
        self.n_states = n_states
        self.lambda_ = float(lambda_)
        self.max_iter = max_iter
        self.random_state = random_state
        self._scaler: StandardScaler | None = None
        self.fit_: JumpModelFit | None = None

    # ------------------------------------------------------------------
    # Internal — DP over a fixed center matrix
    # ------------------------------------------------------------------
    @staticmethod
    def _viterbi(X: np.ndarray, centers: np.ndarray, lam: float) -> np.ndarray:
        T = X.shape[0]
        K = centers.shape[0]
        # Per-(t,k) emission cost: squared L2 to centre k
        cost = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # (T, K)
        V = np.empty((T, K), dtype=float)
        bp = np.empty((T, K), dtype=np.int64)
        V[0] = cost[0]
        bp[0] = -1
        for t in range(1, T):
            # For each candidate state k at time t, the best previous
            # state is either k itself (no jump) or the cheapest other
            # state plus lambda.
            prev = V[t - 1]
            no_jump = prev  # staying in the same state
            best_other_val = np.empty(K)
            best_other_arg = np.empty(K, dtype=np.int64)
            sorted_idx = np.argsort(prev)
            best_idx = sorted_idx[0]
            second_idx = sorted_idx[1] if K > 1 else best_idx
            for k in range(K):
                if k == best_idx:
                    best_other_val[k] = prev[second_idx]
                    best_other_arg[k] = second_idx
                else:
                    best_other_val[k] = prev[best_idx]
                    best_other_arg[k] = best_idx
            with_jump = best_other_val + lam
            stay_better = no_jump <= with_jump
            V[t] = cost[t] + np.where(stay_better, no_jump, with_jump)
            bp[t] = np.where(stay_better, np.arange(K), best_other_arg)

        # Backtrack
        z = np.empty(T, dtype=np.int64)
        z[-1] = int(np.argmin(V[-1]))
        for t in range(T - 2, -1, -1):
            z[t] = bp[t + 1, z[t + 1]]
        return z

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, features: pd.DataFrame) -> "StatisticalJumpModel":
        X_raw = features.values.astype(float)
        self._scaler = StandardScaler().fit(X_raw)
        X = self._scaler.transform(X_raw)

        # Init centers via k-means
        km = KMeans(
            n_clusters=self.n_states,
            n_init=10,
            random_state=self.random_state,
        ).fit(X)
        centers = km.cluster_centers_.copy()
        z = km.labels_.copy()

        prev_z = None
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            # Step A: relabel via Viterbi DP
            z = self._viterbi(X, centers, self.lambda_)
            if prev_z is not None and np.array_equal(z, prev_z):
                break
            prev_z = z.copy()

            # Step B: update centers as per-state means (skip empty states)
            for k in range(self.n_states):
                mask = z == k
                if mask.any():
                    centers[k] = X[mask].mean(axis=0)

        # Final metrics
        cost = ((X - centers[z]) ** 2).sum()
        n_jumps = int((np.diff(z) != 0).sum())
        objective = float(cost + self.lambda_ * n_jumps)
        self.fit_ = JumpModelFit(
            states=z,
            centers=centers,
            n_jumps=n_jumps,
            inertia=float(cost),
            objective=objective,
            lambda_=self.lambda_,
            n_iter=n_iter,
        )
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.fit_ is None or self._scaler is None:
            raise RuntimeError("Call fit() before predict()")
        X = self._scaler.transform(features.values.astype(float))
        return self._viterbi(X, self.fit_.centers, self.lambda_)

    # ------------------------------------------------------------------
    # Diagnostics — small dictionaries only, never raw arrays
    # ------------------------------------------------------------------
    def regime_summary(
        self,
        features: pd.DataFrame,
        labels: np.ndarray | None = None,
    ) -> dict:
        """Return a small dict with per-regime statistics for printing.

        Never returns raw arrays; designed to be safe to dump in chat
        or in commit messages.
        """
        if labels is None:
            if self.fit_ is None:
                raise RuntimeError("fit first")
            labels = self.fit_.states
        out: dict = {
            "n_obs": int(len(labels)),
            "n_states": int(self.n_states),
            "n_jumps": int((np.diff(labels) != 0).sum()),
            "lambda": self.lambda_,
            "regimes": {},
        }
        # Inverse-transform centers to interpret in original units
        if self._scaler is not None and self.fit_ is not None:
            centers_orig = self._scaler.inverse_transform(self.fit_.centers)
        else:
            centers_orig = None

        for k in range(self.n_states):
            mask = labels == k
            count = int(mask.sum())
            share = count / len(labels) if len(labels) else 0.0
            # Average run length: count / number of runs starting in k
            runs = 0
            in_run = False
            for v in labels:
                if v == k and not in_run:
                    runs += 1
                    in_run = True
                elif v != k:
                    in_run = False
            avg_run = (count / runs) if runs > 0 else 0.0
            entry: dict = {
                "count": count,
                "share": round(share, 3),
                "n_runs": runs,
                "avg_run_len": round(avg_run, 1),
            }
            if centers_orig is not None:
                entry["centroid"] = {
                    name: round(float(centers_orig[k, i]), 4)
                    for i, name in enumerate(features.columns)
                }
            out["regimes"][k] = entry
        return out
