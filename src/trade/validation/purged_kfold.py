"""Purged K-Fold cross-validation (Lopez de Prado AFML ch. 7).

When training a classifier on labels that look forward in time
(e.g. triple-barrier labels), naive K-Fold leaks information across
folds because:
  - A training sample at time t_i has a label horizon ending at
    t_i + h. If a test sample falls inside [t_i, t_i + h], the
    training and test labels share information.
  - Serial correlation in returns makes adjacent samples not iid,
    so even disjoint train/test windows can leak.

The fix is two layers around each test fold:

  PURGE  : remove from the training set any sample whose label
           horizon overlaps the test fold.
  EMBARGO: also remove training samples within `embargo` bars of
           the test fold boundary, to absorb residual autocorrelation.

This module implements `purged_kfold_predict_proba` which:
  - Splits a list of signal events into K time-contiguous folds
    by signal index (not by raw bar index).
  - For each fold, builds the train index set with purge + embargo.
  - Trains a classifier on the training samples.
  - Predicts probabilities on the test samples.
  - Returns one OOS probability per signal.

The classifier is supplied as a factory callable so the caller can
parameterise it freely (e.g. RandomForest with different n_estimators).

Reference:
  Lopez de Prado, M. (2018). Advances in Financial Machine Learning,
  ch. 7 ("Cross-Validation in Finance"), Algorithm 7.4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class PurgedKFoldFold:
    fold_id: int
    test_signal_idx: list[int]
    train_signal_idx: list[int]
    n_purged: int       # how many training candidates were removed
    n_embargoed: int    # of those, how many were embargoed (vs purged)


def make_purged_folds(
    signal_bar_indices: np.ndarray,
    label_horizon: int,
    n_splits: int,
    embargo_bars: int,
) -> list[PurgedKFoldFold]:
    """Build the K folds without training the classifier.

    Parameters
    ----------
    signal_bar_indices : array of int
        For each signal, the bar index at which it was emitted.
        Must be sorted ascending (the signals dataframe always is).
    label_horizon : int
        How many bars forward each label looks at.
    n_splits : int
        Number of folds K.
    embargo_bars : int
        Embargo around each test fold.
    """
    n = len(signal_bar_indices)
    if n_splits < 2 or n < n_splits:
        raise ValueError(f"need >= 2 splits and >= n_splits signals; got {n}")

    fold_size = n // n_splits
    folds: list[PurgedKFoldFold] = []
    for fold_id in range(n_splits):
        test_start = fold_id * fold_size
        test_end = (fold_id + 1) * fold_size if fold_id < n_splits - 1 else n
        test_idx = list(range(test_start, test_end))

        # Time interval covered by the test fold
        test_t_min = int(signal_bar_indices[test_start])
        test_t_max = int(signal_bar_indices[test_end - 1])

        # PURGE: drop a training signal if its label horizon
        # [t, t + label_horizon] overlaps the test interval
        # [test_t_min, test_t_max].
        # EMBARGO: also drop training signals within embargo_bars of
        # either side of the test interval.
        purge_lo = test_t_min - label_horizon - embargo_bars
        purge_hi = test_t_max + embargo_bars

        train_idx: list[int] = []
        n_purged = 0
        n_embargoed = 0
        for i in range(n):
            if test_start <= i < test_end:
                continue
            t = int(signal_bar_indices[i])
            t_horizon_end = t + label_horizon
            # Overlap test of [t, t_horizon_end] with [purge_lo, purge_hi]
            if t_horizon_end < purge_lo or t > purge_hi:
                train_idx.append(i)
            else:
                n_purged += 1
                if (test_t_min - embargo_bars <= t <= test_t_min) or \
                   (test_t_max <= t <= test_t_max + embargo_bars):
                    n_embargoed += 1

        folds.append(PurgedKFoldFold(
            fold_id=fold_id,
            test_signal_idx=test_idx,
            train_signal_idx=train_idx,
            n_purged=n_purged,
            n_embargoed=n_embargoed,
        ))
    return folds


def purged_kfold_predict_proba(
    X: np.ndarray,
    y: np.ndarray,
    signal_bar_indices: np.ndarray,
    label_horizon: int,
    n_splits: int,
    embargo_bars: int,
    classifier_factory: Callable[[], object],
    progress_fn: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, list[PurgedKFoldFold]]:
    """Run purged K-Fold and return one OOS probability per signal.

    Parameters
    ----------
    X : (n, d) feature matrix, one row per signal
    y : (n,) binary labels
    signal_bar_indices : (n,) bar index of each signal
    label_horizon : int
        Maximum forward horizon used by the labelling
    n_splits : int
    embargo_bars : int
    classifier_factory : callable returning a fresh sklearn classifier
        with .fit(X, y) and .predict_proba(X) -> (n, 2)
    progress_fn : optional one-line logger called per fold

    Returns
    -------
    (oos_probs, folds)
        oos_probs : (n,) probability of class 1 for each signal,
                    computed entirely OUT-OF-SAMPLE.
        folds : the PurgedKFoldFold list with diagnostics.
    """
    n = len(X)
    out = np.full(n, np.nan, dtype=float)
    folds = make_purged_folds(
        signal_bar_indices, label_horizon, n_splits, embargo_bars
    )
    for f in folds:
        train_idx = f.train_signal_idx
        test_idx = f.test_signal_idx
        if len(train_idx) < 20:
            if progress_fn:
                progress_fn(
                    f"  fold {f.fold_id+1}/{n_splits}: skip "
                    f"(only {len(train_idx)} train samples)"
                )
            continue
        X_train = X[train_idx]
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            if progress_fn:
                progress_fn(
                    f"  fold {f.fold_id+1}/{n_splits}: skip "
                    f"(degenerate target)"
                )
            continue
        clf = classifier_factory()
        clf.fit(X_train, y_train)
        X_test = X[test_idx]
        probs = clf.predict_proba(X_test)
        # Pick the class-1 column robustly (sklearn orders classes_)
        if hasattr(clf, "classes_") and 1 in list(clf.classes_):
            class1_col = list(clf.classes_).index(1)
        else:
            class1_col = probs.shape[1] - 1
        out[test_idx] = probs[:, class1_col]
        if progress_fn:
            progress_fn(
                f"  fold {f.fold_id+1}/{n_splits}: "
                f"train={len(train_idx)} test={len(test_idx)} "
                f"purged={f.n_purged} embargoed={f.n_embargoed}"
            )
    return out, folds
