"""Regime detection — Statistical Jump Model (Nystrup, Lindstrom, Madsen 2020).

The SJM replaces HMM for regime detection because:
- It is convex once the discrete labels are fixed (vs. HMM's
  non-convex EM landscape with label switching).
- The lambda jump penalty is interpretable (cost in feature-distance
  units of switching states), unlike HMM transition probabilities.
- It is more stable on small samples and high-noise financial data.
"""
from trade.research.regimes.features import build_features
from trade.research.regimes.jump_model import StatisticalJumpModel

__all__ = ["build_features", "StatisticalJumpModel"]
