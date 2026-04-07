"""Strategy validation pipeline — institutional rigor, no subscriptions.

Every strategy MUST clear these gates IN ORDER before reaching paper/live:
  1. Paranoid suite (time permutation, bootstrap, DSR, MinBTL, param stability, WFE)
  2. CPCV (WFE >= 0.5, mean OOS Sharpe > 0.5, PBO < 0.5)
  3. NautilusTrader realistic-fill backtest with sign-consistency vs vanilla

Any strategy that fails ANY gate does NOT reach Orchestrator.alpha.
"""

from trade.validation.protocol import StrategyProtocol, PipelineResult
from trade.validation.paranoid import ParanoidSuite, ParanoidResult
from trade.validation.cpcv import CPCV, CPCVResult
from trade.validation.nautilus_harness import NautilusHarness, NautilusResult

__all__ = [
    "StrategyProtocol",
    "PipelineResult",
    "ParanoidSuite",
    "ParanoidResult",
    "CPCV",
    "CPCVResult",
    "NautilusHarness",
    "NautilusResult",
]
