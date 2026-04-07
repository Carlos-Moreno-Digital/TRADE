"""Research strategies — candidates for validation.

NOTHING in this package is wired into Orchestrator.alpha directly.
Every module here is a candidate that must be driven through
trade.validation.run_pipeline() first. The pipeline either blesses it
and the user decides whether to promote, or rejects it and it stays
here as a negative result.
"""
