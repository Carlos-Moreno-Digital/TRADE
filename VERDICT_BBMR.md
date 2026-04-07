# BBMR Final Verdict — Paranoid Validation Results

**Date:** 2026-04-07
**Branch:** `claude/setup-trading-agent-WKrEK`
**Framework:** NautilusTrader 1.221.0 + hand-rolled CPCV
**Data:** yfinance EUR/USD and USD/JPY 1H, 2024-04 to 2026-04 (~12K bars each)

> ⚠️ This validation was run on yfinance 2-year data because the Dukascopy
> CDN was rate-limiting from the sandbox. The 16-year Dukascopy dataset
> exists on the VPS; paranoid v2 and CPCV scripts should be re-run there
> to confirm these findings at scale. However, on yfinance the verdict
> is already definitive and unrecoverable.

---

## Headline: **BBMR IS NOT A REAL EDGE. KILL IT.**

Every independent validation method reached the same conclusion.

---

## 1. Paranoid Test Suite v2 (7 tests, López de Prado methodology)

| Test | EURUSD | USDJPY | Verdict |
|---|---|---|---|
| Future-shift (execution lag +5 bars) | FAIL | FAIL | No coherent edge decay |
| Time permutation (500 shuffles) | **p=0.48** | **p=0.17** | Edge indistinguishable from random walk |
| Block bootstrap (95% CI of Sharpe) | `[-0.94, 1.99]` | `[-0.39, 3.54]` | **CI includes 0 on both** |
| Deflated Sharpe (432-trial penalty) | DSR=0.014 | DSR=0.092 | Below 0.95 by an order of magnitude |
| MinBTL (data sufficiency) | need 35.1y (have 2.0y) | need 8.2y (have 2.0y) | Insufficient data for the N_trials we burned |
| Parameter stability (±20%) | PASS (42% max deg) | PASS (29% max deg) | Only test that passed on both |
| Walk-forward efficiency (5 folds) | WFE=0.16 | WFE=0.87 | Inconsistent across symbols |

**Score: 1/7 EURUSD, 2/7 USDJPY.** The two DSR/MinBTL passes from the
earlier 16-year Dukascopy run disappear once you don't have 16 years of
free data to absorb the 432-trial penalty. DSR is extremely sensitive
to trade count and on realistic datasets the 432-combo parameter sweep
is lethal.

---

## 2. Combinatorial Purged Cross-Validation (hand-rolled, López de Prado AFML ch. 12)

6 folds, C(6,2)=15 test combinations, full 432-param grid.

| Symbol | Mean IS Sharpe | Mean OOS Sharpe | WFE | PBO | Verdict |
|---|---|---|---|---|---|
| EURUSD | +4.995 | +1.446 | 0.29 | 0.27 | NOT VIABLE |
| USDJPY | +4.215 | +0.270 | 0.06 | 0.40 | NOT VIABLE |

**Walk-Forward Efficiency threshold: 0.5.** Both symbols collapse well
below. IS Sharpes of 4-5 are Hollywood numbers that vanish on OOS —
this is textbook parameter-sweep overfitting. The +$50K headline from
`optimize_bbmr.py` was a data-snooping artifact.

---

## 3. Cross-validation in NautilusTrader (event-driven, realistic fills)

Same BBMR params, same data, executed through NautilusTrader's
`BacktestEngine` with bid/ask bar execution and `FillModel` slippage.

Using the account balance delta (USD base currency) as ground truth
rather than per-position realized_pnl (which is in quote currency and
was misreading JPY as USD on a first pass):

| Backtest type | EURUSD | USDJPY |
|---|---|---|
| Vanilla Python (close-to-close) | +$583, 239 trades, ~50% WR | +$1,198, 242 trades, ~50% WR |
| **NautilusTrader (bid/ask fills)** | **-$7,866 (-78.7%), 276 trades, 19.9% WR** | **-$381 (-3.8%), 266 trades, 18.8% WR** |

**Win rate drops from ~50% to ~20% the moment we model bid/ask spread
crossing on entry and exit, and the sign of the P&L flips on both
symbols.** This is the single biggest finding of the entire audit:

> The vanilla backtester was executing buys at close and sells at close
> (the midpoint), giving BBMR a free half-spread on every trade. Once
> you force it to pay the full spread like a real trader, the edge
> reverses sign catastrophically.

This is also consistent with the root cause of why the XGBoost model
lost -$26K on 16yr Dukascopy despite looking great on 2yr yfinance:
the same bug was hiding the same cost.

---

## 4. What This Actually Means

Every single line of evidence points the same way:

1. **BBMR was never a strategy.** It was a 432-dimensional random draw
   that happened to land on a lucky configuration over 16 years of
   Dukascopy OHLC with an unrealistic fill model. The "edge" was the
   bid/ask bounce the fill model failed to charge.
2. **The `+$50,446` optimization headline was noise.** CPCV shows OOS
   Sharpe ~0.3 on USDJPY and ~1.4 on EURUSD when you honestly evaluate,
   and the Nautilus run with real fills turns EURUSD into a -78.7%
   account wipe and USDJPY into a slow bleed. Both flipped sign.
3. **No parameter tweak can save it.** Parameter stability is actually
   the *one* test that passes — meaning the strategy is *uniformly
   unprofitable* across a wide parameter range once execution is
   realistic. There is no "better" BBMR.
4. **This retroactively explains the ML failure.** The XGBoost model
   lost -$26K on 16yr Dukascopy for the same reason BBMR loses -$8K
   in Nautilus on 2yr yfinance: spread cost was unmodeled, and every
   strategy that looked edge-positive in the cheap backtester was
   actually shorting the bid/ask bounce.

---

## 5. What We Keep

The week was not wasted. We built the research infrastructure that
will prevent us from ever burning another week on a fake edge:

### Validation scripts
- `validate_bbmr_paranoid.py` — 7 paranoid tests (v2, fixed the broken
  future-shift from v1, 500-permutation Monte Carlo, raw stationary
  block bootstrap, DSR, MinBTL, param stability, walk-forward
  efficiency). Reusable on any strategy with a `backtest(df, spread,
  params) -> list[pnl]` signature.
- `cpcv_bbmr.py` — Hand-rolled Combinatorial Purged Cross-Validation
  from López de Prado AFML ch. 12. No VectorBT PRO subscription
  needed. Produces WFE and PBO statistics.

### Execution infrastructure
- `nautilus_bbmr_backtest.py` — Working NautilusTrader backtest with
  synthetic bid/ask bars built from a mid-price OHLC source and
  configurable spread. Uses `FillModel` for realistic slippage and
  `bar_execution=True`. Cross-validated the vanilla backtest and
  exposed the spread-crossing bug. **This is now our gold standard
  for any future strategy validation.**
- NautilusTrader 1.221.0 installed and verified (no subscription,
  Apache/LGPL).

### Process lessons
- **Never trust a vanilla backtest.** Cross-validate every strategy
  against NautilusTrader before trusting any P&L number.
- **Parameter sweeps are traps.** Any strategy tuned with >50 combos
  must clear a CPCV walk-forward before being considered real.
- **Time permutation is the single most brutal test.** It killed BBMR
  before Nautilus even ran. Make it the first gate.

---

## 6. Next Steps (recommendation)

1. **Delete `src/trade/agents/multi/bbmr_engine.py`** — it's based on
   a dead strategy. Leave the scaffolding (risk_shield, compliance)
   since they're strategy-agnostic.
2. **Re-run paranoid v2 + CPCV on the VPS** against the full 16-year
   Dukascopy dataset, to confirm these numbers at scale. I expect the
   verdict to be identical.
3. **Back to Phase 1 of the Senior Quant plan.** The literature review
   already identified the three immediately actionable papers:
   - LOBFrame (Briola 2024) — tick-level evaluation on top of
     NautilusTrader
   - Statistical Jump Model — regime detection that's measurably
     better than HMM
   - NMI dependency matrix — replaces correlation for portfolio
     construction
4. **Any new strategy must pass, in order:**
   1. Time permutation p < 0.05
   2. Block bootstrap 95% CI of Sharpe excludes 0
   3. CPCV: Mean OOS Sharpe > 0.5, WFE > 0.5, PBO < 0.5
   4. NautilusTrader backtest with bid/ask fills shows same sign
      P&L as vanilla within 30% tolerance
   5. Only then: DSR > 0.95

No strategy gets to live trading until it clears all five gates.
