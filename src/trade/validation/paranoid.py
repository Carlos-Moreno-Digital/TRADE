"""Paranoid validation suite — strategy-agnostic.

Implements 7 tests from Lopez de Prado's AFML + the canonical quant
validation checklist. Any strategy that does not clear ALL gates is noise.

Tests:
  1. Future-shift (execution lag test)
  2. Time permutation (N shuffles of log-returns)
  3. Stationary block bootstrap of trade P&L (95% CI excludes 0)
  4. Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
  5. MinBTL (enough data for N_trials?)
  6. Parameter stability (+/-20% sweep)
  7. Walk-forward efficiency (5 sequential folds)

Usage
-----
    strat = MyStrategy()
    suite = ParanoidSuite(strat, df, spread=0.00008, n_trials=strat.n_grid)
    result = suite.run_all()
    if not result.all_passed:
        raise SystemExit(f"Rejected: {result.failed_tests}")
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

from trade.validation.protocol import StrategyProtocol


PERIODS_PER_YEAR = 220  # average trades/year sanity


def _sharpe(pnls: np.ndarray, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    if len(pnls) < 2 or np.std(pnls) == 0:
        return 0.0
    return float(np.mean(pnls) / np.std(pnls) * np.sqrt(periods_per_year))


@dataclass
class ParanoidResult:
    per_test: dict[str, dict] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        return all(r.get("passed", False) for r in self.per_test.values())

    @property
    def failed_tests(self) -> list[str]:
        return [k for k, r in self.per_test.items() if not r.get("passed", False)]

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.per_test.values() if r.get("passed", False))

    def summary(self) -> str:
        lines = [f"Paranoid: {self.passed_count}/{len(self.per_test)} passed"]
        for name, r in self.per_test.items():
            mark = "PASS" if r.get("passed") else "FAIL"
            lines.append(f"  [{mark}] {name}: {r.get('verdict', '?')}")
        return "\n".join(lines)


class ParanoidSuite:
    def __init__(
        self,
        strategy: StrategyProtocol,
        df: pd.DataFrame,
        spread: float,
        n_trials: int,
        params: dict[str, Any] | None = None,
        seed: int = 42,
    ):
        self.strategy = strategy
        self.df = df
        self.spread = spread
        self.n_trials = n_trials
        self.params = params or strategy.default_params
        self.rng = np.random.default_rng(seed)
        self._pnls_cache: list[float] | None = None

    def _pnls(self) -> list[float]:
        if self._pnls_cache is None:
            self._pnls_cache = self.strategy.backtest(
                self.df, self.spread, self.params
            )
        return self._pnls_cache

    # -------- Test 1 --------
    def future_shift(self) -> dict:
        normal = self.strategy.backtest(self.df, self.spread, self.params, 0)
        lagged = self.strategy.backtest(self.df, self.spread, self.params, 5)
        cheat = self.strategy.backtest(self.df, self.spread, self.params, -5)
        n, l, c = sum(normal), sum(lagged), sum(cheat)
        denom = max(abs(n), 1.0)
        lag_decay = (n - l) / denom
        cheat_boost = (c - n) / denom
        passed = lag_decay > 0.1 and cheat_boost > 0.5
        return {
            "normal_pnl": round(n, 2),
            "lagged_pnl": round(l, 2),
            "cheat_pnl": round(c, 2),
            "lag_decay": round(lag_decay, 3),
            "cheat_boost": round(cheat_boost, 3),
            "passed": passed,
            "verdict": (
                f"PASS (decay={lag_decay:.2f} cheat={cheat_boost:.2f})"
                if passed
                else f"FAIL (decay={lag_decay:.2f} cheat={cheat_boost:.2f})"
            ),
        }

    # -------- Test 2 --------
    def time_permutation(self, n_permutations: int = 500) -> dict:
        """Time-permutation test.

        For pair-trading or otherwise multi-input strategies, the
        strategy may declare a `with_permuted_state(rng)` context
        manager that the suite enters BEFORE running each permuted
        backtest. The hook is responsible for swapping any internal
        secondary inputs (e.g. partner price series) with shuffled
        versions, then restoring them on exit. This is the lockstep
        fix for the single-leg pair trading p=1.000 caveat.
        """
        normal_pnls = np.array(self._pnls())
        normal_sr = _sharpe(normal_pnls)
        close = self.df["close"].values.astype(float)
        log_rets = np.diff(np.log(close))
        spread_pct = ((self.df["high"] - self.df["low"]) / self.df["close"]).values

        has_state_hook = hasattr(self.strategy, "with_permuted_state")

        shuffled_srs = np.zeros(n_permutations)
        for k in range(n_permutations):
            sr = self.rng.permutation(log_rets)
            new_close = close[0] * np.exp(np.concatenate([[0], np.cumsum(sr)]))
            shuf_spread = self.rng.permutation(spread_pct)
            df_s = pd.DataFrame({
                "open": new_close,
                "close": new_close,
                "high": new_close * (1 + shuf_spread / 2),
                "low": new_close * (1 - shuf_spread / 2),
                "volume": self.df.get("volume", 1),
            }, index=self.df.index)
            if has_state_hook:
                with self.strategy.with_permuted_state(self.rng):
                    pnls = self.strategy.backtest(df_s, self.spread, self.params)
            else:
                pnls = self.strategy.backtest(df_s, self.spread, self.params)
            shuffled_srs[k] = _sharpe(np.array(pnls))

        p_value = float((shuffled_srs >= normal_sr).mean())
        passed = p_value < 0.05
        return {
            "normal_sr": round(normal_sr, 3),
            "mean_shuf_sr": round(float(shuffled_srs.mean()), 3),
            "max_shuf_sr": round(float(shuffled_srs.max()), 3),
            "p_value": round(p_value, 4),
            "n_permutations": n_permutations,
            "passed": passed,
            "verdict": (
                f"PASS (p={p_value:.3f} < 0.05)"
                if passed
                else f"FAIL (p={p_value:.3f} >= 0.05)"
            ),
        }

    # -------- Test 3 --------
    def block_bootstrap(self, n_bootstrap: int = 1000, block_size: int = 20) -> dict:
        pnls = np.array(self._pnls())
        if len(pnls) < block_size * 5:
            return {"passed": False, "verdict": "SKIP (too few trades)"}

        observed = _sharpe(pnls)
        boot = np.zeros(n_bootstrap)
        n = len(pnls)
        for i in range(n_bootstrap):
            sample = []
            while len(sample) < n:
                start = self.rng.integers(0, n)
                length = max(1, int(self.rng.geometric(1.0 / block_size)))
                sample.extend(pnls[start:start + length].tolist())
            boot[i] = _sharpe(np.array(sample[:n]))
        ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        passed = ci_lo > 0
        return {
            "observed_sr": round(observed, 3),
            "ci_95_lo": round(ci_lo, 3),
            "ci_95_hi": round(ci_hi, 3),
            "passed": passed,
            "verdict": (
                f"PASS (95% CI [{ci_lo:.2f}, {ci_hi:.2f}] excludes 0)"
                if passed
                else f"FAIL (95% CI [{ci_lo:.2f}, {ci_hi:.2f}] includes 0)"
            ),
        }

    # -------- Test 4 --------
    def deflated_sharpe(self) -> dict:
        pnls = np.array(self._pnls())
        if len(pnls) < 30:
            return {"passed": False, "verdict": "SKIP (<30 trades)"}
        T = len(pnls)
        sr = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0.0
        skew = float(stats.skew(pnls))
        kurt = float(stats.kurtosis(pnls, fisher=False))

        gamma_e = 0.5772
        exp_max = (
            (1 - gamma_e) * norm.ppf(1 - 1.0 / self.n_trials)
            + gamma_e * norm.ppf(1 - 1.0 / (self.n_trials * np.e))
        )
        sr0 = exp_max / np.sqrt(T)
        try:
            denom = np.sqrt(max(1e-10, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
            dsr = float(norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom))
        except Exception:
            dsr = 0.0
        passed = dsr > 0.95
        return {
            "trades": T,
            "sr_per_trade": round(sr, 4),
            "skew": round(skew, 2),
            "kurt": round(kurt, 2),
            "dsr": round(dsr, 3),
            "passed": passed,
            "verdict": f"PASS (DSR={dsr:.3f})" if passed else f"FAIL (DSR={dsr:.3f})",
        }

    # -------- Test 5 --------
    def minbtl(self) -> dict:
        pnls = np.array(self._pnls())
        years = (self.df.index[-1] - self.df.index[0]).days / 365
        if len(pnls) < 30 or np.std(pnls) == 0:
            return {"passed": False, "verdict": "SKIP"}
        tpy = len(pnls) / years
        sr_ann = np.mean(pnls) / np.std(pnls) * np.sqrt(tpy)
        required = (2 * math.log(self.n_trials)) / (sr_ann ** 2) if sr_ann > 0 else 9999.0
        passed = years >= required
        return {
            "sr_ann": round(float(sr_ann), 2),
            "data_years": round(years, 1),
            "required_years": round(float(required), 1),
            "passed": passed,
            "verdict": (
                f"PASS ({years:.1f}y >= {required:.1f}y)"
                if passed
                else f"FAIL ({years:.1f}y < {required:.1f}y)"
            ),
        }

    # -------- Test 6 --------
    def param_stability(self) -> dict:
        base = sum(self._pnls())
        if base <= 0:
            return {"passed": False, "verdict": "SKIP (base P&L <= 0)"}
        max_deg = 0.0
        for key, val in self.params.items():
            if not isinstance(val, (int, float)):
                continue
            for mult in (0.8, 1.2):
                new_val = val * mult
                if isinstance(val, int):
                    new_val = max(2, int(round(new_val)))
                p = dict(self.params)
                p[key] = new_val
                pnl = sum(self.strategy.backtest(self.df, self.spread, p))
                deg = (base - pnl) / base
                max_deg = max(max_deg, deg)
        passed = max_deg < 0.5
        return {
            "base_pnl": round(base, 2),
            "max_degradation": round(float(max_deg), 3),
            "passed": passed,
            "verdict": (
                f"PASS (max deg={max_deg:.1%} < 50%)"
                if passed
                else f"FAIL (max deg={max_deg:.1%} >= 50%)"
            ),
        }

    # -------- Test 7 --------
    def wfe(self, n_splits: int = 5) -> dict:
        n = len(self.df)
        fold_size = n // (n_splits + 1)
        is_srs, oos_srs = [], []
        for k in range(n_splits):
            is_end = fold_size * (k + 1)
            oos_end = min(n, is_end + fold_size)
            if oos_end - is_end < 2000:
                continue
            is_df = self.df.iloc[:is_end]
            oos_df = self.df.iloc[is_end:oos_end]
            is_pnls = self.strategy.backtest(is_df, self.spread, self.params)
            oos_pnls = self.strategy.backtest(oos_df, self.spread, self.params)
            if len(is_pnls) > 10 and len(oos_pnls) > 10:
                is_srs.append(_sharpe(np.array(is_pnls)))
                oos_srs.append(_sharpe(np.array(oos_pnls)))
        if not is_srs or not oos_srs:
            return {"passed": False, "verdict": "SKIP"}
        mean_is = float(np.mean(is_srs))
        mean_oos = float(np.mean(oos_srs))
        wfe_val = mean_oos / mean_is if mean_is > 0 else 0.0
        passed = wfe_val >= 0.5 and mean_oos > 0
        return {
            "mean_is_sr": round(mean_is, 3),
            "mean_oos_sr": round(mean_oos, 3),
            "wfe": round(float(wfe_val), 3),
            "passed": passed,
            "verdict": (
                f"PASS (WFE={wfe_val:.2f} OOS={mean_oos:.2f})"
                if passed
                else f"FAIL (WFE={wfe_val:.2f} OOS={mean_oos:.2f})"
            ),
        }

    def run_all(self) -> ParanoidResult:
        r = ParanoidResult()
        r.per_test["future_shift"] = self.future_shift()
        r.per_test["time_permutation"] = self.time_permutation()
        r.per_test["block_bootstrap"] = self.block_bootstrap()
        r.per_test["deflated_sharpe"] = self.deflated_sharpe()
        r.per_test["minbtl"] = self.minbtl()
        r.per_test["param_stability"] = self.param_stability()
        r.per_test["wfe"] = self.wfe()
        return r
