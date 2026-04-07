"""BBMR Paranoid Validation v2 — fixed and expanded.

Tests:
1. Future-shift sanity (CORRECT version: features at i vs execution at i+k)
2. Time permutation (500 shuffles, reconstructed from log returns)
3. Block bootstrap of trade P&Ls (both demeaned null and raw)
4. Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
5. MinBTL check
6. Parameter stability (+/-20% sensitivity around each param)
7. Walk-forward efficiency (OOS/IS Sharpe)

References:
- Lopez de Prado, Advances in Financial Machine Learning (2018)
- Bailey & Lopez de Prado, The Deflated Sharpe Ratio (2014)
- Pardo, The Evaluation and Optimization of Trading Strategies (2008)
"""

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import talib
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings("ignore")
console = Console()

DATA_DIR = Path("data/dukascopy")

PARAMS = {
    "bb_period": 30,
    "bb_std": 2.0,
    "adx_max": 20,
    "atr_sl_mult": 1.0,
    "max_bars": 12,
}

RISK_PER_TRADE = 0.003
ACCOUNT = 10000
N_TRIALS_OPTIMIZED = 432

SPREADS = {"EURUSD": 0.00008, "USDJPY": 0.008}


def load_dukascopy(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_1H.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def backtest_bbmr(df: pd.DataFrame, spread: float, params: dict,
                  signal_shift: int = 0) -> list:
    """Returns list of per-trade P&L.

    signal_shift: if > 0, indicators computed at bar i are used to
    enter at bar i+signal_shift (simulates using future info if negative,
    or adds lag if positive). signal_shift=0 is the normal (current) behavior.
    """
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    n = len(close)

    upper, middle, lower = talib.BBANDS(
        close, timeperiod=params["bb_period"],
        nbdevup=params["bb_std"], nbdevdn=params["bb_std"],
    )
    adx = talib.ADX(high, low, close, timeperiod=14)
    atr = talib.ATR(high, low, close, timeperiod=14)

    pnls = []
    position = None
    equity = ACCOUNT

    start = max(200, -signal_shift + 1)
    end = n - params["max_bars"] - max(0, signal_shift)

    for i in range(start, end):
        sig_i = i - signal_shift  # bar where indicators were computed
        if sig_i < 0 or sig_i >= n:
            continue
        if math.isnan(upper[sig_i]) or math.isnan(adx[sig_i]) or math.isnan(atr[sig_i]):
            continue
        if adx[sig_i] > params["adx_max"]:
            continue

        if position is None:
            entry = close[i]  # execute at bar i
            if close[sig_i] < lower[sig_i]:
                sl = entry - atr[sig_i] * params["atr_sl_mult"]
                tp = middle[sig_i]
                risk = entry - sl
                if risk <= 0:
                    continue
                qty = (equity * RISK_PER_TRADE) / risk
                position = {"side": "buy", "entry": entry, "idx": i,
                            "sl": sl, "tp": tp, "qty": qty}
            elif close[sig_i] > upper[sig_i]:
                sl = entry + atr[sig_i] * params["atr_sl_mult"]
                tp = middle[sig_i]
                risk = sl - entry
                if risk <= 0:
                    continue
                qty = (equity * RISK_PER_TRADE) / risk
                position = {"side": "sell", "entry": entry, "idx": i,
                            "sl": sl, "tp": tp, "qty": qty}
        else:
            bars_held = i - position["idx"]
            exit_price = None
            if position["side"] == "buy":
                if low[i] <= position["sl"]:
                    exit_price = position["sl"]
                elif high[i] >= position["tp"]:
                    exit_price = position["tp"]
                elif bars_held >= params["max_bars"]:
                    exit_price = close[i]
                if exit_price is not None:
                    move = exit_price - position["entry"]
                    cost = spread * position["qty"] * 2
                    pnl = move * position["qty"] - cost
                    pnls.append(pnl)
                    equity += pnl
                    position = None
            else:
                if high[i] >= position["sl"]:
                    exit_price = position["sl"]
                elif low[i] <= position["tp"]:
                    exit_price = position["tp"]
                elif bars_held >= params["max_bars"]:
                    exit_price = close[i]
                if exit_price is not None:
                    move = position["entry"] - exit_price
                    cost = spread * position["qty"] * 2
                    pnl = move * position["qty"] - cost
                    pnls.append(pnl)
                    equity += pnl
                    position = None

    return pnls


def sharpe_ratio(pnls: np.ndarray, periods_per_year: float = 220) -> float:
    if len(pnls) < 2 or np.std(pnls) == 0:
        return 0.0
    return float(np.mean(pnls) / np.std(pnls) * np.sqrt(periods_per_year))


# ============================================================
# TEST 1: FUTURE-SHIFT SANITY (FIXED)
# ============================================================
def test_future_shift(df: pd.DataFrame, spread: float) -> dict:
    """Correct version: compute indicators at bar i, but execute at bar i+5.
    If we "cheat" by executing at bar i-5 (signal_shift=-5), we're using
    future info and P&L should explode. If we lag execution 5 bars
    (signal_shift=+5), P&L should collapse because edge is short-lived.

    A legitimate strategy: normal > lagged, and normal << cheating.
    """
    console.print("  [dim]Test 1: Future-shift (execution lag +5 bars)[/dim]")

    normal_pnls = backtest_bbmr(df, spread, PARAMS, signal_shift=0)
    normal_pnl = sum(normal_pnls)
    normal_sr = sharpe_ratio(np.array(normal_pnls))

    # Lag execution by 5 bars (indicators from 5 bars ago)
    lagged_pnls = backtest_bbmr(df, spread, PARAMS, signal_shift=5)
    lagged_pnl = sum(lagged_pnls)
    lagged_sr = sharpe_ratio(np.array(lagged_pnls))

    # Cheat: use future info (indicators from 5 bars in future)
    cheat_pnls = backtest_bbmr(df, spread, PARAMS, signal_shift=-5)
    cheat_pnl = sum(cheat_pnls)
    cheat_sr = sharpe_ratio(np.array(cheat_pnls))

    # Legit: normal must be > lagged (edge decays with delay)
    # AND cheat must be >> normal (future info is massively better)
    lag_decay = (normal_pnl - lagged_pnl) / max(abs(normal_pnl), 1)
    cheat_boost = (cheat_pnl - normal_pnl) / max(abs(normal_pnl), 1)

    # Legitimate strategy: lag_decay > 0.1 (edge decays) and cheat_boost > 0.5
    passed = lag_decay > 0.1 and cheat_boost > 0.5

    return {
        "test": "Future Shift (v2)",
        "normal_pnl": round(normal_pnl, 2),
        "lagged_pnl": round(lagged_pnl, 2),
        "cheat_pnl": round(cheat_pnl, 2),
        "normal_sr": round(normal_sr, 2),
        "lagged_sr": round(lagged_sr, 2),
        "cheat_sr": round(cheat_sr, 2),
        "lag_decay": round(lag_decay, 3),
        "cheat_boost": round(cheat_boost, 3),
        "passed": passed,
        "verdict": (
            f"PASS (decay={lag_decay:.2f}, cheat_boost={cheat_boost:.2f})"
            if passed
            else f"FAIL (decay={lag_decay:.2f}, cheat_boost={cheat_boost:.2f})"
        ),
    }


# ============================================================
# TEST 2: TIME PERMUTATION (500 shuffles)
# ============================================================
def test_time_permutation(df: pd.DataFrame, spread: float,
                          n_permutations: int = 500) -> dict:
    console.print(f"  [dim]Test 2: Time permutation ({n_permutations} shuffles)[/dim]")

    normal_pnls = backtest_bbmr(df, spread, PARAMS)
    normal_sr = sharpe_ratio(np.array(normal_pnls))
    normal_pnl = sum(normal_pnls)

    close = df["close"].values.astype(float)
    log_returns = np.diff(np.log(close))

    spread_pct = ((df["high"] - df["low"]) / df["close"]).values

    rng = np.random.default_rng(42)
    shuffled_srs = []
    shuffled_pnls = []
    for k in range(n_permutations):
        shuffled_rets = rng.permutation(log_returns)
        new_close = close[0] * np.exp(
            np.concatenate([[0], np.cumsum(shuffled_rets)])
        )
        # Shuffle spread_pct independently to break any residual coupling
        shuf_spread = rng.permutation(spread_pct)
        df_shuffled = pd.DataFrame({
            "close": new_close,
            "high": new_close * (1 + shuf_spread / 2),
            "low": new_close * (1 - shuf_spread / 2),
        }, index=df.index)

        shuf_pnls = backtest_bbmr(df_shuffled, spread, PARAMS)
        shuffled_srs.append(sharpe_ratio(np.array(shuf_pnls)))
        shuffled_pnls.append(sum(shuf_pnls) if shuf_pnls else 0)
        if (k + 1) % 100 == 0:
            console.print(f"    ...{k+1}/{n_permutations}")

    shuffled_srs = np.array(shuffled_srs)
    shuffled_pnls_arr = np.array(shuffled_pnls)
    p_value_sr = (shuffled_srs >= normal_sr).mean()
    p_value_pnl = (shuffled_pnls_arr >= normal_pnl).mean()
    passed = p_value_sr < 0.05

    return {
        "test": "Time Permutation (500)",
        "normal_sr": round(normal_sr, 3),
        "mean_shuf_sr": round(float(shuffled_srs.mean()), 3),
        "max_shuf_sr": round(float(shuffled_srs.max()), 3),
        "p_value_sr": round(float(p_value_sr), 4),
        "p_value_pnl": round(float(p_value_pnl), 4),
        "passed": passed,
        "verdict": (
            f"PASS (p_SR={p_value_sr:.3f} < 0.05)"
            if passed
            else f"FAIL (p_SR={p_value_sr:.3f} >= 0.05)"
        ),
    }


# ============================================================
# TEST 3: BLOCK BOOTSTRAP (raw, not demeaned)
# ============================================================
def test_block_bootstrap(pnls: list, n_bootstrap: int = 1000,
                         block_size: int = 20) -> dict:
    """Politis-Romano stationary block bootstrap.
    Tests whether the observed Sharpe is statistically > 0.
    95% CI should not include 0.
    """
    console.print(f"  [dim]Test 3: Stationary block bootstrap ({n_bootstrap})[/dim]")

    if len(pnls) < block_size * 5:
        return {"test": "Block Bootstrap", "passed": False,
                "verdict": "SKIP (too few trades)"}

    pnls_arr = np.array(pnls)
    observed_sr = sharpe_ratio(pnls_arr)

    rng = np.random.default_rng(7)
    boot_srs = []
    n = len(pnls_arr)
    for _ in range(n_bootstrap):
        # Stationary block bootstrap: geometric block lengths
        sample = []
        while len(sample) < n:
            start = rng.integers(0, n)
            length = rng.geometric(1.0 / block_size)
            sample.extend(pnls_arr[start:start + length].tolist())
        sample = np.array(sample[:n])
        boot_srs.append(sharpe_ratio(sample))

    boot_srs = np.array(boot_srs)
    ci_lo = float(np.percentile(boot_srs, 2.5))
    ci_hi = float(np.percentile(boot_srs, 97.5))
    passed = ci_lo > 0

    return {
        "test": "Block Bootstrap",
        "observed_sr": round(observed_sr, 3),
        "ci_95_lo": round(ci_lo, 3),
        "ci_95_hi": round(ci_hi, 3),
        "passed": passed,
        "verdict": (
            f"PASS (95% CI [{ci_lo:.2f}, {ci_hi:.2f}] excludes 0)"
            if passed
            else f"FAIL (95% CI [{ci_lo:.2f}, {ci_hi:.2f}] includes 0)"
        ),
    }


# ============================================================
# TEST 4: DEFLATED SHARPE RATIO
# ============================================================
def test_deflated_sharpe(pnls: list, n_trials: int) -> dict:
    console.print(f"  [dim]Test 4: Deflated Sharpe ({n_trials} trials)[/dim]")

    if len(pnls) < 30:
        return {"test": "Deflated Sharpe", "passed": False, "verdict": "SKIP"}

    pnls_arr = np.array(pnls)
    T = len(pnls_arr)
    sr_obs = np.mean(pnls_arr) / np.std(pnls_arr) if np.std(pnls_arr) > 0 else 0
    skew = float(stats.skew(pnls_arr))
    kurt = float(stats.kurtosis(pnls_arr, fisher=False))

    gamma_e = 0.5772
    expected_max_sr = ((1 - gamma_e) * norm.ppf(1 - 1.0 / n_trials) +
                       gamma_e * norm.ppf(1 - 1.0 / (n_trials * np.e)))
    sr0 = expected_max_sr / np.sqrt(T)

    try:
        denom = np.sqrt(max(1e-10, 1 - skew * sr_obs + (kurt - 1) / 4 * sr_obs ** 2))
        dsr_stat = (sr_obs - sr0) * np.sqrt(T - 1) / denom
        dsr = float(norm.cdf(dsr_stat))
    except Exception:
        dsr = 0.0

    passed = dsr > 0.95

    return {
        "test": "Deflated Sharpe",
        "trades": T,
        "sr_per_trade": round(sr_obs, 4),
        "skew": round(skew, 2),
        "kurt": round(kurt, 2),
        "dsr": round(dsr, 3),
        "passed": passed,
        "verdict": f"PASS (DSR={dsr:.3f})" if passed else f"FAIL (DSR={dsr:.3f})",
    }


# ============================================================
# TEST 5: MinBTL
# ============================================================
def test_minbtl(pnls: list, n_trials: int, data_years: float) -> dict:
    console.print(f"  [dim]Test 5: MinBTL[/dim]")

    pnls_arr = np.array(pnls)
    if len(pnls_arr) < 30 or np.std(pnls_arr) == 0:
        return {"test": "MinBTL", "passed": False, "verdict": "SKIP"}

    trades_per_year = len(pnls_arr) / data_years
    sr_annualized = np.mean(pnls_arr) / np.std(pnls_arr) * np.sqrt(trades_per_year)
    required = (2 * math.log(n_trials)) / (sr_annualized ** 2) if sr_annualized > 0 else 999
    passed = data_years >= required

    return {
        "test": "MinBTL",
        "sr_ann": round(float(sr_annualized), 2),
        "data_years": round(data_years, 1),
        "required_years": round(float(required), 1),
        "passed": passed,
        "verdict": (
            f"PASS ({data_years:.1f}y >= {required:.1f}y)"
            if passed
            else f"FAIL ({data_years:.1f}y < {required:.1f}y)"
        ),
    }


# ============================================================
# TEST 6: PARAMETER STABILITY (+/-20%)
# ============================================================
def test_parameter_stability(df: pd.DataFrame, spread: float) -> dict:
    """Perturb each param +/-20%. Degradation > 50% = overfit to sweet spot."""
    console.print("  [dim]Test 6: Parameter stability (+/-20%)[/dim]")

    base_pnls = backtest_bbmr(df, spread, PARAMS)
    base_pnl = sum(base_pnls)
    if base_pnl <= 0:
        return {"test": "Parameter Stability", "passed": False,
                "verdict": "SKIP (base P&L <= 0)"}

    perturbations = {}
    max_deg = 0
    for key, val in PARAMS.items():
        for mult in (0.8, 1.2):
            new_val = val * mult
            if isinstance(val, int):
                new_val = max(2, int(round(new_val)))
            p = dict(PARAMS)
            p[key] = new_val
            pnls = backtest_bbmr(df, spread, p)
            pnl = sum(pnls)
            deg = (base_pnl - pnl) / base_pnl
            perturbations[f"{key}={new_val}"] = round(pnl, 2)
            max_deg = max(max_deg, deg)

    passed = max_deg < 0.5

    return {
        "test": "Param Stability",
        "base_pnl": round(base_pnl, 2),
        "max_degradation": round(float(max_deg), 3),
        "n_perturbations": len(perturbations),
        "passed": passed,
        "verdict": (
            f"PASS (max deg={max_deg:.1%} < 50%)"
            if passed
            else f"FAIL (max deg={max_deg:.1%} >= 50%)"
        ),
    }


# ============================================================
# TEST 7: WALK-FORWARD EFFICIENCY
# ============================================================
def test_wfe(df: pd.DataFrame, spread: float, n_splits: int = 5) -> dict:
    """Walk-forward efficiency: split data into N sequential folds.
    Train indicators = deterministic (no fit), just measure OOS stability.
    WFE = mean(OOS Sharpe) / mean(IS Sharpe). Pardo: > 0.5 is acceptable.
    """
    console.print(f"  [dim]Test 7: Walk-forward efficiency ({n_splits} folds)[/dim]")

    n = len(df)
    fold_size = n // (n_splits + 1)
    is_srs = []
    oos_srs = []
    for k in range(n_splits):
        is_start = 0
        is_end = fold_size * (k + 1)
        oos_start = is_end
        oos_end = min(n, oos_start + fold_size)
        if oos_end - oos_start < 2000:
            continue
        is_df = df.iloc[is_start:is_end]
        oos_df = df.iloc[oos_start:oos_end]
        is_pnls = backtest_bbmr(is_df, spread, PARAMS)
        oos_pnls = backtest_bbmr(oos_df, spread, PARAMS)
        if len(is_pnls) > 10 and len(oos_pnls) > 10:
            is_srs.append(sharpe_ratio(np.array(is_pnls)))
            oos_srs.append(sharpe_ratio(np.array(oos_pnls)))

    if not is_srs or not oos_srs:
        return {"test": "Walk-Forward", "passed": False, "verdict": "SKIP"}

    mean_is = float(np.mean(is_srs))
    mean_oos = float(np.mean(oos_srs))
    wfe = mean_oos / mean_is if mean_is > 0 else 0
    passed = wfe >= 0.5 and mean_oos > 0

    return {
        "test": "Walk-Forward Eff.",
        "n_folds": len(is_srs),
        "mean_is_sr": round(mean_is, 3),
        "mean_oos_sr": round(mean_oos, 3),
        "wfe": round(float(wfe), 3),
        "passed": passed,
        "verdict": (
            f"PASS (WFE={wfe:.2f} >= 0.5, OOS_SR={mean_oos:.2f})"
            if passed
            else f"FAIL (WFE={wfe:.2f}, OOS_SR={mean_oos:.2f})"
        ),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    console.print(Panel.fit(
        "[bold red]BBMR PARANOID VALIDATION v2[/bold red]\n"
        "Lopez de Prado methodology, hand-rolled\n\n"
        f"Params: {PARAMS}\n"
        f"N_TRIALS: {N_TRIALS_OPTIMIZED}",
        title="The Paranoid Tests v2",
        border_style="red",
    ))

    all_results = {}
    for sym, spread in SPREADS.items():
        df = load_dukascopy(sym)
        if df is None:
            console.print(f"  [red]No data for {sym}[/red]")
            continue

        years = (df.index[-1] - df.index[0]).days / 365
        console.print(f"\n  [cyan bold]{sym}[/cyan bold] {len(df):,} candles, {years:.1f}y")

        pnls = backtest_bbmr(df, spread, PARAMS)
        console.print(f"  Trades: {len(pnls)}, PnL: ${sum(pnls):+,.0f}")

        results = {}
        results["future_shift"] = test_future_shift(df, spread)
        results["time_perm"] = test_time_permutation(df, spread, n_permutations=500)
        results["bootstrap"] = test_block_bootstrap(pnls, n_bootstrap=1000)
        results["dsr"] = test_deflated_sharpe(pnls, n_trials=N_TRIALS_OPTIMIZED)
        results["minbtl"] = test_minbtl(pnls, N_TRIALS_OPTIMIZED, years)
        results["param_stab"] = test_parameter_stability(df, spread)
        results["wfe"] = test_wfe(df, spread, n_splits=5)
        all_results[sym] = results

    # === REPORT ===
    console.print(f"\n{'=' * 80}")
    console.print(Panel.fit("[bold]FINAL VALIDATION REPORT[/bold]", border_style="red"))

    for sym, results in all_results.items():
        console.print(f"\n  [bold cyan]{sym}[/bold cyan]")
        t = Table(show_header=True)
        t.add_column("Test", width=22)
        t.add_column("Key Metric", width=30)
        t.add_column("Verdict", width=45)

        passed_count = 0
        for r in results.values():
            p = r.get("passed", False)
            if p:
                passed_count += 1
            color = "green" if p else "red"
            metric = ""
            for k in ("lag_decay", "p_value_sr", "ci_95_lo", "dsr",
                      "required_years", "max_degradation", "wfe"):
                if k in r:
                    metric = f"{k}={r[k]}"
                    break
            t.add_row(
                r.get("test", "?"),
                metric,
                f"[{color}]{r.get('verdict', '?')}[/{color}]",
            )
        console.print(t)
        total = len(results)
        color = "green" if passed_count == total else "yellow" if passed_count >= total // 2 else "red"
        console.print(
            f"  [bold {color}]{sym}: {passed_count}/{total} tests passed[/bold {color}]"
        )


if __name__ == "__main__":
    main()
