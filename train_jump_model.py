"""Train Statistical Jump Model on EURUSD and USDJPY daily.

Following the token-optimization rule: this script does ALL the
heavy lifting on the VPS and prints ONLY the summarized results
(per-regime statistics, lambda sweep). No dataframes are returned.

Usage:
    python train_jump_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trade.research.regimes import StatisticalJumpModel, build_features

DATA_DIR = Path("data/dukascopy")
SYMBOLS = ["EURUSD", "USDJPY"]
LAMBDA_GRID = [0.5, 1.0, 2.5, 5.0, 10.0, 25.0]
N_STATES = 2  # Risk-On vs Risk-Off as the user requested


def load_daily_close(symbol: str) -> pd.Series:
    df = pd.read_csv(DATA_DIR / f"{symbol}_1H.csv", parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    daily = df["close"].resample("1D").last().dropna()
    return daily


def fmt_regime_summary(s: dict) -> str:
    lines = [
        f"  obs={s['n_obs']}  states={s['n_states']}  "
        f"lambda={s['lambda']:.2f}  jumps={s['n_jumps']}"
    ]
    for k, r in s["regimes"].items():
        c = r.get("centroid", {})
        cen_txt = " ".join(
            f"{name}={val:+.3f}" for name, val in c.items()
        )
        lines.append(
            f"    state {k}: count={r['count']:4d} share={r['share']:.2f} "
            f"runs={r['n_runs']:3d} avg_run={r['avg_run_len']:5.1f}"
        )
        lines.append(f"      centroid: {cen_txt}")
    return "\n".join(lines)


def main() -> None:
    print("=" * 72)
    print(f"Statistical Jump Model — train on {SYMBOLS}, k={N_STATES}")
    print("=" * 72)

    for sym in SYMBOLS:
        close = load_daily_close(sym)
        feats = build_features(close)
        print(f"\n{sym}: {len(feats)} daily obs after warmup, "
              f"{feats.index[0].date()} -> {feats.index[-1].date()}")

        # Lambda sweep — find the elbow where regimes stabilize
        sweep = []
        for lam in LAMBDA_GRID:
            model = StatisticalJumpModel(
                n_states=N_STATES, lambda_=lam, random_state=42
            )
            model.fit(feats)
            f = model.fit_
            sweep.append({
                "lambda": lam,
                "n_jumps": f.n_jumps,
                "inertia": round(f.inertia, 2),
                "objective": round(f.objective, 2),
                "n_iter": f.n_iter,
            })
        print("  Lambda sweep (jumps per lambda):")
        print(f"  {'lambda':>8} {'jumps':>6} {'inertia':>10} {'objective':>10} {'iter':>5}")
        for row in sweep:
            print(
                f"  {row['lambda']:8.2f} {row['n_jumps']:6d} "
                f"{row['inertia']:10.2f} {row['objective']:10.2f} "
                f"{row['n_iter']:5d}"
            )

        # Pick a "production" lambda — first lambda where #jumps stabilizes
        # to a reasonable count (between 4 and 25 transitions over ~500
        # daily obs ≈ regimes lasting 20-125 days, which is sensible).
        chosen = None
        for row in sweep:
            if 4 <= row["n_jumps"] <= 25:
                chosen = row["lambda"]
                break
        if chosen is None:
            chosen = LAMBDA_GRID[len(LAMBDA_GRID) // 2]
        print(f"  -> chosen lambda = {chosen:.2f}")

        model = StatisticalJumpModel(
            n_states=N_STATES, lambda_=chosen, random_state=42
        )
        model.fit(feats)
        summary = model.regime_summary(feats)
        print(fmt_regime_summary(summary))

        # Sanity: highest-vol state should be the "Risk-Off" state
        regimes = summary["regimes"]
        vol_by_state = {
            k: r["centroid"]["realized_vol_20"]
            for k, r in regimes.items()
        }
        risk_off = max(vol_by_state, key=vol_by_state.get)
        risk_on = min(vol_by_state, key=vol_by_state.get)
        print(
            f"  -> state {risk_off} = Risk-Off (vol={vol_by_state[risk_off]:.4f})"
        )
        print(
            f"  -> state {risk_on}  = Risk-On  (vol={vol_by_state[risk_on]:.4f})"
        )


if __name__ == "__main__":
    main()
