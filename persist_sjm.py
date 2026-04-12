"""Train SJM per symbol and persist as JSON for the RegimeGate.

Outputs data/regimes/{symbol}.json with the centers + scaler stats so
the gate can load and predict without retraining.

Token-budget: prints only the persisted summary (one line per symbol).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trade.research.regimes import StatisticalJumpModel, build_features

DATA_DIR = Path("data/dukascopy")
OUT_DIR = Path("data/regimes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Auto-tuned lambdas from train_jump_model.py sweep (the elbows).
# GBPNZD, AUDNZD, XAUUSD use default lambda=5.0 (EURUSD elbow) until
# their own sweep is run on the VPS.
SYMBOLS = {
    "EURUSD": 5.0,
    "USDJPY": 2.5,
    "GBPNZD": 5.0,
    "AUDNZD": 5.0,
    "XAUUSD": 5.0,
}
N_STATES = 2


def load_daily_close(symbol: str) -> pd.Series:
    df = pd.read_csv(DATA_DIR / f"{symbol}_1H.csv", parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df["close"].resample("1D").last().dropna()


def main() -> None:
    for sym, lam in SYMBOLS.items():
        close = load_daily_close(sym)
        feats = build_features(close)
        model = StatisticalJumpModel(
            n_states=N_STATES, lambda_=lam, random_state=42
        )
        model.fit(feats)

        # Identify which state is Risk-Off (highest realized vol centroid)
        centers_orig = model._scaler.inverse_transform(model.fit_.centers)
        rv20_idx = list(feats.columns).index("realized_vol_20")
        risk_off = int(centers_orig[:, rv20_idx].argmax())
        risk_on = int(1 - risk_off) if N_STATES == 2 else None

        payload = model.to_dict(list(feats.columns))
        payload["risk_off_state"] = risk_off
        payload["risk_on_state"] = risk_on

        out = OUT_DIR / f"{sym}.json"
        out.write_text(json.dumps(payload, indent=2))
        print(
            f"  {sym}: lambda={lam:.2f} jumps={model.fit_.n_jumps} "
            f"risk_off=state{risk_off} -> {out}"
        )


if __name__ == "__main__":
    main()
