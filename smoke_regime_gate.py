"""Smoke test for RegimeGate.

Verifies:
  - Models load from data/regimes/{sym}.json
  - Latest regime can be predicted from disk price data
  - Fabricated low-vol vs high-vol price series force Risk-On vs Risk-Off
  - Strategies whose supported_regimes mismatch get BLOCKED
  - Strategies whose supported_regimes match get ALLOWED
  - supported_regimes=[] always blocks
  - Disarmed gate (missing model) fails OPEN

Prints PASS/FAIL only — no dataframes, no arrays.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trade.agents.multi.regime_gate import RegimeGate

DATA_REG = Path("data/regimes")
MODELS_OK = (DATA_REG / "EURUSD.json").exists() and (DATA_REG / "USDJPY.json").exists()
if not MODELS_OK:
    print("FAIL: SJM model files missing; run persist_sjm.py first")
    sys.exit(2)

# Read the persisted "risk_on / risk_off" labels so the smoke test
# does not have to know which integer is which.
risk_off = {}
risk_on = {}
for sym in ("EURUSD", "USDJPY"):
    payload = json.loads((DATA_REG / f"{sym}.json").read_text())
    risk_off[sym] = int(payload["risk_off_state"])
    risk_on[sym] = int(payload["risk_on_state"])
print(f"  models loaded: EURUSD risk_on={risk_on['EURUSD']} risk_off={risk_off['EURUSD']} "
      f"| USDJPY risk_on={risk_on['USDJPY']} risk_off={risk_off['USDJPY']}")


def calm_series(n: int = 200, seed: int = 1) -> pd.Series:
    """Synthetic low-vol drift -> Risk-On regime."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0001, 0.0015, n)  # ~1.5 bps daily vol
    px = 1.10 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2024-01-01", periods=n, freq="1D")
    return pd.Series(px, index=idx)


def stormy_series(n: int = 200, seed: int = 2) -> pd.Series:
    """Synthetic high-vol negative-skew drawdown -> Risk-Off regime."""
    rng = np.random.default_rng(seed)
    base = rng.normal(-0.001, 0.012, n)  # 12 bps vol, negative drift
    # Add fat-tail negative shocks
    shocks_idx = rng.choice(n, size=n // 10, replace=False)
    base[shocks_idx] -= rng.uniform(0.01, 0.04, len(shocks_idx))
    px = 1.10 * np.exp(np.cumsum(base))
    idx = pd.date_range("2024-01-01", periods=n, freq="1D")
    return pd.Series(px, index=idx)


gate = RegimeGate()

# Sanity: model returns SOMETHING for the on-disk EURUSD/USDJPY data
gate.reset_cache()
live_eur = gate.current_regime("EURUSD=X")
gate.reset_cache()
live_jpy = gate.current_regime("JPY=X")
print(f"  live regimes from disk: EURUSD={live_eur}, USDJPY={live_jpy}")

# Force regimes via synthetic data
gate.reset_cache()
calm_eur = gate.current_regime("EURUSD=X", recent_close=calm_series())
gate.reset_cache()
storm_eur = gate.current_regime("EURUSD=X", recent_close=stormy_series())
gate.reset_cache()
calm_jpy = gate.current_regime("JPY=X", recent_close=calm_series(seed=11))
gate.reset_cache()
storm_jpy = gate.current_regime("JPY=X", recent_close=stormy_series(seed=12))
print(f"  forced regimes: EURUSD calm={calm_eur} stormy={storm_eur} "
      f"| USDJPY calm={calm_jpy} stormy={storm_jpy}")

CASES = []

# Risk-On strategy on calm market -> ALLOW
CASES.append((
    "Risk-On strategy on calm EURUSD",
    "EURUSD=X", [risk_on["EURUSD"]], calm_series(), True,
))
# Risk-On strategy on stormy market -> BLOCK
CASES.append((
    "Risk-On strategy on stormy EURUSD",
    "EURUSD=X", [risk_on["EURUSD"]], stormy_series(), False,
))
# Risk-Off strategy on stormy market -> ALLOW
CASES.append((
    "Risk-Off strategy on stormy EURUSD",
    "EURUSD=X", [risk_off["EURUSD"]], stormy_series(), True,
))
# Risk-Off strategy on calm market -> BLOCK
CASES.append((
    "Risk-Off strategy on calm EURUSD",
    "EURUSD=X", [risk_off["EURUSD"]], calm_series(), False,
))
# Multi-regime strategy [0,1] -> always allow
CASES.append((
    "Both-regime strategy on stormy EURUSD",
    "EURUSD=X", [0, 1], stormy_series(), True,
))
# supported_regimes=[] -> always BLOCK
CASES.append((
    "Empty supported_regimes",
    "EURUSD=X", [], calm_series(), False,
))
# Same logic but on USDJPY
CASES.append((
    "Risk-On strategy on calm USDJPY",
    "JPY=X", [risk_on["USDJPY"]], calm_series(seed=21), True,
))
CASES.append((
    "Risk-Off strategy on stormy USDJPY",
    "JPY=X", [risk_off["USDJPY"]], stormy_series(seed=22), True,
))
CASES.append((
    "Risk-On strategy on stormy USDJPY",
    "JPY=X", [risk_on["USDJPY"]], stormy_series(seed=23), False,
))
# Disarmed: unknown symbol -> fails open with allow
CASES.append((
    "Disarmed gate (unknown symbol)",
    "ZZZUSD=X", [risk_on["EURUSD"]], None, True,
))

passed = 0
for label, sym, supported, series, expect in CASES:
    gate.reset_cache()
    d = gate.check(sym, supported, recent_close=series)
    ok = (d.allowed == expect)
    passed += int(ok)
    mark = "PASS" if ok else "FAIL"
    info = f"regime={d.current_regime} supported={d.supported_regimes}"
    print(f"  [{mark}] {label}: allowed={d.allowed} ({info})")

print(f"\n{passed}/{len(CASES)} cases correct")
sys.exit(0 if passed == len(CASES) else 1)
