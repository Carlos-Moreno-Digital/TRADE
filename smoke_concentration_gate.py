"""Smoke test for ConcentrationGate.

Three scenarios — prints PASS/FAIL only, no dataframes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trade.agents.multi.concentration_gate import ConcentrationGate

gate = ConcentrationGate()

if not gate.is_armed:
    print("FAIL: NMI matrix not loaded; run analyze_nmi.py first")
    sys.exit(2)

CASES = [
    # (label, candidate, open, expect_allowed)
    ("EURUSD with empty book",                "EURUSD=X", [],                       True),
    ("GBPUSD on top of EURUSD long",          "GBPUSD=X", ["EURUSD=X"],             False),
    ("AUDUSD on top of NZDUSD long",          "AUDUSD=X", ["NZDUSD=X"],             False),
    ("XAUUSD on top of EURUSD long",          "XAUUSD",   ["EURUSD=X"],             True),
    ("SPX on top of EURUSD long",             "GSPC",     ["EURUSD=X"],             True),
    ("USDJPY on top of XAUUSD",               "JPY=X",    ["XAUUSD"],               True),
    ("EURUSD with EUR + AUD + XAU stack",     "EURUSD=X", ["AUDUSD=X", "XAUUSD"],   True),
    ("GBPUSD with EUR + XAU stack",           "GBPUSD=X", ["EURUSD=X", "XAUUSD"],   False),
]

passed = 0
for label, cand, book, expect in CASES:
    d = gate.check(cand, book)
    ok = (d.allowed == expect)
    passed += int(ok)
    mark = "PASS" if ok else "FAIL"
    detail = f"NMI={d.nmi_value:.3f} vs {d.conflicting_symbol}" if d.nmi_value else "n/a"
    print(f"  [{mark}] {label}: allowed={d.allowed} ({detail})")

print(f"\n{passed}/{len(CASES)} cases correct")
sys.exit(0 if passed == len(CASES) else 1)
