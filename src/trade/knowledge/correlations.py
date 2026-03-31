"""Inter-market correlations and risk-on/risk-off dynamics.

Understanding correlations helps:
1. Avoid taking correlated trades (doubles risk)
2. Confirm trade direction using correlated assets
3. Understand broader market sentiment
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MarketRegime(str, Enum):
    RISK_ON = "risk_on"     # Optimism: stocks up, safe havens down
    RISK_OFF = "risk_off"   # Fear: stocks down, safe havens up
    NEUTRAL = "neutral"


class Correlation(BaseModel):
    """Correlation between two assets."""

    asset_a: str
    asset_b: str
    correlation: float  # -1.0 (inverse) to 1.0 (perfect)
    description: str
    regime_dependent: bool = False  # Changes with market regime?


# Major forex correlations
FOREX_CORRELATIONS: list[Correlation] = [
    # Positive correlations (move together)
    Correlation(
        asset_a="EURUSD", asset_b="GBPUSD", correlation=0.85,
        description="Strong positive. Both are anti-USD. If EUR/USD rises, GBP/USD usually follows.",
    ),
    Correlation(
        asset_a="AUDUSD", asset_b="NZDUSD", correlation=0.90,
        description="Very strong. Both are commodity/risk currencies in same region.",
    ),
    Correlation(
        asset_a="EURUSD", asset_b="AUDUSD", correlation=0.70,
        description="Moderate positive. Both anti-USD but AUD more risk-sensitive.",
    ),

    # Negative correlations (move opposite)
    Correlation(
        asset_a="EURUSD", asset_b="USDCHF", correlation=-0.95,
        description="Near-perfect inverse. EUR/USD up = USD/CHF down (USD weakens in both).",
    ),
    Correlation(
        asset_a="EURUSD", asset_b="USDJPY", correlation=-0.60,
        description="Moderate inverse. But breaks down during risk-off (JPY safe haven).",
        regime_dependent=True,
    ),
    Correlation(
        asset_a="GBPUSD", asset_b="USDCAD", correlation=-0.70,
        description="Both USD-denominated. GBP up usually means CAD up (USD weak).",
    ),
]

# Cross-asset correlations
CROSS_ASSET_CORRELATIONS: list[Correlation] = [
    Correlation(
        asset_a="DXY", asset_b="EURUSD", correlation=-0.95,
        description="DXY (USD index) up = EUR/USD down. DXY is ~57% weighted to EUR.",
    ),
    Correlation(
        asset_a="DXY", asset_b="XAUUSD", correlation=-0.80,
        description="Strong inverse. Dollar strength = Gold weakness (gold priced in USD).",
        regime_dependent=True,
    ),
    Correlation(
        asset_a="US10Y", asset_b="USDJPY", correlation=0.75,
        description="US 10Y yield up = USD/JPY up (carry trade). Higher US yields attract capital.",
    ),
    Correlation(
        asset_a="US10Y", asset_b="XAUUSD", correlation=-0.70,
        description="Rising yields = Gold down (opportunity cost of non-yielding gold).",
    ),
    Correlation(
        asset_a="SPX", asset_b="USDJPY", correlation=0.65,
        description="Risk-on: stocks up, JPY weakens (carry trade unwind on risk-off).",
        regime_dependent=True,
    ),
    Correlation(
        asset_a="SPX", asset_b="XAUUSD", correlation=-0.40,
        description="Weak inverse. Both can rise in easy-money environment.",
        regime_dependent=True,
    ),
    Correlation(
        asset_a="OIL", asset_b="USDCAD", correlation=-0.70,
        description="Oil up = CAD strengthens (Canada is major exporter) = USDCAD down.",
    ),
    Correlation(
        asset_a="VIX", asset_b="SPX", correlation=-0.85,
        description="Fear index. VIX up = stocks down. VIX spike = risk-off.",
    ),
]


# Risk-on / Risk-off behavior
RISK_REGIME_BEHAVIOR: dict[MarketRegime, dict[str, str]] = {
    MarketRegime.RISK_ON: {
        "description": "Optimism. Investors seek higher returns in riskier assets.",
        "stocks": "UP (SPX, NAS100, DAX)",
        "usd": "MIXED (depends on cause)",
        "jpy": "DOWN (carry trade: borrow JPY to buy risky assets)",
        "chf": "DOWN (safe haven outflow)",
        "aud_nzd": "UP (commodity/risk currencies benefit)",
        "gold": "DOWN or FLAT (no need for safe haven)",
        "oil": "UP (economic activity = demand)",
        "bonds": "DOWN (yields UP, investors leave bonds for stocks)",
        "crypto": "UP (speculative risk appetite)",
        "vix": "DOWN (low fear)",
        "best_trades": [
            "LONG AUDUSD", "LONG NZDUSD", "SHORT USDJPY (JPY weakens)",
            "LONG US30/NAS100", "SHORT XAUUSD",
        ],
    },
    MarketRegime.RISK_OFF: {
        "description": "Fear. Investors flee to safety.",
        "stocks": "DOWN",
        "usd": "UP (world reserve currency = safe haven)",
        "jpy": "UP (carry trade unwind: sell risky assets, buy back JPY)",
        "chf": "UP (safe haven)",
        "aud_nzd": "DOWN (risk currencies sold)",
        "gold": "UP (ultimate safe haven)",
        "oil": "DOWN (recession fears = less demand)",
        "bonds": "UP (yields DOWN, flight to safety)",
        "crypto": "DOWN (speculative selling)",
        "vix": "UP (fear spikes)",
        "best_trades": [
            "LONG USDJPY caution / SHORT AUDJPY", "LONG XAUUSD",
            "SHORT NAS100/US30", "LONG USDCHF (USD strength)",
            "SHORT AUDUSD", "SHORT NZDUSD",
        ],
    },
}


def get_correlated_pairs(pair: str) -> list[dict[str, Any]]:
    """Get all pairs correlated with a given pair."""
    results = []
    for corr in FOREX_CORRELATIONS + CROSS_ASSET_CORRELATIONS:
        if pair.upper() in (corr.asset_a.upper(), corr.asset_b.upper()):
            other = corr.asset_b if corr.asset_a.upper() == pair.upper() else corr.asset_a
            results.append({
                "pair": other,
                "correlation": corr.correlation,
                "description": corr.description,
                "regime_dependent": corr.regime_dependent,
            })
    return sorted(results, key=lambda x: abs(x["correlation"]), reverse=True)


def check_correlation_conflict(pairs: list[str]) -> list[str]:
    """Check if multiple trades have dangerous correlation overlap."""
    warnings = []
    for i, pair_a in enumerate(pairs):
        for pair_b in pairs[i + 1:]:
            for corr in FOREX_CORRELATIONS:
                a, b = corr.asset_a.upper(), corr.asset_b.upper()
                if {pair_a.upper(), pair_b.upper()} == {a, b}:
                    if abs(corr.correlation) >= 0.75:
                        warnings.append(
                            f"HIGH correlation ({corr.correlation:.2f}) between "
                            f"{pair_a} and {pair_b}: {corr.description}. "
                            f"Taking both doubles your risk!"
                        )
    return warnings


def get_regime_trades(regime: MarketRegime) -> list[str]:
    """Get recommended trades for current market regime."""
    behavior = RISK_REGIME_BEHAVIOR.get(regime)
    if behavior:
        return behavior.get("best_trades", [])
    return []
