"""RegimeGate — block strategies that fire in unsupported regimes.

Loads a per-symbol Statistical Jump Model from data/regimes/{sym}.json
and reports the current regime. The trading pipeline blocks any signal
whose owning strategy did not declare the active regime in
`supported_regimes`.

Two operating modes for the "current regime" reading:

  Mode A — payload-driven (preferred in production)
    Caller passes a `recent_close` pandas Series in the signal payload.
    The gate computes features from that series and predicts the
    latest regime label. Stateless and unit-testable.

  Mode B — file-driven (default fallback for research / smoke tests)
    The gate reads data/dukascopy/{sym}_1H.csv, resamples to daily,
    builds features, and predicts the latest regime.

Fails OPEN if either the SJM payload or the price data is missing
(research mode), so the trading pipeline does not break while models
are being recalibrated.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trade.research.regimes.features import build_features
from trade.research.regimes.jump_model import StatisticalJumpModel

DEFAULT_MODELS_DIR = Path("data/regimes")
DEFAULT_DATA_DIR = Path("data/dukascopy")


def _normalize(symbol: str) -> str:
    s = symbol.upper().replace("=X", "").replace("^", "")
    aliases = {
        "JPY": "USDJPY",
        "CAD": "USDCAD",
        "GC=F": "XAUUSD",
        "GC": "XAUUSD",
        "GSPC": "SPX",
    }
    return aliases.get(s, s)


@dataclass
class RegimeDecision:
    allowed: bool
    reason: str
    current_regime: int | None = None
    supported_regimes: list[int] | None = None
    symbol: str | None = None


class RegimeGate:
    def __init__(
        self,
        models_dir: Path = DEFAULT_MODELS_DIR,
        data_dir: Path = DEFAULT_DATA_DIR,
    ):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self._models: dict[str, StatisticalJumpModel] = {}
        # Cached "latest regime" per symbol so successive checks within
        # the same bar do not recompute features.
        self._regime_cache: dict[str, int] = {}

    # ------------------------------------------------------------------
    def _load_model(self, symbol_norm: str) -> StatisticalJumpModel | None:
        if symbol_norm in self._models:
            return self._models[symbol_norm]
        path = self.models_dir / f"{symbol_norm}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
            model = StatisticalJumpModel.from_dict(payload)
            self._models[symbol_norm] = model
            return model
        except Exception:
            return None

    def _close_series(
        self,
        symbol_norm: str,
        recent_close: pd.Series | None,
    ) -> pd.Series | None:
        if recent_close is not None and len(recent_close) > 0:
            s = recent_close.copy()
            if not isinstance(s.index, pd.DatetimeIndex):
                return None
            return s
        path = self.data_dir / f"{symbol_norm}_1H.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        return df["close"].resample("1D").last().dropna()

    def current_regime(
        self,
        symbol: str,
        recent_close: pd.Series | None = None,
        use_cache: bool = True,
    ) -> int | None:
        sym = _normalize(symbol)
        if use_cache and sym in self._regime_cache:
            return self._regime_cache[sym]
        model = self._load_model(sym)
        if model is None:
            return None
        close = self._close_series(sym, recent_close)
        if close is None or len(close) < 80:
            return None
        feats = build_features(close)
        if feats.empty:
            return None
        labels = model.predict(feats)
        latest = int(labels[-1])
        self._regime_cache[sym] = latest
        return latest

    def reset_cache(self) -> None:
        self._regime_cache.clear()

    # ------------------------------------------------------------------
    def check(
        self,
        symbol: str,
        supported_regimes: list[int] | None,
        recent_close: pd.Series | None = None,
    ) -> RegimeDecision:
        if supported_regimes is None:
            return RegimeDecision(
                allowed=True,
                reason="Strategy did not declare supported_regimes (legacy)",
                symbol=symbol,
            )
        if not supported_regimes:
            return RegimeDecision(
                allowed=False,
                reason="Strategy supported_regimes=[] -> not authorized in any regime",
                symbol=symbol,
                supported_regimes=supported_regimes,
            )
        regime = self.current_regime(symbol, recent_close=recent_close)
        if regime is None:
            return RegimeDecision(
                allowed=True,
                reason=f"RegimeGate disarmed for {symbol} (no SJM or no price data)",
                symbol=symbol,
                supported_regimes=supported_regimes,
            )
        if regime not in supported_regimes:
            return RegimeDecision(
                allowed=False,
                reason=(
                    f"Active regime={regime} not in strategy's supported "
                    f"regimes {supported_regimes}"
                ),
                current_regime=regime,
                supported_regimes=supported_regimes,
                symbol=symbol,
            )
        return RegimeDecision(
            allowed=True,
            reason=f"Active regime={regime} in {supported_regimes}",
            current_regime=regime,
            supported_regimes=supported_regimes,
            symbol=symbol,
        )
