"""Fetch the validation universe from yfinance.

Downloads 2 years of 1H OHLCV for the canonical 8-asset basket used
for NMI dependency analysis and cross-sectional research:

  FX majors:   EURUSD, USDJPY, GBPUSD, AUDUSD, NZDUSD, USDCAD
  Commodities: XAUUSD (gold)
  Indices:     SPX (S&P 500)

yfinance caveats for this project:
- 1H data capped at 730 days (period='2y').
- Non-FX tickers live on different calendars (SPX and gold futures
  follow the CME/NYSE calendar, FX is nearly 24/7). The NMI analysis
  later will align and inner-join all series by timestamp.
- XAUUSD is proxied via GC=F (COMEX gold front-month) because
  yfinance's XAUUSD=X is often flat.
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

OUT = Path("data/dukascopy")
OUT.mkdir(parents=True, exist_ok=True)

UNIVERSE = {
    # FX majors
    "EURUSD": "EURUSD=X",
    "USDJPY": "JPY=X",
    "GBPUSD": "GBPUSD=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCAD": "CAD=X",
    # Commodities
    "XAUUSD": "GC=F",
    # Indices
    "SPX": "^GSPC",
}


def fetch(name: str, ticker: str) -> None:
    df = yf.download(
        ticker, period="2y", interval="1h",
        auto_adjust=False, progress=False,
    )
    if df.empty:
        print(f"  {name} ({ticker}): no data")
        return
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    needed = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in needed if c in df.columns]]
    df.index.name = "timestamp"
    df.index = df.index.tz_localize(None)
    out = OUT / f"{name}_1H.csv"
    df.to_csv(out)
    print(f"  {name:7s} ({ticker:10s}): {len(df):6d} bars -> {out.name}")


def main() -> None:
    for name, ticker in UNIVERSE.items():
        try:
            fetch(name, ticker)
        except Exception as e:
            print(f"  {name}: ERROR {e}")


if __name__ == "__main__":
    main()
