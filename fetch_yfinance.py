"""Quick yfinance fallback — 2 years of 1H for validation.

IMPORTANT: yfinance has known quirks vs true tick data.
This is for paranoid validation infrastructure testing, NOT for
final production validation. Final validation must use Dukascopy
tick-derived OHLC (run on VPS with full 16yr dataset).
"""
from pathlib import Path
import yfinance as yf
import pandas as pd

OUT = Path("data/dukascopy")
OUT.mkdir(parents=True, exist_ok=True)

SYMBOLS = {"EURUSD": "EURUSD=X", "USDJPY": "JPY=X"}

for name, ticker in SYMBOLS.items():
    df = yf.download(ticker, period="2y", interval="1h",
                     auto_adjust=False, progress=False)
    if df.empty:
        print(f"  {name}: no data")
        continue
    # Flatten multiindex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df = df[["open", "high", "low", "close", "volume"]]
    df.index.name = "timestamp"
    df.index = df.index.tz_localize(None)
    out = OUT / f"{name}_1H.csv"
    df.to_csv(out)
    print(f"  {name}: {len(df)} bars -> {out}")
