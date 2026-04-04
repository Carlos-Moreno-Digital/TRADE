"""Download Dukascopy 1H data directly from CDN — fast parallel version.

Downloads tick data hour by hour, aggregates to 1H OHLCV candles.
Uses ThreadPool for 10x speed.
"""

import struct
import lzma
import os
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

SYMBOLS = {
    "EURUSD": 0.00001,
    "USDJPY": 0.001,
    "GBPNZD": 0.00001,
    "AUDNZD": 0.00001,
    "XAUUSD": 0.01,
}

CDN = "https://datafeed.dukascopy.com/datafeed"
SESSION = requests.Session()


def download_one_hour(args):
    """Download and convert one hour of tick data to OHLCV."""
    symbol, pip, dt = args
    # Dukascopy months are 0-indexed!
    url = f"{CDN}/{symbol}/{dt.year}/{dt.month - 1:02d}/{dt.day:02d}/{dt.hour:02d}h_ticks.bi5"
    try:
        r = SESSION.get(url, timeout=10)
        if r.status_code != 200 or len(r.content) < 20:
            return None
        data = lzma.decompress(r.content)
        if len(data) < 20:
            return None

        bids = []
        vols = []
        for i in range(0, len(data), 20):
            if i + 20 > len(data):
                break
            ms, ask, bid, avol, bvol = struct.unpack('>IIIff', data[i:i+20])
            bids.append(bid * pip)
            vols.append(bvol)

        if not bids:
            return None

        return {
            "timestamp": dt,
            "open": bids[0],
            "high": max(bids),
            "low": min(bids),
            "close": bids[-1],
            "volume": sum(vols),
        }
    except Exception:
        return None


def download_symbol(symbol, start_year=2010, end_year=2026, workers=20):
    """Download all 1H candles for a symbol using parallel threads."""
    pip = SYMBOLS[symbol]

    # Build list of all hours to download
    hours = []
    dt = datetime(start_year, 1, 1)
    end = datetime(end_year, 4, 1)
    while dt < end:
        # Skip weekends (Sat=5, Sun=6) to save time
        if dt.weekday() < 5:
            hours.append((symbol, pip, dt))
        dt += timedelta(hours=1)

    print(f"    {len(hours)} hours to download ({workers} threads)...")

    candles = []
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one_hour, h): h for h in hours}
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result:
                candles.append(result)
            if done % 2000 == 0:
                pct = done / len(hours) * 100
                print(f"    {pct:.0f}% ({done}/{len(hours)}, {len(candles)} candles so far)")

    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles).set_index("timestamp").sort_index()
    return df


def main():
    output_dir = Path("data/dukascopy")
    output_dir.mkdir(parents=True, exist_ok=True)

    for symbol in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"Downloading {symbol} (2010-2026, 1H)...")
        df = download_symbol(symbol, start_year=2010, end_year=2026, workers=100)
        if not df.empty:
            path = output_dir / f"{symbol}_1H.csv"
            df.to_csv(path)
            years = (df.index[-1] - df.index[0]).days / 365
            print(f"    DONE: {len(df)} candles, {years:.1f} years → {path}")
        else:
            print(f"    NO DATA")

    print(f"\n{'='*60}")
    print("All downloads complete!")
    for f in output_dir.glob("*.csv"):
        size = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size:.1f} MB")


if __name__ == "__main__":
    main()
