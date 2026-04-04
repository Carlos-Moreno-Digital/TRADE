"""Download Dukascopy 1H data — sequential with rate limiting.

Previous version with 30-100 threads got blocked by Dukascopy CDN.
This version downloads sequentially with small delays, much more reliable.
Uses 5 threads max + 50ms delay between requests.
"""

import struct
import lzma
import time
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def download_one_hour(symbol, pip, dt):
    """Download one hour of tick data, convert to OHLCV."""
    url = f"{CDN}/{symbol}/{dt.year}/{dt.month - 1:02d}/{dt.day:02d}/{dt.hour:02d}h_ticks.bi5"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=15)
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
        except (requests.exceptions.RequestException, lzma.LZMAError):
            time.sleep(1 * (attempt + 1))
        except Exception:
            return None
    return None


def download_year(symbol, pip, year):
    """Download one year of 1H data using 5 threads with rate limiting."""
    start = datetime(year, 1, 1)
    if year >= 2026:
        end = datetime(2026, 4, 4)
    else:
        end = datetime(year + 1, 1, 1)

    # Build hour list (include all hours, even weekends — Dukascopy handles it)
    hours = []
    dt = start
    while dt < end:
        hours.append(dt)
        dt += timedelta(hours=1)

    candles = []
    batch_size = 5  # Only 5 concurrent requests

    for i in range(0, len(hours), batch_size):
        batch = hours[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = {pool.submit(download_one_hour, symbol, pip, h): h for h in batch}
            for f in as_completed(futures):
                result = f.result()
                if result:
                    candles.append(result)
        # Small delay between batches to avoid rate limiting
        time.sleep(0.05)

        # Progress every 500 hours
        if (i + batch_size) % 500 == 0:
            pct = (i + batch_size) / len(hours) * 100
            print(f"\r      {pct:.0f}% ({len(candles)} candles)", end="", flush=True)

    return candles


def main():
    output_dir = Path("data/dukascopy")
    output_dir.mkdir(parents=True, exist_ok=True)

    for symbol, pip in SYMBOLS.items():
        path = output_dir / f"{symbol}_1H.csv"
        print(f"\n{'='*60}")
        print(f"Downloading {symbol} (2010-2026, 1H)...")

        all_candles = []

        for year in range(2010, 2027):
            print(f"    {year}...", end=" ", flush=True)
            candles = download_year(symbol, pip, year)
            all_candles.extend(candles)
            print(f"{len(candles)} candles")

            # Save progress after each year
            if all_candles:
                df = pd.DataFrame(all_candles).set_index("timestamp").sort_index()
                df = df[~df.index.duplicated(keep='first')]
                df.to_csv(path)

        if all_candles:
            df = pd.DataFrame(all_candles).set_index("timestamp").sort_index()
            df = df[~df.index.duplicated(keep='first')]
            df.to_csv(path)
            years = (df.index[-1] - df.index[0]).days / 365
            print(f"    TOTAL: {len(df)} candles, {years:.1f} years")
        else:
            print(f"    NO DATA")

    print(f"\n{'='*60}")
    print("Complete!")
    for f in output_dir.glob("*.csv"):
        lines = sum(1 for _ in open(f)) - 1
        size = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {lines} candles, {size:.1f} MB")


if __name__ == "__main__":
    main()
