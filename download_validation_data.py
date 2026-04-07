"""Fast download of EURUSD+USDJPY 2020-2026 for validation.

Uses 20 concurrent workers. ~4 years x 2 symbols ~= 70K hours.
"""
import struct
import lzma
import time
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

import sys
SYMBOLS = {"EURUSD": 0.00001, "USDJPY": 0.001}
CDN = "https://datafeed.dukascopy.com/datafeed"
START_YEAR = 2019  # 7 years of data
END = datetime(2026, 4, 1)


def fetch(symbol, pip, dt):
    url = f"{CDN}/{symbol}/{dt.year}/{dt.month - 1:02d}/{dt.day:02d}/{dt.hour:02d}h_ticks.bi5"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=60)
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
                ms, ask, bid, avol, bvol = struct.unpack(">IIIff", data[i:i + 20])
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
            time.sleep(0.5)
    return None


def download_symbol(symbol, pip, out_path):
    hours = []
    dt = datetime(START_YEAR, 1, 1)
    while dt < END:
        hours.append(dt)
        dt += timedelta(hours=1)
    print(f"  {symbol}: {len(hours)} hours to fetch", flush=True)

    candles = []
    batch = 8  # conservative to avoid CDN throttling
    start_t = time.time()
    # Process in chunks to get faster feedback and bounded memory
    chunk = 1000
    for ci in range(0, len(hours), chunk):
        sub = hours[ci:ci + chunk]
        with ThreadPoolExecutor(max_workers=batch) as pool:
            for r in pool.map(lambda h: fetch(symbol, pip, h), sub):
                if r:
                    candles.append(r)
        done = ci + len(sub)
        elapsed = time.time() - start_t
        rate = done / max(elapsed, 0.1)
        eta = (len(hours) - done) / max(rate, 0.1)
        print(f"    {symbol}: {done}/{len(hours)} "
              f"({len(candles)} valid) rate={rate:.0f}/s eta={eta:.0f}s",
              flush=True)
        sys.stdout.flush()

    if candles:
        df = pd.DataFrame(candles).set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        df.to_csv(out_path)
        print(f"  {symbol}: saved {len(df)} candles to {out_path}")


def main():
    out_dir = Path("data/dukascopy")
    out_dir.mkdir(parents=True, exist_ok=True)
    for sym, pip in SYMBOLS.items():
        out = out_dir / f"{sym}_1H_dukas.csv"  # separate from yfinance
        if out.exists():
            df = pd.read_csv(out)
            if len(df) > 30000:
                print(f"  {sym}: already have {len(df)} candles, skipping")
                continue
        download_symbol(sym, pip, out)


if __name__ == "__main__":
    main()
