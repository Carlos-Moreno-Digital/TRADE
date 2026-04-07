"""Robust Dukascopy downloader for LOB calibration.

Pulls 1 month of 1H data per symbol (real bid OHLCV with TICK VOLUME).
Conservative concurrency + generous backoff so the CDN does not
rate-limit us:

  - 3 worker threads (Dukascopy throttles aggressive parallelism)
  - 60s read timeout per request
  - exponential backoff: 1s, 4s, 16s on transient failure
  - per-day batching with intermediate save after each day
  - prints only one line per day (token rule)

Output: data/dukascopy/{SYMBOL}_calib.csv with columns
    timestamp,open,high,low,close,volume

The volume column is the SUM of bid tick volumes within the hour.
This is the real liquidity proxy we need for depth calibration.
"""
from __future__ import annotations

import lzma
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

OUT = Path("data/dukascopy")
OUT.mkdir(parents=True, exist_ok=True)
CDN = "https://datafeed.dukascopy.com/datafeed"

# (display name, dukas symbol, pip multiplier)
SYMBOLS = [
    ("EURUSD", "EURUSD", 1e-5),
    ("USDJPY", "USDJPY", 1e-3),
]

# 1 month is enough for spread/volume calibration
START = datetime(2024, 6, 1)
END = datetime(2024, 7, 1)


def fetch_hour(symbol: str, pip: float, dt: datetime) -> dict | None:
    url = (
        f"{CDN}/{symbol}/{dt.year}/{dt.month - 1:02d}/{dt.day:02d}/"
        f"{dt.hour:02d}h_ticks.bi5"
    )
    backoff = [1, 4, 16]
    for i, wait in enumerate([0] + backoff):
        if wait:
            time.sleep(wait)
        try:
            r = requests.get(url, timeout=60)
            if r.status_code != 200 or len(r.content) < 20:
                if i == len(backoff):
                    return None
                continue
            data = lzma.decompress(r.content)
            if len(data) < 20:
                return None
            bids: list[float] = []
            vols: list[float] = []
            for off in range(0, len(data), 20):
                if off + 20 > len(data):
                    break
                _, _, bid, _, bvol = struct.unpack(">IIIff", data[off:off + 20])
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
                "volume": float(sum(vols)),
                "n_ticks": len(bids),
            }
        except Exception:
            if i == len(backoff):
                return None
    return None


def download_symbol(name: str, dukas: str, pip: float) -> Path:
    out_path = OUT / f"{name}_calib.csv"

    if out_path.exists():
        existing = pd.read_csv(out_path, parse_dates=["timestamp"])
        if len(existing) > 500:
            print(f"  {name}: existing {len(existing)} rows, skipping")
            return out_path

    rows: list[dict] = []
    day = START
    total_hours = 0
    while day < END:
        hours_today = [day + timedelta(hours=h) for h in range(24)]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(fetch_hour, dukas, pip, h): h for h in hours_today}
            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    rows.append(result)
        total_hours += 24
        # Intermediate save every day
        if rows:
            df = pd.DataFrame(rows).set_index("timestamp").sort_index()
            df = df[~df.index.duplicated(keep="first")]
            df.to_csv(out_path)
        print(
            f"    {name} {day.date()}: {len(rows)} valid / {total_hours} requested",
            flush=True,
        )
        day += timedelta(days=1)

    if rows:
        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        df.to_csv(out_path)
        print(
            f"  {name}: saved {len(df)} bars, "
            f"avg ticks/hour={df['n_ticks'].mean():.0f}, "
            f"avg vol/hour={df['volume'].mean():.1f}"
        )
    return out_path


def main() -> int:
    for name, dukas, pip in SYMBOLS:
        try:
            download_symbol(name, dukas, pip)
        except Exception as e:
            print(f"  {name}: ERROR {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
