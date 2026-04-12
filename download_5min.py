"""Download Dukascopy 5-minute bars for the 3 validated pairs.

Reuses the same CDN tick downloader as download_dukascopy.py but
aggregates ticks into 5-minute OHLCV bars instead of 1-hour.

Each hour of ticks produces up to 12 five-minute bars. Over 16 years
(2010-2026) this is ~1.2M bars per pair (~100MB CSV per pair).

Conservative settings for a 2-vCPU VPS:
  - 5 concurrent workers per hour-batch
  - 0.1s pause between batches
  - 3 retries with 1/4/16s backoff
  - Intermediate save every 30 days (so progress isn't lost on crash)

Run:
    python download_5min.py

Produces: data/dukascopy/{SYMBOL}_5M.csv
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

SYMBOLS = {
    "EURUSD": 1e-5,
    "USDJPY": 1e-3,
    "XAUUSD": 1e-2,
}

START_YEAR = 2010
END = datetime(2026, 4, 1)
WORKERS = 5
SAVE_EVERY_DAYS = 30


def fetch_hour_ticks(symbol: str, pip: float, dt: datetime) -> list[dict] | None:
    """Download one hour of ticks, return list of raw tick dicts."""
    url = (f"{CDN}/{symbol}/{dt.year}/{dt.month-1:02d}/{dt.day:02d}/"
           f"{dt.hour:02d}h_ticks.bi5")
    for attempt in range(4):
        if attempt > 0:
            time.sleep([0, 1, 4, 16][attempt])
        try:
            r = requests.get(url, timeout=60)
            if r.status_code != 200 or len(r.content) < 20:
                if attempt == 3:
                    return None
                continue
            data = lzma.decompress(r.content)
            if len(data) < 20:
                return None
            ticks = []
            for off in range(0, len(data), 20):
                if off + 20 > len(data):
                    break
                ms, ask, bid, avol, bvol = struct.unpack(">IIIff", data[off:off+20])
                tick_time = dt + timedelta(milliseconds=ms)
                ticks.append({
                    "time": tick_time,
                    "bid": bid * pip,
                    "volume": bvol,
                })
            return ticks if ticks else None
        except Exception:
            if attempt == 3:
                return None
    return None


def ticks_to_5min(ticks: list[dict], hour_dt: datetime) -> list[dict]:
    """Aggregate raw ticks into up to 12 five-minute OHLCV bars."""
    if not ticks:
        return []
    bars = []
    for m in range(0, 60, 5):
        bar_start = hour_dt.replace(minute=m, second=0, microsecond=0)
        bar_end = bar_start + timedelta(minutes=5)
        window = [t for t in ticks if bar_start <= t["time"] < bar_end]
        if not window:
            continue
        prices = [t["bid"] for t in window]
        vols = [t["volume"] for t in window]
        bars.append({
            "timestamp": bar_start,
            "open": prices[0],
            "high": max(prices),
            "low": min(prices),
            "close": prices[-1],
            "volume": sum(vols),
        })
    return bars


def download_symbol(symbol: str, pip: float):
    out_path = OUT / f"{symbol}_5M.csv"

    # Resume support: check existing data
    existing_rows = 0
    last_date = datetime(START_YEAR, 1, 1)
    if out_path.exists():
        try:
            df = pd.read_csv(out_path, parse_dates=["timestamp"])
            existing_rows = len(df)
            if existing_rows > 0:
                last_date = pd.Timestamp(df["timestamp"].max()).to_pydatetime()
                last_date = last_date + timedelta(hours=1)
                print(f"  {symbol}: resuming from {last_date.date()} "
                      f"({existing_rows:,} existing rows)", flush=True)
        except Exception:
            pass

    all_bars: list[dict] = []
    if existing_rows > 0:
        df = pd.read_csv(out_path, parse_dates=["timestamp"])
        all_bars = df.to_dict("records")

    # Build hour list from resume point
    hours = []
    dt = last_date
    while dt < END:
        hours.append(dt)
        dt += timedelta(hours=1)

    print(f"  {symbol}: {len(hours):,} hours to fetch "
          f"({last_date.date()} -> {END.date()})", flush=True)

    batch_hours = []
    last_save = time.time()
    t0 = time.time()

    for hi, hour_dt in enumerate(hours):
        batch_hours.append(hour_dt)

        # Process in batches of WORKERS
        if len(batch_hours) >= WORKERS or hi == len(hours) - 1:
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = {
                    pool.submit(fetch_hour_ticks, symbol, pip, h): h
                    for h in batch_hours
                }
                for f in as_completed(futures):
                    ticks = f.result()
                    if ticks:
                        bars_5m = ticks_to_5min(ticks, futures[f])
                        all_bars.extend(bars_5m)
            batch_hours = []
            time.sleep(0.1)

        # Progress + intermediate save
        done = hi + 1
        if done % 500 == 0 or time.time() - last_save > SAVE_EVERY_DAYS * 86400 / len(hours) * 500:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1)
            eta = (len(hours) - done) / max(rate, 0.01)
            print(f"    {symbol}: {done:,}/{len(hours):,} hours "
                  f"({len(all_bars):,} bars) "
                  f"rate={rate:.0f}h/s eta={eta/60:.0f}min", flush=True)

        # Save every SAVE_EVERY_DAYS worth of hours
        if done % (SAVE_EVERY_DAYS * 24) == 0:
            df = pd.DataFrame(all_bars)
            df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            df.to_csv(out_path, index=False)
            last_save = time.time()

    # Final save
    if all_bars:
        df = pd.DataFrame(all_bars)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df.to_csv(out_path, index=False)
        years = (df["timestamp"].max() - df["timestamp"].min()).days / 365
        print(f"  {symbol}: DONE {len(df):,} bars, {years:.1f} years -> {out_path}",
              flush=True)


def main():
    print("=" * 60, flush=True)
    print("Dukascopy 5-minute downloader (3 validated pairs)", flush=True)
    print("=" * 60, flush=True)
    for symbol, pip in SYMBOLS.items():
        try:
            download_symbol(symbol, pip)
        except Exception as e:
            print(f"  {symbol}: ERROR {e}", flush=True)
    # Summary
    print("\n=== Summary ===", flush=True)
    for f in sorted(OUT.glob("*_5M.csv")):
        try:
            n = sum(1 for _ in open(f)) - 1
            size = f.stat().st_size / 1024 / 1024
            print(f"  {f.name}: {n:,} bars, {size:.1f} MB", flush=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
