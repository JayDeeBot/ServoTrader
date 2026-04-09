#!/usr/bin/env python3
"""
fetch_btc_5min_historical.py

Downloads the full available 5-minute BTCUSDT OHLCV history from the
Binance public data archive (data.binance.vision).

This approach uses static ZIP file downloads rather than the Binance REST API,
which means it is NOT affected by geo-restrictions (e.g. US / Australia blocks).
It is also significantly faster than paginated API calls.

Archive structure used
----------------------
  Monthly files (complete months):
    https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m/BTCUSDT-5m-YYYY-MM.zip

  Daily files (current incomplete month):
    https://data.binance.vision/data/spot/daily/klines/BTCUSDT/5m/BTCUSDT-5m-YYYY-MM-DD.zip

Coverage
--------
  Binance lists BTCUSDT 5-minute data from August 2017.
  Monthly archives are available up to (but not including) the current month.
  Daily archives fill in the current month up to yesterday.

Output
------
    /home/jarred/git/ServoTrader/data/5min_historical/BTCUSDT.csv
    Columns: timestamp, open, high, low, close, vwap, volume, count, symbol

Usage
-----
    python fetch_btc_5min_historical.py

Author: Jarred Deluca
Project: ServoTrader
License: MIT
"""

import io
import os
import time
import zipfile
from datetime import datetime, date, timezone, timedelta

import pandas as pd
import requests


# =============================================================================
# Configuration
# =============================================================================

SYMBOL      = "BTCUSDT"
INTERVAL    = "5m"
OUTPUT_PATH = "/home/jarred/git/ServoTrader/data/5min_historical/BTCUSDT.csv"

# Binance public data archive — no API key or geo-restriction required
BASE_URL = "https://data.binance.vision/data/spot"

# First month Binance has 5-minute BTCUSDT data
FIRST_YEAR  = 2017
FIRST_MONTH = 8

# HTTP settings
REQUEST_TIMEOUT    = 60    # seconds per download
SLEEP_BETWEEN_REQS = 0.1   # seconds between file downloads


# =============================================================================
# Column names for the Binance kline CSV format
# =============================================================================

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
]


# =============================================================================
# Helpers
# =============================================================================

def _download_zip(url: str) -> bytes | None:
    """
    Download a ZIP file from the archive. Returns the raw bytes on success,
    or None if the file does not exist (404) or the request fails.
    """
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.content
    except requests.RequestException as exc:
        print(f"    ⚠  Download failed: {url} — {exc}")
        return None


def _zip_to_df(raw_bytes: bytes) -> pd.DataFrame:
    """
    Extract a Binance kline CSV from a ZIP archive and return a DataFrame.

    Binance added a header row to their archive CSVs from 2025 onwards.
    Older files have no header. We detect which format we have by checking
    whether the first value in column 0 is numeric (old) or a string (new).
    """
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            # Peek at the first row to decide
            raw_df = pd.read_csv(f, header=None, names=KLINE_COLS, nrows=1)

        first_val = str(raw_df.iloc[0]["open_time"]).strip()
        has_header = not first_val.isdigit()

        with zf.open(csv_name) as f:
            if has_header:
                # Skip the header row; our own column names take precedence
                df = pd.read_csv(f, header=0, names=KLINE_COLS)
            else:
                df = pd.read_csv(f, header=None, names=KLINE_COLS)

    return df


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw kline DataFrame to the project's standard OHLCV format."""
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)

    # VWAP = quote_volume / base_volume
    df["vwap"] = df["quote_volume"] / df["volume"].replace(0, float("nan"))
    df["vwap"] = df["vwap"].replace([float("inf"), -float("inf")], float("nan"))

    # Binance archive files from 2025 onwards store timestamps in microseconds
    # rather than milliseconds. Detect by magnitude and normalise to milliseconds.
    #   ms timestamp for 2025  ≈ 1.7e12
    #   µs timestamp for 2025  ≈ 1.7e15  (1000× larger)
    raw_ts = pd.to_numeric(df["open_time"], errors="coerce")
    if raw_ts.iloc[0] > 1e14:          # value is in microseconds — divide to ms
        raw_ts = raw_ts // 1000

    df["timestamp"] = pd.to_datetime(
        raw_ts.astype("int64"), unit="ms", utc=True
    ).dt.tz_localize(None)   # timezone-naive UTC, consistent with prepare script

    df["symbol"] = SYMBOL

    return df[["timestamp", "open", "high", "low", "close", "vwap", "volume", "count", "symbol"]]


# =============================================================================
# Download logic
# =============================================================================

def _iter_monthly_urls() -> list:
    """
    Return (url, label) pairs for every complete calendar month from
    FIRST_YEAR/FIRST_MONTH up to (but not including) the current month.
    """
    today = date.today()
    pairs = []
    year, month = FIRST_YEAR, FIRST_MONTH

    while (year, month) < (today.year, today.month):
        label = f"{year}-{month:02d}"
        url   = (
            f"{BASE_URL}/monthly/klines/{SYMBOL}/{INTERVAL}/"
            f"{SYMBOL}-{INTERVAL}-{label}.zip"
        )
        pairs.append((url, label))
        month += 1
        if month > 12:
            month = 1
            year += 1

    return pairs


def _iter_daily_urls() -> list:
    """
    Return (url, label) pairs for every day in the current month up to
    (but not including) today, to fill in the gap left by monthly archives.
    """
    today = date.today()
    pairs = []
    d     = date(today.year, today.month, 1)

    while d < today:
        label = d.strftime("%Y-%m-%d")
        url   = (
            f"{BASE_URL}/daily/klines/{SYMBOL}/{INTERVAL}/"
            f"{SYMBOL}-{INTERVAL}-{label}.zip"
        )
        pairs.append((url, label))
        d += timedelta(days=1)

    return pairs


def fetch_all() -> pd.DataFrame:
    """Download all monthly + daily archives and return a combined DataFrame."""
    monthly_urls = _iter_monthly_urls()
    daily_urls   = _iter_daily_urls()
    all_urls     = monthly_urls + daily_urls

    total   = len(all_urls)
    frames  = []
    missing = []

    print(f"  Monthly archives: {len(monthly_urls)}")
    print(f"  Daily  archives : {len(daily_urls)}  (current month fill-in)")
    print(f"  Total files     : {total}\n")

    for i, (url, label) in enumerate(all_urls, 1):
        raw = _download_zip(url)

        if raw is None:
            missing.append(label)
            continue

        try:
            df = _zip_to_df(raw)
            df = _clean_df(df)
            frames.append(df)
        except Exception as exc:
            print(f"    ⚠  Parse error for {label}: {exc}")
            missing.append(label)
            continue

        # Progress every 12 files (~1 year of monthly files)
        if i % 12 == 0 or i == total:
            candles_so_far = sum(len(f) for f in frames)
            print(f"  [{i:>4}/{total}]  {label}  |  {candles_so_far:>9,} candles collected")

        time.sleep(SLEEP_BETWEEN_REQS)

    if missing:
        print(f"\n  ⚠  {len(missing)} files not found (may be expected for recent days):")
        for m in missing[:10]:
            print(f"       {m}")
        if len(missing) > 10:
            print(f"       ... and {len(missing) - 10} more")

    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return combined


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    SEP = "=" * 70

    print(f"\n{SEP}")
    print("  ServoTrader — 5-Minute BTC Historical Data Fetcher")
    print("  Source: data.binance.vision (no API key / no geo-restriction)")
    print(SEP)
    print(f"  Symbol  : {SYMBOL}")
    print(f"  Interval: {INTERVAL}")
    print(f"  Output  : {OUTPUT_PATH}")
    print(SEP + "\n")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    t0 = time.time()
    df = fetch_all()
    elapsed = timedelta(seconds=int(time.time() - t0))

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_PATH, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    approx_days = len(df) * 5 / 60 / 24

    print(f"\n{SEP}")
    print("  ✅  COMPLETE")
    print(SEP)
    print(f"  Candles saved  : {len(df):,}")
    print(f"  Date range     : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Coverage       : ≈{approx_days:.0f} days  ({approx_days / 365:.1f} years)")
    print(f"  Fetch time     : {elapsed}")
    print(f"  Saved to       : {OUTPUT_PATH}")
    print(SEP)

    # ── Gap check ─────────────────────────────────────────────────────────────
    expected_gap = pd.Timedelta(minutes=5)
    ts           = df["timestamp"]
    gaps         = ts.diff().dropna()
    large_gaps   = gaps[gaps > expected_gap * 2]

    if large_gaps.empty:
        print("\n  ✅  No significant timestamp gaps detected.")
    else:
        print(f"\n  ⚠   {len(large_gaps)} gaps larger than 10 minutes detected:")
        for idx in large_gaps.index[:10]:
            print(f"       {ts.iloc[idx - 1]} → {ts.iloc[idx]}  ({gaps.iloc[idx - 1]})")
        if len(large_gaps) > 10:
            print(f"       ... and {len(large_gaps) - 10} more")

    print()


if __name__ == "__main__":
    main()