#!/usr/bin/env python3
"""
collect_btc_sentiment_hourly.py

Collects BTC sentiment metrics and resamples to hourly frequency for hourly prediction models.

**IMPORTANT LIMITATION**: Most sentiment data is only available at DAILY granularity.
This script collects daily sentiment metrics and forward-fills them to hourly frequency.

Sources:
- Alternative.me Fear & Greed (daily)
- Google Trends (daily)
- Wikipedia views (daily)
- GitHub activity (daily/weekly)

For hourly predictions, sentiment serves as daily context.
Primary signal comes from OHLCV technical indicators.

Author: Jarred Deluca
Created: 2025
License: MIT
"""

import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_CSV = "/home/jarred/git/ServoTrader/data/btc_sentiment_hourly.csv"
START_DATE = "2020-01-01"  # ~3 years to match hourly OHLCV
END_DATE   = datetime.now().strftime("%Y-%m-%d")

# GitHub token (optional)
GITHUB_TOKEN = ""

# Rate limits
RATE_COINGECKO = 2.0
RATE_WIKIPEDIA = 0.5
RATE_GITHUB = 1.0
RATE_TRENDS = 5.0

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def safe_get(url, params=None, headers=None, timeout=30, retries=3, delay=2.0):
    """HTTP GET with retry logic."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r
            elif r.status_code == 429:
                wait = delay * (attempt + 1) * 3
                print(f"    ⏳ Rate limited. Waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                print(f"    ⚠️  HTTP {r.status_code}")
                time.sleep(delay)
        except Exception as e:
            print(f"    ❌ Request error: {e}")
            time.sleep(delay * (attempt + 1))
    return None


# ============================================================================
# COLLECT DAILY SENTIMENT DATA
# ============================================================================

def collect_fear_greed():
    """Fetch Fear & Greed Index (daily)."""
    section("SOURCE 1: Fear & Greed Index (Daily)")
    
    r = safe_get("https://api.alternative.me/fng/", params={"limit": 0})
    if r is None:
        return pd.DataFrame()
    
    data = r.json().get("data", [])
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
    df["fear_greed_value"] = pd.to_numeric(df["value"], errors="coerce")
    
    class_map = {"Extreme Fear": 0, "Fear": 1, "Neutral": 2, "Greed": 3, "Extreme Greed": 4}
    df["fear_greed_ordinal"] = df["value_classification"].map(class_map)
    df = df[["date", "fear_greed_value", "fear_greed_ordinal"]]
    
    print(f"  ✅ {len(df)} days")
    return df


def collect_google_trends():
    """Fetch Google Trends (daily)."""
    section("SOURCE 2: Google Trends (Daily)")
    
    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("  ⚠️  pytrends not installed. Skipping.")
        return pd.DataFrame()
    
    keywords = ["bitcoin", "buy bitcoin", "bitcoin price"]
    pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25), retries=2, backoff_factor=0.5)
    
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")
    
    all_chunks = []
    chunk_start = start_dt
    chunk_days = 269  # Max for daily resolution
    
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), end_dt)
        timeframe = f"{chunk_start.strftime('%Y-%m-%d')} {chunk_end.strftime('%Y-%m-%d')}"
        
        try:
            pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo="", gprop="")
            df_chunk = pytrends.interest_over_time()
            
            if not df_chunk.empty:
                df_chunk = df_chunk.drop(columns=["isPartial"], errors="ignore")
                df_chunk.index = pd.to_datetime(df_chunk.index).date
                df_chunk.index.name = "date"
                df_chunk = df_chunk.reset_index()
                all_chunks.append(df_chunk)
            
            time.sleep(RATE_TRENDS)
        except Exception as e:
            print(f"    ❌ Error: {e}")
            time.sleep(RATE_TRENDS * 2)
        
        chunk_start = chunk_end + timedelta(days=1)
    
    if not all_chunks:
        print("  ❌ No data collected")
        return pd.DataFrame()
    
    df = pd.concat(all_chunks).drop_duplicates(subset="date").sort_values("date")
    
    rename_map = {
        "bitcoin": "trends_bitcoin",
        "buy bitcoin": "trends_buy_bitcoin",
        "bitcoin price": "trends_bitcoin_price",
    }
    df = df.rename(columns=rename_map)
    
    core_cols = [c for c in rename_map.values() if c in df.columns]
    df["trends_composite"] = df[core_cols].mean(axis=1)
    
    print(f"  ✅ {len(df)} days")
    return df


def collect_wikipedia_pageviews():
    """Fetch Wikipedia page views (daily)."""
    section("SOURCE 3: Wikipedia Page Views (Daily)")
    
    base_url = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
    headers = {"User-Agent": "btc_sentiment_collector/1.0"}
    
    start_dt = max(datetime.strptime(START_DATE, "%Y-%m-%d"), datetime(2015, 7, 1))
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")
    
    all_chunks = []
    chunk_start = start_dt
    
    while chunk_start <= end_dt:
        chunk_end = min(chunk_start.replace(year=chunk_start.year + 1) - timedelta(days=1), end_dt)
        start_str = chunk_start.strftime("%Y%m%d")
        end_str = chunk_end.strftime("%Y%m%d")
        
        url = f"{base_url}/en.wikipedia.org/all-access/all-agents/Bitcoin/daily/{start_str}00/{end_str}00"
        r = safe_get(url, headers=headers)
        
        if r:
            items = r.json().get("items", [])
            if items:
                df_chunk = pd.DataFrame(items)
                df_chunk["date"] = pd.to_datetime(df_chunk["timestamp"], format="%Y%m%d00").dt.date
                df_chunk = df_chunk.rename(columns={"views": "wikipedia_views"})
                df_chunk = df_chunk[["date", "wikipedia_views"]]
                all_chunks.append(df_chunk)
        
        time.sleep(RATE_WIKIPEDIA)
        chunk_start = chunk_end + timedelta(days=1)
    
    if not all_chunks:
        print("  ❌ No data collected")
        return pd.DataFrame()
    
    df = pd.concat(all_chunks).drop_duplicates(subset="date").sort_values("date")
    df["wikipedia_views_ma7"] = df["wikipedia_views"].rolling(7).mean()
    
    print(f"  ✅ {len(df)} days")
    return df


def collect_github_activity():
    """Fetch GitHub activity (daily/weekly)."""
    section("SOURCE 4: GitHub Activity (Daily/Weekly)")
    
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    
    # Commit activity (last 52 weeks)
    r = safe_get("https://api.github.com/repos/bitcoin/bitcoin/stats/commit_activity", headers=headers)
    
    commit_df = pd.DataFrame()
    if r:
        try:
            weeks = r.json()
            rows = []
            for week in weeks:
                week_start = datetime.fromtimestamp(week["week"]).date()
                daily_commits = week.get("days", [0]*7)
                for i, count in enumerate(daily_commits):
                    day = week_start + timedelta(days=i)
                    rows.append({"date": day, "github_commits": count})
            commit_df = pd.DataFrame(rows)
            print(f"  ✅ {len(commit_df)} days of commit data")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    time.sleep(RATE_GITHUB)
    
    # Repo metadata
    r = safe_get("https://api.github.com/repos/bitcoin/bitcoin", headers=headers)
    repo_meta = {}
    if r:
        try:
            data = r.json()
            repo_meta["github_stars"] = data.get("stargazers_count", np.nan)
            repo_meta["github_forks"] = data.get("forks_count", np.nan)
            repo_meta["github_watchers"] = data.get("watchers_count", np.nan)
            repo_meta["github_open_issues"] = data.get("open_issues_count", np.nan)
        except Exception:
            pass
    
    if commit_df.empty:
        print("  ❌ No GitHub data")
        return pd.DataFrame()
    
    # Add metadata as constant columns
    for col, val in repo_meta.items():
        commit_df[col] = val
    
    return commit_df


def engineer_sentiment_features(df):
    """Engineer features from raw sentiment."""
    section("ENGINEERING SENTIMENT FEATURES")
    
    # Fear & Greed momentum
    if "fear_greed_value" in df.columns:
        df["fear_greed_ma7"] = df["fear_greed_value"].rolling(7).mean()
        df["fear_greed_ma14"] = df["fear_greed_value"].rolling(14).mean()
        df["fear_greed_change"] = df["fear_greed_value"].diff()
        df["fear_greed_momentum"] = df["fear_greed_ma7"] - df["fear_greed_ma14"]
    
    # Wikipedia attention spikes
    if "wikipedia_views" in df.columns:
        df["wiki_views_ma7"] = df["wikipedia_views"].rolling(7).mean()
        df["wiki_views_zscore"] = (
            df["wikipedia_views"] - df["wikipedia_views"].rolling(30).mean()
        ) / (df["wikipedia_views"].rolling(30).std() + 1e-9)
    
    # Google Trends momentum
    if "trends_bitcoin" in df.columns:
        df["trends_bitcoin_ma7"] = df["trends_bitcoin"].rolling(7).mean()
        df["trends_bitcoin_zscore"] = (
            df["trends_bitcoin"] - df["trends_bitcoin"].rolling(30).mean()
        ) / (df["trends_bitcoin"].rolling(30).std() + 1e-9)
    
    # Composite social volume
    social_volume_cols = [c for c in ["trends_bitcoin", "wikipedia_views"] if c in df.columns]
    if social_volume_cols:
        normalised = pd.DataFrame()
        for col in social_volume_cols:
            col_max = df[col].max()
            normalised[col] = df[col] / col_max if col_max > 0 else df[col]
        df["composite_social_volume"] = normalised.mean(axis=1)
    
    print(f"  ✅ Engineered {len(df.columns)} total columns")
    return df


# ============================================================================
# RESAMPLE TO HOURLY
# ============================================================================

def resample_to_hourly(daily_df):
    """Resample daily sentiment to hourly by forward-filling."""
    section("RESAMPLING TO HOURLY FREQUENCY")
    
    print("  ⚠️  IMPORTANT: Sentiment metrics are DAILY aggregates")
    print("     Each hour within a day will have the SAME values")
    print("     Values update once per day")
    
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df = daily_df.set_index('date')
    
    start = daily_df.index.min()
    end = daily_df.index.max() + timedelta(days=1)
    hourly_index = pd.date_range(start=start, end=end, freq='1H')
    
    hourly_df = daily_df.reindex(hourly_index, method='ffill')
    hourly_df.index.name = 'timestamp'
    hourly_df = hourly_df.reset_index()
    
    print(f"\n  ✅ Resampled from {len(daily_df)} days to {len(hourly_df)} hours")
    return hourly_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    section("BTC HOURLY SENTIMENT DATA COLLECTOR")
    
    print(f"  ⚠️  LIMITATION:")
    print(f"     Sentiment data is DAILY only")
    print(f"     This script forward-fills daily values to hourly")
    
    print(f"\n  Date range: {START_DATE} → {END_DATE}")
    print(f"  Output: {OUTPUT_CSV}")
    
    # Collect daily data
    datasets = {}
    datasets["fear_greed"] = collect_fear_greed()
    datasets["google_trends"] = collect_google_trends()
    datasets["wikipedia"] = collect_wikipedia_pageviews()
    datasets["github"] = collect_github_activity()
    
    # Merge
    section("MERGING DAILY DATASETS")
    
    successful = {k: v for k, v in datasets.items() if not v.empty}
    if not successful:
        print("  ❌ No data collected")
        return
    
    sources = list(successful.values())
    merged = sources[0]
    for df in sources[1:]:
        merged = pd.merge(merged, df, on="date", how="outer")
    
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged[(merged["date"] >= START_DATE) & (merged["date"] <= END_DATE)]
    merged = merged.sort_values("date").reset_index(drop=True)
    
    print(f"  ✅ Merged: {len(merged)} days × {len(merged.columns)} columns")
    
    # Engineer features
    merged = engineer_sentiment_features(merged)
    
    # Resample to hourly
    hourly_df = resample_to_hourly(merged)
    
    # Save
    section("SAVING OUTPUT")
    hourly_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  ✅ Saved to: {OUTPUT_CSV}")
    print(f"     Rows: {len(hourly_df):,} hours")
    print(f"     Cols: {len(hourly_df.columns)}")
    
    section("✅ COLLECTION COMPLETE")
    
    print(f"\n  ℹ️  Usage Notes:")
    print(f"     1. Sentiment features update once per day")
    print(f"     2. For intraday predictions, these serve as daily context")
    print(f"     3. Primary hourly signal comes from OHLCV technical indicators")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()