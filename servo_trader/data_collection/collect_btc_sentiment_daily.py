#!/usr/bin/env python3
"""
collect_btc_sentiment_daily.py

Collects daily market sentiment data for Bitcoin from multiple free sources.
Sources: Alternative.me, Google Trends, Wikipedia, CryptoCompare, GDELT,
         CoinGecko, Reddit (PRAW), Stocktwits, GitHub

Author: Jarred Deluca
Created: 2025
License: MIT
"""

import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta, date
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_CSV = "/home/jarred/git/ServoTrader/data/btc_sentiment_daily.csv"
START_DATE = "2017-01-01"
END_DATE   = datetime.now().strftime("%Y-%m-%d")

# Reddit credentials (optional — leave empty strings to skip PRAW)
# Get free credentials at: https://www.reddit.com/prefs/apps
REDDIT_CLIENT_ID     = ""
REDDIT_CLIENT_SECRET = ""
REDDIT_USER_AGENT    = "btc_sentiment_collector/1.0"

# GitHub token (optional — leave empty to use unauthenticated 60 req/hr limit)
# Get free token at: https://github.com/settings/tokens
GITHUB_TOKEN = ""

# Rate limits (seconds between requests)
RATE_COINGECKO    = 2.0
RATE_CRYPTOCOMPARE = 1.0
RATE_GDELT        = 1.5
RATE_WIKIPEDIA    = 0.5
RATE_GITHUB       = 1.0
RATE_STOCKTWITS   = 1.0
RATE_TRENDS       = 5.0   # Google Trends is aggressive with rate limiting

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def date_range_list(start, end):
    """Returns list of date strings between start and end (inclusive)."""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    delta    = end_dt - start_dt
    return [(start_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(delta.days + 1)]


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
            elif r.status_code in (401, 403):
                print(f"    🔒 Auth required (HTTP {r.status_code}) — skipping")
                return None
            else:
                print(f"    ⚠️  HTTP {r.status_code} on attempt {attempt+1}")
                time.sleep(delay)
        except Exception as e:
            print(f"    ❌ Request error (attempt {attempt+1}): {e}")
            time.sleep(delay * (attempt + 1))
    return None


def section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


# ============================================================================
# SOURCE 1 — ALTERNATIVE.ME FEAR & GREED INDEX
# ============================================================================

def collect_fear_greed():
    """
    Fetches full history of the Crypto Fear & Greed Index.
    Available from: Feb 2018
    No key required.
    """
    section("SOURCE 1: Alternative.me — Fear & Greed Index")

    r = safe_get("https://api.alternative.me/fng/", params={"limit": 0})
    if r is None:
        print("  ❌ Failed")
        return pd.DataFrame()

    data = r.json().get("data", [])
    df = pd.DataFrame(data)
    df["date"]             = pd.to_datetime(df["timestamp"], unit="s").dt.date
    df["fear_greed_value"] = pd.to_numeric(df["value"], errors="coerce")
    df["fear_greed_class"] = df["value_classification"]
    df = df[["date", "fear_greed_value", "fear_greed_class"]]

    # Encode classification as ordinal
    class_map = {
        "Extreme Fear": 0, "Fear": 1, "Neutral": 2,
        "Greed": 3, "Extreme Greed": 4
    }
    df["fear_greed_ordinal"] = df["fear_greed_class"].map(class_map)
    df = df.drop(columns=["fear_greed_class"])

    print(f"  ✅ {len(df)} days  |  {df['date'].min()} → {df['date'].max()}")
    return df


# ============================================================================
# SOURCE 2 — GOOGLE TRENDS (via pytrends)
# ============================================================================

def collect_google_trends():
    """
    Fetches historical Google Trends data for Bitcoin-related search terms.
    Uses pytrends (unofficial wrapper). Collects in 90-day chunks to get
    daily granularity rather than weekly.
    
    Terms: bitcoin, buy bitcoin, bitcoin price, bitcoin crash, bitcoin halving
    """
    section("SOURCE 2: Google Trends — Bitcoin Search Interest")

    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("  ⚠️  pytrends not installed. Run: pip install pytrends")
        print("  ⏭️  Skipping Google Trends")
        return pd.DataFrame()

    keywords = ["bitcoin", "buy bitcoin", "bitcoin price", "bitcoin crash", "bitcoin halving"]

    pytrends   = TrendReq(hl="en-US", tz=0, timeout=(10, 25), retries=2, backoff_factor=0.5)
    start_dt   = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt     = datetime.strptime(END_DATE,   "%Y-%m-%d")

    all_chunks = []
    chunk_start = start_dt

    # Collect in 269-day chunks (max for daily granularity from Google)
    chunk_days = 269

    total_chunks = max(1, int((end_dt - start_dt).days / chunk_days) + 1)
    print(f"  📦 Collecting {total_chunks} chunks (269 days each for daily resolution)...")

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
                print(f"    ✅ {timeframe}: {len(df_chunk)} records")
            else:
                print(f"    ⚠️  {timeframe}: No data returned")

            time.sleep(RATE_TRENDS)

        except Exception as e:
            print(f"    ❌ Error on {timeframe}: {e}")
            time.sleep(RATE_TRENDS * 2)

        chunk_start = chunk_end + timedelta(days=1)

    if not all_chunks:
        print("  ❌ No Google Trends data collected")
        return pd.DataFrame()

    df = pd.concat(all_chunks).drop_duplicates(subset="date").sort_values("date")

    # Rename columns
    rename_map = {
        "bitcoin":         "trends_bitcoin",
        "buy bitcoin":     "trends_buy_bitcoin",
        "bitcoin price":   "trends_bitcoin_price",
        "bitcoin crash":   "trends_bitcoin_crash",
        "bitcoin halving": "trends_bitcoin_halving",
    }
    df = df.rename(columns=rename_map)

    # Composite: general interest score
    core_cols = [c for c in rename_map.values() if c in df.columns]
    df["trends_composite"] = df[core_cols].mean(axis=1)

    print(f"  ✅ {len(df)} days  |  {df['date'].min()} → {df['date'].max()}")
    return df


# ============================================================================
# SOURCE 3 — WIKIPEDIA PAGE VIEWS
# ============================================================================

def collect_wikipedia_pageviews():
    """
    Fetches daily Wikipedia page views for the 'Bitcoin' article (English).
    Available from: 2015-07-01
    No key required.
    """
    section("SOURCE 3: Wikipedia — Bitcoin Article Daily Views")

    base_url = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
    headers  = {"User-Agent": "btc_sentiment_collector/1.0 (research project)"}

    start_dt = max(datetime.strptime(START_DATE, "%Y-%m-%d"), datetime(2015, 7, 1))
    end_dt   = datetime.strptime(END_DATE, "%Y-%m-%d")

    # Collect in yearly chunks
    all_chunks = []
    chunk_start = start_dt

    while chunk_start <= end_dt:
        chunk_end = min(chunk_start.replace(year=chunk_start.year + 1) - timedelta(days=1), end_dt)
        start_str = chunk_start.strftime("%Y%m%d")
        end_str   = chunk_end.strftime("%Y%m%d")

        url = f"{base_url}/en.wikipedia.org/all-access/all-agents/Bitcoin/daily/{start_str}00/{end_str}00"
        r   = safe_get(url, headers=headers)

        if r:
            items = r.json().get("items", [])
            if items:
                df_chunk = pd.DataFrame(items)
                df_chunk["date"] = pd.to_datetime(df_chunk["timestamp"], format="%Y%m%d00").dt.date
                df_chunk = df_chunk.rename(columns={"views": "wikipedia_views"})
                df_chunk = df_chunk[["date", "wikipedia_views"]]
                all_chunks.append(df_chunk)
                print(f"    ✅ {start_str}–{end_str}: {len(df_chunk)} records")

        time.sleep(RATE_WIKIPEDIA)
        chunk_start = chunk_end + timedelta(days=1)

    if not all_chunks:
        print("  ❌ No Wikipedia data collected")
        return pd.DataFrame()

    df = pd.concat(all_chunks).drop_duplicates(subset="date").sort_values("date")
    df["wikipedia_views_ma7"] = df["wikipedia_views"].rolling(7).mean()

    print(f"  ✅ {len(df)} days  |  {df['date'].min()} → {df['date'].max()}")
    return df


# ============================================================================
# SOURCE 4 — CRYPTOCOMPARE SOCIAL STATS
# ============================================================================

def collect_cryptocompare_social():
    """
    Fetches aggregated daily social stats for Bitcoin from CryptoCompare.
    Includes Reddit and Twitter metrics without needing NLP.
    Bitcoin CoinId = 1182
    No key required for social stats endpoint.
    """
    section("SOURCE 4: CryptoCompare — Social Stats (Reddit + Twitter)")

    url    = "https://min-api.cryptocompare.com/data/social/coin/histo/day"
    all_records = []
    limit  = 2000
    to_ts  = int(datetime.strptime(END_DATE, "%Y-%m-%d").timestamp())
    start_ts = int(datetime.strptime(START_DATE, "%Y-%m-%d").timestamp())

    print(f"  📦 Paginating backwards from {END_DATE}...")

    while True:
        params = {
            "coinId": 1182,      # Bitcoin's CryptoCompare ID
            "limit":  limit,
            "toTs":   to_ts
        }
        r = safe_get(url, params=params)
        if r is None:
            break

        data = r.json().get("Data", [])
        if not data:
            break

        all_records.extend(data)
        earliest_ts = data[0].get("time", 0)
        print(f"    ✅ Fetched {len(data)} records | earliest: {datetime.fromtimestamp(earliest_ts).strftime('%Y-%m-%d')}")

        if earliest_ts <= start_ts:
            break

        to_ts = earliest_ts - 1
        time.sleep(RATE_CRYPTOCOMPARE)

    if not all_records:
        print("  ❌ No CryptoCompare data collected")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["time"], unit="s").dt.date

    # Select available social columns
    social_cols = {
        "Reddit.posts_per_day":       "reddit_posts_per_day",
        "Reddit.comments_per_day":    "reddit_comments_per_day",
        "Reddit.active_users":        "reddit_active_users",
        "Reddit.subscribers":         "reddit_subscribers",
        "Twitter.statuses":           "twitter_statuses",
        "Twitter.followers":          "twitter_followers",
        "Facebook.likes":             "facebook_likes",
        "CodeRepository.forks":       "github_forks",
        "CodeRepository.stars":       "github_stars",
    }

    # Flatten nested dict columns
    for source_key, col_name in social_cols.items():
        parts = source_key.split(".")
        if parts[0] in df.columns:
            try:
                df[col_name] = df[parts[0]].apply(
                    lambda x: x.get(parts[1], np.nan) if isinstance(x, dict) else np.nan
                )
            except Exception:
                pass
        else:
            # Already flat from some API responses
            flat_key = source_key.replace(".", "_")
            if flat_key in df.columns:
                df[col_name] = pd.to_numeric(df[flat_key], errors="coerce")

    available = ["date"] + [v for v in social_cols.values() if v in df.columns]
    df = df[available].drop_duplicates(subset="date").sort_values("date")

    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  ✅ {len(df)} days  |  {df['date'].min()} → {df['date'].max()}")
    print(f"     Columns: {[c for c in df.columns if c != 'date']}")
    return df


# ============================================================================
# SOURCE 5 — GDELT PROJECT (NEWS VOLUME & SENTIMENT TONE)
# ============================================================================

def collect_gdelt():
    """
    Fetches Bitcoin news volume and average sentiment tone from GDELT.
    Uses the GDELT 2.0 Doc API timeline queries.
    No key required.
    
    Tone range: -100 (very negative) to +100 (very positive)
    """
    section("SOURCE 5: GDELT — Global News Volume & Sentiment Tone")

    # GDELT timeline volume
    url_vol = "https://api.gdeltproject.org/api/v2/doc/doc"

    # GDELT can only return ~3 years at a time reliably
    # We'll collect in 1-year chunks
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt   = datetime.strptime(END_DATE,   "%Y-%m-%d")

    all_vol  = []
    all_tone = []

    chunk_start = start_dt
    while chunk_start <= end_dt:
        chunk_end = min(chunk_start + timedelta(days=365), end_dt)
        start_str = chunk_start.strftime("%Y%m%d%H%M%S")
        end_str   = chunk_end.strftime("%Y%m%d%H%M%S")

        # Volume timeline
        params_vol = {
            "query":      "bitcoin",
            "mode":       "timelinevol",
            "startdatetime": start_str,
            "enddatetime":   end_str,
            "format":     "json",
            "smoothing":  0,
        }
        r = safe_get(url_vol, params=params_vol)
        if r:
            try:
                data = r.json()
                timeline = data.get("timeline", [{}])[0].get("data", [])
                for item in timeline:
                    all_vol.append({
                        "date":  pd.to_datetime(item.get("date"), format="%Y%m%dT%H%M%S").date(),
                        "gdelt_news_volume": item.get("value", np.nan)
                    })
                print(f"    ✅ Volume {chunk_start.strftime('%Y-%m-%d')}–{chunk_end.strftime('%Y-%m-%d')}: {len(timeline)} records")
            except Exception as e:
                print(f"    ⚠️  Volume parse error: {e}")

        time.sleep(RATE_GDELT)

        # Tone timeline
        params_tone = {**params_vol, "mode": "timelinetone"}
        r = safe_get(url_vol, params=params_tone)
        if r:
            try:
                data = r.json()
                timeline = data.get("timeline", [{}])[0].get("data", [])
                for item in timeline:
                    all_tone.append({
                        "date":       pd.to_datetime(item.get("date"), format="%Y%m%dT%H%M%S").date(),
                        "gdelt_avg_tone": item.get("value", np.nan)
                    })
                print(f"    ✅ Tone   {chunk_start.strftime('%Y-%m-%d')}–{chunk_end.strftime('%Y-%m-%d')}: {len(timeline)} records")
            except Exception as e:
                print(f"    ⚠️  Tone parse error: {e}")

        time.sleep(RATE_GDELT)
        chunk_start = chunk_end + timedelta(days=1)

    if not all_vol and not all_tone:
        print("  ❌ No GDELT data collected")
        return pd.DataFrame()

    df_vol  = pd.DataFrame(all_vol).drop_duplicates(subset="date")  if all_vol  else pd.DataFrame()
    df_tone = pd.DataFrame(all_tone).drop_duplicates(subset="date") if all_tone else pd.DataFrame()

    if not df_vol.empty and not df_tone.empty:
        df = pd.merge(df_vol, df_tone, on="date", how="outer")
    elif not df_vol.empty:
        df = df_vol
    else:
        df = df_tone

    df = df.sort_values("date")
    print(f"  ✅ {len(df)} days  |  {df['date'].min()} → {df['date'].max()}")
    return df


# ============================================================================
# SOURCE 6 — COINGECKO (BTC DOMINANCE + COMMUNITY STATS)
# ============================================================================

def collect_coingecko():
    """
    Fetches Bitcoin dominance and community/developer stats from CoinGecko.
    Free tier: 10-50 calls/min. No key required.
    
    - BTC dominance from /global (snapshot only — need historical workaround)
    - Community stats from /coins/bitcoin (Reddit, Twitter metrics)
    """
    section("SOURCE 6: CoinGecko — BTC Dominance & Community Stats")

    all_records = []

    # --- 6a: Market Chart (volume + market cap for dominance proxy) ---
    print("  🔍 Fetching BTC market chart (max history)...")
    url_market = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params_market = {
        "vs_currency": "usd",
        "days":        "max",
        "interval":    "daily"
    }
    r = safe_get(url_market, params=params_market)
    market_df = pd.DataFrame()
    if r:
        data = r.json()
        prices     = pd.DataFrame(data.get("prices",      []), columns=["ts", "price_cg"])
        volumes    = pd.DataFrame(data.get("total_volumes", []), columns=["ts", "volume_cg"])
        mktcap     = pd.DataFrame(data.get("market_caps",  []), columns=["ts", "market_cap_cg"])

        prices["date"]  = pd.to_datetime(prices["ts"],  unit="ms").dt.date
        volumes["date"] = pd.to_datetime(volumes["ts"], unit="ms").dt.date
        mktcap["date"]  = pd.to_datetime(mktcap["ts"],  unit="ms").dt.date

        market_df = prices[["date", "price_cg"]] \
            .merge(volumes[["date", "volume_cg"]], on="date", how="outer") \
            .merge(mktcap[["date",  "market_cap_cg"]], on="date", how="outer")
        print(f"    ✅ {len(market_df)} days of market data")
    time.sleep(RATE_COINGECKO)

    # --- 6b: Global dominance (current snapshot — limited historical) ---
    print("  🔍 Fetching global BTC dominance snapshot...")
    r = safe_get("https://api.coingecko.com/api/v3/global")
    dominance_today = np.nan
    if r:
        dominance_today = r.json().get("data", {}).get("market_cap_percentage", {}).get("btc", np.nan)
        print(f"    ✅ Current BTC dominance: {dominance_today:.1f}%")
    time.sleep(RATE_COINGECKO)

    # --- 6c: Historical global market chart (total market cap) ---
    print("  🔍 Fetching total crypto market cap (for dominance calculation)...")
    url_global = "https://api.coingecko.com/api/v3/global/market_cap_chart"
    params_global = {"days": "max"}
    r = safe_get(url_global, params=params_global)
    global_df = pd.DataFrame()
    if r:
        data = r.json().get("market_cap_chart", {})
        total_mktcap_data = data.get("market_cap", [])
        if total_mktcap_data:
            global_df = pd.DataFrame(total_mktcap_data, columns=["ts", "total_market_cap"])
            global_df["date"] = pd.to_datetime(global_df["ts"], unit="ms").dt.date
            global_df = global_df[["date", "total_market_cap"]]
            print(f"    ✅ {len(global_df)} days of total market cap data")
    time.sleep(RATE_COINGECKO)

    # Merge and compute BTC dominance
    cg_df = market_df.copy() if not market_df.empty else pd.DataFrame()
    if not global_df.empty and not cg_df.empty:
        cg_df = cg_df.merge(global_df, on="date", how="left")
        cg_df["btc_dominance_pct"] = (cg_df["market_cap_cg"] / cg_df["total_market_cap"]) * 100
        cg_df = cg_df.drop(columns=["total_market_cap"], errors="ignore")
        print(f"    ✅ BTC dominance calculated for {cg_df['btc_dominance_pct'].notna().sum()} days")

    if cg_df.empty:
        print("  ❌ No CoinGecko data collected")
        return pd.DataFrame()

    cg_df = cg_df.sort_values("date")
    print(f"  ✅ {len(cg_df)} days  |  {cg_df['date'].min()} → {cg_df['date'].max()}")
    return cg_df


# ============================================================================
# SOURCE 7 — REDDIT (via PRAW or Pushshift fallback)
# ============================================================================

def collect_reddit_sentiment():
    """
    Collects Reddit sentiment from r/Bitcoin and r/CryptoCurrency.
    Requires PRAW credentials for live data.
    Falls back to PullPush (Pushshift successor) for historical data.
    Applies VADER NLP sentiment to post titles.
    """
    section("SOURCE 7: Reddit — r/Bitcoin Sentiment (VADER NLP)")

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        print("  ⚠️  vaderSentiment not installed. Run: pip install vaderSentiment")
        print("  ⏭️  Skipping Reddit sentiment")
        return pd.DataFrame()

    analyzer = SentimentIntensityAnalyzer()

    # --- Try PullPush (Pushshift successor) for historical data ---
    print("  🔍 Attempting PullPush API for historical Reddit data...")

    start_ts = int(datetime.strptime(START_DATE, "%Y-%m-%d").timestamp())
    end_ts   = int(datetime.strptime(END_DATE,   "%Y-%m-%d").timestamp())

    daily_records = {}
    current_ts = start_ts
    subreddits = ["Bitcoin", "CryptoCurrency"]

    # Collect in 7-day windows to manage volume
    window_days = 7
    window_sec  = window_days * 86400
    total_windows = int((end_ts - start_ts) / window_sec) + 1
    windows_done  = 0
    consecutive_errors = 0

    print(f"  📦 Collecting {total_windows} weekly windows across {subreddits}...")

    while current_ts < end_ts:
        window_end = min(current_ts + window_sec, end_ts)

        for subreddit in subreddits:
            url = "https://api.pullpush.io/reddit/search/submission/"
            params = {
                "subreddit": subreddit,
                "after":     current_ts,
                "before":    window_end,
                "size":      100,
                "fields":    "title,score,num_comments,created_utc",
            }
            r = safe_get(url, params=params, retries=2, delay=1.0)

            if r is None:
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    print("  ⚠️  Too many consecutive errors — PullPush may be down, stopping Reddit collection")
                    break
                continue

            consecutive_errors = 0

            try:
                posts = r.json().get("data", [])
                for post in posts:
                    post_date = datetime.fromtimestamp(post.get("created_utc", 0)).date()
                    title     = post.get("title", "")
                    score     = post.get("score", 0)
                    comments  = post.get("num_comments", 0)
                    sentiment = analyzer.polarity_scores(title)

                    if post_date not in daily_records:
                        daily_records[post_date] = {
                            "post_count": 0, "total_score": 0,
                            "total_comments": 0, "sentiment_compound": [],
                            "sentiment_positive": [], "sentiment_negative": []
                        }
                    daily_records[post_date]["post_count"]          += 1
                    daily_records[post_date]["total_score"]          += score
                    daily_records[post_date]["total_comments"]        += comments
                    daily_records[post_date]["sentiment_compound"].append(sentiment["compound"])
                    daily_records[post_date]["sentiment_positive"].append(sentiment["pos"])
                    daily_records[post_date]["sentiment_negative"].append(sentiment["neg"])

            except Exception as e:
                print(f"    ⚠️  Parse error: {e}")

            time.sleep(0.5)

        if consecutive_errors >= 5:
            break

        current_ts = window_end
        windows_done += 1

        if windows_done % 52 == 0:
            print(f"    📅 Progress: {windows_done}/{total_windows} windows ({(windows_done/total_windows)*100:.0f}%)")

    if not daily_records:
        print("  ❌ No Reddit data collected")
        return pd.DataFrame()

    rows = []
    for day, stats in sorted(daily_records.items()):
        rows.append({
            "date":                  day,
            "reddit_post_count":     stats["post_count"],
            "reddit_avg_score":      stats["total_score"] / max(stats["post_count"], 1),
            "reddit_avg_comments":   stats["total_comments"] / max(stats["post_count"], 1),
            "reddit_sentiment_compound": np.mean(stats["sentiment_compound"]) if stats["sentiment_compound"] else np.nan,
            "reddit_sentiment_positive": np.mean(stats["sentiment_positive"]) if stats["sentiment_positive"] else np.nan,
            "reddit_sentiment_negative": np.mean(stats["sentiment_negative"]) if stats["sentiment_negative"] else np.nan,
        })

    df = pd.DataFrame(rows).sort_values("date")
    print(f"  ✅ {len(df)} days  |  {df['date'].min()} → {df['date'].max()}")
    return df


# ============================================================================
# SOURCE 8 — STOCKTWITS (BTC.X Bullish/Bearish Ratio)
# ============================================================================

def collect_stocktwits():
    """
    Fetches current BTC.X stream from Stocktwits for sentiment.
    Free public endpoint — no key required for stream.
    Note: Historical data via public API is very limited (recent only).
    Collects current snapshot for ongoing data collection going forward.
    """
    section("SOURCE 8: Stocktwits — BTC.X Sentiment Snapshot")

    url = "https://api.stocktwits.com/api/2/streams/symbol/BTC.X.json"
    r   = safe_get(url)

    if r is None:
        print("  ❌ Failed to fetch Stocktwits data")
        return pd.DataFrame()

    try:
        data     = r.json()
        messages = data.get("messages", [])

        if not messages:
            print("  ⚠️  No messages returned")
            return pd.DataFrame()

        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()

        daily_records = {}

        for msg in messages:
            created_at = msg.get("created_at", "")
            body       = msg.get("body", "")
            entities   = msg.get("entities", {})
            sentiment  = entities.get("sentiment", {})
            label      = sentiment.get("basic", None) if sentiment else None  # "Bullish" or "Bearish"

            try:
                msg_date = datetime.strptime(created_at[:10], "%Y-%m-%d").date()
            except Exception:
                continue

            vader_score = analyzer.polarity_scores(body)["compound"]

            if msg_date not in daily_records:
                daily_records[msg_date] = {
                    "bullish": 0, "bearish": 0, "neutral": 0,
                    "vader_scores": []
                }

            if label == "Bullish":
                daily_records[msg_date]["bullish"] += 1
            elif label == "Bearish":
                daily_records[msg_date]["bearish"] += 1
            else:
                daily_records[msg_date]["neutral"] += 1

            daily_records[msg_date]["vader_scores"].append(vader_score)

        rows = []
        for day, stats in sorted(daily_records.items()):
            total = stats["bullish"] + stats["bearish"]
            rows.append({
                "date":                      day,
                "stocktwits_bullish_count":  stats["bullish"],
                "stocktwits_bearish_count":  stats["bearish"],
                "stocktwits_bull_bear_ratio": stats["bullish"] / max(total, 1),
                "stocktwits_vader_compound": np.mean(stats["vader_scores"]) if stats["vader_scores"] else np.nan,
            })

        df = pd.DataFrame(rows).sort_values("date")
        print(f"  ✅ {len(df)} days (recent snapshot)  |  {df['date'].min()} → {df['date'].max()}")
        print(f"     Note: Stocktwits historical data limited — use for real-time collection going forward")
        return df

    except Exception as e:
        print(f"  ❌ Error parsing Stocktwits: {e}")
        return pd.DataFrame()


# ============================================================================
# SOURCE 9 — GITHUB BITCOIN CORE ACTIVITY
# ============================================================================

def collect_github_activity():
    """
    Fetches Bitcoin Core GitHub repository activity.
    Includes commit frequency, contributor count, stars, forks.
    Free: 60 req/hr unauthenticated, 5000/hr with token.
    """
    section("SOURCE 9: GitHub — Bitcoin Core Repository Activity")

    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
        print("  🔑 Using GitHub token (5000 req/hr)")
    else:
        print("  ℹ️  No GitHub token — using unauthenticated (60 req/hr)")

    # --- 9a: Weekly commit activity (last 52 weeks) ---
    print("  🔍 Fetching commit activity...")
    r = safe_get(
        "https://api.github.com/repos/bitcoin/bitcoin/stats/commit_activity",
        headers=headers
    )

    commit_df = pd.DataFrame()
    if r:
        try:
            weeks = r.json()
            rows  = []
            for week in weeks:
                week_start = datetime.fromtimestamp(week["week"]).date()
                daily_commits = week.get("days", [0]*7)
                for i, count in enumerate(daily_commits):
                    day = week_start + timedelta(days=i)
                    rows.append({"date": day, "github_commits": count})
            commit_df = pd.DataFrame(rows)
            print(f"    ✅ {len(commit_df)} days of commit data")
        except Exception as e:
            print(f"    ⚠️  Commit parse error: {e}")
    time.sleep(RATE_GITHUB)

    # --- 9b: Contributor count over time ---
    print("  🔍 Fetching contributor stats...")
    r = safe_get(
        "https://api.github.com/repos/bitcoin/bitcoin/stats/contributors",
        headers=headers
    )

    contrib_df = pd.DataFrame()
    if r and r.status_code == 200:
        try:
            contributors = r.json()
            weekly_contribs = {}
            for contributor in contributors:
                for week in contributor.get("weeks", []):
                    w_ts = week.get("w", 0)
                    if week.get("c", 0) > 0:  # only weeks they committed
                        week_date = datetime.fromtimestamp(w_ts).date()
                        weekly_contribs[week_date] = weekly_contribs.get(week_date, 0) + 1

            rows = [{"date": d, "github_active_contributors": c}
                    for d, c in sorted(weekly_contribs.items())]
            contrib_df = pd.DataFrame(rows)
            print(f"    ✅ {len(contrib_df)} weeks of contributor data")
        except Exception as e:
            print(f"    ⚠️  Contributor parse error: {e}")
    time.sleep(RATE_GITHUB)

    # --- 9c: Repo metadata (stars, forks, watchers) ---
    print("  🔍 Fetching repo metadata...")
    r = safe_get("https://api.github.com/repos/bitcoin/bitcoin", headers=headers)
    repo_meta = {}
    if r:
        try:
            data             = r.json()
            repo_meta["github_stars"]    = data.get("stargazers_count", np.nan)
            repo_meta["github_forks"]    = data.get("forks_count",      np.nan)
            repo_meta["github_watchers"] = data.get("watchers_count",   np.nan)
            repo_meta["github_open_issues"] = data.get("open_issues_count", np.nan)
            print(f"    ✅ Stars: {repo_meta['github_stars']:,}  |  Forks: {repo_meta['github_forks']:,}")
        except Exception as e:
            print(f"    ⚠️  Metadata parse error: {e}")

    # Merge commit and contributor data
    if commit_df.empty and contrib_df.empty:
        print("  ❌ No GitHub data collected")
        return pd.DataFrame()

    if not commit_df.empty and not contrib_df.empty:
        # Contributors are weekly — expand to daily by forward-fill
        contrib_df["date"] = pd.to_datetime(contrib_df["date"])
        date_index = pd.DataFrame(
            {"date": pd.date_range(contrib_df["date"].min(), datetime.today())}
        )
        contrib_daily = date_index.merge(contrib_df, on="date", how="left").ffill()
        contrib_daily["date"] = contrib_daily["date"].dt.date
        df = commit_df.merge(contrib_daily, on="date", how="outer")
    elif not commit_df.empty:
        df = commit_df
    else:
        df = contrib_df

    # Append today's snapshot metadata as a constant (best we can do without historical API)
    for col, val in repo_meta.items():
        df[col] = val

    df = df.sort_values("date")
    print(f"  ✅ {len(df)} days  |  {df['date'].min()} → {df['date'].max()}")
    return df


# ============================================================================
# FEATURE ENGINEERING ON SENTIMENT DATA
# ============================================================================

def engineer_sentiment_features(df):
    """
    Derives additional features from raw sentiment data.
    """
    section("FEATURE ENGINEERING")

    df = df.sort_values("date").reset_index(drop=True)

    # Fear & Greed momentum
    if "fear_greed_value" in df.columns:
        df["fear_greed_ma7"]     = df["fear_greed_value"].rolling(7).mean()
        df["fear_greed_ma14"]    = df["fear_greed_value"].rolling(14).mean()
        df["fear_greed_change"]  = df["fear_greed_value"].diff()
        df["fear_greed_momentum"] = df["fear_greed_ma7"] - df["fear_greed_ma14"]
        print("  ✅ Fear & Greed: MA7, MA14, momentum")

    # Wikipedia attention spike detection
    if "wikipedia_views" in df.columns:
        df["wiki_views_ma7"]     = df["wikipedia_views"].rolling(7).mean()
        df["wiki_views_zscore"]  = (
            df["wikipedia_views"] - df["wikipedia_views"].rolling(30).mean()
        ) / (df["wikipedia_views"].rolling(30).std() + 1e-9)
        print("  ✅ Wikipedia: MA7, Z-score (attention spikes)")

    # Google Trends momentum
    if "trends_bitcoin" in df.columns:
        df["trends_bitcoin_ma7"]    = df["trends_bitcoin"].rolling(7).mean()
        df["trends_bitcoin_zscore"] = (
            df["trends_bitcoin"] - df["trends_bitcoin"].rolling(30).mean()
        ) / (df["trends_bitcoin"].rolling(30).std() + 1e-9)
        print("  ✅ Google Trends: MA7, Z-score")

    # Reddit engagement ratio
    if "reddit_post_count" in df.columns:
        df["reddit_post_count_ma7"]  = df["reddit_post_count"].rolling(7).mean()
        df["reddit_engagement_spike"] = (
            df["reddit_post_count"] / (df["reddit_post_count"].rolling(14).mean() + 1e-9)
        )
        print("  ✅ Reddit: post count MA7, engagement spike ratio")

    # GDELT sentiment change
    if "gdelt_avg_tone" in df.columns:
        df["gdelt_tone_ma7"]    = df["gdelt_avg_tone"].rolling(7).mean()
        df["gdelt_tone_change"] = df["gdelt_avg_tone"].diff()
        print("  ✅ GDELT: tone MA7, change")

    # BTC dominance momentum
    if "btc_dominance_pct" in df.columns:
        df["btc_dominance_ma7"]    = df["btc_dominance_pct"].rolling(7).mean()
        df["btc_dominance_change"] = df["btc_dominance_pct"].diff()
        print("  ✅ BTC dominance: MA7, change")

    # Composite social volume (normalised)
    social_volume_cols = [c for c in [
        "trends_bitcoin", "wikipedia_views",
        "reddit_post_count", "gdelt_news_volume",
    ] if c in df.columns]

    if social_volume_cols:
        normalised = pd.DataFrame()
        for col in social_volume_cols:
            col_max = df[col].max()
            normalised[col] = df[col] / col_max if col_max > 0 else df[col]
        df["composite_social_volume"] = normalised.mean(axis=1)
        print(f"  ✅ Composite social volume ({len(social_volume_cols)} sources)")

    print(f"\n  ✅ Engineering complete: {len(df.columns)} total columns")
    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    section("BITCOIN DAILY SENTIMENT DATA COLLECTOR")
    print(f"  Date range : {START_DATE} → {END_DATE}")
    print(f"  Output     : {OUTPUT_CSV}")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    datasets = {}

    # Collect each source — continue even if individual sources fail
    datasets["fear_greed"]    = collect_fear_greed()
    datasets["google_trends"] = collect_google_trends()
    datasets["wikipedia"]     = collect_wikipedia_pageviews()
    datasets["cryptocompare"] = collect_cryptocompare_social()
    datasets["gdelt"]         = collect_gdelt()
    datasets["coingecko"]     = collect_coingecko()
    datasets["reddit"]        = collect_reddit_sentiment()
    datasets["stocktwits"]    = collect_stocktwits()
    datasets["github"]        = collect_github_activity()

    # --- Merge all datasets ---
    section("MERGING ALL DATASETS")

    successful = {k: v for k, v in datasets.items() if not v.empty}
    failed     = [k for k, v in datasets.items() if v.empty]

    if failed:
        print(f"  ⚠️  Sources with no data: {failed}")

    if not successful:
        print("  ❌ CRITICAL: No data collected from any source. Exiting.")
        return

    print(f"  ✅ Successful sources: {list(successful.keys())}")

    # Start from the first successful dataset
    sources = list(successful.values())
    merged  = sources[0]
    for df in sources[1:]:
        merged = pd.merge(merged, df, on="date", how="outer")

    # Filter to date range and sort
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged[(merged["date"] >= START_DATE) & (merged["date"] <= END_DATE)]
    merged["date"] = merged["date"].dt.date
    merged = merged.sort_values("date").reset_index(drop=True)

    # Engineer features
    merged = engineer_sentiment_features(merged)

    # Save
    merged.to_csv(OUTPUT_CSV, index=False)

    # --- Final summary ---
    section("COLLECTION COMPLETE — SUMMARY")
    print(f"  Total records : {len(merged):,}")
    print(f"  Date range    : {merged['date'].min()} → {merged['date'].max()}")
    print(f"  Total features: {len(merged.columns)}")

    print(f"\n  📊 Feature Completeness Report:")
    print(f"  {'-'*70}")
    for col in merged.columns:
        if col != "date":
            nn  = merged[col].notna().sum()
            pct = (nn / len(merged)) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {col:45s} {bar} {pct:5.1f}%")

    print(f"\n  📁 Saved to: {OUTPUT_CSV}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()