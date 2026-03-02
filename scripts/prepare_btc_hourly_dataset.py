"""
prepare_btc_hourly_dataset.py

Engineers the 14 Boruta-confirmed features for BTCTradingEnv from the three
existing data files produced by the ServoTrader collection pipeline.

Source files (all already on disk)
------------------------------------
1. OHLCV (timezone-aware UTC):
     /home/jarred/git/ServoTrader/data/hourly_historical/BTCUSDT.csv
     Columns: timestamp, open, high, low, close, vwap, volume, count, symbol

2. Sentiment (timezone-naive, daily forward-filled to hourly):
     /home/jarred/git/ServoTrader/data/btc_sentiment_hourly.csv
     Relevant column: trends_bitcoin_zscore  (already z-scored — no extra work needed)

3. On-chain (timezone-naive, daily forward-filled to hourly):
     /home/jarred/git/ServoTrader/data/btc_onchain_hourly_metrics.csv
     Not required for the 14 confirmed features. Loaded only so future
     extensions can add on-chain columns without restructuring this script.

Output
------
     /home/jarred/git/ServoTrader/data/btc_hourly_features.csv
     Contains OHLCV base columns + 14 Boruta-confirmed features.
     NaN warm-up rows (from rolling windows) are dropped.

14 Confirmed Boruta Features (in importance order)
----------------------------------------------------
 1  rsi_24               — RSI(24) on close              [technical]
 2  adx_14               — ADX(14) trend strength         [technical]
 3  stoch_k              — Stochastic %K (14, 3)          [technical]
 4  stoch_d              — Stochastic %D (3-period SMA)   [technical]
 5  bb_percent_b_48      — Bollinger %B, window=48        [technical]
 6  bb_percent_b_24      — Bollinger %B, window=24        [technical]
 7  volume_ratio_24      — volume / rolling_mean(24)      [volume]
 8  vwap_deviation       — (close − vwap) / |vwap|       [other]
 9  returns_6h           — 6-period log return            [momentum]
10  bb_percent_b_12      — Bollinger %B, window=12        [technical]
11  returns_1h           — 1-period log return            [momentum]
12  volatility_6h        — rolling std of 1h log returns  [volatility]
13  rsi_6                — RSI(6) on close                [technical]
14  trends_bitcoin_zscore— Google Trends z-score          [sentiment]
     (already computed and present in btc_sentiment_hourly.csv)

Timestamp Normalisation
-----------------------
The OHLCV file has UTC-aware timestamps (datetime64[ns, UTC]).
The sentiment and on-chain files have timezone-naive timestamps.
This script strips timezone info from OHLCV timestamps before merging to
avoid the ValueError: "merging on datetime64[ns, UTC] and datetime64[ns]".

Usage
-----
    python prepare_btc_hourly_dataset.py

Author: Jarred Deluca
Project: ServoTrader
License: MIT
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------

OHLCV_PATH     = "/home/jarred/git/ServoTrader/data/hourly_historical/BTCUSDT.csv"
SENTIMENT_PATH = "/home/jarred/git/ServoTrader/data/btc_sentiment_hourly.csv"
ONCHAIN_PATH   = "/home/jarred/git/ServoTrader/data/btc_onchain_hourly_metrics.csv"
OUTPUT_PATH    = "/home/jarred/git/ServoTrader/data/btc_hourly_features.csv"

# Boruta-confirmed feature columns (must match BORUTA_FEATURES in btc_trading_env.py)
FEATURE_COLS = [
    "rsi_24",
    "adx_14",
    "stoch_k",
    "stoch_d",
    "bb_percent_b_48",
    "bb_percent_b_24",
    "volume_ratio_24",
    "vwap_deviation",
    "returns_6h",
    "bb_percent_b_12",
    "returns_1h",
    "volatility_6h",
    "rsi_6",
    "trends_bitcoin_zscore",
]


# ---------------------------------------------------------------------------
#  Timestamp helper
# ---------------------------------------------------------------------------

def _normalise_ts(series: pd.Series) -> pd.Series:
    """
    Parse a timestamp series and return timezone-naive UTC datetime64[ns].

    Converts both tz-aware and tz-naive inputs to a consistent naive dtype
    so that pd.merge(on='timestamp') works without a dtype mismatch error.

    Strategy:
      1. Parse with utc=True  → everything becomes UTC-aware
      2. Strip tzinfo          → becomes naive (still represents the same UTC moment)
    """
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None)


# ---------------------------------------------------------------------------
#  Technical indicator helpers
# ---------------------------------------------------------------------------

def _wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder-smoothed RSI (EWM alpha = 1/period, matches Boruta training run)."""
    d        = close.diff()
    gain     = d.clip(lower=0)
    loss     = (-d).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index using Wilder smoothing (alpha = 1/period)."""
    ph, pl, pc = high.shift(1), low.shift(1), close.shift(1)

    tr = pd.concat([
        (high - low),
        (high - pc).abs(),
        (low  - pc).abs(),
    ], axis=1).max(axis=1)

    dm_p = (high - ph).clip(lower=0)
    dm_m = (pl - low).clip(lower=0)
    # Where both DMs are positive keep the larger, zero out the smaller
    overlap = (dm_p > 0) & (dm_m > 0)
    dm_p    = dm_p.where(~overlap | (dm_p >= dm_m), 0.0)
    dm_m    = dm_m.where(~overlap | (dm_m > dm_p),  0.0)

    a    = 1.0 / period
    atr  = tr.ewm(alpha=a, min_periods=period, adjust=False).mean()
    di_p = 100.0 * dm_p.ewm(alpha=a, min_periods=period, adjust=False).mean() / (atr + 1e-12)
    di_m = 100.0 * dm_m.ewm(alpha=a, min_periods=period, adjust=False).mean() / (atr + 1e-12)
    dx   = 100.0 * (di_p - di_m).abs() / (di_p + di_m + 1e-12)
    return dx.ewm(alpha=a, min_periods=period, adjust=False).mean()


def _stochastic(high, low, close, k_period=14, d_period=3):
    """Stochastic %K and %D (SMA of %K over d_period)."""
    lo = low.rolling(k_period, min_periods=k_period).min()
    hi = high.rolling(k_period, min_periods=k_period).max()
    k  = 100.0 * (close - lo) / (hi - lo + 1e-12)
    d  = k.rolling(d_period, min_periods=d_period).mean()
    return k, d


def _bb_percent_b(close: pd.Series, window: int, n_std: float = 2.0) -> pd.Series:
    """Bollinger Band %B = (close − lower) / (upper − lower)."""
    ma    = close.rolling(window, min_periods=window).mean()
    sigma = close.rolling(window, min_periods=window).std()
    upper = ma + n_std * sigma
    lower = ma - n_std * sigma
    return (close - lower) / (upper - lower + 1e-12)


def _volume_ratio(volume: pd.Series, window: int = 24) -> pd.Series:
    """volume / rolling_mean_volume(window)."""
    return volume / (volume.rolling(window, min_periods=window).mean() + 1e-12)


def _vwap_deviation(close: pd.Series, vwap: pd.Series) -> pd.Series:
    """(close − vwap) / |vwap|."""
    return (close - vwap) / (vwap.abs() + 1e-12)


def _log_return(close: pd.Series, periods: int = 1) -> pd.Series:
    """Log return over `periods` candles."""
    return np.log(close / (close.shift(periods) + 1e-12))


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def prepare_dataset(
    ohlcv_path:     str  = OHLCV_PATH,
    sentiment_path: str  = SENTIMENT_PATH,
    onchain_path:   str  = ONCHAIN_PATH,
    output_path:    str  = OUTPUT_PATH,
    verbose:        bool = True,
) -> pd.DataFrame:
    """
    Load source CSVs, engineer 14 features, and save the training-ready dataset.
    Returns the final DataFrame.
    """

    sep = "=" * 70

    # ── 1. OHLCV ─────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n{sep}")
        print(f"  LOADING OHLCV")
        print(f"  {ohlcv_path}")
        print(sep)

    if not os.path.exists(ohlcv_path):
        raise FileNotFoundError(f"OHLCV file not found: {ohlcv_path}")

    ohlcv = pd.read_csv(ohlcv_path)
    ohlcv.columns = [c.lower().strip() for c in ohlcv.columns]

    # --- Normalise timestamp to tz-naive ---
    ohlcv["timestamp"] = _normalise_ts(ohlcv["timestamp"])
    ohlcv = ohlcv.sort_values("timestamp").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        ohlcv[col] = pd.to_numeric(ohlcv[col], errors="coerce")

    # VWAP — prefer existing column; approximate if absent/bad
    if "vwap" in ohlcv.columns:
        ohlcv["vwap"] = pd.to_numeric(ohlcv["vwap"], errors="coerce")
        bad_frac = ohlcv["vwap"].isna().mean()
        if bad_frac > 0.5:
            if verbose:
                print(f"  ⚠  vwap column is {bad_frac:.0%} NaN → approximating as (H+L+C)/3")
            ohlcv["vwap"] = (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3.0
    else:
        if verbose:
            print("  ⚠  vwap column missing → approximating as (H+L+C)/3")
        ohlcv["vwap"] = (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3.0

    for col in ["open", "high", "low", "close", "vwap"]:
        ohlcv[col] = ohlcv[col].ffill().bfill()
    ohlcv["volume"] = ohlcv["volume"].fillna(0.0)
    ohlcv = ohlcv.drop(columns=["symbol", "count"], errors="ignore")

    if verbose:
        print(f"  ✅  {len(ohlcv):,} rows | "
              f"{ohlcv['timestamp'].min()} → {ohlcv['timestamp'].max()}")

    # ── 2. Sentiment — extract trends_bitcoin_zscore ─────────────────────────
    if verbose:
        print(f"\n{sep}")
        print(f"  LOADING SENTIMENT (trends_bitcoin_zscore)")
        print(f"  {sentiment_path}")
        print(sep)

    sentiment_col = None
    if os.path.exists(sentiment_path):
        sent = pd.read_csv(sentiment_path)
        sent.columns = [c.lower().strip() for c in sent.columns]

        if "trends_bitcoin_zscore" in sent.columns:
            sent["timestamp"] = _normalise_ts(sent["timestamp"])
            sent = (
                sent[["timestamp", "trends_bitcoin_zscore"]]
                .copy()
                .sort_values("timestamp")
                .drop_duplicates("timestamp")
            )
            sent["trends_bitcoin_zscore"] = (
                pd.to_numeric(sent["trends_bitcoin_zscore"], errors="coerce")
                .ffill()
                .bfill()
                .fillna(0.0)
            )
            sentiment_col = sent
            if verbose:
                print(f"  ✅  {len(sent):,} rows | trends_bitcoin_zscore found and loaded")
        else:
            if verbose:
                print(f"  ⚠  trends_bitcoin_zscore column not in sentiment CSV — filling zeros")
                print(f"      Available columns: {list(sent.columns[:10])} …")
    else:
        if verbose:
            print(f"  ⚠  Sentiment file not found — filling zeros")

    # ── 3. Merge OHLCV + sentiment ────────────────────────────────────────────
    # All timestamps are now timezone-naive → merge is safe
    if sentiment_col is not None:
        df = ohlcv.merge(sentiment_col, on="timestamp", how="left")
        # Sentiment is daily so many hours won't match directly; forward-fill
        df["trends_bitcoin_zscore"] = (
            df["trends_bitcoin_zscore"].ffill().bfill().fillna(0.0)
        )
    else:
        df = ohlcv.copy()
        df["trends_bitcoin_zscore"] = 0.0

    if verbose:
        print(f"\n  After merge: {len(df):,} rows")

    # ── 4. Engineer the 13 OHLCV-based features ──────────────────────────────
    if verbose:
        print(f"\n{sep}")
        print("  ENGINEERING FEATURES")
        print(sep)

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]
    vwap   = df["vwap"]

    df["rsi_24"]          = _wilder_rsi(close, period=24)
    df["rsi_6"]           = _wilder_rsi(close, period=6)
    df["adx_14"]          = _adx(high, low, close, period=14)
    df["stoch_k"], \
    df["stoch_d"]         = _stochastic(high, low, close, k_period=14, d_period=3)
    df["bb_percent_b_12"] = _bb_percent_b(close, window=12)
    df["bb_percent_b_24"] = _bb_percent_b(close, window=24)
    df["bb_percent_b_48"] = _bb_percent_b(close, window=48)
    df["volume_ratio_24"] = _volume_ratio(volume, window=24)
    df["vwap_deviation"]  = _vwap_deviation(close, vwap)
    log_ret_1h            = _log_return(close, periods=1)
    df["returns_1h"]      = log_ret_1h
    df["returns_6h"]      = _log_return(close, periods=6)
    df["volatility_6h"]   = log_ret_1h.rolling(6, min_periods=6).std()

    if verbose:
        print("  ✅  All 14 features computed")

    # ── 5. Drop NaN warm-up rows ──────────────────────────────────────────────
    before = len(df)
    df     = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    if verbose:
        print(f"  Dropped {before - len(df):,} warm-up rows | Remaining: {len(df):,}")

    # ── 6. Sanitise residual Inf/NaN ─────────────────────────────────────────
    for col in FEATURE_COLS:
        n_inf = np.isinf(df[col]).sum()
        n_nan = df[col].isna().sum()
        if n_inf > 0 or n_nan > 0:
            warnings.warn(
                f"[DataPrep] {col}: {n_inf} Inf, {n_nan} NaN remain — replacing with 0"
            )
            df[col] = df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # ── 7. Save ───────────────────────────────────────────────────────────────
    keep  = ["timestamp", "open", "high", "low", "close", "vwap", "volume"] + FEATURE_COLS
    keep  = [c for c in keep if c in df.columns]
    out   = df[keep].copy()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)

    # ── 8. Summary ────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n{sep}")
        print(f"  ✅  COMPLETE")
        print(f"  Saved {len(out):,} rows → {output_path}")
        print(f"  Date range: {out['timestamp'].min()} → {out['timestamp'].max()}")
        print(sep)
        print("\n  Feature statistics:")
        print(out[FEATURE_COLS].describe().round(4).to_string())
        print(f"\n{sep}")
        print("  Value range check (flag if |value| > 1000):")
        for col in FEATURE_COLS:
            mn = out[col].min()
            mx = out[col].max()
            ok = abs(mn) < 1000 and abs(mx) < 1000
            flag = "✅" if ok else "⚠ "
            print(f"    {flag}  {col:<25}  min={mn:>10.4f}   max={mx:>10.4f}")
        print(sep + "\n")

    return out


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prepare_dataset()