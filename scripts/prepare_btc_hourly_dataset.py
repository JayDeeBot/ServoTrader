"""
prepare_btc_hourly_dataset.py

Takes raw hourly BTC OHLCV data (from Binance via the existing pipeline) and
engineers the 14 Boruta-confirmed features required by BTCTradingEnv.

Confirmed Features (in order)
------------------------------
1.  rsi_24               – RSI(24) on close
2.  adx_14               – ADX(14) — trend strength
3.  stoch_k              – Stochastic %K (14,3)
4.  stoch_d              – Stochastic %D (3-period SMA of %K)
5.  bb_percent_b_48      – Bollinger %B with window=48
6.  bb_percent_b_24      – Bollinger %B with window=24
7.  volume_ratio_24      – volume / rolling_mean_volume(24)
8.  vwap_deviation       – (close − vwap) / vwap
9.  returns_6h           – 6-period log return
10. bb_percent_b_12      – Bollinger %B with window=12
11. returns_1h           – 1-period log return
12. volatility_6h        – 6-period rolling std of log returns
13. rsi_6                – RSI(6) on close
14. trends_bitcoin_zscore– Google Trends index, z-scored (forward-filled daily → hourly)

Input Data
----------
Expects a CSV with at minimum: timestamp, open, high, low, close, volume, [vwap]
If vwap is missing it is computed from (high + low + close) / 3 as an approximation.
If trends_bitcoin_zscore is missing it is filled with zeros (neutral) with a warning.

Output
------
CSV saved to `output_path` containing the original OHLCV columns PLUS all 14
engineered features plus 'close' (for trade execution). Rows with NaN features
(warm-up period) are dropped.

Usage
-----
  python prepare_btc_hourly_dataset.py

  or import and call:
    from prepare_btc_hourly_dataset import prepare_dataset
    df = prepare_dataset(input_path, trends_path, output_path)

Author: Jarred Deluca
Project: ServoTrader
License: MIT
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
#  Configuration — adjust paths as needed
# ---------------------------------------------------------------------------

# Raw hourly BTC OHLCV data (from Binance download scripts)
INPUT_PATH  = "/home/jarred/git/ServoTrader/data/BTCUSDT_1h.csv"

# Optional: daily Google Trends CSV with columns ['date', 'bitcoin_trend']
# If not available, leave as None — the feature will be filled with zeros.
TRENDS_PATH = "/home/jarred/git/ServoTrader/data/bitcoin_google_trends.csv"

# Output path for the feature-engineered dataset used by BTCTradingEnv
OUTPUT_PATH = "/home/jarred/git/ServoTrader/data/btc_hourly_features.csv"


# ---------------------------------------------------------------------------
#  Indicator helpers
# ---------------------------------------------------------------------------

def _wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Wilder-smoothed RSI.  Using EWM with alpha=1/period matches the original
    Wilder definition and the computation used in the Boruta training run.
    """
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average Directional Index (ADX).
    Uses Wilder smoothing (EWM alpha=1/period).
    """
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    dm_plus  = (high - prev_high).clip(lower=0)
    dm_minus = (prev_low - low).clip(lower=0)
    # Zero out whichever directional move is smaller
    dm_plus  = dm_plus.where(dm_plus >= dm_minus, 0.0)
    dm_minus = dm_minus.where(dm_minus > dm_plus,  0.0)

    alpha = 1.0 / period
    atr   = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    di_p  = 100.0 * dm_plus.ewm( alpha=alpha, min_periods=period, adjust=False).mean() / (atr + 1e-12)
    di_m  = 100.0 * dm_minus.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / (atr + 1e-12)

    dx    = 100.0 * (di_p - di_m).abs() / (di_p + di_m + 1e-12)
    adx   = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return adx


def _stochastic(
    high:  pd.Series,
    low:   pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic oscillator %K and %D."""
    lowest  = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k       = 100.0 * (close - lowest) / (highest - lowest + 1e-12)
    d       = k.rolling(d_period).mean()
    return k, d


def _bollinger_percent_b(close: pd.Series, window: int, n_std: float = 2.0) -> pd.Series:
    """
    Bollinger Band %B.
    %B = (close − lower) / (upper − lower)
    where upper/lower = MA ± n_std * σ
    """
    ma    = close.rolling(window).mean()
    sigma = close.rolling(window).std()
    upper = ma + n_std * sigma
    lower = ma - n_std * sigma
    return (close - lower) / (upper - lower + 1e-12)


def _volume_ratio(volume: pd.Series, window: int = 24) -> pd.Series:
    """Current volume relative to its rolling mean."""
    return volume / (volume.rolling(window).mean() + 1e-12)


def _vwap_deviation(close: pd.Series, vwap: pd.Series) -> pd.Series:
    """Normalised deviation of close from VWAP."""
    return (close - vwap) / (vwap.abs() + 1e-12)


def _log_return(close: pd.Series, periods: int = 1) -> pd.Series:
    """Log return over `periods` steps."""
    return np.log(close / close.shift(periods))


def _rolling_volatility(log_ret: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation of log returns."""
    return log_ret.rolling(window).std()


# ---------------------------------------------------------------------------
#  Main preparation function
# ---------------------------------------------------------------------------

def prepare_dataset(
    input_path:  str = INPUT_PATH,
    trends_path: str | None = TRENDS_PATH,
    output_path: str = OUTPUT_PATH,
    verbose:     bool = True,
) -> pd.DataFrame:
    """
    Loads raw OHLCV data, engineers all 14 Boruta features, and writes
    the result to `output_path`.

    Parameters
    ----------
    input_path  : path to raw hourly OHLCV CSV
    trends_path : path to daily Google Trends CSV (or None to skip)
    output_path : destination for the feature-engineered CSV
    verbose     : print progress messages

    Returns
    -------
    pd.DataFrame with OHLCV columns + 14 features, NaN warm-up rows dropped.
    """

    # ── Load raw data ────────────────────────────────────────────────────────
    if verbose:
        print(f"[DataPrep] Loading {input_path}…")

    df = pd.read_csv(input_path)

    # Normalise column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]

    # Ensure timestamp is parsed and sorted chronologically
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
    elif "open_time" in df.columns:
        # Binance raw format
        df.rename(columns={"open_time": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Required base columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"[DataPrep] Required column '{col}' not found in {input_path}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # VWAP — use existing or approximate
    if "vwap" not in df.columns or df["vwap"].isna().all():
        if verbose:
            print("[DataPrep] VWAP not found — approximating as (H+L+C)/3")
        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0
    else:
        df["vwap"] = pd.to_numeric(df["vwap"], errors="coerce")

    # Forward-fill prices for any gaps
    for col in ["open", "high", "low", "close", "vwap"]:
        df[col] = df[col].ffill().bfill()
    df["volume"] = df["volume"].fillna(0.0)

    if verbose:
        print(f"[DataPrep] Rows after load: {len(df):,}")

    # ── Engineer features ────────────────────────────────────────────────────
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]
    vwap   = df["vwap"]

    # 1.  RSI(24)
    df["rsi_24"]          = _wilder_rsi(close, period=24)

    # 2.  ADX(14)
    df["adx_14"]          = _adx(high, low, close, period=14)

    # 3 & 4.  Stochastic %K / %D
    df["stoch_k"], df["stoch_d"] = _stochastic(high, low, close, k_period=14, d_period=3)

    # 5 & 6 & 10.  Bollinger %B at three windows
    df["bb_percent_b_48"] = _bollinger_percent_b(close, window=48)
    df["bb_percent_b_24"] = _bollinger_percent_b(close, window=24)
    df["bb_percent_b_12"] = _bollinger_percent_b(close, window=12)

    # 7.  Volume ratio(24)
    df["volume_ratio_24"] = _volume_ratio(volume, window=24)

    # 8.  VWAP deviation
    df["vwap_deviation"]  = _vwap_deviation(close, vwap)

    # 9 & 11.  Log returns
    log_ret_1h            = _log_return(close, periods=1)
    df["returns_1h"]      = log_ret_1h
    df["returns_6h"]      = _log_return(close, periods=6)

    # 12.  6h volatility (std of hourly log returns over 6 periods)
    df["volatility_6h"]   = _rolling_volatility(log_ret_1h, window=6)

    # 13.  RSI(6)
    df["rsi_6"]           = _wilder_rsi(close, period=6)

    # 14.  Google Trends z-score (daily → forward-filled to hourly)
    df["trends_bitcoin_zscore"] = _attach_trends(df, trends_path, verbose)

    # ── Drop warm-up NaN rows ────────────────────────────────────────────────
    feature_cols = [
        "rsi_24", "adx_14", "stoch_k", "stoch_d",
        "bb_percent_b_48", "bb_percent_b_24", "volume_ratio_24",
        "vwap_deviation", "returns_6h", "bb_percent_b_12",
        "returns_1h", "volatility_6h", "rsi_6", "trends_bitcoin_zscore",
    ]
    before = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    if verbose:
        print(f"[DataPrep] Rows after dropping warm-up NaNs: {len(df):,} (dropped {before - len(df):,})")

    # ── Sanity checks ────────────────────────────────────────────────────────
    for col in feature_cols:
        n_inf = np.isinf(df[col]).sum()
        n_nan = df[col].isna().sum()
        if n_inf > 0 or n_nan > 0:
            warnings.warn(f"[DataPrep] {col}: {n_inf} Infs, {n_nan} NaNs remain after clean — filling with 0")
            df[col] = df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # ── Save ─────────────────────────────────────────────────────────────────
    keep_cols = ["timestamp", "open", "high", "low", "close", "vwap", "volume"] + feature_cols
    keep_cols = [c for c in keep_cols if c in df.columns]
    out_df    = df[keep_cols]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_df.to_csv(output_path, index=False)

    if verbose:
        print(f"[DataPrep] ✅ Saved {len(out_df):,} rows → {output_path}")
        print(f"[DataPrep] Columns: {list(out_df.columns)}")
        print("\n[DataPrep] Feature statistics:")
        print(out_df[feature_cols].describe().round(4).to_string())

    return out_df


def _attach_trends(df: pd.DataFrame, trends_path: str | None, verbose: bool) -> pd.Series:
    """
    Loads Google Trends data, z-scores it, and forward-fills to hourly frequency.
    Returns a pd.Series aligned to df's index. If trends_path is None or the file
    does not exist, returns zeros with a warning.
    """
    if trends_path is None or not os.path.exists(trends_path):
        if verbose:
            warnings.warn(
                "[DataPrep] trends_bitcoin_zscore: Google Trends file not found. "
                "Filling with zeros (neutral). This will reduce model signal quality. "
                f"Expected path: {trends_path}"
            )
        return pd.Series(0.0, index=df.index)

    trends = pd.read_csv(trends_path)
    trends.columns = [c.lower().strip() for c in trends.columns]

    # Accept common column name variants
    date_col  = next((c for c in trends.columns if "date" in c),  None)
    val_col   = next((c for c in trends.columns if "trend" in c or "bitcoin" in c or "value" in c), None)

    if date_col is None or val_col is None:
        warnings.warn(
            f"[DataPrep] Trends CSV must have a date column and a value column. "
            f"Found: {list(trends.columns)}. Filling with zeros."
        )
        return pd.Series(0.0, index=df.index)

    trends["date"]  = pd.to_datetime(trends[date_col], utc=True, errors="coerce")
    trends["value"] = pd.to_numeric(trends[val_col], errors="coerce")
    trends = trends[["date", "value"]].dropna().sort_values("date").set_index("date")

    # Z-score the raw trend index
    mu  = trends["value"].mean()
    sig = trends["value"].std() + 1e-8
    trends["zscore"] = (trends["value"] - mu) / sig

    # Reindex to hourly timestamps in df and forward-fill
    if "timestamp" not in df.columns:
        warnings.warn("[DataPrep] No 'timestamp' column in OHLCV data — cannot align Trends. Filling zeros.")
        return pd.Series(0.0, index=df.index)

    ts_index    = df["timestamp"].dt.normalize()   # floor to day for join
    merged      = ts_index.map(trends["zscore"].to_dict())
    merged      = merged.ffill().bfill().fillna(0.0)

    if verbose:
        n_matched = merged.notna().sum()
        print(f"[DataPrep] Trends: matched {n_matched:,}/{len(df):,} rows")

    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prepare_dataset(
        input_path  = INPUT_PATH,
        trends_path = TRENDS_PATH,
        output_path = OUTPUT_PATH,
        verbose     = True,
    )