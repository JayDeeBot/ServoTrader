"""
prepare_btc_5min_dataset.py

Engineers the 38 Boruta-confirmed features for the 5-minute BTCTradingEnv
from the ServoTrader 5-minute OHLCV file, then produces a clean chronological
train / test split for objective model evaluation.

Source file
-----------
    /home/jarred/git/ServoTrader/data/5min_historical/BTCUSDT.csv
    Columns: timestamp, open, high, low, close, vwap, volume, count, symbol

Outputs
-------
    /home/jarred/git/ServoTrader/data/btc_5min_features.csv   — full prepared dataset
    /home/jarred/git/ServoTrader/data/btc_5min_train.csv      — training split  (first 80%)
    /home/jarred/git/ServoTrader/data/btc_5min_test.csv       — test split      (last  20%)

Split policy
------------
The split is strictly chronological. The training set contains the oldest 80%
of candles; the test set contains the most recent 20%. The test set is held out
completely — it must never be used for training or hyperparameter tuning.

  80 % train  ≈ 6.8 years  (with ~8.5 years of full history)
  20 % test   ≈ 1.7 years

38 Boruta-Confirmed Features (importance order)
------------------------------------------------
 1  bb_bandwidth_48     — Bollinger Bandwidth, window=48           [technical]
 2  bb_percent_b_48     — Bollinger %B, window=48                  [technical]
 3  returns_24p         — Log return over 24 candles (2h)          [momentum]
 4  returns_12p         — Log return over 12 candles (1h)          [momentum]
 5  bb_bandwidth_72     — Bollinger Bandwidth, window=72           [technical]
 6  bb_percent_b_72     — Bollinger %B, window=72                  [technical]
 7  returns_6p          — Log return over 6 candles (30 min)       [momentum]
 8  returns_3p          — Log return over 3 candles (15 min)       [momentum]
 9  returns_1p          — Log return over 1 candle  (5 min)        [momentum]
10  vwap_deviation      — (close − vwap) / |vwap|                 [other]
11  stoch_k             — Stochastic %K (14, 3)                    [technical]
12  obv_ma_48           — OBV 48-period SMA                        [technical]
13  returns_mean_24p    — Rolling mean of 1p returns, window=24    [momentum]
14  volume_ma_48        — Volume SMA, window=48                    [volume]
15  volume_ma_24        — Volume SMA, window=24                    [volume]
16  volume_ma_12        — Volume SMA, window=12                    [volume]
17  returns_std_144p    — Rolling std of 1p returns, window=144    [momentum]
18  returns_mean_144p   — Rolling mean of 1p returns, window=144   [momentum]
19  atr_12              — ATR, period=12                            [technical]
20  atr_24              — ATR, period=24                            [technical]
21  atr_48              — ATR, period=48                            [technical]
22  obv                 — On-Balance Volume                         [technical]
23  returns_48p         — Log return over 48 candles (4h)          [momentum]
24  stoch_d             — Stochastic %D (3-period SMA of %K)       [technical]
25  bb_percent_b_24     — Bollinger %B, window=24                  [technical]
26  rsi_12              — Wilder RSI, period=12                    [technical]
27  rsi_24              — Wilder RSI, period=24                    [technical]
28  returns_72p         — Log return over 72 candles (6h)          [momentum]
29  rsi_48              — Wilder RSI, period=48                    [technical]
30  rsi_72              — Wilder RSI, period=72                    [technical]
31  macd                — MACD line (EMA12 − EMA26)               [technical]
32  macd_signal         — MACD signal (EMA9 of MACD)              [technical]
33  macd_histogram      — MACD − Signal                            [technical]
34  bb_percent_b_12     — Bollinger %B, window=12                  [technical]
35  returns_mean_48p    — Rolling mean of 1p returns, window=48    [momentum]
36  bb_bandwidth_24     — Bollinger Bandwidth, window=24           [technical]
37  rsi_6               — Wilder RSI, period=6                     [technical]
38  count               — Number of trades per candle               [ohlcv_base]

At 5-minute resolution:
  12 candles  = 1 hour
  24 candles  = 2 hours
  48 candles  = 4 hours
  72 candles  = 6 hours
  144 candles = 12 hours

Author: Jarred Deluca
Project: ServoTrader
License: MIT
"""

import os
import warnings
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------

OHLCV_PATH   = "/home/jarred/git/ServoTrader/data/5min_historical/BTCUSDT.csv"
OUTPUT_FULL  = "/home/jarred/git/ServoTrader/data/btc_5min_features.csv"
OUTPUT_TRAIN = "/home/jarred/git/ServoTrader/data/btc_5min_train.csv"
OUTPUT_TEST  = "/home/jarred/git/ServoTrader/data/btc_5min_test.csv"

# Fraction of prepared rows assigned to the training set.
# The remaining (1 - TRAIN_RATIO) fraction becomes the held-out test set.
# The split is strictly chronological: train = oldest, test = most recent.
TRAIN_RATIO = 0.80

# Must match BORUTA_FEATURES in btc_trading_env_5m.py exactly
FEATURE_COLS = [
    "bb_bandwidth_48",
    "bb_percent_b_48",
    "returns_24p",
    "returns_12p",
    "bb_bandwidth_72",
    "bb_percent_b_72",
    "returns_6p",
    "returns_3p",
    "returns_1p",
    "vwap_deviation",
    "stoch_k",
    "obv_ma_48",
    "returns_mean_24p",
    "volume_ma_48",
    "volume_ma_24",
    "volume_ma_12",
    "returns_std_144p",
    "returns_mean_144p",
    "atr_12",
    "atr_24",
    "atr_48",
    "obv",
    "returns_48p",
    "stoch_d",
    "bb_percent_b_24",
    "rsi_12",
    "rsi_24",
    "returns_72p",
    "rsi_48",
    "rsi_72",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bb_percent_b_12",
    "returns_mean_48p",
    "bb_bandwidth_24",
    "rsi_6",
    "count",
]


# ---------------------------------------------------------------------------
#  Timestamp helper
# ---------------------------------------------------------------------------

def _normalise_ts(series: pd.Series) -> pd.Series:
    """Parse timestamps to timezone-naive UTC datetime64."""
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None)


# ---------------------------------------------------------------------------
#  Indicator helpers
# ---------------------------------------------------------------------------

def _log_return(close: pd.Series, periods: int) -> pd.Series:
    return np.log(close / (close.shift(periods) + 1e-12))


def _wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    d        = close.diff()
    gain     = d.clip(lower=0)
    loss     = (-d).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _bollinger(close: pd.Series, window: int, n_std: float = 2.0):
    """Returns (percent_b, bandwidth)."""
    ma    = close.rolling(window, min_periods=window).mean()
    sigma = close.rolling(window, min_periods=window).std()
    upper = ma + n_std * sigma
    lower = ma - n_std * sigma
    pct_b     = (close - lower) / (upper - lower + 1e-12)
    bandwidth = (upper - lower) / (ma + 1e-12)
    return pct_b, bandwidth


def _stochastic(high, low, close, k_period=14, d_period=3):
    lo = low.rolling(k_period, min_periods=k_period).min()
    hi = high.rolling(k_period, min_periods=k_period).max()
    k  = 100.0 * (close - lo) / (hi - lo + 1e-12)
    d  = k.rolling(d_period, min_periods=d_period).mean()
    return k, d


def _atr(high, low, close, period):
    ph  = close.shift(1)
    tr  = pd.concat([
        (high - low),
        (high - ph).abs(),
        (low  - ph).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _obv(close, volume):
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def _macd(close, fast=12, slow=26, signal=9):
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    sig_line   = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - sig_line
    return macd_line, sig_line, histogram


def _vwap_deviation(close, vwap):
    return (close - vwap) / (vwap.abs() + 1e-12)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def prepare_dataset(
    ohlcv_path:   str   = OHLCV_PATH,
    output_full:  str   = OUTPUT_FULL,
    output_train: str   = OUTPUT_TRAIN,
    output_test:  str   = OUTPUT_TEST,
    train_ratio:  float = TRAIN_RATIO,
    verbose:      bool  = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw OHLCV, engineer all 38 features, save the full dataset, then
    produce and save a chronological train/test split.

    Returns
    -------
    (train_df, test_df) — both include timestamp + OHLCV base columns + features.
    """

    SEP = "=" * 70

    # ── 1. Load OHLCV ─────────────────────────────────────────────────────────
    if verbose:
        print(f"\n{SEP}")
        print("  LOADING 5-MINUTE OHLCV")
        print(f"  {ohlcv_path}")
        print(SEP)

    if not os.path.exists(ohlcv_path):
        raise FileNotFoundError(
            f"5-minute OHLCV file not found: {ohlcv_path}\n"
            "Run fetch_btc_5min_historical.py first to download the raw data."
        )

    df = pd.read_csv(ohlcv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    df["timestamp"] = _normalise_ts(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # VWAP
    if "vwap" in df.columns:
        df["vwap"] = pd.to_numeric(df["vwap"], errors="coerce")
        if df["vwap"].isna().mean() > 0.5:
            if verbose:
                print("  ⚠  vwap mostly NaN — approximating as (H+L+C)/3")
            df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0
    else:
        if verbose:
            print("  ⚠  vwap missing — approximating as (H+L+C)/3")
        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0

    # Count (trade count per candle — confirmed Boruta feature)
    if "count" not in df.columns:
        if verbose:
            print("  ⚠  count column missing — filling zeros")
        df["count"] = 0.0
    else:
        df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0.0)

    # Fill gaps
    for col in ["open", "high", "low", "close", "vwap"]:
        df[col] = df[col].ffill().bfill()
    df["volume"] = df["volume"].fillna(0.0)
    df = df.drop(columns=["symbol"], errors="ignore")

    if verbose:
        raw_days = len(df) * 5 / 60 / 24
        print(f"  ✅  {len(df):,} rows | {df['timestamp'].min()} → {df['timestamp'].max()}")
        print(f"      ≈ {raw_days:.0f} days ({raw_days / 365:.1f} years) of 5-min bars")

    # ── 2. Engineer features ──────────────────────────────────────────────────
    if verbose:
        print(f"\n{SEP}")
        print("  ENGINEERING 38 FEATURES")
        print(SEP)

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]
    vwap   = df["vwap"]

    # --- Bollinger Bands (4 windows) ---
    df["bb_percent_b_12"], df["bb_bandwidth_12"] = _bollinger(close, 12)
    df["bb_percent_b_24"], df["bb_bandwidth_24"] = _bollinger(close, 24)
    df["bb_percent_b_48"], df["bb_bandwidth_48"] = _bollinger(close, 48)
    df["bb_percent_b_72"], df["bb_bandwidth_72"] = _bollinger(close, 72)

    # --- RSI (5 periods) ---
    df["rsi_6"]  = _wilder_rsi(close, 6)
    df["rsi_12"] = _wilder_rsi(close, 12)
    df["rsi_24"] = _wilder_rsi(close, 24)
    df["rsi_48"] = _wilder_rsi(close, 48)
    df["rsi_72"] = _wilder_rsi(close, 72)

    # --- Stochastic (k=14, d=3) ---
    df["stoch_k"], df["stoch_d"] = _stochastic(high, low, close, k_period=14, d_period=3)

    # --- ATR ---
    df["atr_12"] = _atr(high, low, close, 12)
    df["atr_24"] = _atr(high, low, close, 24)
    df["atr_48"] = _atr(high, low, close, 48)

    # --- MACD (EMA 12/26/9 on 5-min candles) ---
    df["macd"], df["macd_signal"], df["macd_histogram"] = _macd(close, 12, 26, 9)

    # --- OBV and OBV MA ---
    df["obv"]       = _obv(close, volume)
    df["obv_ma_48"] = df["obv"].rolling(48, min_periods=48).mean()

    # --- VWAP deviation ---
    df["vwap_deviation"] = _vwap_deviation(close, vwap)

    # --- Volume MAs ---
    df["volume_ma_12"] = volume.rolling(12, min_periods=12).mean()
    df["volume_ma_24"] = volume.rolling(24, min_periods=24).mean()
    df["volume_ma_48"] = volume.rolling(48, min_periods=48).mean()

    # --- Log returns (multi-period) ---
    ret_1p = _log_return(close, 1)
    df["returns_1p"]  = ret_1p
    df["returns_3p"]  = _log_return(close, 3)
    df["returns_6p"]  = _log_return(close, 6)
    df["returns_12p"] = _log_return(close, 12)
    df["returns_24p"] = _log_return(close, 24)
    df["returns_48p"] = _log_return(close, 48)
    df["returns_72p"] = _log_return(close, 72)

    # --- Rolling return stats ---
    df["returns_mean_24p"]  = ret_1p.rolling(24,  min_periods=24).mean()
    df["returns_mean_48p"]  = ret_1p.rolling(48,  min_periods=48).mean()
    df["returns_mean_144p"] = ret_1p.rolling(144, min_periods=144).mean()
    df["returns_std_144p"]  = ret_1p.rolling(144, min_periods=144).std()

    if verbose:
        print("  ✅  All 38 features computed")

    # ── 3. Drop NaN warm-up rows ──────────────────────────────────────────────
    before = len(df)
    df     = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    if verbose:
        print(f"  Dropped {before - len(df):,} warm-up rows (longest window=144 candles = 12h)")
        print(f"  Remaining: {len(df):,} rows")

    # ── 4. Sanitise Inf / NaN ─────────────────────────────────────────────────
    for col in FEATURE_COLS:
        n_inf = np.isinf(df[col]).sum()
        n_nan = df[col].isna().sum()
        if n_inf > 0 or n_nan > 0:
            warnings.warn(f"[DataPrep] {col}: {n_inf} Inf, {n_nan} NaN remain — replacing with 0")
            df[col] = df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # ── 5. Select output columns ──────────────────────────────────────────────
    keep = ["timestamp", "open", "high", "low", "close", "vwap", "volume"] + FEATURE_COLS
    keep = [c for c in keep if c in df.columns]
    out  = df[keep].copy()

    # ── 6. Save full prepared dataset ─────────────────────────────────────────
    os.makedirs(os.path.dirname(output_full),  exist_ok=True)
    os.makedirs(os.path.dirname(output_train), exist_ok=True)
    os.makedirs(os.path.dirname(output_test),  exist_ok=True)

    out.to_csv(output_full, index=False)

    # ── 7. Chronological train / test split ───────────────────────────────────
    n_total = len(out)
    n_train = int(n_total * train_ratio)
    # Ensure the split lands on a clean boundary — no row belongs to both sets
    n_test  = n_total - n_train

    train_df = out.iloc[:n_train].reset_index(drop=True)
    test_df  = out.iloc[n_train:].reset_index(drop=True)

    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test,   index=False)

    # ── 8. Summary ────────────────────────────────────────────────────────────
    if verbose:
        train_days = n_train * 5 / 60 / 24
        test_days  = n_test  * 5 / 60 / 24

        print(f"\n{SEP}")
        print("  ✅  COMPLETE")
        print(SEP)
        print(f"\n  Full dataset")
        print(f"    Rows      : {n_total:,}")
        print(f"    Date range: {out['timestamp'].min()} → {out['timestamp'].max()}")
        print(f"    Coverage  : ≈{n_total * 5 / 60 / 24:.0f} days ({n_total * 5 / 60 / 24 / 365:.1f} years)")
        print(f"    Saved to  : {output_full}")

        print(f"\n  Train split  ({train_ratio * 100:.0f}%)")
        print(f"    Rows      : {n_train:,}")
        print(f"    Date range: {train_df['timestamp'].min()} → {train_df['timestamp'].max()}")
        print(f"    Coverage  : ≈{train_days:.0f} days ({train_days / 365:.1f} years)")
        print(f"    Saved to  : {output_train}")

        print(f"\n  Test split  ({(1 - train_ratio) * 100:.0f}%)  ← HELD OUT — do not use for training")
        print(f"    Rows      : {n_test:,}")
        print(f"    Date range: {test_df['timestamp'].min()} → {test_df['timestamp'].max()}")
        print(f"    Coverage  : ≈{test_days:.0f} days ({test_days / 365:.1f} years)")
        print(f"    Saved to  : {output_test}")

        print(f"\n{SEP}")
        print("\n  Feature value range check:")
        for col in FEATURE_COLS:
            mn   = float(out[col].min())
            mx   = float(out[col].max())
            flag = "✅" if abs(mn) < 1e6 and abs(mx) < 1e6 else "⚠ "
            print(f"    {flag}  {col:<22}  min={mn:>14.4f}   max={mx:>14.4f}")
        print(f"{SEP}\n")

    return train_df, test_df


if __name__ == "__main__":
    prepare_dataset()