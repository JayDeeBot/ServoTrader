# ServoTrader

*Last updated: 2025-08-28*

ServoTrader is a **reinforcement‑learning crypto trading framework** built around Gymnasium environments and a **MaskablePPO** agent (Stable‑Baselines3). It supports **live trading on Binance**, pretraining environments for **Buy** and **Hold/Sell** decisions, and a full **Tkinter GUI (LiveEnvRenderer)** for real‑time monitoring. The system can train on large historical datasets using **chunked loading** and then switch to **live mode** with the same policy.

> **Author:** Jarred Deluca · **License:** MIT · **Repo:** https://github.com/JayDeeBot/ServoTrader

---

## ✨ What’s new (last 4 months)

- ✅ **RL-first architecture**: MaskablePPO + action masking across multiple environments
- ✅ **LiveCryptoTradingEnv** for **real Binance trades** (paper/live) using the same policy
- ✅ **BuyTrainingEnv** (entry selection) and **SellHoldTrainingEnv** (exit timing) pretraining
- ✅ **Special `NOT_BUY` action** for “skip this opportunity” scenarios (with tailored rewards)
- ✅ **Chunked dataset loading** (e.g., swap 10k-row files every 10k steps) for huge corpora
- ✅ **Sharpe-style risk penalties**, break‑even penalties, non‑linear profit rewards, and **trade cost modeling**
- ✅ **Detailed per‑episode logging** to `servo_trader/env_logs/`
- ✅ **Binance migration** for deep OHLCV history; compliance helpers for **step sizes / filters**
- ✅ Robust **ServoTraderBinance** backend that inherits a common **ServoTraderInterface**
- ✅ End‑to‑end **training scripts** & **live agent runner**; Tkinter **LiveEnvRenderer** GUI

---

## 🧱 Architecture

```
Data (CSV/Chunks, Binance/Kraken)
        │
        ▼
┌───────────────────────────────────────────┐
│  Gym Environments (Gymnasium)             │
│  • BuyTrainingEnv (entry)                 │
│  • SellHoldTrainingEnv (exit)             │
│  • LiveCryptoTradingEnv (buy/hold/sell)   │
│    - Actions: HOLD, BUY[i], SELL, NOT_BUY │
│    - Action masking for invalid actions   │
└───────────────────────────────────────────┘
        │ observations, rewards
        ▼
MaskablePPO (SB3)  ← wrappers, callbacks, logs
        │ actions
        ▼
Trader Backends
• ServoTraderBinance  (live/paper via Binance REST/WebSocket)
• (Legacy) Alpaca/Kraken utilities for history
        │ fills, balances
        ▼
Tkinter LiveEnvRenderer  +  JSONL/CSV logs
```

---

## 📂 Project Structure (key files/directories)

```
servo_trader/
├── envs/
│   ├── buy_training_env.py           # single-decision BUY pretraining (+ NOT_BUY)
│   ├── sell_hold_training_env.py     # single-decision HOLD/SELL pretraining
│   └── live_crypto_trading_env.py    # live/binance buy-hold-sell loop (+ masks)
├── traders/
│   ├── interface.py                  # ServoTraderInterface base class
│   └── binance_trader.py             # ServoTraderBinance (filters, step sizes, fills)
├── gui/
│   └── live_env_renderer.py          # Tkinter dashboard
├── data/
│   ├── split_10k_chunks_modern/      # 001.csv, 002.csv, ... (auto-rotated in training)
│   └── crypto_codes.json             # normalized symbols (e.g., BTCUSDT)
├── scripts/
│   ├── train_ppo_buy_pretrain.py
│   ├── train_ppo_sell_hold_pretrain.py
│   └── run_live_ppo_agent.py         # loads a PPO model and runs LiveCryptoTradingEnv
├── servo_trader/
│   ├── __init__.py
│   ├── params.yaml                   # horizons, rewards, paths, chunking, GUI opts
│   └── env_logs/                     # per-episode JSONL/CSV logs + summaries
├── requirements.txt
├── LICENSE
└── README.md  (this file)
```

> **Note:** Older modules like `crypto_database_init.py`, `servo_predictor.py`, and an Alpaca-only `servo_trader.py` exist historically; the RL path now supersedes them. A new `CryptoDatabaseInitialiser` and data tooling target **Binance** for history.

---

## 🧪 Environments & Actions

### 1) BuyTrainingEnv (single decision)
- **Goal:** Pick the symbol most likely to rise over horizon **H** (or **NOT_BUY**).
- **Actions:** `0..N-1 = BUY:symbol_i`, `N = NOT_BUY`.
- **Reward (sketch):**
  - BUY: rank- or return-based reward on next‑step / configured horizon.
  - NOT_BUY: **positive** if *all* tracked cryptos drop; **negative** if any rise (penalize missed upside).

### 2) SellHoldTrainingEnv (single decision)
- **Goal:** Exit timing for a held position.
- **Actions:** `HOLD`, `SELL`.
- **Reward (sketch):**
  - HOLD: `+future_return` over horizon **H**.
  - SELL: `-future_return` over horizon **H** (inverse reward).

### 3) LiveCryptoTradingEnv (looped)
- **Goal:** Operate continuously with live OHLCV (Binance). 
- **Actions:** `0 = HOLD`, `1..N = BUY:symbol_i`, `N+1 = SELL`, `N+2 = NOT_BUY`.
- **Action Masking:** Prevents invalid actions (e.g., SELL with no position).
- **Reward shaping:** Sharpe-style risk penalty, non-linear profit scaling, break-even penalties, trade fees/slippage.
- **Chunked Data Mode:** For backtests/training, auto-loads a new **10k-row** CSV every **10k steps**, maintaining `global_step` and resetting `current_step` per chunk.

---

## 🧾 Logging & Analytics

- **Step and episode logs** written to `servo_trader/env_logs/` (JSONL/CSV).
- Episode summaries include cumulative profit, reward, win rate, and runtime.
- Compatible with TensorBoard via SB3 callbacks.
- GUI displays connection status, control state, live prices, positions, and action decisions.

---

## 🔧 Installation

```bash
git clone https://github.com/JayDeeBot/ServoTrader
cd ServoTrader
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optional (Poetry):

```bash
poetry install
poetry run python scripts/train_ppo_buy_pretrain.py
```

---

## 🔐 Configuration

### Environment variables (recommended)

```bash
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
export ST_ENV="live"                 # or 'paper'
export ST_SYMBOLS_PATH="data/crypto_codes.json"
export ST_PARAMS_PATH="servo_trader/params.yaml"
```

> You may also use a `.env` loader, or OS keyrings, instead of exporting inline.

### `params.yaml` (example keys)

```yaml
data:
  chunk_dir: "data/split_10k_chunks_modern"
  chunk_size: 10000
  horizon: 1           # steps for reward/labels
  normalize: true

env:
  rewards:
    sharpe_penalty: 0.1
    break_even_penalty: 0.02
    fee_bps: 7.5
    nonlinear_profit_k: 1.6

trading:
  max_buy_limit_usd: 200.0
  daily_cashout_pct: 0.15
  timeout_steps: 300
  allow_not_buy: true

gui:
  enabled: true
```

---

## 🚀 Quickstarts

### A) Pretrain (BUY)
```bash
python scripts/train_ppo_buy_pretrain.py   --symbols data/crypto_codes.json   --params servo_trader/params.yaml   --total-timesteps 2_000_000   --tensorboard ./tb/buy_pretrain
```

### B) Pretrain (HOLD/SELL)
```bash
python scripts/train_ppo_sell_hold_pretrain.py   --symbols data/crypto_codes.json   --params servo_trader/params.yaml   --total-timesteps 1_000_000   --tensorboard ./tb/sellhold_pretrain
```

### C) Live trading / paper trading
```bash
python scripts/run_live_ppo_agent.py   --model checkpoints/ppo_latest.zip   --symbols data/crypto_codes.json   --params servo_trader/params.yaml   --mode live   # or paper
```
> Ensure your Binance keys are set and symbols are exchange‑compatible.

---

## 🧰 Developer Notes

- **Symbols:** `crypto_codes.json` must match exchange filters (e.g., `BTCUSDT`). The loader normalizes `symbol/pair/ticker` columns from data files.
- **Binance filters:** The trader backend enforces step size / lot size / min notional; see `ServoTraderBinance`.
- **Unit tests:** Add coverage for:
  - `set_max_buy_limit`, `set_daily_cashout_percent`
  - `cash_out_to_bank`
  - `find_untradeable_codes`, `find_viable_replacement_codes`
- **Data ingestion:** Historical fetchers exist for Binance (preferred) and legacy Kraken; see `CryptoDatabaseInitialiser`.
- **Scaling:** Use chunked datasets for 1M+ rows. The env rotates chunks every 10k steps and keeps `global_step` for statistics.

---

## ⚠️ Disclaimer

This software is for **research**. Markets are risky. **Use paper trading first.** No warranty or financial advice is provided. You are responsible for API keys, compliance, and tax reporting.

---

## 📝 License

MIT (see [LICENSE](LICENSE)).

---

## 🙋 Contact

**Jarred Deluca** · <jarred.g.deluca@student.uts.edu.au>