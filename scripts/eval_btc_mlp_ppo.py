"""
eval_btc_mlp_ppo.py

Objective evaluation of a trained BTCTradingEnv5m PPO agent on the
held-out test dataset (btc_5min_test.csv).

Design
------
Training uses random episode start positions, which is correct for
exploration but wrong for evaluation — it creates uneven test coverage
and non-reproducible results across runs.

This script instead performs a sequential walk-forward evaluation:
  • The agent starts at the beginning of the test dataset.
  • After each trade completes (sell or forced exit), the next episode
    starts from where the previous one left off.
  • When the dataset is exhausted, evaluation ends.
  • The agent acts greedily (argmax over the masked policy), not by sampling.

This means every candle in the test set is visited exactly once and each
start position is used at most once — an unbiased, reproducible backtest.

The environment used is identical to training (same env class, same
parameters). The only differences are:
  1. Data   → btc_5min_test.csv  (never seen during training)
  2. Starts → sequential, not random
  3. Policy → greedy (deterministic), not sampled

Metrics Reported
----------------
  Trades           — total number of completed trades
  Win Rate         — fraction of trades with profit_pct > 0
  Mean Profit %    — mean realized profit per trade after 0.3% cost
  Median Profit %  — median realized profit per trade
  Total Return %   — sum of all trade profits
  Profit Factor    — gross wins / gross losses (> 1.0 = net profitable)
  Max Drawdown %   — largest peak-to-trough drop in cumulative profit
  Sharpe (trade)   — mean / std of per-trade profits (× √trades)
  Forced Exits %   — fraction of trades closed by the max_hold ceiling
  Mean Hold Steps  — average candles held per trade (5 min each)
  Mean Not-Buy Steps — average Phase 1 wait before entry
  Not-Buy Fraction — Phase 1 decisions that were NOT_BUY (selectivity)
  Vs Buy-and-Hold  — total return vs passively holding BTC over the same period

Output
------
  Console table and a JSONL results file saved to the eval log directory.

Usage
-----
  # Evaluate the latest checkpoint:
  python eval_btc_mlp_ppo.py

  # Evaluate a specific checkpoint:
  python eval_btc_mlp_ppo.py --model /path/to/checkpoint.pt

Author: Jarred Deluca
Project: ServoTrader
License: MIT
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Optional, Tuple

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from servo_trader.envs.btc_trading_env_5m import (
    BTCTradingEnv5m,
    BORUTA_FEATURES,
    OBS_DIM,
    N_FEATURES,
    HISTORY_WINDOW,
    ACTION_NOT_BUY,
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
)


# ============================================================================
#  CONFIGURATION
# ============================================================================

CFG = dict(
    test_data_path = "/home/jarred/git/ServoTrader/data/btc_5min_test.csv",
    model_path     = "/home/jarred/git/ServoTrader/models/btc_mlp_ppo_latest.pt",
    eval_log_dir   = "/home/jarred/git/ServoTrader/logs/eval",

    # Must match training config exactly
    max_hold_steps = 72,
    min_hold_steps = 6,
    hidden_sizes   = [512, 256],
    dropout        = 0.0,    # always 0 for inference — dropout is training-only

    device         = "cuda" if torch.cuda.is_available() else "cpu",
)


# ============================================================================
#  MLP ACTOR-CRITIC  (must match train_btc_mlp_ppo.py exactly)
# ============================================================================

class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, n_actions=4, dropout=0.0):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p=dropout)]
            in_dim = h
        self.backbone    = nn.Sequential(*layers)
        self.actor_head  = nn.Linear(hidden_sizes[-1], n_actions)
        self.critic_head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, obs, masks=None):
        features  = self.backbone(obs)
        logits    = self.actor_head(features)
        if masks is not None:
            logits = logits.masked_fill(~masks.bool(), -1e9)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs     = torch.exp(log_probs)
        entropy   = -(probs * log_probs.clamp(min=-1e9)).sum(dim=-1)
        values    = self.critic_head(features).squeeze(-1)
        return log_probs, values, entropy

    @torch.no_grad()
    def greedy_action(self, obs: torch.Tensor, masks: torch.Tensor) -> int:
        """Return the highest-probability legal action (deterministic)."""
        log_probs, _, _ = self.forward(obs, masks)
        return int(log_probs.argmax(dim=-1).item())


# ============================================================================
#  SEQUENTIAL EVAL ENVIRONMENT
#  Thin subclass of BTCTradingEnv5m that overrides reset() to advance
#  sequentially through the dataset rather than sampling random starts.
#  Everything else — observations, action masks, rewards, JSONL logging —
#  is identical to the training environment.
# ============================================================================

class SequentialBTCEnv(BTCTradingEnv5m):
    """
    Evaluation-mode wrapper around BTCTradingEnv5m.

    On each reset() call the episode starts from _next_start, which is
    advanced to the step after the trade completes. This gives a clean
    walk-forward backtest with no overlap and full dataset coverage.
    """

    def __init__(self, df, max_hold_steps, min_hold_steps, log_dir):
        super().__init__(
            df             = df,
            max_hold_steps = max_hold_steps,
            min_hold_steps = min_hold_steps,
            log_dir        = log_dir,
        )
        self._next_start = HISTORY_WINDOW   # first valid position
        self.exhausted   = False            # True once we've walked the full set

    def reset(self, *, seed=None, options=None):
        # Call the grandparent gym.Env reset to reinitialise the RNG,
        # then set all state manually using our sequential start position.
        gym_reset = super(BTCTradingEnv5m, self).reset  # gym.Env.reset
        gym_reset(self, seed=seed)

        start = self._next_start

        # Check if we have enough runway left for a meaningful episode
        min_runway = self.min_hold_steps + 2
        if start + min_runway >= self.n_rows:
            self.exhausted = True
            # Return a dummy observation — caller must check exhausted flag
            obs  = np.zeros(OBS_DIM, dtype=np.float32)
            info = {"action_mask": np.array([True, True, False, False], dtype=bool)}
            return obs, info

        self.current_step     = start
        self.in_position      = False
        self.buy_price        = 0.0
        self.prev_price       = self.close_prices[self.current_step]
        self.steps_in_trade   = 0
        self.episode_steps    = 0
        self.episode_not_buys = 0
        self._buy_step        = None
        self._buy_price       = 0.0

        self.episode_log = [{
            "type":       "episode_start",
            "episode":    int(self.episode_counter + 1),
            "start_step": int(self.current_step),
            "timestamp":  datetime.now().isoformat(),
        }]

        obs  = self._get_observation()
        info = {"action_mask": self._get_action_mask()}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if terminated or truncated:
            # Advance the sequential cursor to the step after this trade ended
            self._next_start = self.current_step

        return obs, reward, terminated, truncated, info

    def _advance_step(self):
        """Override: never wrap around — stop at the end of the dataset."""
        self.episode_steps += 1
        self.current_step  += 1
        # No wrap — sequential evaluation walks forward only


# ============================================================================
#  METRICS
# ============================================================================

def _max_drawdown(cumulative_profits: list) -> float:
    """Maximum peak-to-trough drop in cumulative profit (%)."""
    if not cumulative_profits:
        return 0.0
    equity = np.cumsum(cumulative_profits)
    peak   = np.maximum.accumulate(equity)
    dd     = peak - equity
    return float(dd.max())


def _sharpe(profits: list) -> float:
    """Trade-level Sharpe: mean / std × sqrt(n). Returns 0 if std is zero."""
    if len(profits) < 2:
        return 0.0
    arr  = np.array(profits)
    std  = arr.std()
    if std < 1e-10:
        return 0.0
    return float(arr.mean() / std * math.sqrt(len(arr)))


def _profit_factor(profits: list) -> float:
    """Gross wins / gross losses. Returns inf if no losing trades."""
    arr   = np.array(profits)
    gross_win  = arr[arr > 0].sum()
    gross_loss = abs(arr[arr < 0].sum())
    if gross_loss < 1e-10:
        return float("inf")
    return float(gross_win / gross_loss)


def compute_metrics(trades: list, df_test: pd.DataFrame) -> dict:
    """
    Compute all evaluation metrics from the list of completed trade dicts.

    Each trade dict has keys: profit_pct, hold_steps, not_buy_steps,
    termination ('sell' | 'forced_sell'), buy_price.
    """
    if not trades:
        return {"error": "no trades completed"}

    profits     = [t["profit_pct"]     for t in trades]
    hold_steps  = [t["hold_steps"]     for t in trades]
    nb_steps    = [t["not_buy_steps"]  for t in trades]
    forced      = [t["termination"] == "forced_sell" for t in trades]

    n_trades    = len(trades)
    wins        = [p for p in profits if p > 0]
    losses      = [p for p in profits if p <= 0]
    win_rate    = len(wins) / n_trades
    mean_nb     = float(np.mean(nb_steps))
    nb_frac     = mean_nb / max(mean_nb + 1.0, 1.0)

    # Buy-and-hold benchmark over the evaluated period
    first_price = df_test["close"].iloc[HISTORY_WINDOW]
    last_price  = df_test["close"].iloc[-1]
    bh_return   = ((last_price / first_price) - 1.0) * 100.0

    return dict(
        n_trades          = n_trades,
        win_rate          = round(win_rate * 100, 2),
        mean_profit_pct   = round(float(np.mean(profits)), 4),
        median_profit_pct = round(float(np.median(profits)), 4),
        total_return_pct  = round(float(np.sum(profits)), 4),
        profit_factor     = round(_profit_factor(profits), 4),
        max_drawdown_pct  = round(_max_drawdown(profits), 4),
        sharpe            = round(_sharpe(profits), 4),
        forced_exit_pct   = round(float(np.mean(forced)) * 100, 2),
        mean_hold_steps   = round(float(np.mean(hold_steps)), 1),
        mean_not_buy_steps= round(mean_nb, 1),
        not_buy_fraction  = round(nb_frac * 100, 2),
        gross_win_pct     = round(float(np.sum(wins)), 4),
        gross_loss_pct    = round(float(np.sum(losses)), 4),
        best_trade_pct    = round(float(max(profits)), 4),
        worst_trade_pct   = round(float(min(profits)), 4),
        buy_and_hold_pct  = round(float(bh_return), 4),
        agent_vs_bh_pct   = round(float(np.sum(profits)) - float(bh_return), 4),
    )


# ============================================================================
#  PRINT REPORT
# ============================================================================

def print_report(metrics: dict, model_path: str, test_data_path: str):
    SEP  = "=" * 64
    SEP2 = "-" * 64

    print(f"\n{SEP}")
    print("  ServoTrader — PPO Agent Evaluation Report")
    print(SEP)
    print(f"  Model : {model_path}")
    print(f"  Data  : {test_data_path}")
    print(f"  Run   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEP)

    if "error" in metrics:
        print(f"  ❌  {metrics['error']}")
        print(SEP + "\n")
        return

    wr    = metrics["win_rate"]
    mp    = metrics["mean_profit_pct"]
    tr    = metrics["total_return_pct"]
    bh    = metrics["buy_and_hold_pct"]
    avbh  = metrics["agent_vs_bh_pct"]

    wr_flag   = "✅" if wr   > 50.0  else "⚠ "
    mp_flag   = "✅" if mp   > 0.0   else "⚠ "
    tr_flag   = "✅" if tr   > 0.0   else "⚠ "
    pf_flag   = "✅" if metrics["profit_factor"] > 1.0 else "⚠ "
    sh_flag   = "✅" if metrics["sharpe"] > 0.0 else "⚠ "

    print(f"\n  TRADE SUMMARY")
    print(SEP2)
    print(f"  Total trades         : {metrics['n_trades']:>10,}")
    print(f"  Forced exits         : {metrics['forced_exit_pct']:>9.1f}%")
    print(f"  Mean hold (candles)  : {metrics['mean_hold_steps']:>10.1f}  ({metrics['mean_hold_steps']*5:.0f} min)")
    print(f"  Mean wait (candles)  : {metrics['mean_not_buy_steps']:>10.1f}")
    print(f"  NOT_BUY fraction     : {metrics['not_buy_fraction']:>9.1f}%  (Phase 1 selectivity)")

    print(f"\n  PROFITABILITY")
    print(SEP2)
    print(f"  {wr_flag} Win rate            : {wr:>9.2f}%")
    print(f"  {mp_flag} Mean profit/trade   : {mp:>+9.4f}%")
    print(f"     Median profit/trade : {metrics['median_profit_pct']:>+9.4f}%")
    print(f"     Best trade          : {metrics['best_trade_pct']:>+9.4f}%")
    print(f"     Worst trade         : {metrics['worst_trade_pct']:>+9.4f}%")
    print(f"  {tr_flag} Total return        : {tr:>+9.4f}%")
    print(f"  {pf_flag} Profit factor       : {metrics['profit_factor']:>10.4f}  (> 1.0 = net profitable)")
    print(f"  {sh_flag} Sharpe (trade-lvl)  : {metrics['sharpe']:>10.4f}")
    print(f"     Max drawdown        : {metrics['max_drawdown_pct']:>+9.4f}%")

    print(f"\n  VS BENCHMARK")
    print(SEP2)
    bh_flag = "✅" if avbh > 0 else "⚠ "
    print(f"     Buy-and-hold return : {bh:>+9.4f}%")
    print(f"     Agent total return  : {tr:>+9.4f}%")
    print(f"  {bh_flag} Agent vs B&H        : {avbh:>+9.4f}%")

    print(f"\n{SEP}\n")


# ============================================================================
#  MAIN
# ============================================================================

def evaluate(model_path: str, cfg: dict):
    device = cfg["device"]

    # ── Load test data ────────────────────────────────────────────────────────
    print(f"[Eval] Loading test data from {cfg['test_data_path']} …")
    df_test = pd.read_csv(cfg["test_data_path"])
    print(f"[Eval] Test rows : {len(df_test):,}")
    print(f"[Eval] Date range: {df_test['timestamp'].iloc[0]} → {df_test['timestamp'].iloc[-1]}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[Eval] Loading model from {model_path} …")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    ckpt  = torch.load(model_path, map_location=device)
    model = MLPActorCritic(
        obs_dim      = OBS_DIM,
        hidden_sizes = cfg["hidden_sizes"],
        n_actions    = 4,
        dropout      = cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    trained_step = ckpt.get("global_step", "unknown")
    print(f"[Eval] Model trained to step: {trained_step:,}" if isinstance(trained_step, int) else
          f"[Eval] Model trained to step: {trained_step}")

    # ── Build sequential environment ──────────────────────────────────────────
    os.makedirs(cfg["eval_log_dir"], exist_ok=True)
    env = SequentialBTCEnv(
        df             = df_test,
        max_hold_steps = cfg["max_hold_steps"],
        min_hold_steps = cfg["min_hold_steps"],
        log_dir        = cfg["eval_log_dir"],
    )

    # ── Walk-forward evaluation loop ──────────────────────────────────────────
    print(f"[Eval] Running sequential walk-forward evaluation …\n")

    trades      = []
    obs, info   = env.reset()
    t_start     = time.time()

    while not env.exhausted:
        mask_t = torch.tensor(
            info["action_mask"], dtype=torch.bool
        ).unsqueeze(0).to(device)
        obs_t  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        action = model.greedy_action(obs_t, mask_t)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            # Extract this trade's stats from the env
            # env._finalise_episode has already run; read from its last log entry
            # which was written to the JSONL file. We reconstruct the dict here
            # from live env state to avoid re-parsing the file.
            trades.append(dict(
                profit_pct    = round(env.total_profit_pct - sum(t["profit_pct"] for t in trades), 6),
                hold_steps    = env.episode_steps,
                not_buy_steps = env.episode_not_buys,
                termination   = "forced_sell" if truncated else "sell",
            ))

            obs, info = env.reset()

            if len(trades) % 200 == 0:
                elapsed = time.time() - t_start
                print(
                    f"  {len(trades):>5} trades | "
                    f"win%={100*np.mean([1 if t['profit_pct']>0 else 0 for t in trades]):.1f} | "
                    f"mean={np.mean([t['profit_pct'] for t in trades]):+.4f}% | "
                    f"elapsed={elapsed:.0f}s"
                )

    elapsed = time.time() - t_start
    print(f"\n[Eval] Completed {len(trades):,} trades in {elapsed:.1f}s")

    # ── Compute and display metrics ───────────────────────────────────────────
    metrics = compute_metrics(trades, df_test)
    print_report(metrics, model_path, cfg["test_data_path"])

    # ── Save results ──────────────────────────────────────────────────────────
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(cfg["eval_log_dir"], f"eval_result_{ts}.json")
    result      = dict(
        model_path     = model_path,
        trained_step   = trained_step,
        test_data_path = cfg["test_data_path"],
        eval_timestamp = ts,
        n_test_rows    = len(df_test),
        metrics        = metrics,
    )
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[Eval] Results saved → {result_path}\n")

    return metrics


# ============================================================================
#  CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained ServoTrader PPO agent")
    parser.add_argument(
        "--model", type=str, default=CFG["model_path"],
        help=f"Path to model checkpoint (default: {CFG['model_path']})"
    )
    args = parser.parse_args()

    evaluate(model_path=args.model, cfg=CFG)