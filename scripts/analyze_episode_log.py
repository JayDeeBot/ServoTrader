"""
analyze_episode_log.py  —  v2 (multi-trade episodes)

Reads the ServoTrader JSONL episode log and prints a clean summary of
training performance. Compatible with both single-trade (v1-v4) and
multi-trade (v5+) episode formats.

Usage
-----
    python scripts/analyze_episode_log.py              # latest log
    python scripts/analyze_episode_log.py --log /path/to/file.jsonl
    python scripts/analyze_episode_log.py --actions    # show action breakdown

Author: Jarred Deluca
Project: ServoTrader
License: MIT
"""

import os
import json
import glob
import argparse
import numpy as np
from collections import Counter

DEFAULT_LOG_DIR = "/home/jarred/git/ServoTrader/logs"


def find_latest_log(log_dir: str) -> str:
    pattern = os.path.join(log_dir, "btc5m_env_*.jsonl")
    logs = sorted(glob.glob(pattern))
    if not logs:
        raise FileNotFoundError(f"No JSONL logs found in {log_dir}")
    return logs[-1]


def analyse(log_path: str, show_actions: bool = False):
    SEP = "=" * 62

    episodes  = []
    all_steps = []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["type"] == "episode_end":
                episodes.append(rec)
            elif rec["type"] == "step" and show_actions:
                all_steps.append(rec)

    if not episodes:
        print("No episode_end records found — training may not have completed any episodes yet.")
        return

    # Detect format: v5 multi-trade has 'n_trades' and 'trades' fields
    is_multitrade = "n_trades" in episodes[0]

    print(f"\n{SEP}")
    print(f"  EPISODE LOG ANALYSIS  ({'multi-trade v5+' if is_multitrade else 'single-trade v1-v4'})")
    print(f"  {log_path}")
    print(SEP)

    if is_multitrade:
        _analyse_multitrade(episodes)
    else:
        _analyse_singletrade(episodes)

    if show_actions and all_steps:
        print(f"\n  ACTION BREAKDOWN (from {len(all_steps):,} logged steps)")
        counts = Counter(s["action_str"] for s in all_steps)
        total  = sum(counts.values())
        for action in ["not_buy", "buy", "hold", "sell", "sell_stoploss", "sell_forced", "sell_ep_end"]:
            c = counts.get(action, 0)
            if c > 0:
                print(f"    {action:<16} : {c:>8,}  ({c/total*100:5.1f}%)")

    print(f"\n{SEP}\n")


def _analyse_multitrade(episodes: list):
    """Analyse v5+ multi-trade episode logs."""
    n_ep = len(episodes)

    # Flatten all individual trades across all episodes
    all_trades = []
    for ep in episodes:
        for t in ep.get("trades", []):
            all_trades.append(t)

    profits     = [t["profit_pct"] for t in all_trades]
    termination = Counter(t["termination"] for t in all_trades)
    wins        = [p for p in profits if p > 0]
    losses      = [p for p in profits if p <= 0]

    steps        = [e["steps"]       for e in episodes]
    not_buy_s    = [e.get("not_buy_steps", 0) for e in episodes]
    trades_per_ep = [e.get("n_trades", 0) for e in episodes]

    print(f"\n  EPISODES")
    print(f"    Total episodes       : {n_ep:,}")
    print(f"    Total completed trades: {len(all_trades):,}")
    print(f"    Mean trades/episode  : {np.mean(trades_per_ep):.1f}")
    print(f"    Episodes with 0 trades: {sum(1 for t in trades_per_ep if t == 0):,}")

    print(f"\n  TRADE EXITS")
    for t_type, count in sorted(termination.items(), key=lambda x: -x[1]):
        print(f"    {t_type:<18}: {count:>7,}  ({count/len(all_trades)*100:.1f}%)")

    if profits:
        print(f"\n  PROFITABILITY (per trade)")
        print(f"    Win rate             : {len(wins)/len(profits)*100:.1f}%  "
              f"({len(wins):,} wins / {len(losses):,} losses)")
        print(f"    Mean profit/trade    : {np.mean(profits):+.4f}%")
        print(f"    Median profit/trade  : {np.median(profits):+.4f}%")
        print(f"    Profit std dev       : {np.std(profits):.4f}%")
        print(f"    Best trade           : {max(profits):+.3f}%")
        print(f"    Worst trade          : {min(profits):+.3f}%")
        if wins and losses:
            print(f"    Avg win size         : {np.mean(wins):+.4f}%")
            print(f"    Avg loss size        : {np.mean(losses):+.4f}%")
            print(f"    Win/loss ratio       : {abs(np.mean(wins)/np.mean(losses)):.2f}x")
            breakeven = abs(np.mean(losses)) / (np.mean(wins) + abs(np.mean(losses)))
            print(f"    Breakeven win rate   : {breakeven*100:.1f}%")

    print(f"\n  EPISODE LENGTH")
    print(f"    Mean steps/episode   : {np.mean(steps):.1f}")
    print(f"    Min / Max            : {min(steps)} / {max(steps)}")

    print(f"\n  NOT_BUY BEHAVIOUR")
    print(f"    Mean NOT_BUY steps/ep : {np.mean(not_buy_s):.1f}")
    print(f"    Median NOT_BUY steps  : {np.median(not_buy_s):.1f}")
    nb_frac = np.mean(not_buy_s) / max(np.mean(steps), 1) * 100
    print(f"    NOT_BUY fraction      : {nb_frac:.1f}% of episode steps")
    print(f"    Episodes with 0 nb    : {sum(1 for v in not_buy_s if v==0):,}  "
          f"({sum(1 for v in not_buy_s if v==0)/n_ep*100:.1f}%)")

    # Profit trajectory by episode quarter
    n = len(episodes)
    print(f"\n  PROFIT TRAJECTORY (episode quarters — mean profit/trade)")
    for label, sl in [("Q1 (0–25%)",   episodes[:n//4]),
                      ("Q2 (25–50%)",  episodes[n//4:n//2]),
                      ("Q3 (50–75%)",  episodes[n//2:3*n//4]),
                      ("Q4 (75–100%)", episodes[3*n//4:])]:
        q_trades  = [t for ep in sl for t in ep.get("trades", [])]
        q_profits = [t["profit_pct"] for t in q_trades]
        if q_profits:
            wr = sum(1 for p in q_profits if p > 0) / len(q_profits) * 100
            print(f"    {label:<18} mean={np.mean(q_profits):+.4f}%  win={wr:.1f}%  "
                  f"n={len(q_trades):,}")
        else:
            print(f"    {label:<18} (no trades)")


def _analyse_singletrade(episodes: list):
    """Analyse v1-v4 single-trade episode logs (one trade per episode)."""
    profits   = [e["profit_pct"]  for e in episodes]
    steps     = [e["steps"]       for e in episodes]
    not_buy_s = [e.get("not_buy_steps", None) for e in episodes]
    forced    = [e for e in episodes if e.get("termination") == "forced_sell"]
    wins      = [p for p in profits if p > 0]
    losses    = [p for p in profits if p <= 0]
    no_buy_ep = [e for e in episodes if e.get("buy_step") is None]

    print(f"\n  EPISODES")
    print(f"    Total episodes       : {len(episodes):,}")
    print(f"    Episodes with no buy : {len(no_buy_ep):,}  ({len(no_buy_ep)/len(episodes)*100:.1f}%)")
    print(f"    Forced sells         : {len(forced):,}  ({len(forced)/len(episodes)*100:.1f}%)")

    print(f"\n  PROFITABILITY")
    print(f"    Win rate             : {len(wins)/len(episodes)*100:.1f}%  "
          f"({len(wins):,} wins / {len(losses):,} losses)")
    print(f"    Mean profit          : {np.mean(profits):+.4f}%")
    print(f"    Median profit        : {np.median(profits):+.4f}%")
    print(f"    Profit std dev       : {np.std(profits):.4f}%")
    print(f"    Best trade           : {max(profits):+.3f}%")
    print(f"    Worst trade          : {min(profits):+.3f}%")
    if wins and losses:
        print(f"    Avg win size         : {np.mean(wins):+.4f}%")
        print(f"    Avg loss size        : {np.mean(losses):+.4f}%")
        print(f"    Win/loss ratio       : {abs(np.mean(wins)/np.mean(losses)):.2f}x")

    valid_steps = [s for s in steps if s >= 0]
    print(f"\n  EPISODE LENGTH")
    print(f"    Mean steps           : {np.mean(valid_steps):.1f}")
    print(f"    Min / Max            : {min(valid_steps)} / {max(valid_steps)}")

    nb_values = [v for v in not_buy_s if v is not None]
    if nb_values:
        print(f"\n  NOT_BUY BEHAVIOUR")
        print(f"    Mean NOT_BUY steps/ep : {np.mean(nb_values):.1f}")
        print(f"    Median NOT_BUY steps  : {np.median(nb_values):.1f}")
        print(f"    Episodes with 0 nb    : {sum(1 for v in nb_values if v==0):,}  "
              f"({sum(1 for v in nb_values if v==0)/len(nb_values)*100:.1f}%)")
        nb_frac = np.mean(nb_values) / max(np.mean(valid_steps), 1) * 100
        print(f"    NOT_BUY fraction      : {nb_frac:.1f}% of episode steps")

    n = len(profits)
    print(f"\n  PROFIT TRAJECTORY (quarters)")
    for label, sl in [("Q1 (0–25%)",   profits[:n//4]),
                      ("Q2 (25–50%)",  profits[n//4:n//2]),
                      ("Q3 (50–75%)",  profits[n//2:3*n//4]),
                      ("Q4 (75–100%)", profits[3*n//4:])]:
        wr = sum(1 for p in sl if p > 0) / max(len(sl), 1) * 100
        print(f"    {label:<18} mean={np.mean(sl):+.4f}%  win={wr:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Analyse ServoTrader JSONL episode log.")
    parser.add_argument("--log",     default=None, help="Path to JSONL log (default: latest)")
    parser.add_argument("--logdir",  default=DEFAULT_LOG_DIR)
    parser.add_argument("--actions", action="store_true", help="Show action breakdown")
    args = parser.parse_args()

    log_path = args.log or find_latest_log(args.logdir)
    print(f"Log: {log_path}")
    analyse(log_path, show_actions=args.actions)


if __name__ == "__main__":
    main()