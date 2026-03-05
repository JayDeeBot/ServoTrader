"""
analyze_episode_log.py

Reads the ServoTrader JSONL episode log and prints a clean summary of
training performance. Replaces the inline bash snippet used in earlier
analysis sessions.

Usage
-----
    # Analyse the most recent log automatically
    python3 scripts/analyze_episode_log.py

    # Analyse a specific log file
    python3 scripts/analyze_episode_log.py --log /path/to/btc5m_env_20240301_120000.jsonl

    # Show action breakdown (NOT_BUY / BUY / HOLD / SELL counts)
    python3 scripts/analyze_episode_log.py --actions

Author: Jarred Deluca
Project: ServoTrader
License: MIT
"""

import os
import json
import glob
import argparse
import numpy as np
from datetime import datetime

DEFAULT_LOG_DIR = "/home/jarred/git/ServoTrader/logs"


def find_latest_log(log_dir: str) -> str:
    pattern = os.path.join(log_dir, "btc5m_env_*.jsonl")
    logs = sorted(glob.glob(pattern))
    if not logs:
        raise FileNotFoundError(f"No JSONL logs found in {log_dir}")
    return logs[-1]


def analyse(log_path: str, show_actions: bool = False):
    SEP = "=" * 62

    episodes   = []
    all_steps  = []

    with open(log_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec["type"] == "episode_end":
                episodes.append(rec)
            elif rec["type"] == "step" and show_actions:
                all_steps.append(rec)

    if not episodes:
        print("No episode_end records found — training may not have completed any trades yet.")
        return

    profits      = [e["profit_pct"]  for e in episodes]
    steps        = [e["steps"]       for e in episodes]
    not_buy_s    = [e.get("not_buy_steps", None) for e in episodes]
    forced       = [e for e in episodes if e.get("termination") == "forced_sell"]
    wins         = [p for p in profits if p > 0]
    losses       = [p for p in profits if p <= 0]
    no_buy_ep    = [e for e in episodes if e.get("buy_step") is None]

    # NOT_BUY data — only available in v4+ logs
    has_nb_data  = any(v is not None for v in not_buy_s)
    nb_values    = [v for v in not_buy_s if v is not None]

    print(f"\n{SEP}")
    print(f"  EPISODE LOG ANALYSIS")
    print(f"  {log_path}")
    print(SEP)

    print(f"\n  {'EPISODES':}")
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

    print(f"\n  EPISODE LENGTH")
    valid_steps = [s for s in steps if s >= 0]
    if len(valid_steps) < len(steps):
        print(f"    ⚠  {len(steps)-len(valid_steps)} episodes had negative step counts (wrap bug)")
    print(f"    Mean steps           : {np.mean(valid_steps):.1f}")
    print(f"    Median steps         : {np.median(valid_steps):.1f}")
    print(f"    Min / Max            : {min(valid_steps)} / {max(valid_steps)}")

    if has_nb_data:
        print(f"\n  NOT_BUY BEHAVIOUR  (v4+ logs)")
        print(f"    Mean NOT_BUY steps/ep : {np.mean(nb_values):.1f}")
        print(f"    Median NOT_BUY steps  : {np.median(nb_values):.1f}")
        print(f"    Max NOT_BUY steps     : {max(nb_values)}")
        print(f"    Episodes with 0 nb    : {sum(1 for v in nb_values if v==0):,}  "
              f"({sum(1 for v in nb_values if v==0)/len(nb_values)*100:.1f}%)")
        mean_ep_len = np.mean(valid_steps) if valid_steps else 1
        print(f"    NOT_BUY fraction      : {np.mean(nb_values)/max(mean_ep_len,1)*100:.1f}% of episode steps")
    else:
        print(f"\n  NOT_BUY BEHAVIOUR")
        print(f"    ℹ  not_buy_steps field not present (pre-v4 log)")
        print(f"    Count NOT_BUY from step records instead (use --actions)")

    # Profit trajectory — split into quarters
    n = len(profits)
    print(f"\n  PROFIT TRAJECTORY (quarters)")
    for label, sl in [("Q1 (0–25%)",   profits[:n//4]),
                      ("Q2 (25–50%)",  profits[n//4:n//2]),
                      ("Q3 (50–75%)",  profits[n//2:3*n//4]),
                      ("Q4 (75–100%)", profits[3*n//4:])]:
        wr = sum(1 for p in sl if p > 0) / max(len(sl), 1) * 100
        print(f"    {label:<18} mean={np.mean(sl):+.4f}%  win={wr:.1f}%")

    if show_actions and all_steps:
        print(f"\n  ACTION BREAKDOWN (from {len(all_steps):,} logged steps)")
        from collections import Counter
        counts = Counter(s["action_str"] for s in all_steps)
        total  = sum(counts.values())
        for action in ["not_buy", "buy", "hold", "sell", "sell_forced"]:
            c = counts.get(action, 0)
            print(f"    {action:<14} : {c:>8,}  ({c/total*100:5.1f}%)")

    print(f"\n{SEP}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyse ServoTrader JSONL episode log.")
    parser.add_argument("--log",     default=None, help="Path to JSONL log file (default: latest)")
    parser.add_argument("--logdir",  default=DEFAULT_LOG_DIR, help="Log directory to search")
    parser.add_argument("--actions", action="store_true", help="Show action breakdown from step records")
    args = parser.parse_args()

    log_path = args.log or find_latest_log(args.logdir)
    print(f"Log: {log_path}")
    analyse(log_path, show_actions=args.actions)


if __name__ == "__main__":
    main()