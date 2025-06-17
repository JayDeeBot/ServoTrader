import json
import pandas as pd

def load_jsonl_log(log_path):
    episodes, training_summary = [], {}
    current = None

    with open(log_path, 'r', encoding='utf-8') as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)

            if record["type"] == "episode_start":
                if current:
                    episodes.append(current)
                current = {
                    "episode": record["episode"],
                    "start_step": record["start_step"],
                    "steps": [],
                    "summary": {}
                }

            elif record["type"] == "step":
                current["steps"].append(record)

            elif record["type"] == "episode_end":
                current["summary"].update({
                    "steps": record["steps"],
                    "final_value": record["final_value"],
                    "return_pct": record["episode_return_pct"],
                })
                episodes.append(current)
                current = None

            elif record["type"] == "training_complete":
                training_summary = {
                    "total_episodes": record["total_episodes"],
                    "total_return": record["total_return_pct"],
                    # removed sim_days based on your request
                }

    return episodes, training_summary

def validate_episode(ep, lookup):
    cash = 1.0

    for step in ep["steps"]:
        action = step["action_type"]
        symbol = step.get("symbol")
        price = step.get("price")
        profit = step.get("profit_pct", 0.0)
        raw_idx = step.get("raw_index")

        # If symbol & price are present, raw_index must exist and price must match raw data
        if symbol and price is not None:
            assert raw_idx is not None, "No raw_index logged!"
            actual_price = lookup[symbol].iloc[raw_idx]["close"]
            assert abs(actual_price - price) < 1e-3, (
                f"Price mismatch at epi {ep['episode']}, step {step['offset']}: "
                f"{actual_price} vs {price}"
            )

        # If action is sell, update portfolio value
        if action == "sell":
            cash *= (1 + float(profit) / 100)

    final = round(cash, 4)
    expected = round(ep["summary"]["final_value"], 4)
    assert abs(final - expected) < 1e-3, (
        f"Final value mismatch epi {ep['episode']}: {final} != {expected}"
    )
    ret = (final - 1) * 100
    assert abs(ret - ep["summary"]["return_pct"]) < 0.1, (
        f"Return pct mismatch epi {ep['episode']}: {ret} != {ep['summary']['return_pct']}"
    )

def validate_training_summary(eps, summary, timeout):
    tot = sum(ep["summary"]["return_pct"] for ep in eps)
    # Only check episodes count and total return
    assert abs(summary["total_episodes"] - len(eps)) < 1, "Episode count mismatch"
    assert abs(summary["total_return"] - tot) < 0.5, "Total return mismatch"
    # sim_days check removed

def run_log_validation_test(log_path, data_path, timeout=15):
    episodes, summ = load_jsonl_log(log_path)
    df = pd.read_csv(data_path)
    df["symbol"] = df["symbol"].str.strip()
    lookup = {s: df[df.symbol == s].reset_index(drop=True) for s in df.symbol.unique()}

    for ep in episodes:
        validate_episode(ep, lookup)
    validate_training_summary(episodes, summ, timeout)
    print("✅ All JSONL log checks passed!")

if __name__ == "__main__":
    run_log_validation_test(
        log_path="/home/jarred/git/ServoTrader/logs/env_log_20250616_130039.jsonl",
        data_path="/home/jarred/git/ServoTrader/data/historical_crypto_data.csv",
        timeout=15,
    )