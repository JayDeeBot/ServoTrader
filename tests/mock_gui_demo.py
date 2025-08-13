#!/usr/bin/env python3
"""
mock_gui_demo.py

Wizard-of-Oz GUI demo for LiveEnvRenderer:
- Produces MANY episodes quickly to test the Session Stats scrolled log.
- Writes JSONL exactly like the live env so the GUI can render newest-first.
- Keeps appending while the GUI is open.

Run:
  /bin/python3.11 /home/jarred/git/ServoTrader/scripts/mock_gui_demo.py
"""

import os
import json
import time
import threading
import random
from datetime import datetime
import pandas as pd
import sys

# --- repo import path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gui.live_renderer import LiveEnvRenderer


class MockTrader:
    def __init__(self):
        self.max_buy_limit = 1e6
        self.daily_cashout_percent = 0.0

    def set_max_buy_limit(self, v: float): self.max_buy_limit = float(v)
    def set_daily_cashout_percent(self, v: float): self.daily_cashout_percent = float(v)
    def cash_out_to_bank(self): print("[MockTrader] cash_out_to_bank called")


class MockEnv:
    """
    Minimal mock that implements all attributes the GUI reads and writes a
    long JSONL trail to test scrolling + newest-first rendering.
    """
    def __init__(self):
        # Paths
        self.root_dir = "/home/jarred/git/ServoTrader"
        os.makedirs(os.path.join(self.root_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.root_dir, "servo_trader", "config"), exist_ok=True)

        # Codes file (GUI reads this from disk)
        self.json_path = os.path.join(self.root_dir, "servo_trader", "config", "mock_crypto_codes.json")
        self._write_codes_file([
            "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","ADAUSDT",
            "AVAXUSDT","DOGEUSDT","BNBUSDT","LINKUSDT","LTCUSDT"
        ])

        # Logging
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.root_dir, "logs", f"env_log_{stamp}.jsonl")
        self.episode_log = []  # in-memory buffer we also persist

        # Session / episode state
        self.episode_counter = 0
        self.current_step = 0
        self.timeout_steps = 12          # short episodes → fast log growth
        self.feature_window = self.timeout_steps
        self.history_window = 5

        # Position state
        self.active_crypto_index = None
        self.active_crypto_code = None
        self.buy_price = 0.0
        self.sell_price = 0.0

        # Profit and balances
        self.current_ep_profit_decimal = 0.0
        self.episode_profit_decimal = 0.0
        self.total_profit_decimal = 0.0
        self.estimated_balance_usdt = 10000.00

        # Misc
        self.current_action = "HOLD"
        self.seconds_left = 0.0
        self.crypto_codes = self._load_codes()
        self.trader = MockTrader()

        # raw_lookup: per-symbol DataFrame with at least a 'close' column
        self.raw_lookup = {sym: pd.DataFrame({"close": [100.0]}) for sym in self.crypto_codes}

        # Start simulation threads
        self._alive = True
        threading.Thread(target=self._simulate_episodes, daemon=True).start()
        threading.Thread(target=self._tick_seconds_left, daemon=True).start()
        threading.Thread(target=self._occasionally_swap_code, daemon=True).start()

    # ---- Helpers for codes file ----
    def _write_codes_file(self, codes):
        with open(self.json_path, "w") as f:
            json.dump({"crypto_codes": codes}, f, indent=2)

    def _load_codes(self):
        try:
            with open(self.json_path, "r") as f:
                data = json.load(f)
            return data["crypto_codes"] if isinstance(data, dict) else data
        except Exception:
            return []

    # ---- Smooth countdown display ----
    def _tick_seconds_left(self):
        # emulate a ~60s cadence visually (GUI shows seconds)
        start = time.monotonic()
        while self._alive:
            period = 60.0
            elapsed = time.monotonic() - start
            self.seconds_left = max(0.0, period - (elapsed % period))
            time.sleep(0.25)

    # ---- Occasionally mutate codes to test the Codes tab meta ----
    def _occasionally_swap_code(self):
        while self._alive:
            time.sleep(8.0)
            codes = self._load_codes()
            # swap last code once in a while
            bump = random.choice(["ARBUSDT", "NEARUSDT", "ATOMUSDT", "FILUSDT"])
            if codes and codes[-1] != bump:
                codes[-1] = bump
                self._write_codes_file(codes)

    # ---- Main episode simulator ----
    def _simulate_episodes(self):
        """
        Generate many short episodes:
          - BUY on step 1
          - HOLD several steps (fake price drift)
          - SELL at the end
          - Append full JSONL trail continuously
        This gives the Session tab lots of data to render & scroll.
        """
        # pre-bake 12 episodes fast, then keep going slowly
        fast_bootstrap = 12
        while self._alive:
            self._new_episode()

            # short pause so GUI can show "start"
            time.sleep(0.3)

            # BUY early
            self.active_crypto_index = 0
            self.active_crypto_code = self.crypto_codes[self.active_crypto_index]
            self.buy_price = float(self.raw_lookup[self.active_crypto_code].iloc[-1]["close"])
            self.current_action = f"BUY: {self.active_crypto_code}"
            self._log_step(action=1, action_type="buy", reward=0.0,
                           symbol=self.active_crypto_code, price=self.buy_price)
            self.current_step += 1
            self._flush_log()

            # HOLD a few steps with price drift
            hold_steps = self.timeout_steps - 2
            for _ in range(hold_steps):
                time.sleep(0.2 if fast_bootstrap > 0 else 1.0)
                self.current_action = "HOLD"

                # drift price
                df = self.raw_lookup[self.active_crypto_code]
                last = float(df.iloc[-1]["close"])
                drift = random.uniform(-0.6, 0.9)  # mild upward bias
                new_price = max(1e-6, last * (1.0 + drift / 100.0))
                df.loc[len(df)] = {"close": new_price}

                # update current ep profit
                if self.buy_price:
                    self.current_ep_profit_decimal = (new_price - self.buy_price) / self.buy_price

                self._log_step(
                    action=0, action_type="hold", reward=float(self.current_ep_profit_decimal),
                    symbol=self.active_crypto_code, price=new_price,
                    profit_pct=self.current_ep_profit_decimal * 100.0, buy_price=self.buy_price
                )
                self.current_step += 1
                self._flush_log()

            # SELL end
            self.sell_price = float(self.raw_lookup[self.active_crypto_code].iloc[-1]["close"])
            self.current_action = f"SELL: {self.active_crypto_code}"
            if self.buy_price:
                self.episode_profit_decimal = (self.sell_price - self.buy_price) / self.buy_price
            else:
                self.episode_profit_decimal = 0.0
            self.total_profit_decimal = (1.0 + self.total_profit_decimal) * (1.0 + self.episode_profit_decimal) - 1.0

            self._log_step(
                action=999, action_type="sell", reward=float(self.episode_profit_decimal),
                symbol=self.active_crypto_code, price=self.sell_price,
                profit_pct=self.episode_profit_decimal * 100.0,
                buy_price=self.buy_price, sell_price=self.sell_price
            )
            self._flush_log()

            # episode_end summary
            summary = {
                "type": "episode_end",
                "episode": self.episode_counter,
                "episode_return_pct": self.episode_profit_decimal * 100.0,
                "total_profit_pct": self.total_profit_decimal * 100.0,
                "termination": "sell",
                "steps": int(self.current_step + 1),
                "timestamp": datetime.now().isoformat()
            }
            self.episode_log.append(summary)
            self._flush_log()

            # small equity wobble
            self.estimated_balance_usdt *= (1.0 + random.uniform(-0.06, 0.10) / 100.0)

            # brief pause before next episode
            time.sleep(0.4 if fast_bootstrap > 0 else 2.0)

            fast_bootstrap = max(0, fast_bootstrap - 1)

    def _new_episode(self):
        self.episode_counter += 1
        self.current_step = 0
        self.active_crypto_index = None
        self.active_crypto_code = None
        self.buy_price = 0.0
        self.sell_price = 0.0
        self.current_ep_profit_decimal = 0.0
        self.episode_profit_decimal = 0.0
        self.current_action = "HOLD"
        self.seconds_left = 0.0

        # Ensure raw_lookup has frames for all codes
        self.crypto_codes = self._load_codes()
        for sym in self.crypto_codes:
            if sym not in self.raw_lookup:
                self.raw_lookup[sym] = pd.DataFrame({"close": [100.0]})

        start_rec = {
            "type": "episode_start",
            "episode": self.episode_counter,
            "start_timestamp": datetime.now().isoformat(),
            "cash_balance": 10000.0
        }
        self.episode_log.append(start_rec)
        self._flush_log()

    def _log_step(self, *, action, action_type, reward,
                  symbol=None, price=None, profit_pct=None,
                  buy_price=None, sell_price=None):
        step_no = int(self.current_step + 1)  # 1-based
        rec = {
            "type": "step",
            "episode": self.episode_counter,
            "step": step_no,
            "action": int(action),
            "action_type": str(action_type),
            "reward": float(reward),
        }
        if symbol is not None: rec["symbol"] = symbol
        if price is not None: rec["price"] = float(price)
        if profit_pct is not None: rec["profit_pct"] = float(profit_pct)
        if buy_price is not None: rec["buy_price"] = float(buy_price)
        if sell_price is not None: rec["sell_price"] = float(sell_price)

        self.episode_log.append(rec)

    def _flush_log(self):
        with open(self.log_path, "w") as f:
            for rec in self.episode_log:
                f.write(json.dumps(rec) + "\n")


def main():
    env = MockEnv()
    gui = LiveEnvRenderer(env)  # or LiveEnvRenderer(env, logo_path="/path/to/logo.png")
    gui.run()


if __name__ == "__main__":
    main()
