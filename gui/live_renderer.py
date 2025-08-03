"""
live_renderer.py

A GUI-based live renderer for the LiveCryptoTradingEnv environment using Tkinter.

This module provides a real-time visual interface for monitoring trading activity,
including the current action, held asset, PnL, portfolio value, and recent episode
summaries. The interface auto-refreshes every 2 seconds and is designed to run
concurrently with a live trading agent using reinforcement learning.

Dependencies:
- tkinter
- datetime
- json
- os

Author: Jarred Deluca
Created: 2025
License: MIT
"""

import tkinter as tk
from tkinter import ttk
from datetime import datetime
import json
import os


class LiveEnvRenderer:
    """
    Tkinter GUI renderer for visualizing live trading performance from a LiveCryptoTradingEnv instance.

    This class creates a compact real-time dashboard that displays:
        - Current step, episode, and timestamp
        - Current action and reward
        - Held symbol and profit/loss
        - Portfolio value and cumulative profit
        - Last 5 episode summaries
    """

    def __init__(self, env):
        """
        Initializes the LiveEnvRenderer GUI.

        Args:
            env (LiveCryptoTradingEnv): The active environment to observe and extract values from.
        """
        self.env = env
        self.root = tk.Tk()
        self.root.title("Live Trading Environment Monitor")

        # Configure window appearance
        self.root.geometry("600x500")
        self.root.configure(bg="#1e1e1e")
        self.labels = {}  # Holds the reference to value labels

        # Build the GUI layout and start auto-updating
        self._create_widgets()
        self._update_gui()

    def _create_widgets(self):
        """
        Creates and arranges all labels, layout separators, and episode summary display boxes.
        """

        def make_label(key, row):
            """
            Helper function to create a key-value row in the GUI.

            Args:
                key (str): The field name to display.
                row (int): The row index in the grid.
            """
            # Label for the field name
            label = tk.Label(self.root, text=key + ": ", font=("Courier", 12),
                             anchor="w", bg="#1e1e1e", fg="#d0d0d0")
            label.grid(row=row, column=0, sticky="w", padx=10)

            # Corresponding label for the value
            value = tk.Label(self.root, text="...", font=("Courier", 12),
                             anchor="w", bg="#1e1e1e", fg="#ffffff")
            value.grid(row=row, column=1, sticky="w")

            # Store the reference
            self.labels[key] = value

        # Keys to display in the top section
        keys = [
            "Time", "Step", "Episode", "Action", "Reward",
            "Held Symbol", "Buy Price", "Current Price", "PnL %",
            "Portfolio Value", "Session Profit %"
        ]
        for i, key in enumerate(keys):
            make_label(key, i)

        # Add a horizontal separator
        ttk.Separator(self.root, orient="horizontal").grid(
            row=len(keys), columnspan=2, sticky="ew", pady=10
        )

        # Label for episode summaries
        self.episode_summary_label = tk.Label(
            self.root,
            text="Last 5 Episodes",
            font=("Courier", 12, "bold"),
            bg="#1e1e1e",
            fg="#00ff00"
        )
        self.episode_summary_label.grid(row=len(keys) + 1, column=0, columnspan=2)

        # Text box for displaying episode history
        self.episode_summary_text = tk.Text(
            self.root,
            width=70,
            height=8,
            bg="#121212",
            fg="#c0ffc0",
            font=("Courier", 10)
        )
        self.episode_summary_text.grid(row=len(keys) + 2, column=0, columnspan=2, padx=10)

    def _update_gui(self):
        """
        Refreshes the GUI by updating all value labels and reloading recent episode summaries.
        Automatically scheduled every 2 seconds via Tkinter's `after()` method.
        """
        env = self.env
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Default/fallback values
        action = "N/A"
        reward = "N/A"
        symbol = "-"
        buy_price = "-"
        current_price = "-"
        pnl_pct = "-"

        # Pull last recorded step data from episode log
        if env.episode_log and env.episode_log[-1]['type'] == 'step':
            entry = env.episode_log[-1]
            action = f"{entry.get('action_type', 'N/A').upper()} {entry.get('symbol', '')}"
            reward = f"{entry.get('reward', 0.0):+.4f}"

        # If a crypto is currently held, fetch latest price and compute profit
        if env.active_crypto_index is not None:
            symbol = env.crypto_codes[env.active_crypto_index]
            buy_price = f"{env.buy_price:.4f}"
            try:
                current_price_val = float(env.raw_lookup[symbol].iloc[-1]["close"])
                pnl = ((current_price_val - env.buy_price) / env.buy_price) * 100 if env.buy_price else 0.0
                pnl_pct = f"{pnl:+.2f}%"
                current_price = f"{current_price_val:.4f}"
            except:
                current_price = "-"
                pnl_pct = "-"

        # Populate all live stats
        updates = {
            "Time": now,
            "Step": str(env.current_step),
            "Episode": str(env.episode_counter + 1),
            "Action": action,
            "Reward": reward,
            "Held Symbol": symbol,
            "Buy Price": buy_price,
            "Current Price": current_price,
            "PnL %": pnl_pct,
            "Portfolio Value": f"${env.portfolio_value:.2f}",
            "Session Profit %": f"{env.total_profit_percent:+.2f}%"
        }

        # Apply updated values to GUI labels
        for key, val in updates.items():
            self.labels[key].config(text=val)

        # Refresh episode summaries
        self._update_episode_summaries()

        # Schedule the next update
        self.root.after(2000, self._update_gui)

    def _update_episode_summaries(self):
        """
        Loads the last 5 episode summaries from the log file and updates the GUI text box.
        """
        self.episode_summary_text.delete(1.0, tk.END)
        try:
            with open(self.env.log_path, "r") as f:
                lines = [json.loads(line) for line in f.readlines()]
                summaries = [l for l in lines if l.get("type") == "episode_end"]
                for ep in summaries[-5:][::-1]:
                    line = f"[Ep {ep['episode']:>3}] Return: {ep['episode_return_pct']:+.2f}%, Steps: {ep['steps']}, End: {ep['termination']}\n"
                    self.episode_summary_text.insert(tk.END, line)
        except Exception as e:
            self.episode_summary_text.insert(tk.END, f"Unable to load summaries: {str(e)}\n")

    def run(self):
        """
        Starts the Tkinter main loop, which blocks until the window is closed.
        """
        self.root.mainloop()
