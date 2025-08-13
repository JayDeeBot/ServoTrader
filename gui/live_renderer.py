"""
live_renderer.py

ServoTrader GUI for monitoring and controlling a live trading session.
Adds a header with logo, a tabbed interface (Live Trading, Session Stats,
Current Crypto Codes, Settings, License), and a bottom "End Trading Session" button.

Author: Jarred Deluca
Created: 2025
License: MIT
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
from threading import Thread
from datetime import datetime, timedelta
import time
from tkinter import scrolledtext

class LiveEnvRenderer:
    """
    Tkinter GUI renderer for visualizing live trading performance from a LiveCryptoTradingEnv instance.
    """

    def __init__(self, env, logo_path: str | None = "None"):
        """
        Args:
            env: LiveCryptoTradingEnv instance
            logo_path: optional path to a PNG logo to show in the header
        """
        self.env = env
        self.logo_path = logo_path
        self._last_codes = None         # track code changes
        self._session_start_dt = datetime.now()  # GUI session start (used for days elapsed)
        self._codes_json_path = getattr(env, "json_path", None)

        # --- Root window ---
        self.root = tk.Tk()
        self.root.title("ServoTrader")
        self.root.geometry("880x640")
        self.root.configure(bg="#1e1e1e")

        # --- Header ---
        self._build_header()

        # --- Notebook (tabs) ---
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=(0,8))

        # Create tab frames
        self.tab_live = ttk.Frame(self.notebook)
        self.tab_session = ttk.Frame(self.notebook)
        self.tab_codes = ttk.Frame(self.notebook)
        self.tab_settings = ttk.Frame(self.notebook)
        self.tab_license = ttk.Frame(self.notebook)

        # Add tabs in reverse so they appear right→left as requested
        self.notebook.add(self.tab_license, text="License")
        self.notebook.add(self.tab_settings, text="Settings")
        self.notebook.add(self.tab_codes, text="Current Codes")
        self.notebook.add(self.tab_session, text="Session Stats")
        self.notebook.add(self.tab_live, text="Live Trading")

        # --- Build each tab ---
        self._build_live_tab()
        self._build_session_tab()
        self._build_codes_tab()
        self._build_settings_tab()
        self._build_license_tab()

        # --- Bottom action bar ---
        self._build_bottom_bar()

        # --- Start periodic updates ---
        self._update_gui()

        # --- Daily cashout scheduler (kept from your version) ---
        def schedule_cashout():
            while True:
                now = datetime.now()
                target = now.replace(hour=9, minute=0, second=0, microsecond=0)
                if now >= target:
                    target += timedelta(days=1)
                time.sleep((target - now).total_seconds())
                print("💸 Running daily cashout to bank...")
                try:
                    self.env.trader.cash_out_to_bank()
                except Exception as e:
                    print(f"[GUI] Cashout error: {e}")

        Thread(target=schedule_cashout, daemon=True).start()

    # ---------- HEADER ----------

    def _build_header(self):
        header = tk.Frame(self.root, bg="#151515")
        header.pack(fill="x", padx=0, pady=0)

        left = tk.Frame(header, bg="#151515")
        left.pack(side="left", padx=12, pady=8)

        # Try to load logo if provided
        self._logo_img = None
        if self.logo_path and os.path.exists(self.logo_path):
            try:
                self._logo_img = tk.PhotoImage(file=self.logo_path)
            except Exception:
                self._logo_img = None

        if self._logo_img is not None:
            logo_label = tk.Label(left, image=self._logo_img, bg="#151515")
            logo_label.pack(side="left", padx=(0,8))

        title_label = tk.Label(
            left, text="ServoTrader",
            font=("Helvetica", 18, "bold"),
            bg="#151515", fg="#eaeaea"
        )
        title_label.pack(side="left")

        # Right-aligned timestamp
        self.header_time = tk.Label(
            header, text="--:--:--",
            font=("Courier", 12), bg="#151515", fg="#8dd1ff"
        )
        self.header_time.pack(side="right", padx=12)

    # ---------- LIVE TAB ----------

    def _build_live_tab(self):
        # Grid config
        for c in range(2):
            self.tab_live.grid_columnconfigure(c, weight=1)

        # Helper to create a key/value row
        self.live_labels = {}
        def kv(row, key):
            k = tk.Label(self.tab_live, text=key + ":", font=("Courier", 12),
                         anchor="w")
            k.grid(row=row, column=0, sticky="w", padx=14, pady=4)
            v = tk.Label(self.tab_live, text="...", font=("Courier", 12),
                         anchor="w", fg="#2705c0")
            v.grid(row=row, column=1, sticky="w", padx=4, pady=4)
            self.live_labels[key] = v

        r = 0
        kv(r, "Episode"); r+=1
        kv(r, "Step"); r+=1
        kv(r, "Timesteps Left"); r+=1
        kv(r, "Countdown (s)"); r+=1
        ttk.Separator(self.tab_live, orient="horizontal").grid(row=r, columnspan=2, sticky="ew", pady=6); r+=1
        kv(r, "Current Action"); r+=1
        kv(r, "Held Symbol"); r+=1
        kv(r, "Buy Price"); r+=1
        kv(r, "Current Price"); r+=1
        kv(r, "Sell Price"); r+=1
        ttk.Separator(self.tab_live, orient="horizontal").grid(row=r, columnspan=2, sticky="ew", pady=6); r+=1
        kv(r, "Curr. Ep Profit %"); r+=1
        kv(r, "Episode Profit %"); r+=1
        kv(r, "Estimated Balance (USDT)"); r+=1

    # ---------- SESSION TAB ----------

    def _build_session_tab(self):
        for c in range(2):
            self.tab_session.grid_columnconfigure(c, weight=1)

        self.session_labels = {}
        def kv(row, key):
            k = tk.Label(self.tab_session, text=key + ":", font=("Courier", 12))
            k.grid(row=row, column=0, sticky="w", padx=14, pady=4)
            v = tk.Label(self.tab_session, text="...", font=("Courier", 12), fg="black")
            v.grid(row=row, column=1, sticky="w", padx=4, pady=4)
            self.session_labels[key] = v

        r = 0
        kv(r, "Episodes Completed"); r+=1
        kv(r, "Total Profit %"); r+=1
        kv(r, "Days Since Start"); r+=1
        kv(r, "Avg Profit % / Day"); r+=1

        ttk.Separator(self.tab_session, orient="horizontal").grid(row=r, columnspan=2, sticky="ew", pady=8); r+=1

        tk.Label(self.tab_session, text="Session Log (newest first)", font=("Courier", 12, "bold")).grid(
            row=r, column=0, columnspan=2, sticky="w", padx=12
        )
        r += 1

        # Scrolled text to browse all episodes
        self.session_text = scrolledtext.ScrolledText(
            self.tab_session, width=100, height=18, bg="#121212", fg="#c0ffc0", font=("Courier", 10), wrap="none"
        )
        self.session_text.grid(row=r, column=0, columnspan=2, padx=12, pady=(4,10), sticky="nsew")
        self.tab_session.grid_rowconfigure(r, weight=1)

        # Cache the last mtime so we only re-render when file changes
        self._log_mtime = None

    # ---------- CODES TAB ----------

    def _build_codes_tab(self):
        for c in range(2):
            self.tab_codes.grid_columnconfigure(c, weight=1)

        tk.Label(self.tab_codes, text="Current Crypto Codes", font=("Courier", 12, "bold")).grid(
            row=0, column=0, sticky="w", padx=12, pady=(10,4)
        )
        self.codes_list = tk.Text(self.tab_codes, width=60, height=18, bg="#111", fg="#eee", font=("Courier", 11))
        self.codes_list.grid(row=1, column=0, padx=12, sticky="nsew")
        self.tab_codes.grid_rowconfigure(1, weight=1)

        right = tk.Frame(self.tab_codes)
        right.grid(row=1, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)

        self.codes_meta = {}
        def kv(parent, row, key):
            k = tk.Label(parent, text=key + ":", font=("Courier", 11))
            k.grid(row=row, column=0, sticky="w", padx=6, pady=2)
            v = tk.Label(parent, text="...", font=("Courier", 11), fg="#2705c0")
            v.grid(row=row, column=1, sticky="w", padx=6, pady=2)
            self.codes_meta[key] = v

        kv(right, 0, "JSON Path")
        kv(right, 1, "Last Modified")
        kv(right, 2, "Last Change Detected")

    # ---------- SETTINGS TAB ----------

    def _build_settings_tab(self):
        for c in range(2):
            self.tab_settings.grid_columnconfigure(c, weight=1)

        # Features list
        tk.Label(self.tab_settings, text="Observed Features (13)", font=("Courier", 12, "bold")).grid(
            row=0, column=0, sticky="w", padx=12, pady=(10,4)
        )
        features = [
            "open", "high", "low", "close", "vwap", "volume", "count",
            "recent_return", "volatility", "price_position", "volume_surge", "trend_slope", "moving_avg"
        ]
        feat_text = tk.Text(self.tab_settings, width=40, height=8, bg="#111", fg="#eee", font=("Courier", 11))
        feat_text.grid(row=1, column=0, padx=12, sticky="w")
        feat_text.insert("1.0", "\n".join(features))
        feat_text.configure(state="disabled")

        # Runtime settings (readonly)
        tk.Label(self.tab_settings, text="Runtime Settings", font=("Courier", 12, "bold")).grid(
            row=0, column=1, sticky="w", padx=12, pady=(10,4)
        )

        self.settings_labels = {}
        def kv(parent, row, key):
            k = tk.Label(parent, text=key + ":", font=("Courier", 11))
            k.grid(row=row, column=0, sticky="w", padx=6, pady=2)
            v = tk.Label(parent, text="...", font=("Courier", 11), fg="#2705c0")
            v.grid(row=row, column=1, sticky="w", padx=6, pady=2)
            self.settings_labels[key] = v

        right = tk.Frame(self.tab_settings)
        right.grid(row=1, column=1, sticky="nw", padx=6)
        kv(right, 0, "timeout_steps")
        kv(right, 1, "feature_window")
        kv(right, 2, "history_window")

        ttk.Separator(self.tab_settings, orient="horizontal").grid(row=2, column=0, columnspan=2, sticky="ew", pady=8)

        # Editable trader limits
        tk.Label(self.tab_settings, text="Trading Limits", font=("Courier", 12, "bold")).grid(
            row=3, column=0, sticky="w", padx=12, pady=(2,4)
        )

        form = tk.Frame(self.tab_settings)
        form.grid(row=4, column=0, sticky="w", padx=12)

        tk.Label(form, text="Max Buy Limit ($):", font=("Courier", 11)).grid(row=0, column=0, sticky="w", padx=(0,6), pady=2)
        self.buy_limit_entry = tk.Entry(form, font=("Courier", 11), width=12)
        self.buy_limit_entry.grid(row=0, column=1, sticky="w", pady=2)

        tk.Label(form, text="Daily Cashout (%):", font=("Courier", 11)).grid(row=1, column=0, sticky="w", padx=(0,6), pady=2)
        self.cashout_entry = tk.Entry(form, font=("Courier", 11), width=12)
        self.cashout_entry.grid(row=1, column=1, sticky="w", pady=2)

        self.status_label = tk.Label(self.tab_settings, text="", font=("Courier", 11), fg="#00ff00")
        self.status_label.grid(row=5, column=0, sticky="w", padx=12, pady=(4,2))

        def apply_trading_limits():
            try:
                buy_limit = float(self.buy_limit_entry.get())
                cashout_percent = float(self.cashout_entry.get())
                if hasattr(self.env, 'trader'):
                    self.env.trader.set_max_buy_limit(buy_limit)
                    self.env.trader.set_daily_cashout_percent(cashout_percent)
                    self.status_label.config(text="✅ Limits applied successfully!", fg="#00ff00")
                else:
                    self.status_label.config(text="❌ Trader not available in environment.", fg="#ff3333")
            except ValueError:
                self.status_label.config(text="❌ Invalid input. Please enter numeric values.", fg="#ff3333")
            self.root.after(5000, lambda: self.status_label.config(text=""))

        tk.Button(self.tab_settings, text="Apply Limits", font=("Courier", 11),
                  command=apply_trading_limits, bg="#00aa00", fg="#2705c0").grid(
            row=6, column=0, sticky="w", padx=12, pady=(2,8)
        )

    # ---------- LICENSE TAB ----------

    def _build_license_tab(self):
        txt = (
            "ServoTrader — Live Crypto Trading Environment\n"
            "Copyright (c) 2025 Jarred Deluca\n\n"
            "This software is provided under the MIT License.\n"
            "Use at your own risk. Trading cryptocurrencies involves substantial risk of loss.\n"
            "Ensure you understand the risks and have appropriate safeguards in place.\n\n"
            "By using this software you agree that the authors are not responsible for any\n"
            "losses incurred. See the repository LICENSE for the full terms."
        )
        w = tk.Text(self.tab_license, width=100, height=24, bg="#111", fg="#ddd", font=("Courier", 11), wrap="word")
        w.pack(fill="both", expand=True, padx=12, pady=12)
        w.insert("1.0", txt)
        w.configure(state="disabled")

    # ---------- BOTTOM BAR ----------

    def _build_bottom_bar(self):
        bar = tk.Frame(self.root, bg="#1e1e1e")
        bar.pack(fill="x", padx=8, pady=(0,8))

        def end_session():
            if messagebox.askyesno("End Trading Session", "Are you sure you want to stop the session and exit?"):
                self.env.trader.liquify() # Liquify all assets when ending the session
                os._exit(0)  # hard exit to stop the live loop safely from GUI thread

        btn = tk.Button(bar, text="End Trading Session", command=end_session,
                        font=("Helvetica", 12, "bold"), bg="#cc3333", fg="#2705c0")
        btn.pack(side="right", padx=4)

    # ---------- PERIODIC UPDATES ----------

    def _update_gui(self):
        try:
            self._update_header_time()
            self._update_live_tab()
            self._update_session_tab()
            self._update_codes_tab()
            self._update_settings_tab()
        except Exception as e:
            print(f"[GUI] update error: {e}")

        self.root.after(2000, self._update_gui)

    def _update_header_time(self):
        self.header_time.config(text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def _update_live_tab(self):
        env = self.env

        # Episode & steps
        episode = env.episode_counter + 1
        step = env.current_step + 1
        steps_left = max(0, getattr(env, "timeout_steps", 0) - env.current_step)
        countdown = getattr(env, "seconds_left", 0.0)

        # Action & position
        current_action = getattr(env, "current_action", "N/A")
        held_symbol = env.active_crypto_code if env.active_crypto_code else "-"
        buy_price = f"{env.buy_price:.6f}" if env.buy_price else "-"
        sell_price = f"{env.sell_price:.6f}" if env.sell_price else "-"

        # Current price (raw close) if holding
        current_price = "-"
        if env.active_crypto_index is not None and held_symbol:
            try:
                current_price_val = float(env.raw_lookup[held_symbol].iloc[-1]["close"])
                current_price = f"{current_price_val:.6f}"
            except Exception:
                current_price = "-"

        # Profit metrics
        curr_ep_pct = getattr(env, "current_ep_profit_decimal", 0.0) * 100.0
        ep_pct = getattr(env, "episode_profit_decimal", 0.0) * 100.0

        # Estimated balance
        est_bal = getattr(env, "estimated_balance_usdt", 0.0)

        updates = {
            "Episode": str(episode),
            "Step": str(step),
            "Timesteps Left": str(steps_left),
            "Countdown (s)": f"{int(countdown+0.5)}",
            "Current Action": str(current_action),
            "Held Symbol": str(held_symbol),
            "Buy Price": buy_price,
            "Current Price": current_price,
            "Sell Price": sell_price,
            "Curr. Ep Profit %": f"{curr_ep_pct:+.2f}%",
            "Episode Profit %": f"{ep_pct:+.2f}%",
            "Estimated Balance (USDT)": f"${est_bal:,.2f}",
        }

        for k, v in updates.items():
            self.live_labels[k].config(text=v)

    def _update_session_tab(self):
        env = self.env

        # Top summary fields
        episodes_completed = env.episode_counter
        total_profit_pct = getattr(env, "total_profit_decimal", 0.0) * 100.0
        days = (datetime.now() - self._session_start_dt).total_seconds() / 86400.0
        days = max(days, 1e-9)
        avg_per_day = total_profit_pct / days

        self.session_labels["Episodes Completed"].config(text=str(episodes_completed))
        self.session_labels["Total Profit %"].config(text=f"{total_profit_pct:+.2f}%")
        self.session_labels["Days Since Start"].config(text=f"{days:.2f}")
        self.session_labels["Avg Profit % / Day"].config(text=f"{avg_per_day:+.2f}%")

        # Render the JSONL only if modified
        path = getattr(env, "log_path", None)
        if not path or not os.path.exists(path):
            return

        try:
            mtime = os.path.getmtime(path)
            if getattr(self, "_log_mtime", None) is not None and mtime == self._log_mtime:
                return  # no change → keep scroll position
            self._log_mtime = mtime

            with open(path, "r") as f:
                records = [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            self.session_text.delete("1.0", tk.END)
            self.session_text.insert(tk.END, f"Unable to load log: {e}\n")
            return

        # -------- Parse episodes by scanning start → steps → end --------
        episodes = []           # list of {"meta": episode_start dict, "steps": [...], "end": episode_end dict}
        current = None

        for r in records:
            t = r.get("type")
            if t == "episode_start":
                # start a new bucket
                if current is not None:
                    episodes.append(current)  # flush any unterminated episode
                current = {"meta": r, "steps": [], "end": None}
            elif t == "step":
                if current is None:
                    # if steps appear before a start (shouldn't), create a placeholder
                    current = {"meta": {"type": "episode_start", "episode": "?"}, "steps": [], "end": None}
                current["steps"].append(r)
            elif t == "episode_end":
                if current is None:
                    # end without start (shouldn't) — treat as standalone
                    current = {"meta": {"type": "episode_start", "episode": r.get("episode","?")}, "steps": [], "end": r}
                    episodes.append(current)
                    current = None
                else:
                    current["end"] = r
                    episodes.append(current)
                    current = None

        # If file ended mid-episode, include it
        if current is not None:
            episodes.append(current)

        # -------- Newest-first display --------
        yview_before = self.session_text.yview()
        self.session_text.delete("1.0", tk.END)

        for ep in reversed(episodes):  # newest last in file → reverse to show newest first
            meta = ep["meta"] or {}
            end  = ep["end"] or {}

            ep_no = meta.get("episode") or end.get("episode") or "?"
            self.session_text.insert(tk.END, f"=== Episode {ep_no} ===\n")

            # episode_start line
            start_ts = meta.get("start_timestamp", "?")
            cash     = meta.get("cash_balance", "?")
            self.session_text.insert(tk.END, f"  start  @ {start_ts}   cash=${cash}\n")

            # steps (already in file order)
            for s in ep["steps"]:
                step = s.get("step", "?")
                act  = str(s.get("action_type","")).upper()
                sym  = s.get("symbol", "")
                price= s.get("price")
                pnl  = s.get("profit_pct")
                bp   = s.get("buy_price")
                sp   = s.get("sell_price")
                rw   = s.get("reward")

                line = f"  step {step:>3}  {act:<5} {sym:<10}"
                if price is not None: line += f"  px={price:.6f}"
                if pnl   is not None: line += f"  pnl={pnl:+.2f}%"
                if bp    is not None: line += f"  buy={bp:.6f}"
                if sp    is not None: line += f"  sell={sp:.6f}"
                if rw    is not None: line += f"  r={float(rw):+.4f}"
                self.session_text.insert(tk.END, line + "\n")

            # episode_end line (optional if not yet ended)
            if end:
                ts   = end.get("timestamp","?")
                ret  = end.get("episode_return_pct", 0.0)
                tot  = end.get("total_profit_pct", 0.0)
                steps= end.get("steps", "?")  # your current env may not include this; fine if "?"
                why  = end.get("termination","?")
                self.session_text.insert(
                    tk.END,
                    f"  end    @ {ts}   ret={ret:+.2f}%   total={tot:+.2f}%   steps={steps}   reason={why}\n"
                )
            self.session_text.insert(tk.END, "\n")

        self.session_text.yview_moveto(yview_before[0])

    def _update_codes_tab(self):
        # Load codes from JSON file on disk (so we reflect env refresh after episodes)
        codes = []
        last_mod = "-"
        last_change = "-"

        path = self._codes_json_path
        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "crypto_codes" in data:
                        codes = list(sorted(data["crypto_codes"]))
                    elif isinstance(data, list):
                        codes = list(sorted(data))
                mtime = datetime.fromtimestamp(os.path.getmtime(path))
                last_mod = mtime.strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                codes = []
                last_mod = f"error: {e}"

        # Detect change
        if self._last_codes is None:
            last_change = "N/A"
        else:
            if codes != self._last_codes:
                last_change = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._last_codes = codes

        # Render list
        self.codes_list.delete("1.0", tk.END)
        if codes:
            self.codes_list.insert("1.0", "\n".join(codes))
        else:
            self.codes_list.insert("1.0", "(no codes)")

        # Meta
        if "JSON Path" in self.codes_meta:
            self.codes_meta["JSON Path"].config(text=str(path or "-"))
        if "Last Modified" in self.codes_meta:
            self.codes_meta["Last Modified"].config(text=last_mod)
        if "Last Change Detected" in self.codes_meta:
            self.codes_meta["Last Change Detected"].config(text=last_change)

    def _update_settings_tab(self):
        # Read-only labels
        self.settings_labels["timeout_steps"].config(text=str(getattr(self.env, "timeout_steps", "-")))
        self.settings_labels["feature_window"].config(text=str(getattr(self.env, "feature_window", "-")))
        self.settings_labels["history_window"].config(text=str(getattr(self.env, "history_window", "-")))

        # Editable entries (pre-fill once if empty)
        if not self.buy_limit_entry.get():
            try:
                self.buy_limit_entry.insert(0, str(self.env.trader.max_buy_limit))
            except Exception:
                self.buy_limit_entry.insert(0, "1000000")
        if not self.cashout_entry.get():
            try:
                self.cashout_entry.insert(0, str(self.env.trader.daily_cashout_percent))
            except Exception:
                self.cashout_entry.insert(0, "0.0")

    # ---------- Public ----------

    def run(self):
        self.root.mainloop()
