#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
web_live_ppo_agent.py

ServoTrader — Live Web UI (Flask) + Live Agent runner (MaskablePPO on LiveCryptoTradingEnv)

Matches the Tk GUI layout:
- Tabs: Live Trading, Session Stats, Current Codes, Settings, License
- Same fields/labels and update cadence
- "End Trading Session" button: liquifies & exits
- Editable trading limits (max buy $, daily cashout %) with Apply

Also:
- Loads paths from env (Docker-friendly)
- Reads params.yaml + crypto_codes.json from mounted config
- Streams step heartbeats via SSE (optional visual)
- Robust JSON serialization for NumPy & datetimes

Env vars:
    MODEL_PATH=/app/models/ppo_servo_trader_squirtle.zip
    CRYPTO_CODES_PATH=/app/config/crypto_codes.json
    PARAMS_PATH=/app/config/params.yaml
    DATA_DIR=/app/data
    LOG_DIR=/app/logs
    EPISODE_TIMEOUT=120
    WEB_HOST=0.0.0.0
    WEB_PORT=8080
    DEBUG_MASK=0
"""
import os
import sys
import time
import json
import yaml
import queue
import signal
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from flask import Flask, jsonify, request, Response

# Ensure project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# RL imports
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# Custom env
from servo_trader.envs.live_crypto_trading_env import LiveCryptoTradingEnv

# -----------------------------
# Config via environment
# -----------------------------
MODEL_PATH         = os.getenv("MODEL_PATH", "/app/models/ppo_servo_trader_squirtle.zip")
CRYPTO_CODES_PATH  = os.getenv("CRYPTO_CODES_PATH", "/app/config/crypto_codes.json")
PARAMS_PATH        = os.getenv("PARAMS_PATH", "/app/config/params.yaml")
DATA_DIR           = os.getenv("DATA_DIR", "/app/data")
LOG_DIR            = os.getenv("LOG_DIR", "/app/logs")
EPISODE_TIMEOUT    = int(os.getenv("EPISODE_TIMEOUT", "120"))
WEB_HOST           = os.getenv("WEB_HOST", "0.0.0.0")
WEB_PORT           = int(os.getenv("WEB_PORT", "8080"))
DEBUG_MASK         = os.getenv("DEBUG_MASK", "0") == "1"

if os.path.isdir(PARAMS_PATH):
    PARAMS_PATH = os.path.join(PARAMS_PATH, "params.yaml")

# -----------------------------
# JSON helpers
# -----------------------------
def _json_default(o):
    if isinstance(o, np.integer):   return int(o)
    if isinstance(o, np.floating):  return float(o)
    if isinstance(o, np.bool_):     return bool(o)
    if isinstance(o, np.ndarray):   return o.tolist()
    if isinstance(o, datetime):     return o.isoformat()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

def _norm(o):
    if isinstance(o, dict):          return {k: _norm(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_norm(v) for v in o]
    if isinstance(o, np.integer):    return int(o)
    if isinstance(o, np.floating):   return float(o)
    if isinstance(o, np.bool_):      return bool(o)
    if isinstance(o, np.ndarray):    return o.tolist()
    if isinstance(o, datetime):      return o.isoformat()
    return o

# -----------------------------
# Utils
# -----------------------------
def _load_params(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def _load_crypto_codes(path: str) -> List[str]:
    with open(path, "r") as f:
        data = json.load(f)
    return data["crypto_codes"] if isinstance(data, dict) and "crypto_codes" in data else data

def _healthcheck() -> None:
    missing = []
    for p, name in [(MODEL_PATH, "MODEL_PATH"), (CRYPTO_CODES_PATH, "CRYPTO_CODES_PATH"), (PARAMS_PATH, "PARAMS_PATH")]:
        if not os.path.exists(p):
            missing.append(f"{name} not found at: {p}")
    for d, name in [(DATA_DIR, "DATA_DIR"), (LOG_DIR, "LOG_DIR")]:
        if not os.path.isdir(d):
            try:
                os.makedirs(d, exist_ok=True)
            except Exception as e:
                missing.append(f"Could not create {name} at {d}: {e}")
    if missing:
        raise FileNotFoundError("Healthcheck failed:\n" + "\n".join(missing))

# -----------------------------
# Agent service (env + model + loop)
# -----------------------------
class AgentService:
    def __init__(self):
        self.params = _load_params(PARAMS_PATH)
        # Export keys if present
        binance_cfg = self.params.get("binance", {}) or self.params.get("api", {})
        if binance_cfg:
            os.environ["BINANCE_API_KEY"] = str(binance_cfg.get("api_key", ""))
            os.environ["BINANCE_API_SECRET"] = str(binance_cfg.get("api_secret", ""))

        self.crypto_codes = _load_crypto_codes(CRYPTO_CODES_PATH)

        # Only pass kwargs that env supports
        from inspect import signature
        allowed = set(signature(LiveCryptoTradingEnv.__init__).parameters.keys())
        extra = {}
        if "data_dir" in allowed: extra["data_dir"] = DATA_DIR
        if "log_dir" in allowed:  extra["log_dir"]  = LOG_DIR
        if "params"  in allowed:  extra["params"]   = self.params

        self.env_core = LiveCryptoTradingEnv(
            crypto_codes=self.crypto_codes,
            episode_timeout=EPISODE_TIMEOUT,
            **extra
        )
        self.env = ActionMasker(self.env_core, self._action_masks)

        # Load model
        self.model = MaskablePPO.load(MODEL_PATH, env=self.env)

        # UI/loop state
        self.last_reward: float = 0.0
        self.last_action: int = -1
        self.last_mask: List[int] = []
        self.episode = 0
        self.total_steps = 0
        self.trades: List[Dict[str, Any]] = []
        self.stream_q: "queue.Queue[str]" = queue.Queue(maxsize=1024)

        # Session “start time” for web UI (mirrors Tk GUI’s _session_start_dt)
        self.session_start_dt: datetime = datetime.now()

        self._stop = threading.Event()
        self._runner = threading.Thread(target=self._loop, daemon=True)

    # Action mask identical to training
    def _action_masks(self, env):
        core = env.unwrapped
        mask = np.zeros(core.action_space.n, dtype=bool)
        if core.active_crypto_index is None:
            mask[1:core.num_cryptos + 1] = True
        else:
            mask[0] = True
            mask[core.num_cryptos + 1] = True
        if DEBUG_MASK:
            print(f"[MASK] legal: {np.where(mask)[0].tolist()}")
        self.last_mask = np.where(mask)[0].tolist()
        return mask

    def start(self):
        if not self._runner.is_alive():
            self._runner.start()

    def stop(self):
        self._stop.set()
        self._runner.join(timeout=2)

    # ------- helpers that mirror GUI computations -------
    def _current_price(self) -> Optional[float]:
        core = self.env_core
        sym = core.active_crypto_code
        if sym and getattr(core, "raw_lookup", None) is not None:
            try:
                val = core.raw_lookup[sym].iloc[-1]["close"]
                return float(val)
            except Exception:
                return None
        return None

    def live_panel(self) -> Dict[str, Any]:
        """Data for 'Live Trading' tab."""
        env = self.env_core

        episode = int(getattr(env, "episode_counter", 0)) + 1
        step = int(getattr(env, "current_step", -1)) + 1
        timeout_steps = int(getattr(env, "timeout_steps", 0))
        steps_left = max(0, timeout_steps - int(getattr(env, "current_step", 0)) - 1)
        countdown = float(getattr(env, "seconds_left", 0.0))

        current_action = str(getattr(env, "current_action", "N/A"))
        held_symbol = env.active_crypto_code if env.active_crypto_code else "-"
        buy_price = float(env.buy_price) if getattr(env, "buy_price", None) else None
        sell_price = float(env.sell_price) if getattr(env, "sell_price", None) else None
        current_price = self._current_price()

        curr_ep_pct = float(getattr(env, "current_ep_profit_decimal", 0.0)) * 100.0
        ep_pct = float(getattr(env, "episode_profit_decimal", 0.0)) * 100.0
        est_bal = float(getattr(env, "estimated_balance_usdt", 0.0))

        return _norm({
            "Episode": episode,
            "Step": step,
            "Timesteps Left": steps_left,
            "Countdown (s)": round(countdown),
            "Current Action": current_action,
            "Held Symbol": held_symbol,
            "Buy Price": buy_price,
            "Current Price": current_price,
            "Sell Price": sell_price,
            "Curr. Ep Profit %": curr_ep_pct,
            "Episode Profit %": ep_pct,
            "Estimated Balance (USDT)": est_bal,
        })

    def session_panel(self) -> Dict[str, Any]:
        """Data + rendered log text for 'Session Stats' tab."""
        env = self.env_core
        episodes_completed = int(getattr(env, "episode_counter", 0))
        total_profit_pct = float(getattr(env, "total_profit_decimal", 0.0)) * 100.0
        days = (datetime.now() - self.session_start_dt).total_seconds() / 86400.0
        days = max(days, 1e-9)
        avg_per_day = total_profit_pct / days

        log_path = getattr(env, "log_path", None)
        text_block = ""
        if log_path and os.path.exists(log_path):
            try:
                with open(log_path, "r") as f:
                    records = [json.loads(line) for line in f if line.strip()]
                # Build episodes (same logic as Tk)
                episodes = []
                current = None
                for r in records:
                    t = r.get("type")
                    if t == "episode_start":
                        if current is not None:
                            episodes.append(current)
                        current = {"meta": r, "steps": [], "end": None}
                    elif t == "step":
                        if current is None:
                            current = {"meta": {"type": "episode_start", "episode": "?"}, "steps": [], "end": None}
                        current["steps"].append(r)
                    elif t == "episode_end":
                        if current is None:
                            current = {"meta": {"type": "episode_start", "episode": r.get("episode","?")}, "steps": [], "end": r}
                            episodes.append(current)
                            current = None
                        else:
                            current["end"] = r
                            episodes.append(current)
                            current = None
                if current is not None:
                    episodes.append(current)

                # Render newest-first text
                lines = []
                for ep in reversed(episodes):
                    meta = ep["meta"] or {}
                    end  = ep["end"] or {}
                    ep_no = meta.get("episode") or end.get("episode") or "?"
                    lines.append(f"=== Episode {ep_no} ===")
                    start_ts = meta.get("start_timestamp", "?")
                    cash     = meta.get("cash_balance", "?")
                    lines.append(f"  start  @ {start_ts}   cash=${cash}")
                    for s in ep["steps"]:
                        step = s.get("step","?")
                        act  = str(s.get("action_type","")).upper()
                        sym  = s.get("symbol","")
                        price= s.get("price")
                        pnl  = s.get("profit_pct")
                        bp   = s.get("buy_price")
                        sp   = s.get("sell_price")
                        rw   = s.get("reward")
                        line = f"  step {step:>3}  {act:<5} {sym:<10}"
                        if price is not None: line += f"  px={float(price):.6f}"
                        if pnl   is not None: line += f"  pnl={float(pnl):+.2f}%"
                        if bp    is not None: line += f"  buy={float(bp):.6f}"
                        if sp    is not None: line += f"  sell={float(sp):.6f}"
                        if rw    is not None: line += f"  r={float(rw):+.4f}"
                        lines.append(line)
                    if end:
                        ts   = end.get("timestamp","?")
                        ret  = float(end.get("episode_return_pct", 0.0))
                        tot  = float(end.get("total_profit_pct", 0.0))
                        steps= end.get("steps","?")
                        why  = end.get("termination","?")
                        lines.append(f"  end    @ {ts}   ret={ret:+.2f}%   total={tot:+.2f}%   steps={steps}   reason={why}")
                    lines.append("")
                text_block = "\n".join(lines)

            except Exception as e:
                text_block = f"Unable to load log: {e}\n"

        return _norm({
            "Episodes Completed": episodes_completed,
            "Total Profit %": total_profit_pct,
            "Days Since Start": round(days, 2),
            "Avg Profit % / Day": avg_per_day,
            "log_text": text_block,
        })

    def codes_panel(self) -> Dict[str, Any]:
        """Data for 'Current Codes' tab."""
        env = self.env_core
        path = getattr(env, "json_path", None)
        codes: List[str] = []
        last_mod = "-"
        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "crypto_codes" in data:
                    codes = sorted(list(data["crypto_codes"]))
                elif isinstance(data, list):
                    codes = sorted(list(data))
                mtime = datetime.fromtimestamp(os.path.getmtime(path))
                last_mod = mtime.strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                last_mod = f"error: {e}"

        # For "Last Change Detected": we simply send last_mod and let the UI display it when list changes.
        return _norm({
            "json_path": path or "-",
            "last_modified": last_mod,
            "codes": codes
        })

    def settings_panel(self) -> Dict[str, Any]:
        """Data for 'Settings' tab (read-only settings & feature list)."""
        env = self.env_core
        features = [
            "open", "high", "low", "close", "vwap", "volume", "count",
            "recent_return", "volatility", "price_position",
            "volume_surge", "trend_slope", "moving_avg"
        ]
        return _norm({
            "features": features,
            "timeout_steps": getattr(env, "timeout_steps", "-"),
            "feature_window": getattr(env, "feature_window", "-"),
            "history_window": getattr(env, "history_window", "-"),
            "buy_limit": getattr(getattr(env, "trader", None), "max_buy_limit", None),
            "cashout_percent": getattr(getattr(env, "trader", None), "daily_cashout_percent", None),
        })

    # ----------------------------------------------------
    def _push_event(self, kind: str, payload: Dict[str, Any]):
        msg = json.dumps({"type": kind, "ts": time.time(), **payload},
                         default=_json_default, ensure_ascii=False)
        try:
            self.stream_q.put_nowait(msg)
        except queue.Full:
            pass

    def _loop(self):
        obs, info = self.env.reset()
        self.episode = self.env_core.episode_counter
        self._push_event("episode_start", {"episode": int(self.episode)})

        while not self._stop.is_set():
            mask = self._action_masks(self.env)
            action, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
            obs, reward, terminated, truncated, info = self.env.step(action)

            self.last_action = int(action)
            self.last_reward = float(reward)
            self.total_steps += 1

            # Stream a lightweight heartbeat
            self._push_event("step", {
                "episode": int(self.env_core.episode_counter),
                "step": int(self.total_steps),
                "action": int(action),
                "reward": float(reward),
            })

            if terminated or truncated:
                self.episode = self.env_core.episode_counter
                self.trades.append({
                    "episode": int(self.episode),
                    "ended": "truncated" if truncated else "terminated",
                    "time": datetime.utcnow(),
                    "portfolio_value": getattr(self.env_core, "portfolio_value", None),
                })
                self._push_event("episode_end", {
                    "episode": int(self.episode),
                    "portfolio_value": getattr(self.env_core, "portfolio_value", None),
                })
                obs, info = self.env.reset()
                self._push_event("episode_start", {"episode": int(self.env_core.episode_counter)})

            time.sleep(1)

# -----------------------------
# Flask app & endpoints
# -----------------------------
app = Flask(__name__, static_folder=None)
AGENT: Optional[AgentService] = None

@app.get("/")
def index():
    return DASHBOARD_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}

# -------- Live tab data --------
@app.get("/api/live")
def api_live():
    return jsonify(AGENT.live_panel())

# -------- Session tab data --------
@app.get("/api/session")
def api_session():
    return jsonify(AGENT.session_panel())

# -------- Codes tab data --------
@app.get("/api/codes")
def api_codes():
    return jsonify(AGENT.codes_panel())

# -------- Settings tab data / actions --------
@app.get("/api/settings")
def api_settings():
    return jsonify(AGENT.settings_panel())

@app.post("/api/settings/limits")
def api_settings_limits():
    data = request.get_json(force=True, silent=True) or {}
    buy_limit = data.get("buy_limit")
    cashout_percent = data.get("cashout_percent")
    # Validate
    try:
        if buy_limit is not None:
            buy_limit = float(buy_limit)
        if cashout_percent is not None:
            cashout_percent = float(cashout_percent)
    except Exception:
        return jsonify({"ok": False, "error": "Invalid numeric values"}), 400

    trader = getattr(AGENT.env_core, "trader", None)
    if trader is None:
        return jsonify({"ok": False, "error": "Trader not available"}), 400

    try:
        if buy_limit is not None:
            trader.set_max_buy_limit(buy_limit)
        if cashout_percent is not None:
            trader.set_daily_cashout_percent(cashout_percent)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# -------- SSE (optional live events) --------
@app.get("/api/stream")
def api_stream():
    def gen():
        hello = json.dumps({'type': 'hello', 'ts': time.time()},
                           default=_json_default, ensure_ascii=False)
        yield f"data: {hello}\n\n"
        while True:
            msg = AGENT.stream_q.get()
            yield f"data: {msg}\n\n"
    return Response(gen(), mimetype="text/event-stream")

# -------- Graceful shutdown helpers & endpoint --------
def _graceful_shutdown_and_exit(delay_sec: float = 0.25):
    """Liquify (if possible), stop the agent thread, then exit the process."""
    try:
        trader = getattr(AGENT.env_core, "trader", None)
        if trader is not None and hasattr(trader, "liquify"):
            print("[WEB] Liquifying all assets before shutdown…")
            trader.liquify()
    except Exception as e:
        print(f"[WEB] Liquify error before exit: {e}")

    try:
        if AGENT is not None:
            print("[WEB] Stopping agent loop…")
            AGENT.stop()
    except Exception as e:
        print(f"[WEB] Agent stop error: {e}")

    # Small delay to let the HTTP response flush
    time.sleep(delay_sec)
    print("[WEB] Exiting process now.")
    os._exit(0)  # intentional hard exit so Docker stops the container

def _handle_sigterm(*_):
    print("[WEB] SIGTERM received — graceful shutdown.")
    _graceful_shutdown_and_exit(0.1)

def _handle_sigint(*_):
    print("[WEB] SIGINT received — graceful shutdown.")
    _graceful_shutdown_and_exit(0.1)

signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigint)

@app.post("/api/end_session")
def api_end_session():
    # Kick off shutdown in the background so we can return a response promptly
    threading.Thread(target=_graceful_shutdown_and_exit, kwargs={"delay_sec": 0.25}, daemon=True).start()
    return jsonify({"ok": True, "shutting_down": True}), 202

# -----------------------------
# HTML (Tabs that mirror the Tk GUI)
# -----------------------------

DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ServoTrader — Live Web UI</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --bg:#0b0f12; --card:#11171c; --ink:#e7eef5; --muted:#9fb3c8; --accent:#8dd1ff; --line:#24313b; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: var(--bg); color: var(--ink); }
  header { display:flex; align-items:center; justify-content:space-between; padding:12px 16px; background:#151515; position:sticky; top:0; z-index:9; }
  header .title { font-weight:700; font-size:20px; }
  header .clock { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: var(--accent); }
  .tabs { display:flex; flex-wrap:wrap; gap:2px; padding:10px 10px 0; background:var(--bg); border-bottom:1px solid var(--line); }
  .tab-btn { padding:10px 14px; border:none; background:#0f1418; color:var(--ink); border-top-left-radius:8px; border-top-right-radius:8px; cursor:pointer; opacity:.8; }
  .tab-btn.active { background: var(--card); opacity:1; }
  .tab { display:none; padding:16px; }
  .tab.active { display:block; }
  /* Two-column grids that collapse on smaller screens */
  .grid2 { display:grid; gap:16px; align-items:start; }
  .grid2 .col { min-width: 0; } /* allow content to shrink without overflow */
  @media (min-width: 1100px) {
    .grid2 { grid-template-columns: 360px 1fr; }
  }
  @media (max-width: 1099.98px) {
    .grid2 { grid-template-columns: 1fr; }
  }
  .kv { display:grid; grid-template-columns: 200px 1fr; gap:8px 10px; align-items:center; margin:6px 0; }
  .kv-key { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: var(--muted); }
  .kv-val { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: #2705c0; word-break: break-word; }
  .card { background: var(--card); border-radius: 12px; padding: 14px; box-shadow: 0 2px 12px rgba(0,0,0,.35); }
  .sep { height:1px; background: var(--line); margin: 10px 0; }
  textarea, pre { width:100%; background:#121212; color:#c0ffc0; border:1px solid var(--line); border-radius:8px; padding:10px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  pre { margin: 0; }
  table { width:100%; border-collapse: collapse; }
  th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); }
  .muted { color: var(--muted); }
  .btn { padding:10px 14px; border:none; border-radius:10px; font-weight:600; cursor:pointer; }
  .btn-danger { background:#cc3333; color:#2705c0; }
  .btn-primary { background:#00aa00; color:#2705c0; }
  .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  input[type="number"] { background:#111; color:#eee; border:1px solid var(--line); border-radius:8px; padding:8px 10px; width:180px; }
  .status { min-height: 20px; }
  /* Make fixed-height panels scroll internally, never overlap siblings */
  .panel-fixed { max-height: 460px; overflow:auto; }
</style>
</head>
<body>
  <header>
    <div class="title">ServoTrader</div>
    <div class="clock" id="clock">--:--:--</div>
  </header>

  <nav class="tabs">
    <button class="tab-btn active" data-tab="live">Live Trading</button>
    <button class="tab-btn" data-tab="session">Session Stats</button>
    <button class="tab-btn" data-tab="codes">Current Codes</button>
    <button class="tab-btn" data-tab="settings">Settings</button>
    <button class="tab-btn" data-tab="license">License</button>
  </nav>

  <!-- Live Trading -->
  <section id="tab-live" class="tab active">
    <div class="grid2">
      <div class="col">
        <div class="kv"><div class="kv-key">Episode:</div><div class="kv-val" id="Episode">…</div></div>
        <div class="kv"><div class="kv-key">Step:</div><div class="kv-val" id="Step">…</div></div>
        <div class="kv"><div class="kv-key">Timesteps Left:</div><div class="kv-val" id="Timesteps Left">…</div></div>
        <div class="kv"><div class="kv-key">Countdown (s):</div><div class="kv-val" id="Countdown (s)">…</div></div>
        <div class="sep"></div>
        <div class="kv"><div class="kv-key">Current Action:</div><div class="kv-val" id="Current Action">…</div></div>
        <div class="kv"><div class="kv-key">Held Symbol:</div><div class="kv-val" id="Held Symbol">…</div></div>
        <div class="kv"><div class="kv-key">Buy Price:</div><div class="kv-val" id="Buy Price">…</div></div>
        <div class="kv"><div class="kv-key">Current Price:</div><div class="kv-val" id="Current Price">…</div></div>
        <div class="kv"><div class="kv-key">Sell Price:</div><div class="kv-val" id="Sell Price">…</div></div>
        <div class="sep"></div>
        <div class="kv"><div class="kv-key">Curr. Ep Profit %:</div><div class="kv-val" id="Curr. Ep Profit %">…</div></div>
        <div class="kv"><div class="kv-key">Episode Profit %:</div><div class="kv-val" id="Episode Profit %">…</div></div>
        <div class="kv"><div class="kv-key">Estimated Balance (USDT):</div><div class="kv-val" id="Estimated Balance (USDT)">…</div></div>
      </div>
      <div class="col">
        <div class="card">
          <div class="muted">Live heartbeat</div>
          <pre id="live-json" class="panel-fixed">(waiting…)</pre>
        </div>
      </div>
    </div>
  </section>

  <!-- Session Stats -->
  <section id="tab-session" class="tab">
    <div class="grid2">
      <div class="col">
        <div class="kv"><div class="kv-key">Episodes Completed:</div><div class="kv-val" id="Episodes Completed">…</div></div>
        <div class="kv"><div class="kv-key">Total Profit %:</div><div class="kv-val" id="Total Profit %">…</div></div>
        <div class="kv"><div class="kv-key">Days Since Start:</div><div class="kv-val" id="Days Since Start">…</div></div>
        <div class="kv"><div class="kv-key">Avg Profit % / Day:</div><div class="kv-val" id="Avg Profit % / Day">…</div></div>
      </div>
      <div class="col">
        <div class="card">
          <div class="muted">Session Log (newest first)</div>
          <pre id="session-log" class="panel-fixed">(loading…)</pre>
        </div>
      </div>
    </div>
  </section>

  <!-- Current Codes -->
  <section id="tab-codes" class="tab">
    <div class="grid2">
      <div class="col">
        <div class="card">
          <div class="muted">Current Crypto Codes</div>
          <pre id="codes-list" class="panel-fixed">(loading…)</pre>
        </div>
      </div>
      <div class="col">
        <div class="kv"><div class="kv-key">JSON Path:</div><div class="kv-val" id="codes-json-path">…</div></div>
        <div class="kv"><div class="kv-key">Last Modified:</div><div class="kv-val" id="codes-last-mod">…</div></div>
        <div class="kv"><div class="kv-key">Last Change Detected:</div><div class="kv-val" id="codes-last-change">N/A</div></div>
      </div>
    </div>
  </section>

  <!-- Settings -->
  <section id="tab-settings" class="tab">
    <div class="grid2">
      <div class="col">
        <div class="card">
          <div class="muted">Observed Features (13)</div>
          <pre id="features" class="panel-fixed" style="max-height: 260px;"></pre>
        </div>
      </div>
      <div class="col">
        <div class="kv"><div class="kv-key">timeout_steps:</div><div class="kv-val" id="timeout_steps">…</div></div>
        <div class="kv"><div class="kv-key">feature_window:</div><div class="kv-val" id="feature_window">…</div></div>
        <div class="kv"><div class="kv-key">history_window:</div><div class="kv-val" id="history_window">…</div></div>
        <div class="sep"></div>
        <div class="row">
          <label class="kv-key">Max Buy Limit ($):</label>
          <input id="buy_limit" type="number" step="0.01" />
        </div>
        <div class="row" style="margin-top:8px;">
          <label class="kv-key">Daily Cashout (%):</label>
          <input id="cashout_percent" type="number" step="0.01" />
        </div>
        <div class="row" style="margin-top:8px;">
          <button class="btn btn-primary" onclick="applyLimits()">Apply Limits</button>
          <div id="limits-status" class="status"></div>
        </div>
      </div>
    </div>
  </section>

  <!-- License -->
  <section id="tab-license" class="tab">
    <div class="card">
      <pre style="white-space:pre-wrap;">
ServoTrader — Live Crypto Trading Environment
Copyright (c) 2025 Jarred Deluca

This software is provided under the MIT License.
Use at your own risk. Trading cryptocurrencies involves substantial risk of loss.
Ensure you understand the risks and have appropriate safeguards in place.

By using this software you agree that the authors are not responsible for any
losses incurred. See the repository LICENSE for the full terms.
      </pre>
    </div>
  </section>

  <div style="padding: 12px 16px; display:flex; justify-content:flex-end; gap:8px;">
    <button class="btn btn-danger" onclick="endSession()" id="end-btn">End Trading Session</button>
    <span id="end-status" class="status"></span>
  </div>

<script>
const tabs = document.querySelectorAll('.tab-btn');
tabs.forEach(b=>b.addEventListener('click', ()=>{
  tabs.forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  b.classList.add('active');
  document.getElementById('tab-'+b.dataset.tab).classList.add('active');
}));

function fmtMoney(v){ if(v===null || v===undefined || isNaN(v)) return "–"; return "$"+Number(v).toLocaleString(undefined,{maximumFractionDigits:2}); }
function fmtPct(v){ if(v===null || v===undefined || isNaN(v)) return "–"; return Number(v).toFixed(2)+"%"; }
function setKV(id, v){ const el=document.getElementById(id); if(el) el.textContent = (v==null?"–":v); }

async function tickLive(){
  const r = await fetch('/api/live'); const d = await r.json();
  for(const [k,v] of Object.entries(d)){
    if(k === "Estimated Balance (USDT)") setKV(k, fmtMoney(v));
    else if(k.includes("Profit %")) setKV(k, fmtPct(v));
    else setKV(k, (v==null?"–":v));
  }
  document.getElementById('live-json').textContent = JSON.stringify(d, null, 2);
}

async function tickSession(){
  const r = await fetch('/api/session'); const d = await r.json();
  setKV("Episodes Completed", d["Episodes Completed"]);
  setKV("Total Profit %", fmtPct(d["Total Profit %"]));
  setKV("Days Since Start", d["Days Since Start"]);
  setKV("Avg Profit % / Day", fmtPct(d["Avg Profit % / Day"]));
  document.getElementById('session-log').textContent = d.log_text || "";
}

let lastCodes = null;
async function tickCodes(){
  const r = await fetch('/api/codes'); const d = await r.json();
  setKV('codes-json-path', d.json_path || "-");
  setKV('codes-last-mod', d.last_modified || "-");
  const listTxt = (d.codes && d.codes.length) ? d.codes.join("\\n") : "(no codes)";
  document.getElementById('codes-list').textContent = listTxt;
  if(lastCodes !== null && JSON.stringify(lastCodes)!==JSON.stringify(d.codes)){
    setKV('codes-last-change', new Date().toISOString().slice(0,19).replace('T',' '));
  }
  if(lastCodes===null) setKV('codes-last-change', "N/A");
  lastCodes = d.codes || [];
}

async function tickSettings(fillLimits=false){
  const r = await fetch('/api/settings'); const d = await r.json();
  document.getElementById('features').textContent = (d.features||[]).join("\\n");
  setKV('timeout_steps', d.timeout_steps);
  setKV('feature_window', d.feature_window);
  setKV('history_window', d.history_window);
  if(fillLimits){
    if(d.buy_limit!=null) document.getElementById('buy_limit').value = d.buy_limit;
    if(d.cashout_percent!=null) document.getElementById('cashout_percent').value = d.cashout_percent;
  }
}

async function applyLimits(){
  const buy_limit = document.getElementById('buy_limit').value;
  const cashout_percent = document.getElementById('cashout_percent').value;
  const r = await fetch('/api/settings/limits', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({buy_limit, cashout_percent})
  });
  const j = await r.json();
  const el = document.getElementById('limits-status');
  if(j.ok){ el.textContent = "✅ Limits applied successfully!"; el.style.color = "#00ff00"; }
  else { el.textContent = "❌ " + (j.error||"Failed"); el.style.color = "#ff3333"; }
  setTimeout(()=>{ el.textContent=""; }, 5000);
}

async function endSession(){
  if(!confirm("Are you sure you want to stop the session and exit? This will liquify all assets.")) return;
  const btn = document.getElementById('end-btn');
  const st  = document.getElementById('end-status');
  btn.disabled = true;
  st.textContent = "Shutting down…";
  try { await fetch('/api/end_session', {method:'POST'}); } catch(e) {}
  // Container will terminate shortly (unless restart policy restarts it).
}

function clockTick(){
  const d = new Date();
  const s = d.toISOString().slice(0,19).replace('T',' ');
  document.getElementById('clock').textContent = s;
}
setInterval(clockTick, 1000); clockTick();

async function tickAll(){
  try{ await tickLive(); }catch(e){}
  try{ await tickSession(); }catch(e){}
  try{ await tickCodes(); }catch(e){}
  try{ await tickSettings(false); }catch(e){}
}
setInterval(tickAll, 2000);
tickAll();
tickSettings(true);

// Optional SSE hook
const es = new EventSource('/api/stream');
es.onopen = ()=>{};
es.onerror = ()=>{};
</script>
</body>
</html>
"""

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    _healthcheck()
    print("✅ Healthcheck passed.")

    AGENT = AgentService()
    print(f"🧠 Loading model from {MODEL_PATH} and starting agent loop…")
    AGENT.start()

    print(f"🌐 Web UI on http://{WEB_HOST}:{WEB_PORT}")
    app.run(host=WEB_HOST, port=WEB_PORT, debug=False, use_reloader=False)
