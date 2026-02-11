"""
train_ppo_agent.py

This script trains a Proximal Policy Optimization (PPO) reinforcement learning agent
on a custom Gym environment designed for high-frequency crypto trading using historical
OHLCV data. The environment simulates buy, hold, and sell actions across 100 cryptocurrencies,
with dense and risk-adjusted reward strategies.

Features:
- Loads historical crypto data and symbol codes
- Preprocesses and wraps the trading environment
- Supports fresh training or continued training from a saved model
- Saves periodic checkpoints and final model to disk
- Logs progress to TensorBoard for visualization

Usage:
- Set CONTINUE_TRAINING to True to resume from a saved model
- Run the script '/bin/python3.11 /home/jarred/git/ServoTrader/scripts/train_ppo_agent.py'
- Run `tensorboard --logdir /home/jarred/git/ServoTrader/logs --port 6006` to visualize training
- To view logs go to '[http://localhost:6006](http://localhost:6006)'

Author: Jarred Deluca  
Project: ServoTrader  
License: MIT  
"""

# scripts/train_ppo_agent.py

import warnings

# Suppress specific Axes3D warning from matplotlib
warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D",
    category=UserWarning,
    module="matplotlib.projections"
)

# --- Add these imports near your other imports ---
import torch

# --- Add this just before you define ppo_config (or at the very top after imports) ---
# Auto-select GPU if available; fall back to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Device] Using: {DEVICE}")
if DEVICE == "cuda":
    print(f"[CUDA] {torch.cuda.get_device_name(0)}")
    # Optional perf tweaks (PyTorch 2.0+)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.backends.cudnn.benchmark = True  # good for fixed-size LSTM/MLP shapes

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pandas as pd
import json
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import signal

from stable_baselines3 import PPO  # Import the PPO (Proximal Policy Optimization algorithm) Agent
from stable_baselines3.common.vec_env import DummyVecEnv
from servo_trader.envs.crypto_trading_env import CryptoTradingEnv  # Import the custom training environment
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor  # Necessary for monitoring rewards
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from sb3_contrib import MaskablePPO, RecurrentPPO
from sb3_contrib.common.wrappers import ActionMasker
from wrappers.action_mask_wrapper import LSTMActionMaskWrapper
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Torch activation for policy kwargs (SiLU/Swish) ---
# Using SiLU tends to work nicely with recurrent policies.
import torch.nn as nn

print("Loading crypto codes...")

# --- Load crypto codes ---
with open('/home/jarred/git/ServoTrader/servo_trader/config/crypto_codes_ancient.json') as f:
    crypto_codes = json.load(f)

print("Loading initial dataset...")
# --- Load historical data ---
historical_df = pd.read_csv("/home/jarred/git/ServoTrader/data/split_10k_chunks_ancient/000.csv")

# --- Create environment ---
USE_LSTM = True  # Set to False to use MaskablePPO instead

if USE_LSTM:
    # --- New method: RecurrentPPO with LSTMActionMaskWrapper ---
    env = DummyVecEnv([
        lambda: CryptoTradingEnv(data=historical_df, crypto_codes=crypto_codes, episode_timeout=60)
    ])
    env = VecMonitor(env)
    env.envs[0] = LSTMActionMaskWrapper(env.envs[0], model=None)  # Model injected later

else:
    # --- Original method: MaskablePPO with ActionMasker ---
    # Define action masks
    def action_masks(env):
        """
        Masks the action space to ensure only legal moves are possible.

        Legal moves:
            - No Crypto Held: Buy actions only (1 to num_cryptos)
            - Crypto Held: Hold (0) and Sell (num_cryptos + 1)

        Returns:
            mask (np array [bools]): an array of bools for each possible action, where each false index makes the action not possible.
        """
        mask = np.zeros(env.action_space.n, dtype=bool)  # Start all as False

        if env.active_crypto_index is None:
            # No crypto held → only Buy and Not Buy actions are legal
            mask[1:env.num_cryptos + 1] = True
            # mask[2 + env.num_cryptos] = True                 # Not Buy (skip)
        else:
            # Crypto held → only Hold and Sell are legal
            mask[0] = True  # Hold
            mask[env.num_cryptos + 1] = True  # Sell

        return mask

    env = DummyVecEnv([
        lambda: ActionMasker(
            CryptoTradingEnv(data=historical_df, crypto_codes=crypto_codes, episode_timeout=60),
            action_masks
        )
    ])
    env = VecMonitor(env)

# =========================
# SCHEDULE HELPERS (NEW)
# =========================

def make_lr_schedule(lr_start=1e-4, lr_end=3e-5, warmup_steps=10_000, total_timesteps=1_000_000):
    """
    Create a cosine learning rate schedule with linear warmup.

    The schedule is compatible with Stable-Baselines3 callables:
    SB3 calls the schedule with `progress_remaining` decreasing from 1.0 → 0.0 during training.
    We map this to a forward time fraction t in [0.0 → 1.0].

    Args:
        lr_start (float): Peak learning rate after warmup.
        lr_end (float): Final learning rate (floor) at the end of training.
        warmup_steps (int): Number of warmup steps from 0 → lr_start.
        total_timesteps (int): Total planned training timesteps for the run.

    Returns:
        callable: A function f(progress_remaining) -> float LR.
    """
    warmup_frac = float(np.clip(warmup_steps / max(total_timesteps, 1), 0.0, 1.0))

    def lr_schedule(progress_remaining: float) -> float:
        """
        Compute the learning rate for the current training progress.

        Args:
            progress_remaining (float): SB3-provided value, 1.0 at start, 0.0 at end.

        Returns:
            float: Current learning rate.
        """
        # Convert to elapsed fraction t \in [0, 1]
        t = 1.0 - float(progress_remaining)
        if t < warmup_frac:
            # Linear warmup from 0 → lr_start
            return (t / max(warmup_frac, 1e-12)) * lr_start
        # Cosine decay from lr_start → lr_end
        tt = (t - warmup_frac) / max(1.0 - warmup_frac, 1e-12)
        return lr_end + 0.5 * (lr_start - lr_end) * (1.0 + np.cos(np.pi * tt))

    return lr_schedule


def make_entropy_schedule(ent_start=5e-3, ent_end=1e-3, decay_until_frac=0.70):
    """
    Create a linear entropy schedule that decays to a target fraction of training.

    The schedule linearly decays from ent_start → ent_end until `decay_until_frac` of
    training is completed, then holds at ent_end.

    Args:
        ent_start (float): Initial entropy coefficient.
        ent_end (float): Final entropy coefficient.
        decay_until_frac (float): Fraction of total training (0-1) at which decay ends.

    Returns:
        callable: A function f(progress_remaining) -> float entropy coefficient.
    """
    decay_until_frac = float(np.clip(decay_until_frac, 1e-6, 1.0))

    def ent_schedule(progress_remaining: float) -> float:
        """
        Compute the entropy coefficient for the current training progress.

        Args:
            progress_remaining (float): SB3-provided value, 1.0 at start, 0.0 at end.

        Returns:
            float: Current entropy coefficient.
        """
        t = 1.0 - float(progress_remaining)  # elapsed fraction
        if t <= decay_until_frac:
            alpha = t / decay_until_frac
            return ent_start + alpha * (ent_end - ent_start)
        return ent_end

    return ent_schedule


# --- Optional: log schedules to TensorBoard (NEW) ---
from stable_baselines3.common.callbacks import BaseCallback

class LogSchedulesCallback(BaseCallback):
    """
    Callback to log dynamic learning rate and entropy coefficient to TensorBoard.

    SB3 does not log these when provided as callables by default.
    This callback evaluates the schedules at the current training progress and records them.
    NOTE: `ent_coef` is logged as a float updated elsewhere (see UpdateEntCoefCallback).
    """
    def __init__(self, name_lr="lr", name_ent="ent_coef", verbose=0):
        super().__init__(verbose)
        self.name_lr = name_lr
        self.name_ent = name_ent

    def _on_step(self) -> bool:
        # Current remaining progress (1.0 → 0.0)
        progress = float(self.model._current_progress_remaining)
        # Evaluate LR schedule (SB3 exposes model.lr_schedule as a callable)
        lr_now = float(self.model.lr_schedule(progress))
        # Entropy coef is stored as a scalar float (kept up-to-date by UpdateEntCoefCallback)
        ent_now = float(self.model.ent_coef)
        # Record to logger (TensorBoard)
        if self.logger is not None:
            self.logger.record(self.name_lr, lr_now)
            self.logger.record(self.name_ent, ent_now)
        return True


class UpdateEntCoefCallback(BaseCallback):
    """
    Updates PPO/RecurrentPPO's entropy coefficient (ent_coef) from a schedule.

    Why:
        sb3-contrib's RecurrentPPO expects `ent_coef` to be a float during loss computation.
        Passing a callable directly can cause a TypeError at `self.ent_coef * entropy_loss`.

    How:
        At the start of each rollout, evaluate the entropy schedule using the current
        progress (1.0 → 0.0) and write the scalar into `model.ent_coef`.

    Args:
        ent_schedule (callable): Function f(progress_remaining) -> float entropy coefficient.
        log_name (str): TensorBoard series name for the entropy coefficient.
    """
    def __init__(self, ent_schedule, log_name="ent_coef", verbose=0):
        super().__init__(verbose)
        self.ent_schedule = ent_schedule
        self.log_name = log_name

    def _on_rollout_start(self) -> bool:
        progress = float(self.model._current_progress_remaining)  # 1.0 at start → 0.0 at end
        new_ent = float(self.ent_schedule(progress))
        # Important: ensure ent_coef is a scalar for the loss computation
        self.model.ent_coef = new_ent
        if self.logger is not None:
            self.logger.record(self.log_name, new_ent)
        return True

    def _on_step(self) -> bool:
        """
        Required by BaseCallback as an abstract method.
        We do not need per-step logic here, so we simply return True to continue training.
        """
        return True


# =========================
# Pause/Resume + Snapshot Control
# =========================

CONTROL_FILE = Path("/tmp/servotrader_train.ctrl")  # write: pause|resume|save|stop
SNAP_DIR = Path("/home/jarred/git/ServoTrader/models/checkpoints")
SNAP_DIR.mkdir(parents=True, exist_ok=True)

_SIG_SAVE = False
_SIG_TOGGLE_PAUSE = False

def _signal_save(signum, frame):
    global _SIG_SAVE
    _SIG_SAVE = True

def _signal_toggle_pause(signum, frame):
    global _SIG_TOGGLE_PAUSE
    _SIG_TOGGLE_PAUSE = True

# Register Linux signals (optional convenience)
try:
    signal.signal(signal.SIGUSR1, _signal_save)         # kill -USR1 <pid> → snapshot
    signal.signal(signal.SIGUSR2, _signal_toggle_pause) # kill -USR2 <pid> → toggle pause/resume
except Exception:
    # Not all platforms support these (e.g., Windows); safe to ignore
    pass

def safe_snapshot(model, tag="manual"):
    """
    Save a timestamped model snapshot + a small sidecar JSON with training state.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = SNAP_DIR / f"ppo_goku_{tag}_{ts}"
    model.save(str(base))
    meta = {
        "num_timesteps": int(getattr(model, "num_timesteps", 0)),
        "progress_remaining": float(getattr(model, "_current_progress_remaining", 1.0)),
        "ent_coef": float(getattr(model, "ent_coef", 0.0)),
        "timestamp": datetime.now().isoformat(),
    }
    with open(str(base) + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Snapshot] Saved {base}  (steps={meta['num_timesteps']})")
    return base

class PausableCallback(BaseCallback):
    """
    File- and signal-based pause/resume + save/stop control.

    Commands (write one word into /tmp/servotrader_train.ctrl):
      - pause     : pause at next callback point (safely idles GPU)
      - resume    : resume if paused
      - save      : save a snapshot and continue
      - stop      : save a snapshot and end training

    Signals:
      - SIGUSR1   : save a snapshot and continue
      - SIGUSR2   : toggle pause/resume
    """
    def __init__(self, check_interval_steps=1024, verbose=0):
        super().__init__(verbose)
        self.check_interval_steps = int(check_interval_steps)
        self._paused = False
        self._since_check = 0

    def _handle_signals_and_file(self) -> bool:
        """
        Handle signals and control file. Returns False to request training stop.
        """
        global _SIG_SAVE, _SIG_TOGGLE_PAUSE

        # Handle signals
        if _SIG_SAVE:
            _SIG_SAVE = False
            safe_snapshot(self.model, tag="sigusr1")

        if _SIG_TOGGLE_PAUSE:
            _SIG_TOGGLE_PAUSE = False
            self._paused = not self._paused
            print(f"[PauseCtl] {'PAUSED' if self._paused else 'RESUMED'} via SIGUSR2")

        # Handle control file
        if CONTROL_FILE.exists():
            cmd = CONTROL_FILE.read_text().strip().lower()
            # clear it so one write → one action
            try:
                CONTROL_FILE.unlink(missing_ok=True)
            except Exception:
                pass

            if cmd in {"pause", "resume", "save", "stop"}:
                if cmd == "pause":
                    self._paused = True
                    print("[PauseCtl] PAUSED (file)")
                elif cmd == "resume":
                    self._paused = False
                    print("[PauseCtl] RESUMED (file)")
                elif cmd == "save":
                    safe_snapshot(self.model, tag="manual")
                elif cmd == "stop":
                    safe_snapshot(self.model, tag="stop")
                    print("[PauseCtl] STOP requested — ending training after snapshot.")
                    return False

        # Pause loop (idle but responsive)
        while self._paused:
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
            time.sleep(2.0)
            # allow control while paused
            if not self._handle_signals_and_file():
                return False
        return True

    def _on_step(self) -> bool:
        self._since_check += 1
        if self._since_check >= self.check_interval_steps:
            self._since_check = 0
            return self._handle_signals_and_file()
        return True

    def _on_training_end(self) -> None:
        # Optional final snapshot
        safe_snapshot(self.model, tag="final")


# =========================
# PPO HYPERPARAMETERS
# =========================

# --- Training budget and warmup for schedules ---
TOTAL_TIMESTEPS = 1_000_000   # Adjust to your run budget
WARMUP_UPDATES = 5_000       # 5_000–10_000 is typical for warmup

# Build schedules
lr_sched = make_lr_schedule(
    lr_start=5e-4,
    lr_end=3e-5,
    # lr_end=1.5e-5,
    warmup_steps=WARMUP_UPDATES,
    total_timesteps=TOTAL_TIMESTEPS
)
ent_sched = make_entropy_schedule(ent_start=5e-3, ent_end=1e-3, decay_until_frac=0.70)

# Define PPO hyperparameters
ppo_config = {
    # LR is a callable (SB3 supports this)
    # "learning_rate": lr_sched,
    "learning_rate": 3e-5,  # Fixed LR

    # IMPORTANT: keep ent_coef a float; we will update it via UpdateEntCoefCallback
    "ent_coef": 5e-3,

    # LSTM-friendly PPO core knobs
    "n_steps": 500,                 # sequence length per update (unroll)
    "batch_size": 100,              # should divide evenly into rollout_size across envs
    "n_epochs": 10,         # number of times we iterate over the rollout buffer
    "gamma": 0.995,
    "gae_lambda": 0.97,
    "clip_range": 0.2,
    "vf_coef": 0.8,
    "max_grad_norm": 0.3,
    "normalize_advantage": True,
    "target_kl": 0.02,              # mild guardrail on destructive updates

    "device": DEVICE,                # auto-selects GPU/CPU
    "verbose": 1,
    "tensorboard_log": "/home/jarred/git/ServoTrader/logs",

    # Policy network (LSTM + MLP heads)
    "policy_kwargs": dict(
        lstm_hidden_size=100,     # size of LSTM hidden state
        n_lstm_layers=3,            # stick to 1 layer for now
        shared_lstm=False,          # actor-only LSTM
        enable_critic_lstm=False,   # be explicit to avoid assertion ambiguity
        ortho_init=False,           # orthogonal + LSTM can over-scale early steps
        activation_fn=nn.SiLU,      # SiLU/Swish
        net_arch=dict(pi=[128, 128], vf=[256, 256]),
    ),
}

# --- Set this flag to True if continuing training ---
CONTINUE_TRAINING = False

if CONTINUE_TRAINING:
    # --- Load existing model ---
    model_cls = RecurrentPPO if USE_LSTM else MaskablePPO
    model = model_cls.load(
        "/home/jarred/git/ServoTrader/models/ppo_goku",
        env=env,
        tensorboard_log=ppo_config["tensorboard_log"],
        device=ppo_config["device"]
    )
    # Re-inject model into the LSTM action-mask wrapper if used
    if USE_LSTM:
        env.envs[0].model = model
else:
    # --- Train new model ---
    if USE_LSTM:
        # MlpLstmPolicy = Multilayer Perceptron (i.e., a fully connected feedforward neural network)
        # verbose=1: This turns on basic logging output, which prints training information (episode rewards, losses, etc.) to the console during training.
        # tensorboard_log="./ppo_logs": This logs training metrics (e.g., rewards, losses, episode lengths) to a directory called ppo_logs/ for use with TensorBoard — a tool for visualizing training progress over time.
        model = RecurrentPPO("MlpLstmPolicy", env, **ppo_config)
        env.envs[0].model = model  # Inject model into wrapper
    else:
        model = MaskablePPO("MlpPolicy", env, **ppo_config)

# --- Set up checkpointing ---
checkpoint = CheckpointCallback(
    save_freq=100_000, save_path="/home/jarred/git/ServoTrader/models/", name_prefix="ppo_goku"
)

# --- Train model ---
# Add callbacks:
#  - UpdateEntCoefCallback: applies entropy schedule safely each rollout
#  - LogSchedulesCallback: logs LR + current entropy coef to TensorBoard
#  - PausableCallback: enables pause/resume/save/stop control
callbacks = [
    checkpoint,
    UpdateEntCoefCallback(ent_schedule=ent_sched),
    LogSchedulesCallback(),
    PausableCallback(check_interval_steps=512),  # NEW
]

if CONTINUE_TRAINING:
    # Continue from current num_timesteps so schedules/progress align
    remaining = max(TOTAL_TIMESTEPS - int(getattr(model, "num_timesteps", 0)), 0)
    print(f"[Continue] Completed={model.num_timesteps}  Remaining={remaining}")
    if remaining > 0:
        model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=False)
else:
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)

# Save the total number of timesteps completed during training
actual_timesteps = model.num_timesteps  # Real number of steps — could be > planned due to n_steps rounding
simulated_minutes = actual_timesteps  # Since each step = 1 minute in your env
simulated_days = simulated_minutes / (60 * 24)  # Compute how many trading days were simulated

print(f"[TensorBoard] Logs saved to: {ppo_config['tensorboard_log']}")  # Check where the logs are going

# --- Log final training results as JSON ---
env_instance = env.envs[0].env  # Unwrap the inner CryptoTradingEnv from the VecEnv stack

if env_instance.take_logs: # Only if logging was enabled
    total_return_pct = float(env_instance.total_profit_percent)  # grab the total return

    # Geometric (compounded) daily return
    if simulated_days > 0 and total_return_pct > -100:
        total_growth = 1 + (total_return_pct / 100)
        daily_growth_factor = total_growth ** (1 / simulated_days)
        compounded_daily_return_pct = (daily_growth_factor - 1) * 100
    else:
        compounded_daily_return_pct = 0.0

    # Build a structured summary object
    training_summary = {
        "type": "training_complete",                              # event type
        "total_episodes": env_instance.episode_counter,           # number of completed episodes
        "total_return_pct": float(env_instance.total_profit_percent),  # cumulative return (float)
        "total_timesteps": int(actual_timesteps),  # total simulated minutes
        "simulated_days": simulated_days,  # total simulated days
        "compounded_daily_return_pct": compounded_daily_return_pct,  # compound daily return
        "timestamp": datetime.now().isoformat()  # timestamp of training end
    }

    # Append this record at the end of the JSONL log
    with open(env_instance.log_path, "a") as f:
        f.write(json.dumps(training_summary) + "\n\n")

# --- Save final model ---
model.save("/home/jarred/git/ServoTrader/models/ppo_goku")
