"""
train_btc_transformer_ppo.py

Trains a Transformer-based PPO agent on the BTCTradingEnv.

Architecture
------------
The policy uses a small Transformer encoder as the shared backbone, with
separate actor and critic heads. This follows the actor-critic with shared
feature extractor pattern common in modern deep RL.

Transformer Policy Network
  Input  : [batch, HISTORY_WINDOW=24, N_FEATURES=14] (reshaped from flat obs)
  Encoder: Linear projection (14 → d_model=64)
           + Sinusoidal positional encoding
           + 2× TransformerEncoderLayer (4 heads, ff_dim=128, dropout=0.1)
           + Mean pool over sequence → (batch, 64)
  Actor  : Linear(64, 4)  + action masking  → log-probs
  Critic : Linear(64, 1)                    → V(s)

  Total parameters ≈ 100K — appropriate for ~26K training rows.

PPO Configuration (starting point — tune from logs)
  n_steps      : 2048   (rollout length before each update)
  batch_size   : 256
  n_epochs     : 10     (passes over the rollout buffer per update)
  gamma        : 0.99
  gae_lambda   : 0.95
  clip_epsilon : 0.2
  ent_coef     : 0.01   (decays over training)
  vf_coef      : 0.5
  max_grad_norm: 0.5

Action Masking
  Masked actions receive logit = −1e9 before softmax so they have
  effectively zero probability. The action mask is stored in the rollout
  buffer and applied during each update pass.

Logging
  TensorBoard-compatible scalar logging via a lightweight helper.
  Episode stats (profit, win rate, avg trade length) are printed every
  `log_interval` episodes.

Usage
-----
  python train_btc_transformer_ppo.py

  Monitor logs:
    tensorboard --logdir /home/jarred/git/ServoTrader/logs/tb --port 6006

Author: Jarred Deluca
Project: ServoTrader
License: MIT
"""

import os
import sys
import json
import time
import math
import random
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter   # pip install tensorboard

# ── project imports ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from servo_trader.envs.btc_trading_env import (
    BTCTradingEnv,
    BORUTA_FEATURES,
    HISTORY_WINDOW,
    N_FEATURES,
    N_CONTEXT,
    OBS_DIM,
)

# ============================================================================
#  CONFIGURATION
# ============================================================================

CFG = dict(
    # ── Paths ────────────────────────────────────────────────────────────────
    data_path     = "/home/jarred/git/ServoTrader/data/btc_hourly_features.csv",
    model_dir     = "/home/jarred/git/ServoTrader/models",
    log_dir       = "/home/jarred/git/ServoTrader/logs",
    tb_log_dir    = "/home/jarred/git/ServoTrader/logs/tb",
    model_name    = "btc_transformer_ppo",

    # ── Environment ──────────────────────────────────────────────────────────
    max_hold_steps = 200,

    # ── PPO Core ─────────────────────────────────────────────────────────────
    total_timesteps = 2_000_000,
    n_steps         = 2048,          # rollout buffer size
    batch_size      = 256,
    n_epochs        = 10,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_epsilon    = 0.2,
    ent_coef        = 0.01,          # entropy bonus start
    ent_coef_end    = 0.002,         # entropy bonus end (linear decay)
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    target_kl       = 0.02,          # early-stop updates if KL divergence too large

    # ── Learning Rate ────────────────────────────────────────────────────────
    learning_rate   = 3e-4,
    lr_end          = 5e-5,          # linear LR decay to this value

    # ── Transformer ──────────────────────────────────────────────────────────
    d_model         = 64,
    n_heads         = 4,
    n_layers        = 2,
    ff_dim          = 128,
    dropout         = 0.1,

    # ── Training Control ─────────────────────────────────────────────────────
    seed            = 42,
    log_interval    = 10,            # episodes between console log lines
    save_interval   = 100_000,       # timesteps between checkpoint saves
    device          = "cuda" if torch.cuda.is_available() else "cpu",

    # ── Continue training from checkpoint ────────────────────────────────────
    continue_from   = None,          # set to checkpoint path string to resume
)

# ============================================================================
#  TRANSFORMER POLICY NETWORK
# ============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al., 2017).
    Works well for short sequences without needing learned embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerActorCritic(nn.Module):
    """
    Transformer encoder backbone shared between actor and critic heads.

    Observation layout expected (flat, shape [batch, OBS_DIM=339]):
        [:HISTORY_WINDOW*N_FEATURES] → reshaped to [batch, 24, 14] for Transformer
        [HISTORY_WINDOW*N_FEATURES:] → context (pnl, position_flag, steps_norm)
    """

    def __init__(
        self,
        d_model:   int = 64,
        n_heads:   int = 4,
        n_layers:  int = 2,
        ff_dim:    int = 128,
        dropout:   float = 0.1,
        n_actions: int = 4,
    ):
        super().__init__()

        self.d_model    = d_model
        self.n_features = N_FEATURES       # 14
        self.seq_len    = HISTORY_WINDOW   # 24

        # ── Input projection: 14 features → d_model ─────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(N_FEATURES, d_model),
            nn.LayerNorm(d_model),
        )

        # ── Positional encoding ──────────────────────────────────────────────
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=HISTORY_WINDOW + 1, dropout=dropout)

        # ── Transformer encoder ──────────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = ff_dim,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,   # (batch, seq, d_model)
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # ── Context fusion: append context features after pooling ────────────
        # Pool (mean) gives d_model=64; concat context (3) → 67
        self.context_proj = nn.Linear(d_model + N_CONTEXT, d_model)
        self.backbone_norm = nn.LayerNorm(d_model)

        # ── Actor head (policy) ──────────────────────────────────────────────
        self.actor_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_actions),
        )

        # ── Critic head (value function) ─────────────────────────────────────
        self.critic_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # ── Weight initialisation (orthogonal — good for RL) ─────────────────
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        # Final actor/critic layers with smaller gain for better initial entropy
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor_head[-1].bias)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic_head[-1].bias)

    def forward(
        self,
        obs:    torch.Tensor,
        masks:  Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        obs   : (batch, OBS_DIM=339)
        masks : (batch, 4)  bool, True = action is legal

        Returns
        -------
        log_probs : (batch, 4)   log-softmax over masked logits
        values    : (batch,)     state values
        entropy   : (batch,)     policy entropy (for ent_coef bonus)
        """
        batch = obs.shape[0]

        # ── Split observation ────────────────────────────────────────────────
        seq_flat = obs[:, : self.seq_len * self.n_features]          # (B, 336)
        context  = obs[:, self.seq_len * self.n_features :]          # (B, 3)

        seq = seq_flat.view(batch, self.seq_len, self.n_features)     # (B, 24, 14)

        # ── Transformer backbone ─────────────────────────────────────────────
        x = self.input_proj(seq)          # (B, 24, 64)
        x = self.pos_enc(x)               # (B, 24, 64)
        x = self.transformer(x)           # (B, 24, 64)
        x = x.mean(dim=1)                 # mean pool → (B, 64)

        # ── Fuse context ─────────────────────────────────────────────────────
        x = torch.cat([x, context], dim=-1)   # (B, 67)
        x = self.context_proj(x)              # (B, 64)
        x = self.backbone_norm(x)             # (B, 64)

        # ── Actor logits + masking ───────────────────────────────────────────
        logits = self.actor_head(x)           # (B, 4)
        if masks is not None:
            # Set logit to -1e9 for illegal actions so they get ~0 probability
            logits = logits.masked_fill(~masks.bool(), -1e9)
        log_probs = torch.log_softmax(logits, dim=-1)   # (B, 4)

        # ── Entropy ─────────────────────────────────────────────────────────
        probs   = torch.exp(log_probs)
        entropy = -(probs * log_probs.clamp(min=-1e9)).sum(dim=-1)   # (B,)

        # ── Critic value ─────────────────────────────────────────────────────
        values = self.critic_head(x).squeeze(-1)   # (B,)

        return log_probs, values, entropy

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Used during rollout collection for value bootstrapping."""
        _, values, _ = self.forward(obs)
        return values

    def act(
        self,
        obs:   torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the masked policy.

        Returns: action, log_prob_of_action, entropy, value
        """
        log_probs, values, entropy = self.forward(obs, masks)
        dist   = Categorical(logits=log_probs)
        action = dist.sample()   # (B,)
        log_pa = dist.log_prob(action)
        return action, log_pa, entropy, values


# ============================================================================
#  ROLLOUT BUFFER
# ============================================================================

class RolloutBuffer:
    """
    Stores n_steps of transitions for a single environment.
    All tensors live on CPU during collection; moved to device during update.
    """

    def __init__(self, n_steps: int, obs_dim: int, n_actions: int = 4):
        self.n_steps   = n_steps
        self.obs_dim   = obs_dim
        self.n_actions = n_actions
        self.reset()

    def reset(self):
        self.obs      = np.zeros((self.n_steps, self.obs_dim),   dtype=np.float32)
        self.actions  = np.zeros((self.n_steps,),                dtype=np.int64)
        self.rewards  = np.zeros((self.n_steps,),                dtype=np.float32)
        self.dones    = np.zeros((self.n_steps,),                dtype=np.float32)
        self.values   = np.zeros((self.n_steps,),                dtype=np.float32)
        self.log_pis  = np.zeros((self.n_steps,),                dtype=np.float32)
        self.masks    = np.zeros((self.n_steps, self.n_actions), dtype=np.float32)
        self.ptr      = 0
        self.full     = False

    def add(self, obs, action, reward, done, value, log_pi, mask):
        self.obs[self.ptr]     = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr]   = float(done)
        self.values[self.ptr]  = value
        self.log_pis[self.ptr] = log_pi
        self.masks[self.ptr]   = mask.astype(np.float32)
        self.ptr += 1
        if self.ptr == self.n_steps:
            self.full = True
            self.ptr  = 0

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        """Compute generalised advantage estimates (GAE)."""
        advantages = np.zeros_like(self.rewards)
        gae = 0.0
        for t in reversed(range(self.n_steps)):
            next_val = last_value if t == self.n_steps - 1 else self.values[t + 1]
            next_done = self.dones[t]   # 1 if episode ended after this step
            delta = self.rewards[t] + gamma * next_val * (1.0 - next_done) - self.values[t]
            gae   = delta + gamma * gae_lambda * (1.0 - next_done) * gae
            advantages[t] = gae
        self.returns    = advantages + self.values
        self.advantages = advantages

    def get_batches(self, batch_size: int, device: str):
        """Yields shuffled mini-batches as tensors on `device`."""
        indices = np.random.permutation(self.n_steps)
        for start in range(0, self.n_steps, batch_size):
            idx = indices[start: start + batch_size]
            yield (
                torch.tensor(self.obs[idx],        dtype=torch.float32).to(device),
                torch.tensor(self.actions[idx],    dtype=torch.long).to(device),
                torch.tensor(self.advantages[idx], dtype=torch.float32).to(device),
                torch.tensor(self.returns[idx],    dtype=torch.float32).to(device),
                torch.tensor(self.log_pis[idx],    dtype=torch.float32).to(device),
                torch.tensor(self.masks[idx],      dtype=torch.bool).to(device),
            )


# ============================================================================
#  PPO UPDATE
# ============================================================================

def ppo_update(
    model:        TransformerActorCritic,
    optimiser:    optim.Optimizer,
    buffer:       RolloutBuffer,
    last_value:   float,
    cfg:          dict,
    ent_coef:     float,
    global_step:  int,
    total_steps:  int,
) -> dict:
    """
    Performs `n_epochs` PPO update passes over the rollout buffer.

    Returns a dict of training metrics for logging.
    """
    device     = cfg["device"]
    n_epochs   = cfg["n_epochs"]
    clip_eps   = cfg["clip_epsilon"]
    vf_coef    = cfg["vf_coef"]
    max_gnorm  = cfg["max_grad_norm"]
    target_kl  = cfg["target_kl"]
    batch_size = cfg["batch_size"]

    buffer.compute_gae(last_value, cfg["gamma"], cfg["gae_lambda"])

    # Normalise advantages
    adv       = buffer.advantages
    adv_mean  = adv.mean()
    adv_std   = adv.std() + 1e-8
    buffer.advantages = (adv - adv_mean) / adv_std

    metrics = dict(
        policy_loss   = [],
        value_loss    = [],
        entropy       = [],
        approx_kl     = [],
        clip_fraction = [],
    )

    for _epoch in range(n_epochs):
        for obs_b, act_b, adv_b, ret_b, old_lp_b, mask_b in buffer.get_batches(batch_size, device):

            log_probs, values, entropy = model(obs_b, mask_b)

            # Log prob of the action taken
            new_lp = log_probs.gather(1, act_b.unsqueeze(1)).squeeze(1)

            # PPO clipped surrogate loss
            ratio      = (new_lp - old_lp_b).exp()
            clipped    = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_b
            pg_loss    = -torch.min(ratio * adv_b, clipped).mean()

            # Value loss (clipped)
            vf_loss    = 0.5 * (values - ret_b).pow(2).mean()

            # Entropy bonus
            ent_loss   = -entropy.mean()

            loss = pg_loss + vf_coef * vf_loss + ent_coef * ent_loss

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_gnorm)
            optimiser.step()

            # Diagnostics
            with torch.no_grad():
                approx_kl    = ((old_lp_b - new_lp).mean()).item()
                clip_frac    = ((ratio - 1).abs() > clip_eps).float().mean().item()

            metrics["policy_loss"].append(pg_loss.item())
            metrics["value_loss"].append(vf_loss.item())
            metrics["entropy"].append(-ent_loss.item())
            metrics["approx_kl"].append(approx_kl)
            metrics["clip_fraction"].append(clip_frac)

        # Early stop if KL divergence is too large
        if np.mean(metrics["approx_kl"]) > target_kl:
            print(f"  [PPO] Early stop at epoch {_epoch+1} — KL={np.mean(metrics['approx_kl']):.4f}")
            break

    return {k: float(np.mean(v)) for k, v in metrics.items()}


# ============================================================================
#  MAIN TRAINING LOOP
# ============================================================================

def make_env(cfg: dict, df: pd.DataFrame) -> BTCTradingEnv:
    env = BTCTradingEnv(
        df            = df,
        max_hold_steps= cfg["max_hold_steps"],
        log_dir       = cfg["log_dir"],
    )
    return env


def linear_schedule(start: float, end: float, fraction: float) -> float:
    return start + fraction * (end - start)


def train(cfg: dict):
    # ── Reproducibility ──────────────────────────────────────────────────────
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if cfg["device"] == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])
        torch.backends.cudnn.benchmark = True

    device = cfg["device"]
    print(f"[Train] Device: {device}")
    if device == "cuda":
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"[Train] Loading data from {cfg['data_path']}…")
    df = pd.read_csv(cfg["data_path"])
    print(f"[Train] Dataset: {len(df):,} rows | columns: {list(df.columns)}")

    # ── Environment ──────────────────────────────────────────────────────────
    env = make_env(cfg, df)
    obs, info = env.reset()

    # ── Model ────────────────────────────────────────────────────────────────
    model = TransformerActorCritic(
        d_model   = cfg["d_model"],
        n_heads   = cfg["n_heads"],
        n_layers  = cfg["n_layers"],
        ff_dim    = cfg["ff_dim"],
        dropout   = cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model parameters: {n_params:,}")

    # Resume from checkpoint?
    global_step = 0
    if cfg["continue_from"] and os.path.exists(cfg["continue_from"]):
        ckpt = torch.load(cfg["continue_from"], map_location=device)
        model.load_state_dict(ckpt["model"])
        global_step = ckpt.get("global_step", 0)
        print(f"[Train] Resumed from {cfg['continue_from']} at step {global_step:,}")

    optimiser = optim.Adam(model.parameters(), lr=cfg["learning_rate"], eps=1e-5)

    # ── Logging ──────────────────────────────────────────────────────────────
    os.makedirs(cfg["model_dir"],  exist_ok=True)
    os.makedirs(cfg["tb_log_dir"], exist_ok=True)
    writer = SummaryWriter(log_dir=cfg["tb_log_dir"])

    # Episode tracking
    ep_rewards        = deque(maxlen=100)
    ep_lengths        = deque(maxlen=100)
    ep_profits        = deque(maxlen=100)
    ep_wins           = deque(maxlen=100)
    ep_count          = env.episode_counter
    last_save_step    = global_step

    # ── Rollout buffer ────────────────────────────────────────────────────────
    buffer = RolloutBuffer(n_steps=cfg["n_steps"], obs_dim=OBS_DIM)

    total_timesteps = cfg["total_timesteps"]
    print(f"[Train] Total timesteps: {total_timesteps:,}")

    t_start = time.time()

    while global_step < total_timesteps:

        # ── Linear schedules ─────────────────────────────────────────────────
        progress  = global_step / total_timesteps
        lr_now    = linear_schedule(cfg["learning_rate"], cfg["lr_end"], progress)
        ent_now   = linear_schedule(cfg["ent_coef"], cfg["ent_coef_end"], progress)
        for pg in optimiser.param_groups:
            pg["lr"] = lr_now

        # ── Collect rollout ───────────────────────────────────────────────────
        model.eval()
        episode_reward = 0.0
        buffer.reset()

        for _step in range(cfg["n_steps"]):
            obs_t  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            mask_t = torch.tensor(info["action_mask"], dtype=torch.bool).unsqueeze(0).to(device)

            with torch.no_grad():
                action, log_pi, entropy, value = model.act(obs_t, mask_t)

            action_np = int(action.item())
            lp_np     = float(log_pi.item())
            val_np    = float(value.item())

            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            buffer.add(obs, action_np, reward, done, val_np, lp_np, info.get("action_mask", np.ones(4, dtype=bool)))
            episode_reward += reward
            global_step    += 1

            if done:
                ep_rewards.append(episode_reward)
                ep_lengths.append(env.current_step - env.episode_start)
                # Grab last episode profit from env log
                # (env logs asynchronously; approximate via reward sum)
                ep_profits.append(episode_reward * 100)   # rough % proxy
                ep_wins.append(1 if episode_reward > 0 else 0)

                obs, info = env.reset()
                episode_reward = 0.0
            else:
                obs = next_obs

        # Bootstrap value for GAE
        with torch.no_grad():
            obs_t  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            last_v = float(model.get_value(obs_t).item())
            if done:
                last_v = 0.0

        # ── PPO update ────────────────────────────────────────────────────────
        model.train()
        metrics = ppo_update(
            model        = model,
            optimiser    = optimiser,
            buffer       = buffer,
            last_value   = last_v,
            cfg          = cfg,
            ent_coef     = ent_now,
            global_step  = global_step,
            total_steps  = total_timesteps,
        )

        # ── TensorBoard logging ───────────────────────────────────────────────
        writer.add_scalar("train/policy_loss",   metrics["policy_loss"],   global_step)
        writer.add_scalar("train/value_loss",    metrics["value_loss"],    global_step)
        writer.add_scalar("train/entropy",       metrics["entropy"],       global_step)
        writer.add_scalar("train/approx_kl",     metrics["approx_kl"],    global_step)
        writer.add_scalar("train/clip_fraction", metrics["clip_fraction"], global_step)
        writer.add_scalar("train/lr",            lr_now,                  global_step)
        writer.add_scalar("train/ent_coef",      ent_now,                 global_step)

        if ep_rewards:
            writer.add_scalar("rollout/mean_ep_reward", np.mean(ep_rewards), global_step)
            writer.add_scalar("rollout/mean_ep_length", np.mean(ep_lengths), global_step)
            writer.add_scalar("rollout/win_rate",       np.mean(ep_wins),    global_step)

        # ── Console log ──────────────────────────────────────────────────────
        new_episodes = env.episode_counter - ep_count
        ep_count = env.episode_counter
        if new_episodes > 0 and (ep_count % cfg["log_interval"] == 0):
            elapsed  = time.time() - t_start
            fps      = global_step / max(elapsed, 1)
            pct_done = 100 * global_step / total_timesteps
            print(
                f"[{global_step:>8,} | {pct_done:5.1f}%] "
                f"ep={ep_count:,} | "
                f"mean_r={np.mean(ep_rewards) if ep_rewards else 0:.4f} | "
                f"win%={100*np.mean(ep_wins) if ep_wins else 0:.1f} | "
                f"pol={metrics['policy_loss']:.4f} | "
                f"val={metrics['value_loss']:.4f} | "
                f"ent={metrics['entropy']:.4f} | "
                f"kl={metrics['approx_kl']:.4f} | "
                f"fps={fps:.0f}"
            )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if global_step - last_save_step >= cfg["save_interval"]:
            _save_checkpoint(model, optimiser, global_step, cfg)
            last_save_step = global_step

    # ── Final save ────────────────────────────────────────────────────────────
    _save_checkpoint(model, optimiser, global_step, cfg, tag="final")
    writer.close()
    print(f"\n[Train] ✅ Done! Total steps: {global_step:,} | Total episodes: {env.episode_counter:,}")
    print(f"[Train] Total profit (sum): {env.total_profit_pct:.2f}%")


def _save_checkpoint(model, optimiser, step, cfg, tag="ckpt"):
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(cfg["model_dir"], f"{cfg['model_name']}_{tag}_{step}_{ts}.pt")
    torch.save({
        "model":       model.state_dict(),
        "optimiser":   optimiser.state_dict(),
        "global_step": step,
        "cfg":         cfg,
    }, path)
    print(f"  [Checkpoint] Saved → {path}")

    # Also save a 'latest' pointer for easy loading
    latest = os.path.join(cfg["model_dir"], f"{cfg['model_name']}_latest.pt")
    torch.save({
        "model":       model.state_dict(),
        "optimiser":   optimiser.state_dict(),
        "global_step": step,
        "cfg":         cfg,
    }, latest)


# ============================================================================
#  ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    train(CFG)