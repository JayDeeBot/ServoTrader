"""
train_btc_mlp_ppo.py

Trains an MLP-based PPO agent on the 5-minute BTCTradingEnv5m.

Why MLP first?
--------------
The MLP is intentionally simpler than the Transformer so that training failures
point clearly at environment problems (reward shaping, episode structure,
action masking) rather than architectural ones. Once this model achieves
a positive mean episode reward and a win rate above 50%, the environment is
validated and you can swap in the Transformer backbone with confidence.

MLP Architecture
----------------
  Input:  flat obs vector, shape (OBS_DIM=1827,)
  ┌── Shared backbone ──────────────────────────────────────────────┐
  │  Linear(1827 → 512) → LayerNorm → GELU                         │
  │  Linear(512  → 256) → LayerNorm → GELU                         │
  └─────────────────────────────────────────────────────────────────┘
  ┌── Actor head ─────┐     ┌── Critic head ────┐
  │  Linear(256 → 4)  │     │  Linear(256 → 1)  │
  └───────────────────┘     └───────────────────┘

  Total parameters ≈ 1.1M  (well within sample budget for ~1.5M rows)

Key Diagnostics to Watch
------------------------
  rollout/win_rate           → target > 0.53 (breakeven); aim for 0.55+
                               v4 peaked at 58% at step 1.94M — that is the floor to beat
  rollout/mean_profit_pct    → must trend positive; any value > 0.003 beats trading cost
  rollout/not_buy_fraction   → fraction of Phase 1 decisions (NOT_BUY / (NOT_BUY+BUY))
                               target 0.20–0.60; near-zero = always buying immediately (bad)
  rollout/mean_not_buy_steps → mean Phase 1 wait before each entry
  train/entropy              → keep above ent_floor=0.10 throughout training
  train/lr                   → cosine: starts 1e-4, stays higher longer, drops to 1e-5
  train/approx_kl            → < 0.02; spikes above 0.05 indicate gradient instability
  train/clip_fraction        → 0.05–0.20 healthy; near-zero = policy frozen

Fix in This Version (v6 + not_buy logging fix)
-----------------------------------------------
  One-line fix to rollout/not_buy_fraction:

  BEFORE: nb_frac = mean_nb / max(mean_len, 1.0)
    Divided NOT_BUY steps by total episode length, which includes all Phase 2
    HOLD steps. This made the metric read ~8% when the agent was actually
    declining to buy on ~52% of its Phase 1 decisions — a 44pp undercount.

  AFTER:  nb_frac = mean_nb / max(mean_nb + 1.0, 1.0)
    In a single-trade episode there are exactly (mean_nb NOT_BUY + 1 BUY)
    Phase 1 decisions. Dividing by (mean_nb + 1) gives the true fraction of
    entry-decision steps that are NOT_BUY.

  All other code is identical to the version that produced:
    - 51% win rate at step 3.94M
    - +0.028% mean profit at step 3.82M  (first ever positive profit)
    - Q4 JSONL win rate 38.3%

Usage
-----
  python train_btc_mlp_ppo.py

  Monitor with TensorBoard:
    tensorboard --logdir /home/jarred/git/ServoTrader/logs/tb --port 6006

Author: Jarred Deluca
Project: ServoTrader
License: MIT
"""

import os
import sys
import time
import random
import math
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

# ── project import ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from servo_trader.envs.btc_trading_env_5m import (
    BTCTradingEnv5m,
    BORUTA_FEATURES,
    OBS_DIM,
    N_FEATURES,
    HISTORY_WINDOW,
)

# ============================================================================
#  CONFIGURATION
#  Tune these once you have baseline TensorBoard logs showing stable learning.
# ============================================================================

CFG = dict(
    # ── Paths ────────────────────────────────────────────────────────────────
    data_path   = "/home/jarred/git/ServoTrader/data/btc_5min_features.csv",
    model_dir   = "/home/jarred/git/ServoTrader/models",
    log_dir     = "/home/jarred/git/ServoTrader/logs",
    tb_log_dir  = "/home/jarred/git/ServoTrader/logs/tb",
    model_name  = "btc_mlp_ppo",

    # ── Environment ──────────────────────────────────────────────────────────
    max_hold_steps = 72,   # 6-hour forced exit ceiling
    min_hold_steps = 18,   # 90-min minimum hold before SELL is legal.
                           # Increased from 6 (30 min) — at 6 candles the agent
                           # converged to "sell at first legal opportunity" in
                           # every episode, giving a mean hold of only 36 min.
                           # 18 candles forces the agent into a longer learning
                           # regime where the 38 features have more time to
                           # express predictive power and a single noisy candle
                           # cannot dominate the trade outcome.

    # ── PPO Core ─────────────────────────────────────────────────────────────
    total_timesteps = 4_000_000,   # 2M→4M: v4's best result (+0.135%) was at
                                   # step 1.94M still improving — needed more time.
    n_steps         = 2048,
    batch_size      = 512,
    n_epochs        = 8,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_epsilon    = 0.2,
    ent_coef        = 0.10,        # flat for first ent_warmup_frac of training
    ent_coef_end    = 0.02,        # decays to this after warmup
    ent_warmup_frac = 0.30,        # 30% flat warmup → 1.2M steps at full ent_coef
    ent_floor       = 0.10,        # quadratic penalty if entropy drops below this
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    target_kl       = 0.02,

    # ── Learning Rate — cosine schedule ──────────────────────────────────────
    # Cosine annealing keeps LR higher for longer than linear decay, reducing
    # the risk of locking into a local minimum early. In v4, linear decay made
    # LR too small to escape after step ~1.5M — win rate peaked then declined.
    learning_rate   = 1e-4,
    lr_min          = 1e-5,        # cosine floor

    # ── MLP Architecture ─────────────────────────────────────────────────────
    hidden_sizes    = [512, 256],
    dropout         = 0.1,

    # ── Training Control ─────────────────────────────────────────────────────
    seed            = 42,
    log_interval    = 10,          # rollouts between console prints
    save_interval   = 100_000,     # steps between model checkpoints
    device          = "cuda" if torch.cuda.is_available() else "cpu",

    # Resume from a checkpoint (set to path string to continue, or None)
    continue_from   = None,
)


# ============================================================================
#  MLP ACTOR-CRITIC
# ============================================================================

class MLPActorCritic(nn.Module):
    """
    Simple MLP shared-backbone actor-critic for PPO.

    The backbone processes the flat observation vector. Actor and critic heads
    share the backbone features but have independent final layers.

    Orthogonal initialisation throughout (standard for RL, reduces early
    gradient issues and speeds up convergence vs default random init).
    """

    def __init__(
        self,
        obs_dim:      int,
        hidden_sizes: list,
        n_actions:    int   = 4,
        dropout:      float = 0.1,
    ):
        super().__init__()

        self.obs_dim   = obs_dim
        self.n_actions = n_actions

        # ── Shared backbone ──────────────────────────────────────────────────
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(p=dropout),
            ]
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        feat_dim = hidden_sizes[-1]   # output dimension of backbone

        # ── Actor head ───────────────────────────────────────────────────────
        self.actor_head = nn.Linear(feat_dim, n_actions)

        # ── Critic head ──────────────────────────────────────────────────────
        self.critic_head = nn.Linear(feat_dim, 1)

        # ── Orthogonal init ──────────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        # Smaller gain for final heads (common PPO practice)
        nn.init.orthogonal_(self.actor_head.weight,  gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(
        self,
        obs:   torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        obs   : (batch, obs_dim)
        masks : (batch, 4) bool, True = action is legal

        Returns
        -------
        log_probs : (batch, 4)
        values    : (batch,)
        entropy   : (batch,)
        """
        features = self.backbone(obs)                   # (B, feat_dim)

        # --- Actor ---
        logits = self.actor_head(features)              # (B, 4)
        if masks is not None:
            logits = logits.masked_fill(~masks.bool(), -1e9)
        log_probs = torch.log_softmax(logits, dim=-1)  # (B, 4)

        # --- Entropy ---
        probs   = torch.exp(log_probs)
        entropy = -(probs * log_probs.clamp(min=-1e9)).sum(dim=-1)  # (B,)

        # --- Critic ---
        values = self.critic_head(features).squeeze(-1)  # (B,)

        return log_probs, values, entropy

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(obs)
        return self.critic_head(features).squeeze(-1)

    def act(
        self,
        obs:   torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from masked policy. Returns (action, log_pi, entropy, value)."""
        log_probs, values, entropy = self.forward(obs, masks)
        # Use probs= not logits= — log_probs is already log_softmax output.
        # Passing it as logits would apply a second softmax internally (double-transform),
        # making dist.log_prob() inconsistent with the log_probs used in the PPO update.
        dist   = Categorical(probs=torch.exp(log_probs).clamp(min=1e-8))
        action = dist.sample()
        log_pa = dist.log_prob(action)
        return action, log_pa, entropy, values


# ============================================================================
#  ROLLOUT BUFFER
# ============================================================================

class RolloutBuffer:
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

    def add(self, obs, action, reward, done, value, log_pi, mask):
        self.obs[self.ptr]     = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr]   = float(done)
        self.values[self.ptr]  = value
        self.log_pis[self.ptr] = log_pi
        self.masks[self.ptr]   = mask.astype(np.float32)
        self.ptr += 1

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        advantages = np.zeros_like(self.rewards)
        gae = 0.0
        for t in reversed(range(self.n_steps)):
            next_val  = last_value if t == self.n_steps - 1 else self.values[t + 1]
            next_done = self.dones[t]
            delta     = self.rewards[t] + gamma * next_val * (1.0 - next_done) - self.values[t]
            gae       = delta + gamma * gae_lambda * (1.0 - next_done) * gae
            advantages[t] = gae
        self.returns    = advantages + self.values
        self.advantages = advantages

    def get_batches(self, batch_size: int, device: str):
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

def ppo_update(model, optimiser, buffer, last_value, cfg, ent_coef) -> dict:
    device     = cfg["device"]
    n_epochs   = cfg["n_epochs"]
    clip_eps   = cfg["clip_epsilon"]
    vf_coef    = cfg["vf_coef"]
    max_gnorm  = cfg["max_grad_norm"]
    target_kl  = cfg["target_kl"]
    batch_size = cfg["batch_size"]
    ent_floor  = cfg.get("ent_floor", 0.10)   # minimum entropy threshold

    buffer.compute_gae(last_value, cfg["gamma"], cfg["gae_lambda"])

    adv = buffer.advantages
    buffer.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

    metrics = dict(policy_loss=[], value_loss=[], entropy=[], approx_kl=[], clip_fraction=[])

    for _epoch in range(n_epochs):
        for obs_b, act_b, adv_b, ret_b, old_lp_b, mask_b in buffer.get_batches(batch_size, device):

            log_probs, values, entropy = model(obs_b, mask_b)
            new_lp  = log_probs.gather(1, act_b.unsqueeze(1)).squeeze(1)
            ratio   = (new_lp - old_lp_b).exp()
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_b
            pg_loss = -torch.min(ratio * adv_b, clipped).mean()
            vf_loss = 0.5 * (values - ret_b).pow(2).mean()

            # Standard entropy bonus (encourages exploration)
            mean_entropy = entropy.mean()
            ent_loss     = -mean_entropy

            # Entropy floor: apply an extra quadratic penalty when mean entropy
            # drops below ent_floor. This kicks in only when the policy is
            # becoming dangerously deterministic, pushing it back toward
            # exploration without over-regularising when entropy is healthy.
            # In previous runs entropy collapsed to <0.001 by step 313K despite
            # ent_coef=0.05 — the floor catches this before it becomes permanent.
            floor_violation = torch.clamp(ent_floor - mean_entropy, min=0.0)
            floor_penalty   = 2.0 * floor_violation.pow(2)   # quadratic, so gentle near floor

            loss = pg_loss + vf_coef * vf_loss + ent_coef * ent_loss + floor_penalty

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_gnorm)
            optimiser.step()

            with torch.no_grad():
                approx_kl = (old_lp_b - new_lp).mean().item()
                clip_frac  = ((ratio - 1).abs() > clip_eps).float().mean().item()

            metrics["policy_loss"].append(pg_loss.item())
            metrics["value_loss"].append(vf_loss.item())
            metrics["entropy"].append(mean_entropy.item())
            metrics["approx_kl"].append(approx_kl)
            metrics["clip_fraction"].append(clip_frac)

        if np.mean(metrics["approx_kl"]) > target_kl:
            print(f"  [PPO] Early stop at epoch {_epoch+1} — KL={np.mean(metrics['approx_kl']):.4f}")
            break

    return {k: float(np.mean(v)) for k, v in metrics.items()}


# ============================================================================
#  MAIN TRAINING LOOP
# ============================================================================

def linear_schedule(start, end, fraction):
    return start + fraction * (end - start)


def cosine_lr_schedule(cfg: dict, progress: float) -> float:
    """
    Cosine annealing learning rate schedule.

    Keeps LR higher for longer than linear decay, then drops smoothly to
    lr_min. This reduces the risk of converging to a local minimum early —
    in v4, linear decay made LR too small to escape a suboptimal policy in
    the final third of training (win rate peaked then declined).

    Formula: lr_min + 0.5*(lr_max - lr_min)*(1 + cos(π * progress))
    """
    import math
    lr_max = cfg["learning_rate"]
    lr_min = cfg.get("lr_min", 1e-5)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def entropy_schedule(cfg: dict, progress: float) -> float:
    """
    Entropy coefficient schedule with a flat warmup period.

    Holds ent_coef constant for the first `ent_warmup_frac` fraction of
    training, then linearly decays to ent_coef_end over the remainder.

    This is important for this environment because with only 2 legal actions
    at any time (NOT_BUY/BUY in Phase 1, HOLD/SELL in Phase 2), max entropy
    is ln(2)=0.693. Orthogonal init already biases toward one action, so
    entropy collapses before the agent has explored NOT_BUY without a warmup.
    """
    warmup = cfg.get("ent_warmup_frac", 0.30)
    if progress < warmup:
        return cfg["ent_coef"]   # flat — full exploration pressure
    # Remap progress into [0, 1] over the decay window
    decay_progress = (progress - warmup) / (1.0 - warmup)
    return linear_schedule(cfg["ent_coef"], cfg["ent_coef_end"], decay_progress)


def train(cfg: dict):
    # ── Reproducibility ──────────────────────────────────────────────────────
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if cfg["device"] == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])

    device = cfg["device"]
    print(f"[Train] Device : {device}")
    if device == "cuda":
        print(f"[Train] GPU    : {torch.cuda.get_device_name(0)}")

    # ── Data ─────────────────────────────────────────────────────────────────
    print(f"[Train] Loading {cfg['data_path']} …")
    df = pd.read_csv(cfg["data_path"])
    print(f"[Train] Dataset: {len(df):,} rows | {len(BORUTA_FEATURES)} features")

    # Train / validation split (last 20% as holdout — never touch during training)
    split = int(len(df) * 0.80)
    df_train = df.iloc[:split].reset_index(drop=True)
    df_val   = df.iloc[split:].reset_index(drop=True)
    print(f"[Train] Train  : {len(df_train):,} rows | Val: {len(df_val):,} rows")

    # ── Environment ──────────────────────────────────────────────────────────
    env = BTCTradingEnv5m(
        df            = df_train,
        max_hold_steps= cfg["max_hold_steps"],
        min_hold_steps= cfg["min_hold_steps"],
        log_dir       = cfg["log_dir"],
    )
    env._prev_total = 0.0   # used to diff total_profit_pct per episode
    obs, info = env.reset()

    # ── Model ────────────────────────────────────────────────────────────────
    model = MLPActorCritic(
        obs_dim      = OBS_DIM,
        hidden_sizes = cfg["hidden_sizes"],
        n_actions    = 4,
        dropout      = cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] MLP parameters  : {n_params:,}")
    print(f"[Train] Obs dimension   : {OBS_DIM:,}  ({HISTORY_WINDOW} candles × {N_FEATURES} features + 3 context)")
    print(f"[Train] Total timesteps : {cfg['total_timesteps']:,}")

    global_step = 0
    if cfg.get("continue_from") and os.path.exists(cfg["continue_from"]):
        ckpt = torch.load(cfg["continue_from"], map_location=device)
        model.load_state_dict(ckpt["model"])
        global_step = ckpt.get("global_step", 0)
        print(f"[Train] Resumed from step {global_step:,}")

    optimiser = optim.Adam(model.parameters(), lr=cfg["learning_rate"], eps=1e-5)

    # ── Logging ──────────────────────────────────────────────────────────────
    os.makedirs(cfg["model_dir"],  exist_ok=True)
    os.makedirs(cfg["tb_log_dir"], exist_ok=True)
    writer = SummaryWriter(log_dir=cfg["tb_log_dir"])

    ep_rewards  = deque(maxlen=200)
    ep_lengths  = deque(maxlen=200)
    ep_wins     = deque(maxlen=200)
    ep_profits  = deque(maxlen=200)
    ep_not_buys = deque(maxlen=200)   # NOT_BUY steps per episode
    ep_count   = env.episode_counter
    last_save  = global_step

    buffer = RolloutBuffer(n_steps=cfg["n_steps"], obs_dim=OBS_DIM)
    t_start = time.time()

    print(f"\n{'─'*70}")
    print("  Starting training. Key metrics to watch in TensorBoard:")
    print("    rollout/win_rate       → target > 0.50")
    print("    rollout/mean_ep_reward → target > 0.003  (beats 0.3% cost)")
    print("    train/entropy          → should decay slowly from ~1.1")
    print("    train/approx_kl        → should stay < 0.02")
    print(f"{'─'*70}\n")

    while global_step < cfg["total_timesteps"]:

        progress = global_step / cfg["total_timesteps"]
        lr_now   = cosine_lr_schedule(cfg, progress)
        ent_now  = entropy_schedule(cfg, progress)
        for pg in optimiser.param_groups:
            pg["lr"] = lr_now

        # ── Collect rollout ───────────────────────────────────────────────────
        model.eval()
        ep_reward = 0.0
        buffer.reset()

        for _step in range(cfg["n_steps"]):
            obs_t  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            # ── IMPORTANT: capture the mask BEFORE stepping ───────────────────
            # The mask stored in the buffer must be the one used to SELECT the
            # action, not the mask returned by env.step() (which reflects the
            # state AFTER the action and will be a different phase for BUY/SELL
            # transitions). Storing the post-step mask was the original bug:
            # during PPO updates it caused the just-taken action to receive
            # logit=-1e9, blowing up approx_kl to ~250M and corrupting all
            # gradient updates.
            current_mask = info["action_mask"].copy()
            mask_t = torch.tensor(current_mask, dtype=torch.bool).unsqueeze(0).to(device)

            with torch.no_grad():
                action, log_pi, entropy, value = model.act(obs_t, mask_t)

            action_np = int(action.item())
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            buffer.add(
                obs, action_np, reward, done,
                float(value.item()), float(log_pi.item()),
                current_mask,   # pre-step mask — consistent with action selection
            )
            ep_reward  += reward
            global_step += 1
            obs = next_obs

            if done:
                ep_rewards.append(ep_reward)
                ep_lengths.append(env.episode_steps)
                last_profit = env.total_profit_pct - getattr(env, '_prev_total', 0.0)
                env._prev_total = env.total_profit_pct
                ep_wins.append(1 if last_profit > 0 else 0)
                ep_profits.append(last_profit)
                ep_not_buys.append(env.episode_not_buys)
                obs, info = env.reset()
                ep_reward = 0.0

        # Bootstrap value for open episode
        with torch.no_grad():
            obs_t  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            last_v = float(model.get_value(obs_t).item()) if not done else 0.0

        # ── PPO update ────────────────────────────────────────────────────────
        model.train()
        metrics = ppo_update(model, optimiser, buffer, last_v, cfg, ent_now)

        # ── Logging ──────────────────────────────────────────────────────────
        writer.add_scalar("train/policy_loss",   metrics["policy_loss"],   global_step)
        writer.add_scalar("train/value_loss",    metrics["value_loss"],    global_step)
        writer.add_scalar("train/entropy",       metrics["entropy"],       global_step)
        writer.add_scalar("train/approx_kl",     metrics["approx_kl"],    global_step)
        writer.add_scalar("train/clip_fraction", metrics["clip_fraction"], global_step)
        writer.add_scalar("train/lr",            lr_now,                  global_step)
        writer.add_scalar("train/ent_coef",      ent_now,                 global_step)

        if ep_rewards:
            win_r    = float(np.mean(ep_wins))
            mean_r   = float(np.mean(ep_rewards))
            mean_nb  = float(np.mean(ep_not_buys)) if ep_not_buys else 0.0
            mean_len = float(np.mean(ep_lengths))
            mean_mp  = float(np.mean(ep_profits))

            # ── not_buy_fraction fix ──────────────────────────────────────────
            # In a single-trade episode, Phase 1 has exactly (mean_nb NOT_BUY
            # steps + 1 BUY step). The correct fraction of Phase 1 entry
            # decisions that are NOT_BUY is therefore mean_nb / (mean_nb + 1).
            #
            # The previous formula mean_nb / mean_len divided by total episode
            # length, which includes all Phase 2 HOLD steps — this caused the
            # metric to read ~8% when the agent was declining to buy on ~52% of
            # its actual Phase 1 decisions, a 44pp undercount that made it
            # impossible to detect entry selectivity changes in TensorBoard.
            nb_frac = mean_nb / max(mean_nb + 1.0, 1.0)

            writer.add_scalar("rollout/mean_ep_reward",     mean_r,   global_step)
            writer.add_scalar("rollout/mean_ep_length",     mean_len, global_step)
            writer.add_scalar("rollout/win_rate",           win_r,    global_step)
            writer.add_scalar("rollout/mean_profit_pct",    mean_mp,  global_step)
            writer.add_scalar("rollout/mean_not_buy_steps", mean_nb,  global_step)
            writer.add_scalar("rollout/not_buy_fraction",   nb_frac,  global_step)

        new_ep = env.episode_counter - ep_count
        ep_count = env.episode_counter
        if new_ep > 0 and ep_count % cfg["log_interval"] == 0:
            elapsed = time.time() - t_start
            fps     = global_step / max(elapsed, 1)
            pct     = 100 * global_step / cfg["total_timesteps"]
            print(
                f"[{global_step:>9,} | {pct:5.1f}%] "
                f"ep={ep_count:,} | "
                f"mean_r={np.mean(ep_rewards) if ep_rewards else 0:+.4f} | "
                f"win%={100*np.mean(ep_wins) if ep_wins else 0:.1f} | "
                f"nb%={100*nb_frac:.1f} | "
                f"pol={metrics['policy_loss']:.4f} | "
                f"val={metrics['value_loss']:.4f} | "
                f"ent={metrics['entropy']:.3f} | "
                f"kl={metrics['approx_kl']:.4f} | "
                f"fps={fps:.0f}"
            )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if global_step - last_save >= cfg["save_interval"]:
            _save_checkpoint(model, optimiser, global_step, cfg)
            last_save = global_step

    # ── Final save ────────────────────────────────────────────────────────────
    _save_checkpoint(model, optimiser, global_step, cfg, tag="final")
    writer.close()
    print(f"\n[Train] ✅ Done — {global_step:,} steps | {env.episode_counter:,} episodes")
    print(f"[Train] Cumulative env profit: {env.total_profit_pct:.2f}%")
    print(f"[Train] Final model: {cfg['model_dir']}/{cfg['model_name']}_latest.pt")


def _save_checkpoint(model, optimiser, step, cfg, tag="ckpt"):
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(cfg["model_dir"], f"{cfg['model_name']}_{tag}_{step}_{ts}.pt")
    state = dict(model=model.state_dict(), optimiser=optimiser.state_dict(),
                 global_step=step, cfg=cfg)
    torch.save(state, path)
    latest = os.path.join(cfg["model_dir"], f"{cfg['model_name']}_latest.pt")
    torch.save(state, latest)
    print(f"  [Checkpoint] → {path}")


if __name__ == "__main__":
    train(CFG)