#!/usr/bin/env python3
"""
inspect_sb3_model.py
Print the observation/action spaces and key metadata from a saved SB3 model zip.

Usage:
  /bin/python3.11 /home/jarred/git/ServoTrader/tests/inspect_sb3_model.py \
    /home/jarred/git/ServoTrader/models/ppo_servo_trader_squirtle.zip
"""

import argparse
import zipfile
import json
import pprint

# SB3 & SB3-Contrib
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.base_class import BaseAlgorithm

# Recurrent algorithms live in contrib
try:
    from sb3_contrib import RecurrentPPO, MaskablePPO, TRPO
except Exception:  # pragma: no cover
    RecurrentPPO = None
    MaskablePPO = None
    TRPO = None

ALGOS = [
    # contrib first (most likely, given your model)
    ("RecurrentPPO", RecurrentPPO),
    ("MaskablePPO", MaskablePPO),
    ("TRPO", TRPO),
    # core sb3 fallbacks
    ("PPO", PPO),
    ("A2C", A2C),
    ("DQN", DQN),
    ("SAC", SAC),
    ("TD3", TD3),
]

def load_model_any(path: str) -> BaseAlgorithm:
    last_err = None
    for name, cls in ALGOS:
        if cls is None:
            continue
        try:
            # load without env to just restore spaces/hparams
            model = cls.load(path, env=None, print_system_info=False)
            print(f"[OK] Loaded with {name}")
            return model
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load model with any known algo. Last error: {last_err}")

def summarize_space(space):
    import gymnasium as gym
    if isinstance(space, gym.spaces.Box):
        low = space.low
        high = space.high
        shape = getattr(space, "shape", None)
        # Small summary to avoid dumping giant arrays
        low_min = float(low.min())
        low_max = float(low.max())
        high_min = float(high.min())
        high_max = float(high.max())
        neg_count = int((low < 0).sum())
        return {
            "type": "Box",
            "shape": shape,
            "dtype": str(space.dtype),
            "low_min": low_min,
            "low_max": low_max,
            "high_min": high_min,
            "high_max": high_max,
            "negative_entries_in_low": neg_count,
        }
    elif isinstance(space, gym.spaces.Discrete):
        return {"type": "Discrete", "n": space.n}
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return {"type": "MultiDiscrete", "nvec": space.nvec.tolist()}
    elif isinstance(space, gym.spaces.MultiBinary):
        return {"type": "MultiBinary", "n": space.n}
    else:
        return {"type": type(space).__name__, "repr": repr(space)}

def check_vecnormalize_in_zip(path: str):
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
        # Heuristics: SB3 saves VecNormalize under these names
        vn_candidates = [n for n in names if "vecnormalize" in n.lower() or n.endswith("normalize.pkl")]
        return list(names), vn_candidates
    except Exception:
        return [], []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", type=str, help="Path to SB3/SB3-Contrib .zip model")
    args = ap.parse_args()

    # 1) Try load with various algos
    model = load_model_any(args.model_path)

    # 2) Spaces
    obs_summary = summarize_space(model.observation_space)
    act_summary = summarize_space(model.action_space)

    print("\n=== SPACES ===")
    pprint.pprint({"observation_space": obs_summary, "action_space": act_summary})

    # 3) Policy + net arch
    print("\n=== POLICY / ARCH ===")
    print("policy:", type(model.policy).__name__)
    # policy_kwargs may be on model or saved in _hyperparams
    policy_kwargs = getattr(model, "policy_kwargs", None) or model.__dict__.get("_hyperparams", {}).get("policy_kwargs")
    if policy_kwargs:
        try:
            pprint.pprint({"policy_kwargs": policy_kwargs})
        except Exception:
            print("policy_kwargs (raw):", policy_kwargs)

    # 4) Hyperparams snapshot
    hparams = getattr(model, "_hyperparams", {})
    # show a subset for brevity
    keys_of_interest = [
        "n_steps", "batch_size", "learning_rate", "gamma", "gae_lambda",
        "clip_range", "ent_coef", "vf_coef", "max_grad_norm",
        "use_sde", "sde_sample_freq", "normalize_advantage",
    ]
    hp_view = {k: hparams.get(k) for k in keys_of_interest if k in hparams}
    print("\n=== HYPERPARAMS (subset) ===")
    pprint.pprint(hp_view)

    # 5) Check if VecNormalize stats are embedded in the zip
    names, vn = check_vecnormalize_in_zip(args.model_path)
    print("\n=== ZIP CONTENTS (trimmed) ===")
    print("contains files (sample):", names[:10], "...")
    if vn:
        print("VecNormalize-related files found:", vn)
    else:
        print("No VecNormalize pickle found.")

    # 6) Extra hint if obs looks like single-symbol (69-dim) vs multi-asset
    if obs_summary.get("type") == "Box" and obs_summary.get("shape") == (69,):
        print("\nHint: obs shape 69 == 5 * 1 * 13 + 4 (extras). "
              "Model was trained for a single symbol (Discrete(3) action space).")
    elif obs_summary.get("type") == "Box":
        print(f"\nObs shape looks like {obs_summary['shape']}. "
              "If you expected 6603, confirm num_cryptos, history_window, features_per_crypto.")

if __name__ == "__main__":
    main()
