"""
LSTMActionMaskWrapper

This wrapper enforces legal action constraints in environments with dynamically
changing action masks. It is designed specifically for use with LSTM-based policies
in Stable-Baselines3's RecurrentPPO.

Purpose:
    - Prevents illegal actions from being executed during training.
    - Resamples actions from the policy until a valid one is chosen, based on
      the `action_mask` provided by the environment.
    - Maintains internal state tracking (`hidden_state`, `episode_start`)
      to ensure correct sequential LSTM behavior across steps and episodes.

Intended Use:
    - For use in environments where only a subset of the action space is valid
      at any given timestep.
    - Integrates seamlessly with models that support recurrent hidden state handling.
    - Assumes the wrapped environment provides an `action_mask` key in the `info` dict
      from `env.reset()` and `env.step()`.

Requirements:
    - The environment must include a `action_mask` in its `info` dictionary.
    - The model must implement `.predict(obs, state, episode_start)` and return
      `(action, new_hidden_state)`.

Author: Jarred Deluca  
Project: ServoTrader  
License: MIT  
"""

import numpy as np
import gymnasium as gym

class LSTMActionMaskWrapper(gym.Wrapper):
    """
    A wrapper that forces legal actions by repeatedly sampling from the LSTM policy
    until a legal action (according to the environment's action_mask) is selected.
    """
    def __init__(self, env, model):
        """
        Initializes the wrapper.

        Args:
            env (gym.Env): The environment to wrap.
            model: A recurrent policy model that implements `.predict(obs, state, episode_start)`
        """
        super().__init__(env)
        self.model = model                # The LSTM-based RL model
        self.last_mask = None            # Stores the latest action mask from the environment
        self.hidden_state = None         # Hidden state for the LSTM model
        self.episode_start = np.ones((1,), dtype=bool)  # Indicates episode start for LSTM
        self.last_obs = None             # Tracks last observation in case resampling is needed

    def reset(self, **kwargs):
        """
        Resets the environment and internal recurrent state.

        Returns:
            obs (np.ndarray): Initial observation.
            info (dict): Info dictionary including 'action_mask'.
        """
        obs, info = self.env.reset(**kwargs)
        self.last_mask = info.get("action_mask", None)
        self.hidden_state = None
        self.episode_start = np.ones((1,), dtype=bool)
        self.last_obs = obs
        return obs, info

    def step(self, action):
        """
        Executes the given action. If the action is illegal according to `action_mask`,
        it resamples from the policy until a valid action is found.

        Args:
            action (int): The action proposed by the agent.

        Returns:
            obs (np.ndarray): The next observation.
            reward (float): The received reward.
            terminated (bool): Whether the episode has terminated.
            truncated (bool): Whether the episode was truncated.
            info (dict): Info dictionary including next 'action_mask'.
        """
        if self.last_mask is not None and not self.last_mask[action]:
            # Invalid action detected: resample until a valid one is found
            obs = self.last_obs
            while True:
                new_action, self.hidden_state = self.model.predict(
                    obs,
                    state=self.hidden_state,
                    episode_start=self.episode_start,
                    deterministic=False
                )
                if self.last_mask[new_action]:
                    action = new_action
                    break

        # Step the environment with the (possibly resampled) action
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Update LSTM management state
        self.last_obs = obs
        self.episode_start = np.array([terminated or truncated])
        self.last_mask = info.get("action_mask", None)

        return obs, reward, terminated, truncated, info