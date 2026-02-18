"""
Environment wrappers for RL training.
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Tuple
from gymnasium.core import ObsType, ActType


class ObsNormWrapper(gym.Wrapper):
    """Observation normalization wrapper"""

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        self.obs_sum = 0.0
        self.obs_sq_sum = 0.0
        self.obs_count = 0
        self.mean = None
        self.std = None

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        obs, info = self.env.reset(**kwargs)
        return self._normalize_obs(obs), info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize_obs(obs), reward, terminated, truncated, info

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        # Update running statistics
        self.obs_count += 1
        delta = obs - self.mean if self.mean is not None else obs
        self.mean = self.mean + delta / self.obs_count if self.mean is not None else delta / self.obs_count
        delta2 = obs - self.mean
        self.obs_sq_sum = self.obs_sq_sum + delta * delta2
        return obs  # Return original for now, can enable normalization if needed


class RewardNormWrapper(gym.Wrapper):
    """Reward normalization wrapper using running mean and std"""

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        self.ret = 0.0
        self.ret_mean = 0.0
        self.ret_sq_sum = 0.0
        self.ret_count = 0

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        self.ret = 0.0
        return self.env.reset(**kwargs)

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.ret += reward
        return obs, self._normalize_reward(reward), terminated, truncated, info

    def _normalize_reward(self, reward: float) -> float:
        self.ret_count += 1
        delta = reward - self.ret_mean
        self.ret_mean += delta / self.ret_count
        delta2 = reward - self.ret_mean
        self.ret_sq_sum += delta * delta2
        std = np.sqrt(self.ret_sq_sum / self.ret_count + self.epsilon) if self.ret_count > 1 else 1.0
        return (reward - self.ret_mean) / (std + self.epsilon)


class FrameStackWrapper(gym.Wrapper):
    """Frame stacking wrapper for pixel-based environments"""

    def __init__(self, env: gym.Env, num_frames: int = 4):
        super().__init__(env)
        self.num_frames = num_frames
        self.frames = []

        # Modify observation space
        from gymnasium.spaces import Box
        obs_shape = env.observation_space.shape
        new_shape = (obs_shape[0] * num_frames,) + obs_shape[1:]
        self.observation_space = Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_frames):
            self.frames.append(obs)
        return self._stack_frames(), info

    def step(self, action: ActType) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.pop(0)
        self.frames.append(obs)
        return self._stack_frames(), reward, terminated, truncated, info

    def _stack_frames(self) -> np.ndarray:
        return np.stack(self.frames, axis=0)


class ActionNoiseWrapper(gym.Wrapper):
    """Action noise wrapper for exploration in continuous control"""

    def __init__(
        self,
        env: gym.Env,
        noise_type: str = "gaussian",
        noise_scale: float = 0.1,
        noise_decay: float = 1.0,
    ):
        super().__init__(env)
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.noise_decay = noise_decay
        self.current_scale = noise_scale

        if noise_type == "ou":
            from tianshou.exploration import OUNoise
            self.ou_noise = OUNoise(
                sigma=noise_scale,
                action_shape=env.action_space.shape,
            )
        else:
            self.ou_noise = None

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        if self.ou_noise is not None:
            self.ou_noise.reset()
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray) -> Tuple[ObsType, float, bool, bool, dict]:
        # Add noise to action
        if self.noise_type == "gaussian":
            noise = np.random.normal(0, self.current_scale, size=action.shape)
            action = np.clip(action + noise, -1, 1)
        elif self.noise_type == "ou" and self.ou_noise is not None:
            action = action + self.ou_noise()

        # Decay noise scale
        self.current_scale *= self.noise_decay

        return self.env.step(action)


class ClipActionWrapper(gym.Wrapper):
    """Clip actions to valid range for continuous control"""

    def step(self, action: np.ndarray) -> Tuple[ObsType, float, bool, bool, dict]:
        action = np.clip(
            action,
            self.env.action_space.low,
            self.env.action_space.high,
        )
        return self.env.step(action)
