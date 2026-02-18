"""
Environment factory for creating RL environments.
Supports Gymnasium, PyBullet, and MuJoCo environments.
"""

import gymnasium as gym
from typing import Optional, Dict, Any, Tuple, Union, List
from dataclasses import dataclass

try:
    import pybullet_envs_gym
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    print("Warning: pybullet_envs_gym not available. Install with: pip install pybullet")

try:
    import mujoco
    import gymnasium.envs.mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("Warning: mujoco not available. Install with: pip install mujoco")

from gymnasium.wrappers import NormalizeObservation, NormalizeReward, TimeLimit


@dataclass
class EnvInfo:
    """Environment information"""
    name: str
    obs_shape: Tuple[int, ...]
    obs_dtype: str
    action_type: str  # "discrete" or "continuous"
    action_shape: Union[int, Tuple[int, ...]]
    action_dtype: Optional[str] = None
    min_action: Optional[float] = None
    max_action: Optional[float] = None
    max_episode_steps: Optional[int] = None
    reward_range: Tuple[float, float] = (-float('inf'), float('inf'))


def get_env_info(env: gym.Env) -> EnvInfo:
    """
    Extract information from environment.

    Args:
        env: Gymnasium environment

    Returns:
        EnvInfo object with environment details
    """
    obs_shape = env.observation_space.shape
    obs_dtype = str(env.observation_space.dtype)

    if isinstance(env.action_space, gym.spaces.Discrete):
        return EnvInfo(
            name=env.spec.id if env.spec else "unknown",
            obs_shape=obs_shape,
            obs_dtype=obs_dtype,
            action_type="discrete",
            action_shape=env.action_space.n,
            action_dtype="int64",
            max_episode_steps=env.spec.max_episode_steps if env.spec else None,
            reward_range=env.reward_range if hasattr(env, 'reward_range') else (-float('inf'), float('inf')),
        )
    elif isinstance(env.action_space, gym.spaces.Box):
        return EnvInfo(
            name=env.spec.id if env.spec else "unknown",
            obs_shape=obs_shape,
            obs_dtype=obs_dtype,
            action_type="continuous",
            action_shape=env.action_space.shape,
            action_dtype=str(env.action_space.dtype),
            min_action=float(env.action_space.low.min()) if env.action_space.low.ndim > 0 else float(env.action_space.low),
            max_action=float(env.action_space.high.max()) if env.action_space.high.ndim > 0 else float(env.action_space.high),
            max_episode_steps=env.spec.max_episode_steps if env.spec else None,
            reward_range=env.reward_range if hasattr(env, 'reward_range') else (-float('inf'), float('inf')),
        )
    else:
        raise ValueError(f"Unsupported action space: {type(env.action_space)}")


# Environment registry for custom configurations
ENV_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_env(name: str, entry_point: str, **kwargs) -> None:
    """
    Register a custom environment.

    Args:
        name: Environment name
        entry_point: Module path to environment class
        **kwargs: Additional arguments for environment creation
    """
    ENV_REGISTRY[name] = {
        "entry_point": entry_point,
        "kwargs": kwargs,
    }


def _create_env_internal(
    env_name: str,
    render_mode: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
    **kwargs
) -> gym.Env:
    """
    Internal function to create an environment.

    Args:
        env_name: Environment name
        render_mode: Render mode (None, "human", "rgb_array")
        max_episode_steps: Maximum steps per episode
        **kwargs: Additional arguments

    Returns:
        Gymnasium environment
    """
    # Check custom registry
    if env_name in ENV_REGISTRY:
        registry_entry = ENV_REGISTRY[env_name]
        env = gym.make(registry_entry["entry_point"], **registry_entry.get("kwargs", {}))
    else:
        # Standard gymnasium environment
        render_kwargs = {"render_mode": render_mode} if render_mode else {}
        env = gym.make(env_name, **render_kwargs, **kwargs)

    # Apply time limit if specified
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps)

    return env


def make_env(
    env_name: str,
    render_mode: Optional[str] = None,
    normalize_obs: bool = False,
    normalize_reward: bool = False,
    max_episode_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> gym.Env:
    """
    Create a single environment with optional wrappers.

    Args:
        env_name: Environment name (e.g., "CartPole-v1", "HalfCheetah-v4")
        render_mode: Render mode (None, "human", "rgb_array")
        normalize_obs: Whether to normalize observations
        normalize_reward: Whether to normalize rewards (continuous control only)
        max_episode_steps: Maximum steps per episode
        seed: Random seed for reproducibility

    Returns:
        Configured gymnasium environment
    """
    env = _create_env_internal(
        env_name,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )

    # Apply normalization wrappers
    if normalize_obs:
        env = NormalizeObservation(env)
    if normalize_reward:
        # Only apply to continuous control environments
        info = get_env_info(env)
        if info.action_type == "continuous":
            env = NormalizeReward(env)

    # Set seed if provided
    if seed is not None:
        env.reset(seed=seed)

    return env


def make_vector_env(
    env_name: str,
    num_envs: int = 1,
    render_mode: Optional[str] = None,
    normalize_obs: bool = False,
    normalize_reward: bool = False,
    max_episode_steps: Optional[int] = None,
    seed: Optional[int] = None,
    vector_mode: str = "dummy",
) -> Union[gym.Env, "DummyVectorEnv", "SubprocVectorEnv"]:
    """
    Create vectorized environments for parallel training.

    Args:
        env_name: Environment name
        num_envs: Number of parallel environments
        render_mode: Render mode
        normalize_obs: Whether to normalize observations
        normalize_reward: Whether to normalize rewards
        max_episode_steps: Maximum steps per episode
        seed: Random seed (each env gets seed + worker_id)
        vector_mode: "dummy" for single-process, "subproc" for multi-process

    Returns:
        Vectorized environment
    """
    from tianshou.env import DummyVectorEnv, SubprocVectorEnv

    if num_envs == 1:
        return make_env(
            env_name,
            render_mode=render_mode,
            normalize_obs=normalize_obs,
            normalize_reward=normalize_reward,
            max_episode_steps=max_episode_steps,
            seed=seed,
        )

    def make_single_env(idx: int):
        env_seed = (seed + idx) if seed is not None else None
        return make_env(
            env_name,
            render_mode=None,  # No rendering for vector envs
            normalize_obs=normalize_obs,
            normalize_reward=normalize_reward,
            max_episode_steps=max_episode_steps,
            seed=env_seed,
        )

    if vector_mode == "dummy":
        return DummyVectorEnv([lambda i=i: make_single_env(i) for i in range(num_envs)])
    elif vector_mode == "subproc":
        return SubprocVectorEnv([lambda i=i: make_single_env(i) for i in range(num_envs)])
    else:
        raise ValueError(f"Unknown vector_mode: {vector_mode}")


# List of available continuous control environments
CONTINUOUS_ENVS = [
    # PyBullet
    "HalfCheetahBulletEnv-v0",
    "Walker2DBulletEnv-v0",
    "AntBulletEnv-v0",
    "HopperBulletEnv-v0",
    "ReacherBulletEnv-v0",
    # MuJoCo (v4)
    "HalfCheetah-v4",
    "Walker2d-v4",
    "Ant-v4",
    "Hopper-v4",
    "Humanoid-v4",
    "Swimmer-v4",
    # MuJoCo (v3 - legacy)
    "HalfCheetah-v3",
    "Walker2d-v3",
    "Ant-v3",
    "Hopper-v3",
]


def list_available_envs() -> Dict[str, List[str]]:
    """
    List all available environments.

    Returns:
        Dictionary with categories and environment names
    """
    discrete_envs = [
        env.id for env in gym.envs.registry.values()
        if hasattr(env, 'entry_point') and
        isinstance(gym.make(env.id).action_space, gym.spaces.Discrete)
    ]

    continuous_envs = [
        env.id for env in gym.envs.registry.values()
        if hasattr(env, 'entry_point') and
        isinstance(gym.make(env.id).action_space, gym.spaces.Box)
    ]

    return {
        "discrete": discrete_envs[:50],  # Limit for readability
        "continuous": continuous_envs[:50],
        "pybullet": [e for e in CONTINUOUS_ENVS if "Bullet" in e] if PYBULLET_AVAILABLE else [],
        "mujoco": [e for e in CONTINUOUS_ENVS if "Bullet" not in e] if MUJOCO_AVAILABLE else [],
    }
