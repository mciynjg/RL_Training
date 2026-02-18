"""
Pytest configuration and fixtures.
"""

import pytest
import torch
import numpy as np
import random


@pytest.fixture(autouse=True)
def set_seed():
    """Automatically set seed for all tests"""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    yield


@pytest.fixture
def cartpole_env():
    """Create CartPole environment"""
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    yield env
    env.close()


@pytest.fixture
def pendulum_env():
    """Create Pendulum environment"""
    import gymnasium as gym
    env = gym.make("Pendulum-v1")
    yield env
    env.close()


@pytest.fixture
def device():
    """Get available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def dummy_batch():
    """Create a dummy batch for testing"""
    from tianshou.data import Batch
    import numpy as np

    return Batch(
        obs=np.random.randn(4, 4).astype(np.float32),
        act=np.random.randint(0, 2, 4),
        rew=np.random.randn(4),
        done=np.random.rand(4) < 0.1,
        terminated=np.random.rand(4) < 0.1,
        truncated=np.random.rand(4) < 0.1,
        obs_next=np.random.randn(4, 4).astype(np.float32),
    )
