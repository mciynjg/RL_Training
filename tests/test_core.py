"""
Unit tests for RL Training Framework.
"""

import pytest
import torch
import gymnasium as gym
from typing import Dict, Any


class TestConfig:
    """Tests for configuration module"""

    def test_default_config(self):
        """Test default configuration creation"""
        from src.config import Config

        config = Config()

        assert config.env.name == "CartPole-v1"
        assert config.algorithm.name == "dqn"
        assert config.training.epoch == 100
        assert config.seed == 42

    def test_get_config(self):
        """Test config with overrides"""
        from src.config import get_config

        config = get_config(
            env_name="LunarLander-v2",
            algorithm="ppo",
            epoch=50,
            lr=0.001,
        )

        assert config.env.name == "LunarLander-v2"
        assert config.algorithm.name == "ppo"
        assert config.training.epoch == 50
        assert config.algorithm.lr == 0.001

    def test_config_to_dict(self):
        """Test config serialization"""
        from src.config import Config

        config = Config()
        data = config.to_dict()

        assert isinstance(data, dict)
        assert "env" in data
        assert "algorithm" in data

    def test_config_from_dict(self):
        """Test config deserialization"""
        from src.config import Config

        data = {
            "env": {"name": "TestEnv-v0"},
            "algorithm": {"name": "sac"},
        }

        config = Config.from_dict(data)

        assert config.env.name == "TestEnv-v0"
        assert config.algorithm.name == "sac"


class TestEnvFactory:
    """Tests for environment factory"""

    def test_make_env_discrete(self):
        """Test creating discrete environment"""
        from src.envs import make_env, get_env_info

        env = make_env("CartPole-v1")
        info = get_env_info(env)

        assert info.action_type == "discrete"
        assert info.action_shape == 2
        assert len(info.obs_shape) == 1

        env.close()

    def test_make_vector_env(self):
        """Test creating vectorized environment"""
        from src.envs import make_vector_env

        envs = make_vector_env("CartPole-v1", num_envs=4)

        assert len(envs) == 4

        envs.close()

    def test_get_env_info_continuous(self):
        """Test getting environment info for continuous env"""
        from src.envs import make_env, get_env_info

        env = make_env("Pendulum-v1")
        info = get_env_info(env)

        assert info.action_type == "continuous"
        assert len(info.action_shape) == 1

        env.close()


class TestHelpers:
    """Tests for utility functions"""

    def test_set_seed(self):
        """Test random seed setting"""
        from src.utils import set_seed
        import numpy as np
        import random

        set_seed(42)
        val1 = random.randint(0, 100)
        np_val1 = np.random.randint(0, 100)

        set_seed(42)
        val2 = random.randint(0, 100)
        np_val2 = np.random.randint(0, 100)

        assert val1 == val2
        assert np_val1 == np_val2

    def test_format_duration(self):
        """Test duration formatting"""
        from src.utils import format_duration

        assert format_duration(30) == "30s"
        assert format_duration(90) == "1m 30s"
        assert format_duration(3660) == "1h 1m 0s"

    def test_count_parameters(self):
        """Test parameter counting"""
        from src.utils import count_parameters
        import torch.nn as nn

        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        # 10*20 + 20 + 20*5 + 5 = 200 + 20 + 100 + 5 = 325
        assert count_parameters(model) == 325


class TestTrainerFactory:
    """Tests for trainer factory"""

    def test_list_algorithms(self):
        """Test listing available algorithms"""
        from src.trainers import TrainerFactory

        algos = TrainerFactory.list_algorithms()

        assert "dqn" in algos
        assert "ppo" in algos
        assert "sac" in algos
        assert "td3" in algos

        # Check DQN info
        assert algos["dqn"]["action_space"] == "discrete"

        # Check SAC info
        assert algos["sac"]["action_space"] == "continuous"


class TestIntegration:
    """Integration tests"""

    def test_dqn_training_step(self):
        """Test a single DQN training step"""
        from src.config import get_config
        from src.envs import make_vector_env
        from src.trainers.trainer_factory import TrainerFactory
        from tianshou.data import VectorReplayBuffer

        config = get_config(env_name="CartPole-v1", algorithm="dqn")
        config.training.epoch = 1  # Just 1 epoch for testing
        config.training.step_per_epoch = 100

        train_envs = make_vector_env("CartPole-v1", num_envs=2)
        test_envs = make_vector_env("CartPole-v1", num_envs=2)

        buffer = VectorReplayBuffer(total_size=1000, buffer_num=2)

        trainer = TrainerFactory.get_trainer(
            algorithm_name="dqn",
            train_envs=train_envs,
            test_envs=test_envs,
            buffer=buffer,
            config=config,
        )

        # Just verify setup works - don't run full training
        assert trainer is not None
        assert trainer.create_policy() is not None

        train_envs.close()
        test_envs.close()

    def test_ppo_training_step(self):
        """Test a single PPO training step"""
        from src.config import get_config
        from src.envs import make_vector_env
        from src.trainers.trainer_factory import TrainerFactory
        from tianshou.data import VectorReplayBuffer

        config = get_config(env_name="CartPole-v1", algorithm="ppo")
        config.training.epoch = 1
        config.training.step_per_epoch = 100

        train_envs = make_vector_env("CartPole-v1", num_envs=2)
        test_envs = make_vector_env("CartPole-v1", num_envs=2)

        buffer = VectorReplayBuffer(total_size=1000, buffer_num=2)

        trainer = TrainerFactory.get_trainer(
            algorithm_name="ppo",
            train_envs=train_envs,
            test_envs=test_envs,
            buffer=buffer,
            config=config,
        )

        assert trainer is not None
        assert trainer.create_policy() is not None

        train_envs.close()
        test_envs.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
