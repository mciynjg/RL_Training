"""
Factory class for creating trainers based on algorithm name.
"""

import logging
from typing import Any, Optional

from tianshou.data import VectorReplayBuffer
from tianshou.utils import TensorboardLogger

from src.trainers.base import BaseTrainer
from src.trainers.dqn_trainer import DQNTrainer
from src.trainers.ppo_trainer import PPOTrainer
from src.trainers.sac_trainer import SACTrainer
from src.trainers.td3_trainer import TD3Trainer
from src.envs import get_env_info

logger = logging.getLogger(__name__)


class TrainerFactory:
    """Factory for creating RL trainers"""

    @staticmethod
    def get_trainer(
        algorithm_name: str,
        train_envs: Any,
        test_envs: Any,
        buffer: VectorReplayBuffer,
        config: Any,
        logger_instance: Optional[TensorboardLogger] = None,
    ) -> BaseTrainer:
        """
        Create a trainer based on algorithm name.

        Args:
            algorithm_name: Name of algorithm (dqn, ppo, sac, td3)
            train_envs: Training environments
            test_envs: Test environments
            buffer: Replay buffer
            config: Configuration object
            logger_instance: Tensorboard logger

        Returns:
            Trainer instance
        """
        algorithm_name = algorithm_name.lower()

        # Check if environment is compatible with algorithm
        # Get a single environment to check info
        if hasattr(train_envs, 'workers') and len(train_envs.workers) > 0:
            # Tianshou DummyVectorEnv/SubprocVectorEnv
            single_env = train_envs.workers[0].env
        elif hasattr(train_envs, 'observation_space'):
            # Single gym environment
            single_env = train_envs
        elif hasattr(train_envs, '__getitem__'):
            single_env = train_envs[0]
        else:
            single_env = train_envs

        env_info = get_env_info(single_env)

        # Validate algorithm-environment compatibility
        if algorithm_name in ["sac", "td3"] and env_info.action_type != "continuous":
            raise ValueError(
                f"{algorithm_name.upper()} requires continuous action space, "
                f"but got {env_info.action_type} action space"
            )

        if algorithm_name in ["dqn", "ppo"] and env_info.action_type != "discrete":
            # PPO can actually handle continuous actions too, but our simple implementation is discrete-only
            logger.warning(
                f"{algorithm_name.upper()} is typically used with discrete action spaces. "
                f"Consider using SAC or TD3 for continuous control."
            )

        trainers = {
            "dqn": DQNTrainer,
            "ppo": PPOTrainer,
            "sac": SACTrainer,
            "td3": TD3Trainer,
        }

        if algorithm_name not in trainers:
            raise ValueError(
                f"Unknown algorithm: {algorithm_name}. "
                f"Available algorithms: {list(trainers.keys())}"
            )

        trainer_class = trainers[algorithm_name]

        return trainer_class(
            algorithm=None,  # Will be created by trainer
            train_envs=train_envs,
            test_envs=test_envs,
            buffer=buffer,
            config=config,
            logger_instance=logger_instance,
        )

    @staticmethod
    def list_algorithms() -> dict:
        """
        List available algorithms and their descriptions.

        Returns:
            Dictionary of algorithm info
        """
        return {
            "dqn": {
                "name": "Deep Q-Network",
                "type": "off-policy",
                "action_space": "discrete",
                "description": "Value-based method using Q-learning with neural networks",
            },
            "ppo": {
                "name": "Proximal Policy Optimization",
                "type": "on-policy",
                "action_space": "discrete (can be extended to continuous)",
                "description": "Policy gradient method with clipped surrogate objective",
            },
            "sac": {
                "name": "Soft Actor-Critic",
                "type": "off-policy",
                "action_space": "continuous",
                "description": "Maximum entropy RL algorithm for continuous control",
            },
            "td3": {
                "name": "Twin Delayed DDPG",
                "type": "off-policy",
                "action_space": "continuous",
                "description": "Improved DDPG with clipped double Q-learning",
            },
        }
