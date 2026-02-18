"""
Base trainer class for RL algorithms.
"""

import os
import torch
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

from tianshou.data import Collector, ReplayBuffer
from tianshou.env import BaseVectorEnv
from tianshou.utils import TensorboardLogger

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Training result dataclass"""
    best_reward: float
    final_reward: float
    total_episodes: int
    total_steps: int
    training_time: float
    success: bool


class BaseTrainer(ABC):
    """Abstract base class for all RL trainers"""

    def __init__(
        self,
        algorithm: Any,
        train_envs: BaseVectorEnv,
        test_envs: BaseVectorEnv,
        buffer: ReplayBuffer,
        config: Any,
        logger: Optional[TensorboardLogger] = None,
    ):
        self.algorithm = algorithm
        self.train_envs = train_envs
        self.test_envs = test_envs
        self.buffer = buffer
        self.config = config
        self.logger = logger

        self.train_collector: Optional[Collector] = None
        self.test_collector: Optional[Collector] = None

    @abstractmethod
    def create_policy(self) -> Any:
        """Create the RL policy for the algorithm"""
        pass

    @abstractmethod
    def create_collectors(self) -> tuple:
        """Create training and test collectors"""
        pass

    def setup(self) -> None:
        """Setup the trainer before training"""
        self.train_collector, self.test_collector = self.create_collectors()
        logger.info("Trainer setup complete")

    def _stop_fn(self, mean_rewards: float) -> bool:
        """Default stop condition - can be overridden"""
        # Try to get reward threshold from environment spec
        try:
            import gymnasium as gym
            env_spec = gym.spec(self.config.env.name)
            if hasattr(env_spec, 'reward_threshold'):
                return mean_rewards >= env_spec.reward_threshold
        except Exception:
            pass
        return False

    def _save_best_fn(self, algorithm: Any) -> None:
        """Save the best model"""
        checkpoint_dir = self.config.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_path = os.path.join(
            checkpoint_dir,
            f"{self.config.algorithm.name}_{self.config.env.name}_best.pth"
        )

        torch.save({
            'algorithm_state_dict': algorithm.state_dict(),
            'config': self.config.to_dict(),
        }, model_path)
        logger.info(f"Saved best model to {model_path}")

    def _save_checkpoint_fn(
        self,
        algorithm: Any,
        epoch: int,
        reward: float,
    ) -> None:
        """Save periodic checkpoint"""
        if self.config.training.save_freq <= 0:
            return

        if epoch % self.config.training.save_freq != 0:
            return

        checkpoint_dir = os.path.join(self.config.checkpoint_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"{self.config.algorithm.name}_{self.config.env.name}_epoch{epoch}.pth"
        )

        torch.save({
            'epoch': epoch,
            'algorithm_state_dict': algorithm.state_dict(),
            'reward': reward,
            'config': self.config.to_dict(),
        }, checkpoint_path)
        logger.debug(f"Saved checkpoint to {checkpoint_path}")

    @abstractmethod
    def train(self) -> TrainResult:
        """Run training loop"""
        pass

    @abstractmethod
    def test(
        self,
        model_path: Optional[str] = None,
        num_episodes: int = 10,
        render: bool = False,
    ) -> Dict[str, float]:
        """Test the trained policy"""
        pass

    def get_device(self) -> torch.device:
        """Get the device for training"""
        return torch.device(self.config.device.device)
