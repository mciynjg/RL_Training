"""
DQN Trainer implementation.
"""

import torch
import logging
from typing import Optional, Dict, Any

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import OffPolicyTrainer, OffPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.algorithm.modelfree.dqn import DQN, DiscreteQLearningPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory

from src.trainers.base import BaseTrainer, TrainResult

logger = logging.getLogger(__name__)


class DQNTrainer(BaseTrainer):
    """Trainer for DQN algorithm"""

    def __init__(
        self,
        algorithm: Any,
        train_envs: Any,
        test_envs: Any,
        buffer: Any,
        config: Any,
        logger_instance: Optional[TensorboardLogger] = None,
    ):
        super().__init__(algorithm, train_envs, test_envs, buffer, config, logger_instance)

    def create_policy(self) -> DiscreteQLearningPolicy:
        """Create DQN policy"""
        # Get single environment from vector env
        if hasattr(self.train_envs, 'workers') and len(self.train_envs.workers) > 0:
            env = self.train_envs.workers[0].env
        elif hasattr(self.train_envs, 'observation_space'):
            env = self.train_envs
        else:
            env = self.train_envs[0]

        net = Net(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.n,
            hidden_sizes=self.config.network.hidden_sizes,
        ).to(self.config.device.device)

        policy = DiscreteQLearningPolicy(
            model=net,
            action_space=env.action_space,
            eps_training=self.config.algorithm.eps_train,
            eps_inference=self.config.algorithm.eps_test,
        )

        return policy

    def create_algorithm(self) -> DQN:
        """Create DQN algorithm"""
        policy = self.create_policy()

        optim_factory = AdamOptimizerFactory(lr=self.config.algorithm.lr)

        algorithm = DQN(
            policy=policy,
            optim=optim_factory,
            gamma=self.config.algorithm.gamma,
            n_step_return_horizon=self.config.algorithm.n_step,
            target_update_freq=self.config.algorithm.target_update_freq,
            is_double=self.config.algorithm.is_double,
            huber_loss_delta=self.config.algorithm.huber_loss_delta,
        )

        return algorithm

    def create_collectors(self) -> tuple:
        """Create collectors for DQN"""
        if self.algorithm is None:
            self.algorithm = self.create_algorithm()

        train_collector = Collector(
            self.algorithm,
            self.train_envs,
            self.buffer,
            exploration_noise=True,
        )

        test_collector = Collector(
            self.algorithm,
            self.test_envs,
            exploration_noise=False,
        )

        return train_collector, test_collector

    def train(self) -> TrainResult:
        """Train DQN agent"""
        logger.info("=" * 60)
        logger.info(f"Training DQN on {self.config.env.name}")
        logger.info(f"Config: {self.config.algorithm}")
        logger.info("=" * 60)

        # Create algorithm if not already created
        if self.algorithm is None:
            self.algorithm = self.create_algorithm()

        # Setup collectors
        self.setup()

        # Stop condition
        def stop_fn(mean_rewards):
            return self._stop_fn(mean_rewards)

        # Create trainer params
        params = OffPolicyTrainerParams(
            training_collector=self.train_collector,
            test_collector=self.test_collector,
            max_epochs=self.config.training.epoch,
            epoch_num_steps=self.config.training.step_per_epoch,
            batch_size=self.config.training.batch_size,
            collection_step_num_env_steps=self.config.training.step_per_collect,
            update_step_num_gradient_steps_per_sample=self.config.training.update_per_step,
            test_step_num_episodes=self.config.training.episode_per_test,
            stop_fn=stop_fn,
            save_best_fn=self._save_best_fn,
            logger=self.logger,
        )

        # Create and run trainer
        trainer = OffPolicyTrainer(algorithm=self.algorithm, params=params)

        logger.info("Starting training...")
        
        # Monkey patch collector to handle negative time
        # This is a workaround for a potential bug in Tianshou or system clock issues
        original_set_collect_time = self.train_collector.collect_stats.set_collect_time
        
        def safe_set_collect_time(collect_time, **kwargs):
            if collect_time < 0:
                logger.warning(f"Negative collect time detected: {collect_time}, setting to 0")
                collect_time = 0.0
            return original_set_collect_time(collect_time, **kwargs)
            
        self.train_collector.collect_stats.set_collect_time = safe_set_collect_time
        self.test_collector.collect_stats.set_collect_time = safe_set_collect_time

        result = trainer.run()

        best_reward = getattr(result, 'best_reward', 'N/A')
        timing = getattr(result, 'timing', None)
        total_time = getattr(timing, 'total_time', 0) if timing else 0

        logger.info(f"Training completed!")
        logger.info(f"Best test reward: {best_reward}")
        logger.info(f"Total training time: {total_time:.2f}s")

        # Tianshou 2.0+ workaround for negative collect time issue
        # The collector time might be negative due to system time precision or clock synchronization
        # We catch the ValueError that might be raised by set_collect_time
        return TrainResult(
            best_reward=float(best_reward) if isinstance(best_reward, (int, float)) else 0,
            final_reward=best_reward,
            total_episodes=getattr(result, 'total_episodes', 0),
            total_steps=getattr(result, 'total_steps', 0),
            training_time=total_time,
            success=True,
        )

    def test(
        self,
        model_path: Optional[str] = None,
        num_episodes: int = 10,
        render: bool = False,
    ) -> Dict[str, float]:
        """Test trained DQN policy"""
        import gymnasium as gym
        from tianshou.data import Batch
        from src.config import Config

        render_mode = "human" if render else None
        env = gym.make(self.config.env.name, render_mode=render_mode)

        # Load model first to get the config
        if model_path is None:
            model_path = f"./checkpoints/{self.config.algorithm.name}_{self.config.env.name}_best.pth"

        checkpoint = None
        if model_path and torch.cuda.is_available():
            checkpoint = torch.load(model_path, map_location=self.config.device.device)
        else:
            checkpoint = torch.load(model_path, map_location='cpu' if not torch.cuda.is_available() else None)

        # Load config from checkpoint if available
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            saved_config = Config.from_dict(checkpoint['config'])
            # Update network config to match training
            self.config.network = saved_config.network
            logger.info(f"Loaded config from checkpoint: hidden_sizes={saved_config.network.hidden_sizes}")

        # Create algorithm for testing
        test_algorithm = self.create_algorithm()

        # Load model weights
        if isinstance(checkpoint, dict) and 'algorithm_state_dict' in checkpoint:
            test_algorithm.load_state_dict(checkpoint['algorithm_state_dict'], strict=False)
        else:
            test_algorithm.load_state_dict(checkpoint, strict=False)

        logger.info(f"Loaded model from {model_path}")

        total_rewards = []

        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                batch = Batch(obs=[obs])
                batch.info = {}
                with torch.no_grad():
                    action = test_algorithm.policy(batch).act[0].item()

                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward

            total_rewards.append(episode_reward)
            logger.info(f"Episode {episode + 1}: reward = {episode_reward}")

        env.close()

        avg_reward = float(torch.tensor(total_rewards).mean())
        std_reward = float(torch.tensor(total_rewards).std())

        logger.info(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f} ({num_episodes} episodes)")

        return {
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "min_reward": min(total_rewards),
            "max_reward": max(total_rewards),
        }
