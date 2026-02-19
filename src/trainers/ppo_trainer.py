"""
PPO Trainer implementation.
"""

import torch
import logging
import torch.distributions as dist
from typing import Optional, Dict, Any

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import OnPolicyTrainer, OnPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import DiscreteCritic
from tianshou.algorithm.modelfree.ppo import PPO
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory

from src.trainers.base import BaseTrainer, TrainResult

logger = logging.getLogger(__name__)


class PPOTrainer(BaseTrainer):
    """Trainer for PPO algorithm"""

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

    def create_policy(self) -> ProbabilisticActorPolicy:
        """Create PPO policy for both discrete and continuous actions"""
        # Get single environment from vector env
        if hasattr(self.train_envs, 'workers') and len(self.train_envs.workers) > 0:
            env = self.train_envs.workers[0].env
        elif hasattr(self.train_envs, 'observation_space'):
            env = self.train_envs
        else:
            env = self.train_envs[0]

        is_continuous = hasattr(env.action_space, 'shape') and len(env.action_space.shape) > 0 and hasattr(env.action_space, 'high')

        if is_continuous:
            from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic
            
            action_shape = env.action_space.shape
            max_action = float(env.action_space.high.max())
            
            # Actor network
            actor_net = Net(
                state_shape=env.observation_space.shape,
                hidden_sizes=self.config.network.hidden_sizes,
                activation=torch.nn.Tanh,
            ).to(self.config.device.device)
            
            actor = ContinuousActorProbabilistic(
                preprocess_net=actor_net,
                action_shape=action_shape,
                max_action=max_action,
                hidden_sizes=self.config.network.hidden_sizes,
            ).to(self.config.device.device)

            # Critic network
            critic_net = Net(
                state_shape=env.observation_space.shape,
                hidden_sizes=self.config.network.hidden_sizes,
                activation=torch.nn.Tanh,
            ).to(self.config.device.device)
            
            self.critic = ContinuousCritic(
                preprocess_net=critic_net,
                hidden_sizes=self.config.network.hidden_sizes,
            ).to(self.config.device.device)

        if is_continuous:
            # Policy
            policy = ProbabilisticActorPolicy(
                actor=actor,
                dist_fn=dist.Normal,
                action_space=env.action_space,
                deterministic_eval=True,
                action_scaling=True,
                action_bound_method="clip",
            )
            
        else:
            # Discrete action space
            from tianshou.utils.net.discrete import DiscreteCritic
            
            # Actor network
            actor_net = Net(
                state_shape=env.observation_space.shape,
                action_shape=env.action_space.n,
                hidden_sizes=self.config.network.hidden_sizes,
                softmax=True,
            ).to(self.config.device.device)

            # Critic network
            self.critic = DiscreteCritic(
                preprocess_net=Net(
                    state_shape=env.observation_space.shape,
                    hidden_sizes=self.config.network.hidden_sizes,
                ).to(self.config.device.device),
                hidden_sizes=self.config.network.hidden_sizes,
            ).to(self.config.device.device)

            # Policy
            policy = ProbabilisticActorPolicy(
                actor=actor_net,
                dist_fn=dist.categorical.Categorical,
                action_space=env.action_space,
                deterministic_eval=True,
                action_scaling=False,
            )

        return policy

    def create_algorithm(self) -> PPO:
        """Create PPO algorithm"""
        policy = self.create_policy()

        optim_factory = AdamOptimizerFactory(lr=self.config.algorithm.lr)

        algorithm = PPO(
            policy=policy,
            critic=self.critic,
            optim=optim_factory,
            eps_clip=self.config.algorithm.ppo_clip_ratio,
            vf_coef=self.config.algorithm.ppo_value_coef,
            ent_coef=self.config.algorithm.ppo_ent_coef,
            max_grad_norm=self.config.algorithm.ppo_max_grad_norm,
            gae_lambda=self.config.algorithm.ppo_gae_lambda,
            dual_clip=None,
            gamma=self.config.algorithm.gamma,
        )

        return algorithm

    def create_collectors(self) -> tuple:
        """Create collectors for PPO"""
        if self.algorithm is None:
            self.algorithm = self.create_algorithm()

        # On-policy algorithms use fresh buffers per epoch
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
        """Train PPO agent"""
        logger.info("=" * 60)
        logger.info(f"Training PPO on {self.config.env.name}")
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
        params = OnPolicyTrainerParams(
            training_collector=self.train_collector,
            test_collector=self.test_collector,
            max_epochs=self.config.training.epoch,
            epoch_num_steps=self.config.training.step_per_epoch,
            update_step_num_repetitions=self.config.algorithm.ppo_repeat_per_collect,
            test_step_num_episodes=self.config.training.episode_per_test,
            batch_size=self.config.training.batch_size,
            stop_fn=stop_fn,
            save_best_fn=self._save_best_fn,
            logger=self.logger,
        )

        # Create and run trainer
        trainer = OnPolicyTrainer(algorithm=self.algorithm, params=params)

        logger.info("Starting training...")
        result = trainer.run()

        best_reward = getattr(result, 'best_reward', 'N/A')
        timing = getattr(result, 'timing', None)
        total_time = getattr(timing, 'total_time', 0) if timing else 0

        logger.info(f"Training completed!")
        logger.info(f"Best test reward: {best_reward}")
        logger.info(f"Total training time: {total_time:.2f}s")

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
        """Test trained PPO policy"""
        import gymnasium as gym
        import numpy as np
        from tianshou.data import Batch
        from src.config import Config

        render_mode = "human" if render else None
        env = gym.make(self.config.env.name, render_mode=render_mode)

        # Load model first to get the config
        if model_path is None:
            model_path = f"./checkpoints/{self.config.algorithm.name}_{self.config.env.name}_best.pth"

        if model_path:
            if torch.cuda.is_available():
                checkpoint = torch.load(model_path, map_location=self.config.device.device)
            else:
                checkpoint = torch.load(model_path, map_location='cpu')

            # Load config from checkpoint if available
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                saved_config = Config.from_dict(checkpoint['config'])
                # Update network config to match training
                self.config.network = saved_config.network
                logger.info(f"Loaded config from checkpoint: hidden_sizes={saved_config.network.hidden_sizes}")

        # Create algorithm for testing with loaded config
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
                batch = Batch(obs=[obs], info={})
                with torch.no_grad():
                    # For PPO, the policy output might be different
                    # Tianshou's PPO policy returns a Batch with 'act'
                    result = test_algorithm.policy(batch)
                    if hasattr(result, 'act'):
                         action = result.act[0].item() if not isinstance(result.act[0], (np.ndarray, list)) else result.act[0]
                    else:
                         action = result[0] # Fallback
                         
                # Handle numpy/tensor conversion if needed
                if hasattr(action, 'detach'):
                    action = action.detach().cpu().numpy()
                if isinstance(action, np.ndarray) and action.ndim == 0:
                    action = action.item()

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
