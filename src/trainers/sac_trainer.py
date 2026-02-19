"""
SAC (Soft Actor-Critic) Trainer using tianshou 2.0 official APIs.
"""

import torch
import numpy as np
import logging
from typing import Optional, Dict, Any

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import OffPolicyTrainer, OffPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.algorithm.modelfree.sac import SACPolicy, AutoAlpha
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic
from tianshou.algorithm.optim import AdamOptimizerFactory

from src.trainers.base import BaseTrainer, TrainResult

logger = logging.getLogger(__name__)


class SACTrainer(BaseTrainer):
    """Trainer for SAC algorithm using tianshou 2.0 official APIs"""

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

    def _get_env_info(self) -> tuple:
        """Get environment info from vector env"""
        if hasattr(self.train_envs, 'workers') and len(self.train_envs.workers) > 0:
            env = self.train_envs.workers[0].env
        elif hasattr(self.train_envs, 'observation_space'):
            env = self.train_envs
        else:
            env = self.train_envs[0]
        return env, env.observation_space.shape, env.action_space.shape

    def create_policy(self) -> SACPolicy:
        """Create SAC using tianshou 2.0 official APIs"""
        env, state_shape, action_shape = self._get_env_info()

        # Get max action for scaling
        max_action = float(env.action_space.high.max()) if hasattr(env.action_space, 'high') else 1.0

        # Actor network using tianshou's built-in Net
        actor_net = Net(
            state_shape=state_shape,
            hidden_sizes=self.config.network.hidden_sizes,
        ).to(self.config.device.device)

        # Actor: outputs mean and log_std (probabilistic for SAC)
        actor = ContinuousActorProbabilistic(
            preprocess_net=actor_net,
            action_shape=action_shape,
            max_action=max_action,
            hidden_sizes=self.config.network.hidden_sizes,
            conditioned_sigma=True,  # SAC needs conditioned sigma
        ).to(self.config.device.device)

        # Critic 1
        critic1_net = Net(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_sizes=self.config.network.hidden_sizes,
            concat=True,
        ).to(self.config.device.device)

        critic1 = ContinuousCritic(
            preprocess_net=critic1_net,
            hidden_sizes=self.config.network.hidden_sizes,
        ).to(self.config.device.device)

        # Critic 2 (for SAC double Q learning)
        critic2_net = Net(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_sizes=self.config.network.hidden_sizes,
            concat=True,
        ).to(self.config.device.device)

        critic2 = ContinuousCritic(
            preprocess_net=critic2_net,
            hidden_sizes=self.config.network.hidden_sizes,
        ).to(self.config.device.device)

        if self.config.algorithm.sac_automatic_alpha_tuning:
            target_entropy = self.config.algorithm.sac_target_entropy
            if target_entropy is None:
                target_entropy = -float(np.prod(action_shape))
            log_alpha = torch.zeros(1, device=self.config.device.device, requires_grad=True)
            alpha = AutoAlpha(
                target_entropy=target_entropy,
                log_alpha=log_alpha,
                alpha_optim=AdamOptimizerFactory(lr=self.config.algorithm.sac_alpha_lr),
            )
        else:
            alpha = self.config.algorithm.sac_alpha

        # Create policy
        policy = SACPolicy(
            actor=actor,
            actor_optim=AdamOptimizerFactory(lr=self.config.algorithm.sac_policy_lr),
            critic=critic1,
            critic_optim=AdamOptimizerFactory(lr=self.config.algorithm.sac_qf_lr),
            critic2=critic2,
            critic2_optim=AdamOptimizerFactory(lr=self.config.algorithm.sac_qf_lr),
            tau=self.config.algorithm.sac_tau,
            gamma=self.config.algorithm.gamma,
            alpha=alpha,
            exploration_noise=None,  # SAC uses policy noise
            deterministic_eval=True,
            action_space=env.action_space,
        )

        return policy

    def create_algorithm(self) -> SACPolicy:
        """Create SAC algorithm (which is the policy in Tianshou 2.0)"""
        return self.create_policy()

    def create_collectors(self) -> tuple:
        """Create collectors for SAC"""
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
        """Train SAC agent"""
        logger.info("=" * 60)
        logger.info(f"Training SAC on {self.config.env.name}")
        logger.info(f"Config: {self.config.algorithm}")
        logger.info("=" * 60)

        if self.algorithm is None:
            self.algorithm = self.create_algorithm()

        self.setup()

        def stop_fn(mean_rewards):
            return self._stop_fn(mean_rewards)

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

        trainer = OffPolicyTrainer(algorithm=self.algorithm, params=params)

        logger.info("Starting training...")
        
        # Monkey patch collector to handle negative time
        if hasattr(self.train_collector, "collect_stats") and hasattr(self.train_collector.collect_stats, "set_collect_time"):
            original_set_collect_time = self.train_collector.collect_stats.set_collect_time
            
            def safe_set_collect_time(collect_time, **kwargs):
                if collect_time < 0:
                    logger.warning(f"Negative collect time detected: {collect_time}, setting to 0")
                    collect_time = 0.0
                return original_set_collect_time(collect_time, **kwargs)
                
            self.train_collector.collect_stats.set_collect_time = safe_set_collect_time
            if hasattr(self.test_collector, "collect_stats") and hasattr(self.test_collector.collect_stats, "set_collect_time"):
                self.test_collector.collect_stats.set_collect_time = safe_set_collect_time
        
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
        """Test trained SAC policy"""
        import gymnasium as gym
        import numpy as np
        from tianshou.data import Batch
        from src.config import Config

        render_mode = "human" if render else None
        env = gym.make(self.config.env.name, render_mode=render_mode)

        if model_path is None:
            model_path = f"./checkpoints/{self.config.algorithm.name}_{self.config.env.name}_best.pth"

        checkpoint = None
        if model_path:
            state_dict = torch.load(
                model_path,
                map_location=self.config.device.device if torch.cuda.is_available() else 'cpu'
            )

            if isinstance(state_dict, dict) and 'config' in state_dict:
                saved_config = Config.from_dict(state_dict['config'])
                self.config.network = saved_config.network
                logger.info(f"Loaded config from checkpoint: hidden_sizes={saved_config.network.hidden_sizes}")

        test_algorithm = self.create_algorithm()

        if model_path:
            if isinstance(state_dict, dict) and 'algorithm_state_dict' in state_dict:
                test_algorithm.load_state_dict(state_dict['algorithm_state_dict'], strict=False)
            else:
                test_algorithm.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded model from {model_path}")

        total_rewards = []

        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                batch = Batch(obs=[obs], info={})
                with torch.no_grad():
                    action = test_algorithm.policy(batch).act[0].detach().cpu().numpy()

                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                if truncated:
                    logger.info(f"Episode {episode + 1} truncated at step {env._elapsed_steps if hasattr(env, '_elapsed_steps') else 'unknown'} (likely max_episode_steps reached)")

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
