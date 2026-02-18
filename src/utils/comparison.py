"""
Algorithm comparison framework for RL training.
Enables training and comparing multiple algorithms on the same environment.
"""

import os
import json
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time

from src.config import Config, get_config
from src.envs import make_vector_env, get_env_info
from src.trainers.trainer_factory import TrainerFactory
from src.utils.helpers import set_seed, format_duration

logger = logging.getLogger(__name__)


@dataclass
class AlgorithmResult:
    """Result from training a single algorithm"""
    algorithm_name: str
    env_name: str
    best_reward: float
    final_reward: float
    avg_reward: float
    std_reward: float
    total_time: float
    total_episodes: int
    total_steps: int
    reward_history: List[float] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ComparisonResult:
    """Result from comparing multiple algorithms"""
    env_name: str
    results: List[AlgorithmResult]
    timestamp: str
    total_comparison_time: float

    def get_ranking(self) -> List[Tuple[str, float]]:
        """Get algorithms ranked by best reward"""
        ranking = [(r.algorithm_name, r.best_reward) for r in self.results if r.success]
        return sorted(ranking, key=lambda x: x[1], reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "env_name": self.env_name,
            "timestamp": self.timestamp,
            "total_time": self.total_comparison_time,
            "ranking": self.get_ranking(),
            "results": [
                {
                    "algorithm": r.algorithm_name,
                    "best_reward": r.best_reward,
                    "final_reward": r.final_reward,
                    "avg_reward": r.avg_reward,
                    "std_reward": r.std_reward,
                    "total_time": r.total_time,
                    "success": r.success,
                }
                for r in self.results
            ],
        }


class ComparisonRunner:
    """Runner for comparing multiple RL algorithms"""

    def __init__(
        self,
        env_name: str,
        algorithms: List[str] = None,
        base_config: Optional[Config] = None,
        output_dir: str = "./comparison_results",
    ):
        """
        Initialize comparison runner.

        Args:
            env_name: Environment name
            algorithms: List of algorithm names to compare
            base_config: Base configuration (optional)
            output_dir: Directory for saving results
        """
        self.env_name = env_name
        self.algorithms = algorithms or ["dqn", "ppo", "sac", "td3"]
        self.base_config = base_config or get_config(env_name=env_name)
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        self.results: List[AlgorithmResult] = []
        self.training_curves: Dict[str, List[float]] = {}

    def _filter_compatible_algorithms(self) -> List[str]:
        """Filter algorithms compatible with the environment"""
        env_info = get_env_info(make_vector_env(self.env_name, num_envs=1))
        action_type = env_info.action_type

        compatible = []
        for algo in self.algorithms:
            if algo in ["dqn"] and action_type == "discrete":
                compatible.append(algo)
            elif algo in ["ppo"] and action_type == "discrete":
                compatible.append(algo)
            elif algo in ["sac", "td3"] and action_type == "continuous":
                compatible.append(algo)
            elif algo == "ppo" and action_type == "continuous":
                # PPO can work with continuous too, but needs different implementation
                logger.warning(f"PPO with continuous action space needs modification - skipping")
                pass

        logger.info(f"Compatible algorithms for {action_type} action space: {compatible}")
        return compatible

    def train_algorithm(
        self,
        algorithm_name: str,
        config: Config,
    ) -> AlgorithmResult:
        """
        Train a single algorithm.

        Args:
            algorithm_name: Name of algorithm
            config: Configuration

        Returns:
            AlgorithmResult
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {algorithm_name.upper()} on {self.env_name}")
        logger.info(f"{'='*60}\n")

        start_time = time.time()

        try:
            # Set seed for reproducibility
            set_seed(config.seed)

            # Create environments
            train_envs = make_vector_env(
                self.env_name,
                num_envs=config.env.train_num,
                seed=config.seed,
            )
            test_envs = make_vector_env(
                self.env_name,
                num_envs=config.env.test_num,
                seed=config.seed + 1000,
            )

            # Create replay buffer
            buffer_size = config.training.buffer_size
            buffer = torch.zeros(0)  # Placeholder, will be created by trainer

            # Get info about environment
            env_info = get_env_info(
                train_envs if hasattr(train_envs, 'observation_space') else train_envs[0]
            )

            # Update config based on environment
            config.algorithm.name = algorithm_name

            # Create trainer
            trainer = TrainerFactory.get_trainer(
                algorithm_name=algorithm_name,
                train_envs=train_envs,
                test_envs=test_envs,
                buffer=None,  # Will be created internally
                config=config,
            )

            # For now, use a simpler approach - just import and use existing trainers
            # This is a simplified version - full implementation would need
            # proper integration with Tianshou's buffer system

            result = trainer.train()

            total_time = time.time() - start_time

            return AlgorithmResult(
                algorithm_name=algorithm_name,
                env_name=self.env_name,
                best_reward=result.best_reward,
                final_reward=result.final_reward,
                avg_reward=result.final_reward,
                std_reward=0,
                total_time=total_time,
                total_episodes=result.total_episodes,
                total_steps=result.total_steps,
                success=result.success,
            )

        except Exception as e:
            logger.error(f"Error training {algorithm_name}: {e}")
            return AlgorithmResult(
                algorithm_name=algorithm_name,
                env_name=self.env_name,
                best_reward=0,
                final_reward=0,
                avg_reward=0,
                std_reward=0,
                total_time=time.time() - start_time,
                total_episodes=0,
                total_steps=0,
                success=False,
                error_message=str(e),
            )

    def run(
        self,
        epochs: int = 100,
        save_results: bool = True,
    ) -> ComparisonResult:
        """
        Run comparison of all algorithms.

        Args:
            epochs: Number of training epochs per algorithm
            save_results: Whether to save results to file

        Returns:
            ComparisonResult
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Algorithm Comparison")
        logger.info(f"Environment: {self.env_name}")
        logger.info(f"Algorithms: {self.algorithms}")
        logger.info(f"Epochs per algorithm: {epochs}")
        logger.info(f"{'='*60}\n")

        # Filter compatible algorithms
        compatible_algos = self._filter_compatible_algorithms()

        if not compatible_algos:
            raise ValueError(f"No compatible algorithms found for {self.env_name}")

        start_time = time.time()
        self.results = []

        for algo in compatible_algos:
            # Create config for this algorithm
            config = self._create_algorithm_config(algo, epochs)

            # Train
            result = self.train_algorithm(algo, config)
            self.results.append(result)

            # Save intermediate results
            if save_results:
                self._save_intermediate_results()

        total_time = time.time() - start_time

        comparison_result = ComparisonResult(
            env_name=self.env_name,
            results=self.results,
            timestamp=datetime.now().isoformat(),
            total_comparison_time=total_time,
        )

        if save_results:
            self._save_final_results(comparison_result)
            self.plot_comparison(comparison_result)

        return comparison_result

    def _create_algorithm_config(self, algorithm: str, epochs: int) -> Config:
        """Create configuration for a specific algorithm"""
        config = get_config(env_name=self.env_name, algorithm=algorithm, epoch=epochs)

        # Algorithm-specific defaults
        if algorithm == "dqn":
            config.training.step_per_epoch = 10000
            config.algorithm.lr = 1e-3
            config.algorithm.gamma = 0.99
        elif algorithm == "ppo":
            config.training.step_per_epoch = 10000
            config.algorithm.lr = 3e-4
            config.algorithm.gamma = 0.99
        elif algorithm == "sac":
            config.training.step_per_epoch = 15000
            config.algorithm.sac_policy_lr = 3e-4
            config.algorithm.gamma = 0.99
        elif algorithm == "td3":
            config.training.step_per_epoch = 15000
            config.algorithm.td3_actor_lr = 3e-4
            config.algorithm.gamma = 0.99

        return config

    def _save_intermediate_results(self) -> None:
        """Save intermediate results"""
        intermediate_path = os.path.join(self.output_dir, "intermediate_results.json")

        data = {
            "timestamp": datetime.now().isoformat(),
            "env_name": self.env_name,
            "results": [
                {
                    "algorithm": r.algorithm_name,
                    "best_reward": r.best_reward,
                    "success": r.success,
                }
                for r in self.results
            ],
        }

        with open(intermediate_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _save_final_results(self, result: ComparisonResult) -> None:
        """Save final comparison results"""
        final_path = os.path.join(
            self.output_dir,
            f"comparison_{self.env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(final_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Saved final results to {final_path}")

    def plot_comparison(
        self,
        result: ComparisonResult,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot comparison results.

        Args:
            result: ComparisonResult
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Bar chart of best rewards
        ax1 = axes[0, 0]
        successful_results = [(r.algorithm_name, r.best_reward) for r in result.results if r.success]
        if successful_results:
            names = [r[0].upper() for r in successful_results]
            rewards = [r[1] for r in successful_results]
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
            ax1.bar(names, rewards, color=colors)
            ax1.set_xlabel('Algorithm')
            ax1.set_ylabel('Best Reward')
            ax1.set_title('Best Reward by Algorithm')
            ax1.tick_params(axis='x', rotation=0)

        # 2. Training time comparison
        ax2 = axes[0, 1]
        times = [r.total_time for r in result.results if r.success]
        names = [r.algorithm_name.upper() for r in result.results if r.success]
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(times)))
        ax2.bar(names, times, color=colors)
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Training Time (s)')
        ax2.set_title('Training Time Comparison')
        ax2.tick_params(axis='x', rotation=0)

        # 3. Summary table
        ax3 = axes[1, 0]
        ax3.axis('tight')
        ax3.axis('off')

        table_data = []
        headers = ['Algorithm', 'Best Reward', 'Time (s)', 'Episodes']
        for r in result.results:
            status = "OK" if r.success else f"FAIL: {r.error_message}"
            table_data.append([
                r.algorithm_name.upper(),
                f"{r.best_reward:.1f}",
                f"{r.total_time:.1f}",
                str(r.total_episodes),
            ])

        table = ax3.table(
            cellText=table_data,
            colLabels=headers,
            loc='center',
            cellLoc='center',
            colColours=plt.cm.Blues(np.linspace(0.2, 0.5, len(headers))),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        ax3.set_title('Summary', pad=20)

        # 4. Ranking
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')

        ranking = result.get_ranking()
        if ranking:
            rank_data = [[f"#{i+1}", name, f"{reward:.1f}"] for i, (name, reward) in enumerate(ranking)]
            rank_table = ax4.table(
                cellText=rank_data,
                colLabels=['Rank', 'Algorithm', 'Best Reward'],
                loc='center',
                cellLoc='center',
                colColours=plt.cm.Greens(np.linspace(0.2, 0.5, 3)),
            )
            rank_table.auto_set_font_size(False)
            rank_table.set_fontsize(9)
            ax4.set_title('Ranking', pad=20)

        plt.suptitle(f'Algorithm Comparison - {result.env_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {save_path}")

        return fig


def compare_algorithms(
    env_name: str,
    algorithms: List[str] = None,
    epochs: int = 100,
    output_dir: str = "./comparison_results",
) -> ComparisonResult:
    """
    Convenience function to run algorithm comparison.

    Args:
        env_name: Environment name
        algorithms: List of algorithms to compare
        epochs: Training epochs per algorithm
        output_dir: Output directory

    Returns:
        ComparisonResult
    """
    runner = ComparisonRunner(
        env_name=env_name,
        algorithms=algorithms,
        output_dir=output_dir,
    )

    return runner.run(epochs=epochs)


def plot_comparison(
    result: ComparisonResult,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot comparison results from a ComparisonResult.

    Args:
        result: ComparisonResult
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    runner = ComparisonRunner(env_name=result.env_name)
    return runner.plot_comparison(result, save_path)
