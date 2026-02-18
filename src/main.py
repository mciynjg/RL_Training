"""
Main entry point for RL Training Framework.
Supports training, testing, and comparing multiple algorithms.
"""

import argparse
import sys
import os
import logging
import torch
from typing import Optional, List

from src.config import (
    Config,
    get_config,
    load_config_from_yaml,
    save_config_to_yaml,
    get_predefined_config,
    PREDEFINED_ENVS,
)
from src.envs import make_vector_env, get_env_info, list_available_envs
from src.trainers.trainer_factory import TrainerFactory
from src.utils.helpers import setup_logging, set_seed, print_config
from src.utils.comparison import ComparisonRunner, compare_algorithms

logger = logging.getLogger(__name__)


def train(
    env_name: str,
    algorithm: str,
    config: Config,
    render: bool = False,
) -> None:
    """
    Train an RL agent.

    Args:
        env_name: Environment name
        algorithm: Algorithm name
        config: Configuration
        render: Whether to render during training
    """
    logger.info(f"Training {algorithm.upper()} on {env_name}")
    print_config(config)

    # Validate environment
    try:
        test_env = make_vector_env(env_name, num_envs=1)
        env_info = get_env_info(test_env)
        logger.info(f"Environment: {env_name}")
        logger.info(f"  - Observation shape: {env_info.obs_shape}")
        logger.info(f"  - Action type: {env_info.action_type}")
        logger.info(f"  - Action shape: {env_info.action_shape}")
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        return

    # Validate algorithm compatibility
    if algorithm in ["sac", "td3"] and env_info.action_type != "continuous":
        logger.error(f"{algorithm} requires continuous action space")
        return
    if algorithm in ["dqn"] and env_info.action_type != "discrete":
        logger.error(f"{algorithm} requires discrete action space")
        return

    # Set seed
    set_seed(config.seed)

    # Create environments
    train_envs = make_vector_env(
        env_name,
        num_envs=config.env.train_num,
        seed=config.seed,
    )
    test_envs = make_vector_env(
        env_name,
        num_envs=config.env.test_num,
        seed=config.seed + 1000,
    )

    # Create buffer
    from tianshou.data import VectorReplayBuffer
    buffer = VectorReplayBuffer(
        total_size=config.training.buffer_size,
        buffer_num=len(train_envs) if hasattr(train_envs, '__len__') else 1,
    )

    # Create logger
    from tianshou.utils import TensorboardLogger
    from torch.utils.tensorboard import SummaryWriter
    import time

    log_dir = os.path.join(config.log_dir, f"{algorithm}_{env_name}_{int(time.time())}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    tb_logger = TensorboardLogger(writer)

    logger.info(f"TensorBoard logs: {log_dir}")

    # Create trainer
    trainer = TrainerFactory.get_trainer(
        algorithm_name=algorithm,
        train_envs=train_envs,
        test_envs=test_envs,
        buffer=buffer,
        config=config,
        logger_instance=tb_logger,
    )

    # Train
    result = trainer.train()

    logger.info(f"\nTraining completed!")
    logger.info(f"  - Best reward: {result.best_reward}")
    logger.info(f"  - Total episodes: {result.total_episodes}")
    logger.info(f"  - Total time: {result.training_time:.2f}s")


def test(
    env_name: str,
    algorithm: str,
    model_path: str,
    config: Config,
    num_episodes: int = 10,
    render: bool = True,
) -> None:
    """
    Test a trained model.

    Args:
        env_name: Environment name
        algorithm: Algorithm name
        model_path: Path to model weights
        config: Configuration
        num_episodes: Number of test episodes
        render: Whether to render
    """
    from src.trainers.trainer_factory import TrainerFactory
    from src.envs import make_vector_env

    config.algorithm.name = algorithm

    # Create temporary environments for factory to get env info
    temp_train_envs = make_vector_env(env_name, num_envs=1)
    temp_test_envs = make_vector_env(env_name, num_envs=1)

    # Create buffer (required by factory, but not used in test mode)
    from tianshou.data import VectorReplayBuffer
    temp_buffer = VectorReplayBuffer(total_size=1000, buffer_num=1)

    trainer = TrainerFactory.get_trainer(
        algorithm_name=algorithm,
        train_envs=temp_train_envs,
        test_envs=temp_test_envs,
        buffer=temp_buffer,
        config=config,
    )

    results = trainer.test(
        model_path=model_path,
        num_episodes=num_episodes,
        render=render,
    )

    logger.info(f"\nTest Results:")
    logger.info(f"  - Average reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
    logger.info(f"  - Min reward: {results['min_reward']:.2f}")
    logger.info(f"  - Max reward: {results['max_reward']:.2f}")


def compare(
    env_name: str,
    algorithms: List[str],
    epochs: int,
    output_dir: str,
) -> None:
    """
    Compare multiple algorithms.

    Args:
        env_name: Environment name
        algorithms: List of algorithms to compare
        epochs: Training epochs per algorithm
        output_dir: Output directory for results
    """
    logger.info(f"Comparing algorithms: {algorithms}")
    logger.info(f"Environment: {env_name}")
    logger.info(f"Epochs: {epochs}")

    result = compare_algorithms(
        env_name=env_name,
        algorithms=algorithms,
        epochs=epochs,
        output_dir=output_dir,
    )

    # Print ranking
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 60)

    ranking = result.get_ranking()
    for i, (algo, reward) in enumerate(ranking, 1):
        logger.info(f"  #{i}: {algo.upper()} - Best Reward: {reward:.2f}")

    logger.info(f"\nTotal comparison time: {result.total_comparison_time:.2f}s")
    logger.info(f"Results saved to: {output_dir}")


def list_envs() -> None:
    """List available environments"""
    envs = list_available_envs()

    print("\n" + "=" * 60)
    print("Available Environments")
    print("=" * 60)

    print("\nDiscrete Action Space:")
    for env in envs['discrete'][:20]:
        print(f"  - {env}")

    print("\nContinuous Action Space:")
    for env in envs['continuous'][:20]:
        print(f"  - {env}")

    if envs.get('pybullet'):
        print("\nPyBullet Environments:")
        for env in envs['pybullet']:
            print(f"  - {env}")

    if envs.get('mujoco'):
        print("\nMuJoCo Environments:")
        for env in envs['mujoco']:
            print(f"  - {env}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="RL Training Framework - A unified framework for training, testing, and comparing reinforcement learning algorithms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train DQN on CartPole
  python train.py train --env CartPole-v1 --algo dqn --epochs 100

  # Train PPO with config file (recommended for complex setups)
  python train.py train --config configs/ppo_cartpole.yaml --epochs 100

  # Test a trained model
  python train.py test --env CartPole-v1 --algo dqn --model ./checkpoints/dqn_CartPole-v1_best.pth

  # Compare algorithms
  python train.py compare --env HalfCheetah-v4 --algos sac td3 --epochs 100

  # List available environments
  python train.py list-envs

  # Plot training curves
  python train.py plot --algo dqn ppo --env CartPole-v1 --save results.png
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train an RL agent",
        description="Train a reinforcement learning agent using the specified algorithm and environment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple training with command line args
  python train.py train --env CartPole-v1 --algo dqn --epochs 100

  # Training with config file (recommended)
  python train.py train --config configs/ppo_cartpole.yaml

  # Training with custom parameters
  python train.py train --env CartPole-v1 --algo ppo --epochs 200 --lr 0.0003 --device cuda

Note: When using --config, --env and --algo are optional (taken from config file).
      When not using --config, both --env and --algo are required.
        """
    )
    train_parser.add_argument("--env", type=str,
                              help="Environment name (e.g., CartPole-v1, HalfCheetah-v4)")
    train_parser.add_argument("--algo", type=str,
                             choices=["dqn", "ppo", "sac", "td3"],
                             help="Algorithm: dqn (discrete), ppo (discrete/continuous), sac (continuous), td3 (continuous)")
    train_parser.add_argument("--epochs", type=int, default=None,
                              help="Number of training epochs (default: 100)")
    train_parser.add_argument("--lr", type=float, default=None,
                              help="Learning rate (default: 0.001)")
    train_parser.add_argument("--config", type=str,
                              help="Path to YAML config file (optional, overrides other args)")
    train_parser.add_argument("--seed", type=int, default=None,
                              help="Random seed for reproducibility (default: 42)")
    train_parser.add_argument("--device", type=str, default=None,
                              choices=["auto", "cuda", "cpu"],
                              help="Device to use: auto/cuda/cpu (default: auto)")

    # Test command
    test_parser = subparsers.add_parser(
        "test",
        help="Test a trained model",
        description="Evaluate a trained RL model and display performance statistics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with rendering (default)
  python train.py test --env CartPole-v1 --algo dqn --model ./checkpoints/dqn_CartPole-v1_best.pth

  # Test without rendering, more episodes
  python train.py test --env CartPole-v1 --algo ppo --model model.pth --episodes 50
        """
    )
    test_parser.add_argument("--env", type=str, required=True,
                             help="Environment name (must match the environment used during training)")
    test_parser.add_argument("--algo", type=str, required=True,
                             choices=["dqn", "ppo", "sac", "td3"],
                             help="Algorithm name (must match the algorithm used during training)")
    test_parser.add_argument("--model", type=str, required=True,
                             help="Path to the trained model weights (.pth file)")
    test_parser.add_argument("--episodes", type=int, default=10,
                             help="Number of test episodes to run (default: 10)")
    test_parser.add_argument("--render", action="store_true",
                             help="Render the environment during testing (shows GUI window)")

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare multiple algorithms",
        description="Train and compare multiple RL algorithms on the same environment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare SAC and TD3 (default)
  python train.py compare --env HalfCheetah-v4 --algos sac td3

  # Compare all algorithms
  python train.py compare --env CartPole-v1 --algos dqn ppo

  # Custom epochs and output directory
  python train.py compare --env Walker2d-v4 --algos sac td3 --epochs 200 --output ./results
        """
    )
    compare_parser.add_argument("--env", type=str, required=True,
                                help="Environment name for comparison")
    compare_parser.add_argument("--algos", type=str, nargs="+",
                                default=["sac", "td3"],
                                help="Algorithms to compare (default: sac td3)")
    compare_parser.add_argument("--epochs", type=int, default=100,
                                help="Training epochs per algorithm (default: 100)")
    compare_parser.add_argument("--output", type=str, default="./comparison_results",
                                help="Output directory for results (default: ./comparison_results)")

    # List envs command
    subparsers.add_parser(
        "list-envs",
        help="List available environments",
        description="Display all available training environments categorized by action space type."
    )

    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Generate a YAML config file",
        description="Generate a YAML configuration file for a specific environment and algorithm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate PPO config for CartPole
  python train.py config --env CartPole-v1 --algo ppo --output configs/ppo_cartpole.yaml

  # Generate SAC config for continuous control
  python train.py config --env HalfCheetah-v4 --algo sac
        """
    )
    config_parser.add_argument("--env", type=str, required=True,
                               help="Environment name")
    config_parser.add_argument("--algo", type=str, default="sac",
                               help="Algorithm name (default: sac)")
    config_parser.add_argument("--output", type=str, default="configs/generated_config.yaml",
                               help="Output path for the YAML file (default: configs/generated_config.yaml)")

    # Plot command
    plot_parser = subparsers.add_parser(
        "plot",
        help="Plot training reward curves",
        description="Plot and visualize training reward curves from TensorBoard logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot all available logs
  python train.py plot

  # Plot specific algorithms
  python train.py plot --algo dqn ppo --env CartPole-v1

  # Save plot to file
  python train.py plot --algo ppo --save results.png

  # List available log directories
  python train.py plot --list

  # Custom smoothing window
  python train.py plot --smooth 20
        """
    )
    plot_parser.add_argument("--log-dir", type=str, default="./log",
                             help="Base log directory for TensorBoard logs (default: ./log)")
    plot_parser.add_argument("--save", type=str,
                             help="Path to save the figure (e.g., results.png)")
    plot_parser.add_argument("--smooth", type=int, default=10,
                             help="Smoothing window size for moving average (default: 10)")
    plot_parser.add_argument("--algo", type=str, nargs="+",
                             help="Algorithms to compare (e.g., dqn ppo)")
    plot_parser.add_argument("--env", type=str,
                             help="Environment name to filter logs")
    plot_parser.add_argument("--list", action="store_true",
                             help="List available log directories and exit")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Setup logging
    setup_logging()

    if args.command == "list-envs":
        list_envs()
        return

    if args.command == "plot":
        from src.plot_rewards import plot_rewards, compare_runs, find_log_directories

        if args.list:
            logs = find_log_directories(args.log_dir)
            print(f"\nFound {len(logs)} log directories:")
            for log in logs:
                from src.plot_rewards import parse_log_name
                info = parse_log_name(log)
                print(f"  - {info['algorithm']} / {info['environment']} ({log})")
        elif args.algo or args.env:
            compare_runs(
                log_dir=args.log_dir,
                algorithms=args.algo,
                environment=args.env,
                save_path=args.save,
            )
        else:
            plot_rewards(
                log_dir=args.log_dir,
                save_path=args.save,
                smooth_window=args.smooth,
            )
        return

    if args.command == "config":
        config = get_predefined_config(args.env)
        config.algorithm.name = args.algo
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        save_config_to_yaml(config, args.output)
        logger.info(f"Generated config: {args.output}")
        return

    # Load or create config
    if args.command == "train":
        # Validate required args when no config file
        if not args.config and not args.env:
            train_parser.error("--env is required (or use --config)")
        if not args.config and not args.algo:
            train_parser.error("--algo is required (or use --config)")

    if hasattr(args, 'config') and args.config:
        config = load_config_from_yaml(args.config)
    else:
        config = get_predefined_config(getattr(args, 'env', 'CartPole-v1'))

    # Override with command line args
    if hasattr(args, 'epochs') and args.epochs is not None:
        config.training.epoch = args.epochs
    if hasattr(args, 'lr') and args.lr is not None:
        config.algorithm.lr = args.lr
    if hasattr(args, 'seed') and args.seed is not None:
        config.seed = args.seed
    if hasattr(args, 'device') and args.device is not None:
        config.device.device = args.device

    # Resolve device string to actual device
    if config.device.device == "auto":
        config.device.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.command == "train":
        # Get env and algo from args or config
        env_name = args.env if args.env else config.env.name
        algorithm = args.algo if args.algo else config.algorithm.name

        train(
            env_name=env_name,
            algorithm=algorithm,
            config=config,
        )

    elif args.command == "test":
        test(
            env_name=args.env,
            algorithm=args.algo,
            model_path=args.model,
            config=config,
            num_episodes=args.episodes,
            render=args.render,
        )

    elif args.command == "compare":
        compare(
            env_name=args.env,
            algorithms=args.algos,
            epochs=args.epochs,
            output_dir=args.output,
        )


if __name__ == "__main__":
    main()
