"""
Unified configuration management for RL training.
Supports both Python dataclass and YAML configuration.
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import torch


@dataclass
class EnvConfig:
    """Environment configuration"""
    name: str = "CartPole-v1"
    train_num: int = 10
    test_num: int = 100
    max_episode_steps: Optional[int] = None
    normalize_obs: bool = False
    normalize_reward: bool = False
    frame_stack: int = 1
    grayscale: bool = False


@dataclass
class NetworkConfig:
    """Network architecture configuration"""
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    dropout: float = 0.0
    use_layer_norm: bool = False


@dataclass
class AlgorithmConfig:
    """Algorithm-specific configuration"""
    name: str = "dqn"  # dqn, ppo, sac, td3, ddpg

    # Common
    gamma: float = 0.99
    lr: float = 1e-3

    # DQN specific
    n_step: int = 3
    target_update_freq: int = 320
    eps_train: float = 0.1
    eps_test: float = 0.05
    is_double: bool = True
    huber_loss_delta: Optional[float] = None

    # PPO specific
    ppo_clip_ratio: float = 0.2
    ppo_value_coef: float = 0.5
    ppo_ent_coef: float = 0.01
    ppo_max_grad_norm: float = 0.5
    ppo_gae_lambda: float = 0.95
    ppo_repeat_per_collect: int = 2

    # SAC specific
    sac_alpha: float = 0.2
    sac_alpha_lr: float = 3e-4
    sac_policy_lr: float = 3e-4
    sac_qf_lr: float = 3e-4
    sac_buffer_size: int = 1000000
    sac_automatic_alpha_tuning: bool = True
    sac_target_entropy: Optional[float] = None
    sac_target_update_mode: str = "soft"
    sac_tau: float = 0.005

    # TD3 specific
    td3_actor_lr: float = 3e-4
    td3_critic_lr: float = 3e-4
    td3_policy_noise: float = 0.2
    td3_noise_clip: float = 0.5
    td3_policy_freq: int = 2
    td3_tau: float = 0.005


@dataclass
class TrainingConfig:
    """Training loop configuration"""
    epoch: int = 100
    step_per_epoch: int = 10000
    step_per_collect: int = 2000
    update_per_step: float = 1.0
    episode_per_test: int = 100
    batch_size: int = 256
    buffer_size: int = 100000
    warmup_steps: int = 5000
    eval_freq: int = 1
    save_freq: int = 10


@dataclass
class DeviceConfig:
    """Device configuration"""
    device: str = "auto"
    num_threads: int = 4

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_device(self) -> torch.device:
        """Get torch device"""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


@dataclass
class Config:
    """Main configuration class"""
    # Sections
    env: EnvConfig = field(default_factory=EnvConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)

    # Metadata
    seed: int = 42
    log_dir: str = "./log"
    checkpoint_dir: str = "./checkpoints"
    project_name: str = "rl-training"
    run_name: Optional[str] = None
    use_tensorboard: bool = True
    use_wandb: bool = False
    verbose: bool = True

    def __post_init__(self):
        if self.run_name is None:
            self.run_name = f"{self.algorithm.name}_{self.env.name}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary"""
        config = cls()

        if "env" in data:
            config.env = EnvConfig(**data["env"])
        if "network" in data:
            config.network = NetworkConfig(**data["network"])
        if "algorithm" in data:
            config.algorithm = AlgorithmConfig(**data["algorithm"])
        if "training" in data:
            config.training = TrainingConfig(**data["training"])
        if "device" in data:
            config.device = DeviceConfig(**data["device"])

        # Update simple fields
        for key in ["seed", "log_dir", "checkpoint_dir", "project_name",
                    "run_name", "use_tensorboard", "use_wandb", "verbose"]:
            if key in data:
                setattr(config, key, data[key])

        return config


def get_config(
    env_name: str = None,
    algorithm: str = None,
    epoch: int = None,
    lr: float = None,
    **kwargs
) -> Config:
    """
    Get a config with optional overrides.

    Args:
        env_name: Environment name
        algorithm: Algorithm name (dqn, ppo, sac, td3)
        epoch: Number of training epochs
        lr: Learning rate
        **kwargs: Additional config overrides

    Returns:
        Config object
    """
    config = Config()

    if env_name:
        config.env.name = env_name
    if algorithm:
        config.algorithm.name = algorithm
    if epoch:
        config.training.epoch = epoch
    if lr:
        config.algorithm.lr = lr

    # Apply additional overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def load_config_from_yaml(yaml_path: str) -> Config:
    """
    Load configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        Config object
    """
    import yaml

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    return Config.from_dict(data)


def save_config_to_yaml(config: Config, yaml_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Config object
        yaml_path: Path to output YAML file
    """
    import yaml

    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)


# Predefined configurations for common environments
PREDEFINED_ENVS = {
    # Discrete environments
    "CartPole-v1": {
        "env": {"name": "CartPole-v1"},
        "training": {"step_per_epoch": 10000, "episode_per_test": 100},
    },
    "MountainCar-v0": {
        "env": {"name": "MountainCar-v0"},
        "network": {"hidden_sizes": [256, 256]},
        "algorithm": {"gamma": 0.99, "lr": 1e-3, "ppo_ent_coef": 0.03, "ppo_value_coef": 0.5},
        "training": {"step_per_epoch": 10000, "epoch": 200},
    },
    "Acrobot-v1": {
        "env": {"name": "Acrobot-v1"},
        "network": {"hidden_sizes": [256, 128]},
    },
    # Continuous control - PyBullet
    "HalfCheetahBulletEnv-v0": {
        "env": {"name": "HalfCheetahBulletEnv-v0"},
        "algorithm": {"name": "sac"},
        "network": {"hidden_sizes": [256, 256]},
        "training": {"buffer_size": 1000000},
    },
    "Walker2DBulletEnv-v0": {
        "env": {"name": "Walker2DBulletEnv-v0"},
        "algorithm": {"name": "sac"},
        "network": {"hidden_sizes": [256, 256]},
    },
    "AntBulletEnv-v0": {
        "env": {"name": "AntBulletEnv-v0"},
        "algorithm": {"name": "sac"},
        "network": {"hidden_sizes": [256, 256]},
    },
    "HopperBulletEnv-v0": {
        "env": {"name": "HopperBulletEnv-v0"},
        "algorithm": {"name": "sac"},
        "network": {"hidden_sizes": [256, 256]},
    },
    # Continuous control - MuJoCo
    "HalfCheetah-v4": {
        "env": {"name": "HalfCheetah-v4"},
        "algorithm": {"name": "sac"},
        "network": {"hidden_sizes": [256, 256]},
        "training": {"buffer_size": 1000000},
    },
    "Walker2d-v4": {
        "env": {"name": "Walker2d-v4"},
        "algorithm": {"name": "sac"},
        "network": {"hidden_sizes": [256, 256]},
    },
    "Ant-v4": {
        "env": {"name": "Ant-v4"},
        "algorithm": {"name": "sac"},
        "network": {"hidden_sizes": [256, 256]},
    },
    "Hopper-v4": {
        "env": {"name": "Hopper-v4"},
        "algorithm": {"name": "sac"},
        "network": {"hidden_sizes": [256, 256]},
    },
}


def get_predefined_config(env_name: str) -> Config:
    """
    Get predefined configuration for common environments.

    Args:
        env_name: Environment name

    Returns:
        Config object with predefined settings
    """
    config = Config()

    if env_name in PREDEFINED_ENVS:
        return Config.from_dict(PREDEFINED_ENVS[env_name])

    config.env.name = env_name
    return config
