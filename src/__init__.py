"""
RL Training Framework
A comprehensive reinforcement learning training framework based on Tianshou.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from src.config import Config, get_config, load_config_from_yaml
from src.envs import make_env, make_vector_env, get_env_info
from src.trainers import TrainerFactory, BaseTrainer
from src.utils import setup_logging, set_seed, save_checkpoint, load_checkpoint

__all__ = [
    "Config",
    "get_config",
    "load_config_from_yaml",
    "make_env",
    "make_vector_env",
    "get_env_info",
    "TrainerFactory",
    "BaseTrainer",
    "setup_logging",
    "set_seed",
    "save_checkpoint",
    "load_checkpoint",
]
