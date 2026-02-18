"""
Helper utility functions for RL training.
"""

import os
import random
import logging
import torch
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime


def setup_logging(
    log_dir: str = "./log",
    level: int = logging.INFO,
    format_str: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_dir: Directory for log files
        level: Logging level
        format_str: Log format string

    Returns:
        Root logger
    """
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        ],
    )

    return logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(
    algorithm: Any,
    config: Any,
    epoch: int,
    reward: float,
    checkpoint_dir: str,
    filename: Optional[str] = None,
) -> str:
    """
    Save training checkpoint.

    Args:
        algorithm: Algorithm/policy object
        config: Configuration object
        epoch: Current epoch
        reward: Current reward
        checkpoint_dir: Directory to save checkpoint
        filename: Optional filename

    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    if filename is None:
        filename = f"checkpoint_epoch{epoch}.pth"

    checkpoint_path = os.path.join(checkpoint_dir, filename)

    checkpoint = {
        'epoch': epoch,
        'algorithm_state_dict': algorithm.state_dict(),
        'reward': reward,
        'config': config.to_dict() if hasattr(config, 'to_dict') else config,
    }

    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Saved checkpoint to {checkpoint_path}")

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    algorithm: Any,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        algorithm: Algorithm/policy object
        device: Device to load to

    Returns:
        Checkpoint data
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'algorithm_state_dict' in checkpoint:
        algorithm.load_state_dict(checkpoint['algorithm_state_dict'])
    else:
        algorithm.load_state_dict(checkpoint)

    logging.info(f"Loaded checkpoint from {checkpoint_path}")

    return checkpoint


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_config(config: Any) -> None:
    """
    Print configuration in a formatted way.

    Args:
        config: Configuration object
    """
    print("\n" + "=" * 60)
    print("Configuration")
    print("=" * 60)

    if hasattr(config, 'to_dict'):
        data = config.to_dict()
    else:
        data = config

    def print_dict(d, indent=0):
        for key, value in d.items():
            prefix = "  " * indent
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                print_dict(value, indent + 1)
            else:
                print(f"{prefix}{key}: {value}")

    print_dict(data)
    print("=" * 60 + "\n")


def get_model_size(model: torch.nn.Module) -> int:
    """
    Get model size in bytes.

    Args:
        model: PyTorch model

    Returns:
        Model size in bytes
    """
    return sum(p.numel() * p.element_size() for p in model.parameters())


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count number of trainable parameters.

    Args:
        model: PyTorch model

    Returns:
        Number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
