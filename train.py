#!/usr/bin/env python3
"""
RL Training Framework - Main Entry Point
=========================================

A comprehensive reinforcement learning training framework with support for:
- DQN (Discrete control)
- PPO (Discrete control)
- SAC (Continuous control)
- TD3 (Continuous control)

Quick Start:
------------
# Train DQN on CartPole
python train.py --env CartPole-v1 --algo dqn --epochs 100

# Train PPO on CartPole
python train.py --env CartPole-v1 --algo ppo --epochs 100

# Compare algorithms (continuous control)
python train.py compare --env HalfCheetah-v4 --algos sac td3 --epochs 100

# List available environments
python train.py list-envs
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main import main

if __name__ == "__main__":
    main()
