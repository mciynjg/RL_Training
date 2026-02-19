# RL Training Framework - Deployment Guide

This project is containerized using Docker to ensure a consistent environment for training and visualization.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed (for GPU support).

## Quick Start

1.  **Build and Run**:
    ```bash
    docker-compose up --build -d
    ```

2.  **Access the Web UI**:
    Open your browser and navigate to [http://localhost:8501](http://localhost:8501).

3.  **Stop the Container**:
    ```bash
    docker-compose down
    ```

## Directory Structure

- `log/`: Stores TensorBoard logs (persisted on host).
- `checkpoints/`: Stores trained model weights (persisted on host).
- `configs/`: Stores training configurations (persisted on host).

## Troubleshooting

- **GPU Issues**: If you encounter errors related to CUDA or GPU, ensure your NVIDIA drivers are up to date and the NVIDIA Container Toolkit is correctly configured. You can verify GPU visibility inside the container with:
    ```bash
    docker-compose exec rl-training nvidia-smi
    ```

- **Permission Issues**: If you cannot write to `log/` or `checkpoints/`, check the directory permissions on your host machine.
