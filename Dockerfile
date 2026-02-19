# Use nvidia/cuda base image with Python 3.11 support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
# libgl1-mesa-glx: for opencv/render
# libglib2.0-0: for opencv/render
# git: for installing git dependencies
# patchelf: for mujoco
# libosmesa6-dev: for mujoco rendering
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libosmesa6-dev \
    patchelf \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.11 from deadsnakes PPA (using CN mirror if available)
# Note: Tianshou v2.0+ requires Python >= 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11 (using ensurepip)
RUN python3.11 -m ensurepip --upgrade && \
    python3.11 -m pip install --upgrade pip setuptools wheel

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# We upgrade pip, install PyTorch, then install requirements
# We also explicitly install gymnasium[mujoco] to support Walker2d, Ant, etc.
# Use Tsinghua mirror for faster installation in China
RUN pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch==2.1.2 && \
    pip install --no-cache-dir --ignore-installed blinker -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir "gymnasium[mujoco]>=0.29.0" "mujoco>=3.0.0" -i https://pypi.tuna.tsinghua.edu.cn/simple

# Copy the rest of the application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Healthcheck to ensure the container is ready
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
