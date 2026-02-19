# RL Training Framework

基于 Tianshou v2.0 的强化学习训练框架，支持多种算法和连续/离散控制环境。

## 功能特性

- **多算法支持**: DQN, PPO, SAC, TD3
- **环境支持**:
  - 离散控制：CartPole, MountainCar, Acrobot 等
  - 连续控制：PyBullet, MuJoCo 环境
- **Web 可视化界面**: 简洁易用的训练管理界面
- **算法比较**: 自动化多算法比较和可视化
- **配置管理**: YAML 配置文件支持
- **实验追踪**: TensorBoard 集成

## 快速开始

### Web 界面 (推荐)

```bash
# 启动 Web 可视化界面
python run_ui.py

# 或使用批处理文件 (Windows)
start_ui.bat
```

然后在浏览器打开：http://localhost:8501

### 命令行

```bash
# 使用命令行参数训练 DQN
python train.py train --env CartPole-v1 --algo dqn --epochs 100

# 使用配置文件训练 PPO
python train.py train --config configs/ppo_cartpole.yaml
```

## Docker 部署 (推荐)

如果您不想手动配置 Python 环境，可以使用 Docker 快速启动本项目。

### 前置要求

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- (可选) [NVIDIA Docker Support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (用于 GPU 加速)

## Docker 使用指南

### 1. 您作为开发者如何使用

**标准开发流程**：
1. **启动环境**：
   ```bash
   docker-compose up -d --build
   ```
2. **访问界面**：[http://localhost:8501](http://localhost:8501)
3. **查看日志**：
   ```bash
   docker-compose logs -f
   ```

**进阶：实时开发模式**：
如果您正在修改代码（如 `src/` 或 `app.py`），不希望每次修改都重建镜像，可以编辑 `docker-compose.yml`，取消以下注释：
```yaml
    volumes:
      # ...
      - ./src:/app/src        # 挂载源码目录
      - ./app.py:/app/app.py  # 挂载入口文件
```
这样修改本地代码后，容器内会即时生效（Streamlit 会自动重载）。

### 2. 其他人如何使用

#### 场景 A：朋友/同事有代码权限（推荐）
如果他们可以访问您的源码（Git 仓库或压缩包）：

1. **获取代码**：
   ```bash
   git clone <您的项目地址>
   cd RL_Training
   ```
2. **一键启动**：
   确保安装了 Docker Desktop，然后运行：
   ```bash
   docker-compose up -d --build
   ```
3. **开始使用**：
   直接访问 [http://localhost:8501](http://localhost:8501) 即可。

#### 场景 B：纯用户（无代码，仅使用镜像）
如果您想把做好的镜像发给别人，不提供源码：

**方法 1：导出镜像文件（离线分享）**
1. **您导出镜像**：
   ```bash
   # 构建镜像
   docker build -t rl-training:v1 .
   # 导出为文件
   docker save -o rl-training-v1.tar rl-training:v1
   ```
2. **发送文件**：将 `rl-training-v1.tar` 发送给朋友。
3. **朋友导入并运行**：
   ```bash
   # 导入镜像
   docker load -i rl-training-v1.tar
   # 运行容器
   docker run -d -p 8501:8501 --name rl-training rl-training:v1
   ```

**方法 2：推送到 Docker Hub（在线分享）**
1. **您推送镜像**：
   ```bash
   docker login
   docker tag rl-training:v1 <您的用户名>/rl-training:v1
   docker push <您的用户名>/rl-training:v1
   ```
2. **朋友拉取并运行**：
   ```bash
   docker run -d -p 8501:8501 <您的用户名>/rl-training:v1
   ```

### 3. 数据持久化说明

无论哪种方式，以下数据都会保留在您的本地目录中（不会随容器删除而丢失）：
- `log/`: 训练日志（TensorBoard）
- `checkpoints/`: 训练好的模型文件
- `configs/`: 生成的配置文件

## 存储空间说明与清理

Docker 确实会占用一定的磁盘空间，主要包括：
1. **镜像 (Images)**：包含完整的操作系统和运行环境（约 3-5GB）。
2. **容器 (Containers)**：运行时产生的临时文件。
3. **构建缓存 (Build Cache)**：构建过程中产生的中间层。

### 如何节省空间

**1. 使用 CPU 版镜像（轻量级）**
如果您没有 GPU 或者想节省空间，可以使用专门的 CPU 版 Dockerfile，体积会小很多（约 1GB+）：

```bash
# 使用 CPU 版 Dockerfile 构建
docker-compose -f docker-compose.yml build --build-arg DOCKERFILE=Dockerfile.cpu
# 或者直接修改 docker-compose.yml 中的 build context
```
*注：我们为您准备了 `Dockerfile.cpu`，您只需将 `docker-compose.yml` 中的 `build: .` 改为 `build: { context: ., dockerfile: Dockerfile.cpu }` 即可。*

**2. 清理空间**
如果不使用 Docker 了，可以运行以下命令释放空间：

```bash
# 停止并删除容器、网络
docker-compose down

# 删除不再使用的镜像（慎用，会删除所有悬空镜像）
docker image prune
```bash
# 删除所有未使用的数据（包括未使用的镜像、容器、网络、构建缓存）
docker system prune -a
```

### 常见问题：C 盘空间不足怎么办？

Docker Desktop 默认将数据（镜像、容器）存储在 C 盘的 WSL 2 虚拟磁盘文件中。文件路径通常为：
`%LOCALAPPDATA%\Docker\wsl\data\ext4.vhdx`

**解决方法：将 Docker 数据迁移到其他盘（例如 D 盘）**
1. **停止 Docker Desktop**：右键托盘图标 -> Quit Docker Desktop。
2. **确认分发名称**：
   打开 PowerShell 运行：
   ```powershell
   wsl -l -v
   ```
   *确认状态为 `Stopped`。如果没看到 `docker-desktop-data`，说明您可能没开启 WSL 2 后端。*

3. **如果遇到超时或占用错误**：
   运行以下命令强制关闭所有 WSL 实例：
   ```powershell
   wsl --shutdown
   ```

4. **导出数据**：
   ```powershell
   wsl --export docker-desktop-data "D:\docker-desktop-data.tar"
   ```
   *注意：请确保导出的是 `docker-desktop-data`（存数据），而不是 `docker-desktop`（存程序）。*

5. **注销原数据（这会从 C 盘删除）**：
   ```powershell
   wsl --unregister docker-desktop-data
   ```
6. **导入到新位置**（例如 `D:\DockerData`）：
   ```powershell
   mkdir "D:\DockerData"
   wsl --import docker-desktop-data "D:\DockerData" "D:\docker-desktop-data.tar" --version 2
   ```
7. **重启 Docker Desktop**。
8. **删除临时文件**：确认一切正常后，删除 `D:\docker-desktop-data.tar`。

### 常见问题：Docker 下载太慢怎么办？

如果您在构建或拉取镜像时速度极慢，请配置国内加速镜像源。

1. **打开 Docker Desktop 设置**：
   点击右上角齿轮图标 -> **Docker Engine**。

2. **添加镜像加速器**：
   在 JSON配置中添加 `registry-mirrors` 字段：
   ```json
   {
     "builder": {
       "gc": {
         "defaultKeepStorage": "20GB",
         "enabled": true
       }
     },
     "experimental": false,
     "registry-mirrors": [
       "https://docker.m.daocloud.io",
       "https://huecker.io",
       "https://dockerhub.timeweb.cloud",
       "https://noohub.ru"
     ]
   }
   ```
   *注意：请保留原有的其他配置，只添加或修改 `registry-mirrors` 部分。*

3. **点击 "Apply & Restart"**。

此外，本项目已内置了 `pip` 和 `apt` 的清华源加速，构建过程中的软件包下载应该会很快。

### GPU 支持 (关键！)

**好消息：您不需要在电脑上安装 CUDA Toolkit！**
因为我们的 Docker 镜像里已经内置了 CUDA 环境。您只需要满足以下最基本的条件：

#### Windows 用户 (最简单)
1. **安装显卡驱动**：确保您的 NVIDIA 显卡驱动是最新的（去 NVIDIA 官网下载）。
2. **安装 Docker Desktop**：确保使用 **WSL 2** 后端（默认就是）。
   * *这就够了！Docker Desktop 会自动处理 GPU 透传。*

#### Linux 用户
1. **安装显卡驱动**。
2. **安装 NVIDIA Container Toolkit**：这是让 Docker 能识别显卡的插件。
   ```bash
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

#### 开启步骤
1. 编辑 `docker-compose.yml` 文件，取消注释 `deploy` 部分：
   ```yaml
   # 找到并取消注释这几行：
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: all
             capabilities: [gpu]
   ```
2. 重新启动容器：
   ```bash
   docker-compose up -d --force-recreate
   ```

## 安装

```bash
# 基础安装
pip install -r requirements.txt

# 或安装完整依赖（包括 MuJoCo 和 PyBullet）
pip install -e ".[all]"

# 仅 MuJoCo
pip install -e ".[mujoco]"

# 仅 PyBullet
pip install -e ".[pybullet]"
```

## 项目结构

```
D:\Study\code\RL_Training/
├── src/                          # 源代码包
│   ├── __init__.py
│   ├── main.py                   # 主入口
│   ├── config.py                 # 配置管理
│   ├── envs/                     # 环境模块
│   │   ├── env_factory.py        # 环境工厂
│   │   └── wrappers.py           # 环境包装器
│   ├── models/                   # 神经网络模型
│   │   └── networks.py           # 网络定义
│   ├── trainers/                 # 训练器模块
│   │   ├── base.py               # 基础训练器
│   │   ├── dqn_trainer.py        # DQN 训练器
│   │   ├── ppo_trainer.py        # PPO 训练器
│   │   ├── sac_trainer.py        # SAC 训练器
│   │   ├── td3_trainer.py        # TD3 训练器
│   │   └── trainer_factory.py    # 训练器工厂
│   └── utils/                    # 工具函数
│       ├── helpers.py            # 辅助函数
│       └── comparison.py         # 算法比较
├── configs/                      # YAML 配置文件
│   ├── dqn_cartpole.yaml
│   ├── ppo_cartpole.yaml
│   ├── sac_halfcheetah.yaml
│   └── td3_walker2d.yaml
├── tests/                        # 单元测试
│   └── test_core.py
├── log/                          # TensorBoard 日志
├── checkpoints/                  # 训练模型
├── comparison_results/           # 算法比较结果
├── app.py                        # Streamlit Web 界面
├── run_ui.py                     # UI 启动脚本
├── start_ui.bat                  # UI 启动脚本 (Windows)
├── train.py                      # 主训练脚本
├── pyproject.toml                # 项目配置
└── requirements.txt              # 依赖
```

## 使用方法

### Web 界面

启动 Web 界面后，您可以通过图形界面进行以下操作：

1. **训练**: 选择算法和环境，配置参数，点击开始训练
2. **测试**: 选择已训练的模型进行评估
3. **可视化**: 查看训练曲线，比较不同算法
4. **设置**: 生成配置文件，管理系统数据

### 命令行

```bash
# 使用命令行参数训练 DQN
python train.py train --env CartPole-v1 --algo dqn --epochs 100

# 使用配置文件训练 PPO
python train.py train --config configs/ppo_cartpole.yaml

# 训练 SAC 在连续控制环境
python train.py train --env HalfCheetah-v4 --algo sac --epochs 100
```

### 测试模型

```bash
# 测试训练好的模型
python train.py test --env CartPole-v1 --algo dqn --model ./checkpoints/dqn_CartPole-v1_best.pth --episodes 10 --render
```

### 算法比较

```bash
# 比较 SAC 和 TD3 在 HalfCheetah 上的表现
python train.py compare --env HalfCheetah-v4 --algos sac td3 --epochs 100

# 比较结果将保存到 comparison_results/ 目录
```

### 列出可用环境

```bash
python train.py list-envs
```

### 生成配置文件

```bash
python train.py config --env Walker2d-v4 --algo td3 --output configs/td3_walker2d.yaml
```

## 算法选择指南

| 环境类型 | 推荐算法 | 说明 |
|---------|---------|------|
| 离散动作 | DQN | 简单环境，样本效率高 |
| 离散动作 | PPO | 复杂环境，稳定性好 |
| 连续动作 | SAC | 样本效率高，自动熵调节 |
| 连续动作 | TD3 | 性能稳定，超参少 |

## 配置选项

### 环境配置

```yaml
env:
  name: "CartPole-v1"      # 环境名称
  train_num: 10            # 并行训练环境数
  test_num: 100            # 并行测试环境数
  normalize_obs: false     # 观测归一化
  normalize_reward: false  # 奖励归一化
```

### 算法配置

```yaml
algorithm:
  name: "dqn"
  gamma: 0.99         # 折扣因子
  lr: 0.001           # 学习率
  n_step: 3           # n-step returns
  target_update_freq: 320  # 目标网络更新频率
  is_double: true     # Double DQN
```

### 训练配置

```yaml
training:
  epoch: 100              # 训练轮数
  step_per_epoch: 10000   # 每轮步数
  batch_size: 64          # 批次大小
  buffer_size: 10000      # 回放缓冲区大小
  save_freq: 10           # 模型保存频率
```

## 算法比较功能

框架提供自动化的算法比较功能：

1. **自动环境检测**: 根据环境类型（离散/连续）过滤兼容算法
2. **并行训练**: 依次训练多个算法
3. **结果可视化**: 生成对比图表和排名
4. **JSON 报告**: 保存详细结果到文件

示例输出:
```
COMPARISON RESULTS
============================================================
  #1: SAC - Best Reward: 3500.00
  #2: TD3 - Best Reward: 3200.00

Total comparison time: 3600.00s
Results saved to: ./comparison_results
```

## 连续控制环境

### PyBullet 环境

```bash
pip install pybullet gymnasium
```

可用环境:
- `HalfCheetahBulletEnv-v0`
- `Walker2DBulletEnv-v0`
- `AntBulletEnv-v0`
- `HopperBulletEnv-v0`

### MuJoCo 环境

```bash
pip install mujoco gymnasium[mujoco]
```

可用环境:
- `HalfCheetah-v4`
- `Walker2d-v4`
- `Ant-v4`
- `Hopper-v4`
- `Humanoid-v4`

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v

# 代码格式化
black src/ tests/
isort src/ tests/

# 类型检查
mypy src/
```

## 许可证

MIT License

## Web UI 说明

Web 界面基于 Streamlit 构建，提供以下功能：

- **首页**: 功能概览和快速入门
- **训练页面**: 可视化配置和实时训练日志
- **测试页面**: 模型评估和结果展示
- **可视化页面**: 训练曲线绘制和比较
- **设置页面**: 配置文件生成和数据管理

更多详情请参阅 [WEBUI.md](WEBUI.md)
