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
