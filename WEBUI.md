# RL Training Framework - Web UI

一个简洁、高级的强化学习训练可视化界面。

## 🚀 快速开始

### 方法 1: 使用启动脚本 (Windows)
```bash
start_ui.bat
```

### 方法 2: 手动启动
```bash
streamlit run app.py --server.port 8501
```

然后在浏览器打开：http://localhost:8501

## ✨ 功能特性

### 🏠 主页
- 功能概览卡片
- 可用环境列表
- 快速入门指南

### 🎮 训练 (Train)
- 可视化算法和环境选择
- 超参数调节滑块
- 实时训练日志输出
- 进度条显示
- 自动保存配置文件
- 历史训练记录

### 🧪 测试 (Test)
- 自动检测已训练的模型
- 可配置测试参数
- 实时显示测试结果

### 📈 可视化 (Visualize)
- 按算法和环境过滤训练记录
- 生成对比曲线图
- 可调节平滑度
- 支持导出图片

### ⚙️ 设置 (Settings)
- 系统信息查看
- 一键生成配置文件
- 数据管理（清理日志和检查点）

## 🎯 使用示例

### 训练一个模型
1. 点击左侧导航栏的 "🎮 Train"
2. 选择算法（如 PPO）
3. 选择环境（如 CartPole-v1）
4. 调整超参数（可选）
5. 点击 "🚀 Start Training"

### 测试已训练的模型
1. 点击左侧导航栏的 "🧪 Test"
2. 从下拉列表选择模型
3. 设置测试参数
4. 点击 "🧪 Run Test"

### 生成训练曲线
1. 点击左侧导航栏的 "📈 Visualize"
2. 使用过滤器选择要比较的训练
3. 调整平滑度
4. 点击 "📈 Generate Plot"

## 📋 支持的算法

| 算法 | 类型 | 动作空间 |
|------|------|----------|
| DQN | Off-policy | 离散 |
| PPO | On-policy | 离散/连续 |
| SAC | Off-policy | 连续 |
| TD3 | Off-policy | 连续 |

## 🌍 支持的环境

### 离散动作空间
- CartPole-v1 - 平衡杆子
- MountainCar-v0 - 开车上山
- Acrobot-v1 - 摆动机械臂

### 连续动作空间
- HalfCheetah-v4 - 猎豹机器人跑步
- Walker2d-v4 - 双足机器人行走
- Hopper-v4 - 单腿机器人跳跃
- Ant-v4 - 四足机器人行走

## ⌨️ 快捷键

在浏览器中使用 Streamlit 时：
- `R` - 重新运行
- `O` - 显示/隐藏代码

## 🔧 配置

如果端口 8501 被占用，可以修改启动命令：

```bash
streamlit run app.py --server.port 8503
```

## 📝 注意事项

1. 首次启动可能需要安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 训练需要较长时间，请在训练前确保有足够的计算资源

3. 日志文件存储在 `log/` 目录，模型存储在 `checkpoints/` 目录

4. 关闭服务器请按 `Ctrl+C`

## 🎨 界面设计

本次升级采用了 **Apple 风格的高端科技界面**：

- **视觉风格**: 银白渐变色调 (#F5F7FA - #E4E7EB) 搭配冷灰色 (#6E7881)
- **卡片设计**: 毛玻璃效果 (Glassmorphism) + 细腻阴影
- **交互体验**: 按钮按压反馈、悬停提亮、流畅过渡动画
- **深色模式**: 完美适配 Dark Mode，自动切换为深邃黑金风格
- **排版**: 统一使用 San Francisco / Inter 字体，层级分明

## 📊 技术栈

- **前端**: Streamlit
- **后端**: Python + 原生训练框架
- **可视化**: Matplotlib + TensorBoard

---

Built with ❤️ using Streamlit
