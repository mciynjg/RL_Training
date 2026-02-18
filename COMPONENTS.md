# UI 组件使用指南

本文档提供了如何在代码中使用新设计系统的指南。完整的设计规范请参考 `DESIGN_SYSTEM.md`。

## 引入组件

首先，在你的 Streamlit 页面中引入 `ui_components`：

```python
from src.ui_components import card, metric_card, section_header, feature_list
```

## 核心组件

### 1. Card (`card`)

用于展示主要内容的卡片容器。

```python
card(
    title="Train", 
    content="Train agents with DQN...", 
    icon="🎮" # 可选，支持 Emoji 或 HTML
)
```

### 2. Metric Card (`metric_card`)

用于展示关键指标。

```python
metric_card("Total Runs", "124")
```

### 3. Section Header (`section_header`)

用于页面或区块的标题，带有可选的副标题。

```python
section_header("Training Results", "Analysis of the latest run")
```

### 4. Feature List (`feature_list`)

用于展示列表项，通常用于环境列表或功能列表。

```python
feature_list("Environments", [
    {"name": "CartPole", "desc": "Balance pole"},
    {"name": "Ant", "desc": "Quadruped robot"}
])
```

## 布局最佳实践

- **使用列布局**: 总是使用 `st.columns` 来避免内容过宽。
- **留白**: 使用 `section_header` 自带的 margin，避免手动添加过多的 `st.write("")`。
- **一致性**: 所有的配置项应该分组在 Card 中或者使用 Header 分隔。

## CSS 类

如果你需要手动编写 HTML，可以使用以下 CSS 类：

- `.apple-card`: 标准卡片样式
- `.text-gradient`: 渐变文字效果
- `.glass-morphism`: 强制毛玻璃效果

