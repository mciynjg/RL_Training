# Changelog

## [Unreleased]

### UI Upgrade (2026-02-18)

全面升级 Web UI 为 Apple 风格高端科技界面。

#### 新增
- **assets/style.css**: 完整的 CSS 样式系统，包含颜色变量、组件样式和响应式断点。
- **COMPONENTS.md**: UI 组件库文档和设计规范。
- **Glassmorphism**: 所有卡片和容器采用毛玻璃效果。
- **Animations**: 按钮和卡片添加了平滑的过渡动画和触觉反馈。

#### 修改
- **app.py**: 重构 HTML 结构以适配新的 CSS 类 (`.apple-card`)。
- **Typography**: 统一使用系统无衬线字体栈 (San Francisco, Inter)。
- **Color Palette**: 
  - Light Mode: 银白渐变 (#F5F7FA) + Apple Blue (#007AFF)
  - Dark Mode: 深空灰 (#1C1C1E) + 柔和白 (#F2F2F7)

#### 优化
- **Performance**: CSS 文件体积 < 10KB，无外部字体依赖。
- **Responsive**: 优化了移动端和宽屏的显示效果。
