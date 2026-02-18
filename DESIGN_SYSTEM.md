# RL Framework - Design System (v2.0)

## 1. Design Philosophy
**"Crystal Clarity"**
A clean, modern, and data-centric interface inspired by Apple's Glassmorphism and high-end SaaS dashboards. The focus is on readability, hierarchy, and subtle interactions.

- **Keywords**: Translucent, Blur, Depth, Clean, Precise.
- **Core Value**: Reduce cognitive load while maximizing data visibility.

## 2. Color System

### Palette
| Token | Light Mode | Dark Mode | Usage |
|-------|------------|-----------|-------|
| `--bg-primary` | `#F5F7FA` | `#000000` | Main application background |
| `--bg-secondary` | `#E4E7EB` | `#1C1C1E` | Sidebar, secondary panels |
| `--text-main` | `#1D1D1F` | `#F5F5F7` | Primary headings, body text |
| `--text-sub` | `#86868B` | `#86868B` | Labels, captions, metadata |
| `--accent` | `#007AFF` | `#0A84FF` | Primary actions, active states, links |
| `--success` | `#34C759` | `#30D158` | Success states, positive trends |
| `--warning` | `#FF9500` | `#FF9F0A` | Warnings, medium alerts |
| `--error` | `#FF3B30` | `#FF453A` | Errors, critical alerts, negative trends |

### Surface & Depth
- **Card Background**: `rgba(255, 255, 255, 0.7)` (Light) / `rgba(28, 28, 30, 0.7)` (Dark)
- **Glass Effect**: `backdrop-filter: blur(20px)`
- **Border**: `1px solid rgba(0,0,0,0.05)` (Light) / `1px solid rgba(255,255,255,0.1)` (Dark)
- **Shadows**:
  - Soft: `0 2px 8px rgba(0,0,0,0.04)`
  - Hover: `0 8px 24px rgba(0,0,0,0.12)`

## 3. Typography

**Font Family**: System Stack (`-apple-system`, `BlinkMacSystemFont`, `Inter`, `sans-serif`)

| Level | Size | Weight | Tracking | Line Height | Usage |
|-------|------|--------|----------|-------------|-------|
| H1 | 32px | 700 (Bold) | -0.02em | 1.2 | Page Titles |
| H2 | 24px | 600 (Semibold) | -0.01em | 1.3 | Section Headings |
| H3 | 20px | 600 (Semibold) | 0 | 1.4 | Card Titles |
| Body | 16px | 400 (Regular) | 0 | 1.5 | Standard Text |
| Label | 13px | 500 (Medium) | +0.01em | 1.4 | Form Labels, Metadata |
| Code | 14px | 400 (Regular) | 0 | 1.5 | Logs, Configs |

## 4. Spacing & Layout

**Grid System**: 8px baseline grid.
- **XS**: 4px
- **SM**: 8px
- **MD**: 16px
- **LG**: 24px
- **XL**: 32px
- **2XL**: 48px

**Responsive Breakpoints**:
- **Mobile**: < 768px (Single column, stacked)
- **Tablet**: 768px - 1024px (Two columns, flexible)
- **Desktop**: > 1024px (Max width 1200px, centered)

## 5. Components

### Cards (`.apple-card`)
- Rounded corners (`16px`).
- Glassmorphism background.
- Subtle border.
- Hover lift effect (`translateY(-2px)`).

### Buttons
- **Primary**: Accent gradient, white text, shadow.
- **Secondary**: Surface color, subtle border, text color.
- **Height**: 40px (Desktop), 44px (Touch).
- **Radius**: 10px.

### Inputs
- No heavy borders.
- Background blends with card.
- Focus ring: 3px accent color with opacity.

### Navigation
- Sidebar with linear SVG icons.
- Active state: Accent color background (light opacity) + Accent text.

## 6. Accessibility (WCAG 2.1 AA)
- **Contrast**: Text vs Background ratio >= 4.5:1.
- **Focus**: Visible focus indicators for keyboard navigation.
- **Motion**: Respect `prefers-reduced-motion`.
- **Touch Targets**: Minimum 44x44px.
