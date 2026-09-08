---
name: post-figures
description: 博客配图与 Astro 图组件的约定与坑。新建或修改 src/components/{figures,k3,v4,llm}/ 下的 SVG 图、方块流程图、交互 demo 时读；在文章里放图或想用 mermaid 时读。
---

全系列一套风格：手绘 SVG、站点字体、CSS 变量配色，随主题切换。图元在 `src/components/figures/`。

## 三种图，各用什么

| 要画的 | 用 | 位置 |
|---|---|---|
| 方块流程图（节点、边、分组） | `FlowDiagram`：数据驱动，边自动走线，`lines` 里 `$...$` 是公式 | `figures/FlowDiagram.astro`，用法见 `llm/PpEvolution.astro` |
| 任意手绘 SVG 里的公式标签 | `SvgTex`：构建期 MathJax 渲染成 SVG 路径，`(x, y)` 是锚点，`align` 定对齐 | `figures/SvgTex.astro` |
| HTML 场景的公式（figcaption、demo 图例） | `tex.ts` 的 `texMix`（KaTeX） | `figures/tex.ts` |

**mermaid 与 `<foreignObject>` 都已弃用**：mermaid 的字体、配色、尺寸与手绘图不是一套；foreignObject 里的 KaTeX 在 Safari 会整体漂移（WebKit 对缩放后 foreignObject 内 `position: relative` 元素算错坐标），Chromium 截图看不出来。凡涉及 Safari 兼容的改动，用纯 SVG 方案规避，不靠截图验证。

## 连载总图

每个系列一张主架构图（`k3/K3ArchDiagram.astro`、`v4/V4ArchDiagram.astro`），接受 `highlight` 与 `caption`，方块可点击跳文章，`LINKS` 表里为 `null` 的是还没写的篇。新系列照此建一张，每篇开头复用并高亮本篇模块。

## 交互 demo

- 用 innerHTML 动态生成 SVG 的组件必须写 `<style is:global>`：Astro 作用域样式不会匹配动态插入的节点，症状是黑色默认填充。参考 `k3/PpTimeline.astro`、`v4/CsaCell.astro`。
- 交互只做「拖一个参数看曲线/分配怎么变」这种能回答一个问题的，不做装饰。

## Astro 组件里的坑

- `.astro` 的 SVG `<text>` 里 `{...}`（如 `S_{t-1}`）会被当 JSX 解析，包成字符串表达式 `{'S_{t-1}'}`。
- 模板属性里的 LaTeX 用单反斜杠 `$\Psi$`；双反斜杠会原样进 KaTeX。
- 每张图给 `label`（aria）和 `caption`；`viewBox` 配 `minWidth`，窄屏横向滚动而不是缩成看不清。
- 组件文件顶部一段注释写这张图表达什么机制、每个 prop 干什么。

## 验证

图改完 `npm run build`（Node 24，见 CLAUDE.md），再用 Playwright（Chromium）截浅色与深色两份看一眼：公式没有变成原文本、节点文字没溢出、手机宽度下分支仍挂在父节点上。
