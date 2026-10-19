# blog.javazero.top

[Astro](https://astro.build) 静态博客。文章是 `src/content/posts/` 下的 Markdown / MDX。

## 本地开发

需要 Node ≥ 22.12（`.nvmrc` 锁定 24.20.0）。

```bash
npm ci
npx playwright install --with-deps chromium   # 首次：mermaid 构建期渲染需要
npm run dev       # http://localhost:4321
npm run build     # 输出到 dist/
npm run check     # 类型检查
npm run test:smoke # 构建后运行浏览器回归检查（CI 同样运行）
```

## 写文章

在 `src/content/posts/` 新建 `.md`（纯 Markdown）或 `.mdx`（需要 import 组件时）。
**文件名会直接进 URL**，永久链接是 `/:year/:month/:day/:文件名/`。

```yaml
---
title: "标题"
pubDate: 2026-09-03 10:00:00
description: "一句话摘要，用于列表页和 og:description"
categories:
  - "📝学习"
tags:
  - "深度学习"
sticky: 100      # 可选，置顶权重，越大越靠前
draft: true      # 可选，为 true 时不会生成页面
---
```

front-matter 由 `src/content.config.ts` 的 zod schema 校验，字段写错构建会直接报错。

### 能写什么

| 能力 | 写法 | 渲染时机 |
|---|---|---|
| 数学公式 | `$行内$`、`$$块级$$` | 构建期，KaTeX |
| 流程图 | ` ```mermaid ` 代码块 | 构建期，浅/深两份 SVG 随主题切换 |
| 提示框 | `:::note{type="info"}` … `:::` | 构建期 |
| 代码高亮 | ` ```python ` | 构建期，Shiki（Tokyo Night / Catppuccin Latte） |
| 代码标注 | 行尾 `// [!code highlight]`、`[!code ++]` / `[!code --]`、`[!code focus]`、`[!code word:xxx]`；或信息串 ` ```ts {1,4-6} /needle/ ` | 构建期 |
| 代码块标题、行号 | ` ```ts title="src/foo.ts" showLineNumbers ` | 构建期 |
| 交互组件 | `.mdx` 里 `import` 后当标签用 | 浏览器 |

完整示例见 `src/content/posts/interactive-demo.mdx`（当前是草稿，把 `draft` 改成 `false` 即可发布），
交互组件的参考实现见 `src/components/SoftmaxDemo.astro`。

需要 React/Vue/Svelte 组件时先 `npx astro add react`，然后用 `<Component client:visible />`。

## 字体

- 正文/文章标题：系统字体栈，避免中文长文触发大量字体下载。页头和首页品牌名称保留 **HarmonyOS Sans SC Bold**：`src/utils/brand-font.ts` 在构建期按 `AUTHOR` 提取所需字符的声明，只加载命中的分片，不加载完整字体 CSS。品牌使用 `font-display: optional`，慢网下保留系统字体以避免后续跳动。
- HarmonyOS 字体分片仍保留在 `public/fonts/harmonyos-sans-sc/`，供品牌及构建期分享图使用。协议见旁边的 `LICENSE.txt`，页脚有署名。重新切片：`node scripts/split-fonts.mjs <含 HarmonyOS_SansSC_*.ttf 的目录>`（需先 `npm i -D cn-font-split`；有意不进 CI）。
- 代码：**Cascadia Code**（可变字体，fontsource 自托管），默认关闭连字，`global.css` 里 `--code-ligatures: normal` 可打开。
- 站点图标 / 页头小头像 / 首页分享图（`public/og.png`）都由 `node scripts/gen-assets.mjs` 从 `public/images/avatar.jpg` 生成，换头像后重跑一次。
- 每篇文章的分享图由 `src/plugins/og-images.mjs` 在构建末尾用 Chromium 渲染到 `dist/og/<文件名>.png`（不进 git），卡片字体就是站点自己的 HarmonyOS 分片。
- 搜索：Pagefind，`astro build` 后自动生成索引到 `dist/pagefind/`；`npm run dev` 时从上一次 build 的 `dist/` 读取，所以先 build 一次搜索才有结果。

## 部署

`npm run test:smoke` 使用临时本地端口检查 `dist/`，覆盖桌面与 390/320px 手机布局、字体请求预算、图表放大交互、弹窗焦点、主题、站内导航、真实搜索和乱序搜索响应。设置 `SMOKE_SCREENSHOTS=/tmp/blog-smoke` 可保存截图。

EdgeOne 仅对带哈希的字体文件和 `/_astro/` 资源启用一年缓存；字体 CSS 和 Pagefind 资源需要重新验证，避免发布后继续使用旧入口。缓存规则遵循 [EdgeOne URL 路径匹配语法](https://pages.edgeone.ai/document/edgeone-json)。GitHub Pages 的响应头由平台管理，修改此配置不会更改其缓存策略。

推送到 `hexo`（默认分支，或 `main`）后由 `.github/workflows/deploy.yml` 构建并把 `dist/` 推到 `gh-pages` 分支，GitHub Pages 从该分支提供服务（Settings → Pages → Source: Deploy from a branch → `gh-pages`）。

## 目录

```
src/
  assets/images/     文章配图（走 Astro 图片管线：WebP + 宽高）
  content/posts/     文章
  components/        可复用组件（figures/ 是各系列共用的图元：SvgTex、FlowDiagram、tex.ts）
  layouts/           页面骨架
  pages/             路由（[...permalink].astro 负责文章 URL）
  plugins/           remark / rehype 插件与构建后处理（:::note、标题锚点、公式字形去重等）
  utils/             永久链接、分类聚合、feed
public/              原样拷贝到站点根目录（fonts/、files/、CNAME、头像与 og.png）
```
