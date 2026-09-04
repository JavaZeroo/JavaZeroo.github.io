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
| 交互组件 | `.mdx` 里 `import` 后当标签用 | 浏览器 |

完整示例见 `src/content/posts/interactive-demo.mdx`（当前是草稿，把 `draft` 改成 `false` 即可发布），
交互组件的参考实现见 `src/components/SoftmaxDemo.astro`。

需要 React/Vue/Svelte 组件时先 `npx astro add react`，然后用 `<Component client:visible />`。

## 字体

- 正文/标题：**HarmonyOS Sans SC**（Regular / Medium / Bold），用 `cn-font-split` 切成 unicode-range 分片放在 `public/fonts/harmonyos-sans-sc/`，浏览器只拉页面用到的分片。协议见旁边的 `LICENSE.txt`，页脚有署名（协议要求）。重新切片：`node scripts/split-fonts.mjs <含 HarmonyOS_SansSC_*.ttf 的目录>`（需先 `npm i -D cn-font-split`；有意不进 CI）。
- 代码：**Cascadia Code**（可变字体，fontsource 自托管），默认关闭连字，`global.css` 里 `--code-ligatures: normal` 可打开。
- 搜索：Pagefind，`astro build` 后自动生成索引到 `dist/pagefind/`；`npm run dev` 时从上一次 build 的 `dist/` 读取，所以先 build 一次搜索才有结果。

## 部署

推送到 `main`（或 `hexo`）分支后由 `.github/workflows/deploy.yml` 自动构建并发布到 GitHub Pages。

> 仓库 **Settings → Pages → Source** 需要设为 **GitHub Actions**。

## 目录

```
src/
  content/posts/     文章
  components/        可复用组件
  layouts/           页面骨架
  pages/             路由（[...permalink].astro 负责文章 URL）
  plugins/           remark 插件（:::note）
  utils/             永久链接、分类聚合、feed
public/              原样拷贝到站点根目录（images/、files/、CNAME）
```
