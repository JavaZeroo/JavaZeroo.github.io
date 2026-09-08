---
name: new-post
description: 规划并写一篇（或一个连载的）技术文章，从 PLAN 到正文到构建验证
disable-model-invocation: true
---

先读 skill `post-style`（写法）与 `post-figures`（画图），再按下面走。参数：主题或系列名；若是续写已有系列，`drafts/<系列>/PLAN.md` 是依据。

## 1. 规划

新系列先建 `drafts/<系列>/PLAN.md`，续写则核对现有 PLAN 是否覆盖本篇：

- 目标读者一句话，对应论文/资料的哪些节。
- 篇目表：编号 / 标题 / 对应论文章节 / 主要图 / 预估篇幅。连载按论文自己的叙事切（例如序列 → 深度 → 宽度），不按「上下篇」凑。
- 每篇要点清单：讲什么、推哪些公式、哪些数字要从 config 算、引用哪些已有文章。
- 图清单：每张图的机制与组件名。
- 明确「本连载不讲」什么。

没有 `refs/` 精读笔记的主题先 `/research-topic`。规划完成的标准：每篇的要点都能指到 refs 里的出处。

## 2. 落文件

- 文章 `src/content/posts/<系列>-<两位序号>-<slug>.mdx`，文件名即 URL，定了不改。
- front-matter 按 README；连载填 `series` / `part` / `order`，`part` 必须是系列文件 `src/content/series/<系列>.md` 里列出的，否则构建报错。新系列先建系列文件（title / description / parts / color）。
- 图组件放 `src/components/<系列缩写>/`，每篇文章顶部 import。

## 3. 写正文

按 `post-style` 的每节骨架写。写的时候持续做三件事：

- 每个数字对回 refs 或 config；论文没给的留变量。
- 每段问「这是本模型的贡献还是背景知识」，背景知识放 llm 专栏并链接。
- 每张图先想清「回答什么问题」再画。

## 4. 验证

1. `PATH=~/.local/node24/bin:$PATH npm run build` 通过。
2. Playwright 截浅色与深色两份，检查图与公式。
3. `git status` 核对新组件都在，未跟踪的组件一起提交。
4. PLAN 里把该篇标记为已写；主架构图 `LINKS` 表填上这篇的 URL。

完成标准：构建通过、每张图截过图、PLAN 与 LINKS 已更新。
