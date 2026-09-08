# blog.javazero.top

Astro 静态博客，中文技术文章为主。站点怎么跑、front-matter 字段、Markdown 能力见 `README.md`，这里只写 README 没有的。

## 环境

- 系统 node 是 18，构建要用 `~/.local/node24/bin`（`.nvmrc` 锁 24）：`PATH=~/.local/node24/bin:$PATH npm run build`。
- 改完文章或组件必须 `npm run build` 过一遍再交付，MDX 与 front-matter 错误只在构建期暴露。
- 截图验证：用 python `http.server` 起 `dist/`，Playwright 从项目 `node_modules` 的绝对路径 import（ESM 不认 NODE_PATH）。本地只有 Chromium，WebKit 跑不起来。

## 写文章

- 长文写作规范（推导深度、段落骨架、通用与模型特有的边界）在 skill `post-style`；画图与 Astro 组件约定在 skill `post-figures`。动笔前先读。
- 流程：`/research-topic` 精读一手资料写 `drafts/<专栏>/refs/<主题>.md` → `/new-post` 规划并写正文 → `/review-post` 按清单审。
- 每个连载在 `drafts/<系列>/` 下有 `PLAN.md`（大纲、每篇要点、图清单）和 `refs/`（论文摘录、config、代码）。续写系列先读 PLAN 再读对应 refs，PLAN 是后续各篇的依据。
- 系列图组件放 `src/components/<系列缩写>/`（k3、v4、llm），共用图元在 `src/components/figures/`。

## 提交

- Conventional commits，scope 用系列名：`feat(k3): …`、`fix(llm): …`、`feat(deepseek-v4): …`。
- 文章引用的组件要一起提交：曾发生过正文 import 了七个未提交的组件，CI 直接挂。提交前 `git status` 核对 `src/components/`。
- 推到 `hexo` 分支即部署（GitHub Pages + EdgeOne Pages 双目标）。DNS 在 NameSilo，动 DNS 前先列全 26 条记录，不迁 NS。
