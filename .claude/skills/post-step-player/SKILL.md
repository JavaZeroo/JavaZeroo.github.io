---
name: post-step-player
description: 用 figures/StepPlayer 给文章开头做「一步一改」的分步演进动画（例：CSA2 怎么从 V4 的 CSA 变来），以及按帧重排正文。要做引入动画、分步讲解图、改 StepPlayer 或它的场景时读。
---

先读 `post-figures`（图的通用约定）。样板：`src/components/v41/Csa2Evolution.astro` + `src/content/posts/deepseek-v41-02-csa2.mdx` 开头。

## 什么时候用

结构的演进能拆成 4–8 步、每步只改一处时用它：开头先让读者看到「长什么样、每步回答什么问题」，正文再按同样的步骤讲「为什么这样改」。连续调参数看曲线的用交互 demo，不用它。

## 场景怎么写

```astro
---
import StepPlayer from '../figures/StepPlayer.astro';
import type { Step, ElState } from '../figures/StepPlayer.astro';
const { step, caption } = Astro.props;            // step 固定某一帧，正文里当静态图
const base: Record<string, ElState> = { 'bus-kv': 'hide', 'dec-panel': 'hide' };   // 开头不显示的
const steps: Step[] = [
  { title: '共用 main KV', caption: '做了什么。为什么，代价。', facts: ['6 份 → 1 份'],
    state: { 'l1-kv': 'dim', 'bus-kv': 'on' } },
];
---
<StepPlayer id="x" label="…" steps={steps} base={base} step={step} caption={caption}>
  <div class="sp-panels">
    <div class="sp-panel" data-el="enc-panel"><svg viewBox="0 0 490 420">…<g data-el="l1-kv">…</g></svg></div>
    <div class="sp-panel" data-el="dec-panel"><svg viewBox="0 0 470 440">…</svg></div>
  </div>
</StepPlayer>
```

- 受控元素加 `data-el`，状态只有 `on / dim / hide`；没写的沿用 `base`，`base` 没写的是 `on`。**每一帧把要显示的东西写全**：第 4 帧要显示 decoder 面板，就得把它所有元素显式置 `on`，否则继承 `base` 的 `hide`（踩过）。用 `allOf()` 这类小函数批量生成状态表。
- 多条「总线」（源层给后面各层送箭头）用一个 helper 生成路径，一组一个 `data-el`。
- 一个场景拆成几张 SVG，每张包一层 `.sp-panel`：宽屏并排，窄屏只显示该帧为 `on` 的面板、上下排列。单张 viewBox 控制在 500 以内，文字 ≥ 11px，不要写 `min-width`。
- 字幕和 `facts` 里可以写 `$...$`，构建期用 KaTeX 渲染。

## 帧的设计

- 第 0 帧是改动之前的样子，最后一帧是合起来的结果加总数。中间每帧只改一处。
- `title` 写动作（「共用 main KV」「Reuse：选出的 512 条也共用」），不写名词。
- `caption` 两句：这一步做了什么；为什么可以、代价是什么。
- `facts` 是这一步改变的数字，写成 `A → B`。
- 正文每帧一个 h2，节首放 `<Scene step={n} />`，再讲推导、代码、代价。

## 验证

1. `npm run check`（CI 在 build 前跑它，类型错本地 build 看不出）再 `npm run build`。
2. 用 `scripts/shot_steps.mjs` 截图：桌面每一帧浅色、深色各一份，手机 390px 宽抽几帧：
   `node .claude/skills/post-step-player/scripts/shot_steps.mjs <slug> <data-stepplayer id> <输出目录>`。逐帧看：该淡的淡了、该出的箭头出了、面板切换对不对、手机上有没有横向滚动。
3. 本地起服务用 `run_in_background`，别在同一条命令里 `pkill`（会把后面的命令一起杀掉）。

## 坑

- `.astro` 的 frontmatter 不能写 JSX；要包动态标签就把标签名当字符串传给模板里的回调，回调参数标 `any`。
- 类名别用单字母：`.b.b` 会匹配到所有 `.b`。
- 元素截图会把站点的 sticky 页头截进去，用 fullPage + 页面坐标裁剪。
- 手机上全站的图已统一去掉最小宽度、缩到屏宽，靠放大按钮读细节；播放器自己不走这条路，靠拆面板。
