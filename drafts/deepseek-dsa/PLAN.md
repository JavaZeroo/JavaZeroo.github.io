# DeepSeek DSA 独立解读

状态：已完成正文、配图、教学脚本与验证（2026-11-12）。

文章路径：`/2026/11/12/deepseek-dsa/`；单篇独立文章，不涉及系列总图 LINKS。

读者有本科线性代数基础，想理解一个结构如何从计算约束与数学目标推出来。文章对应 DeepSeek-V3.2 报告 §2.1–2.3、Appendix A，以及 V3.2-Exp 官方推理实现。独立文章，不新建单篇系列。

## 正文

文件：`src/content/posts/deepseek-dsa.mdx`。

1. 先确认 DSA 首发于 2025-09-29 的 V3.2-Exp，正式 V3.2 继承结构。说明推导是作者的解释路径，非 DeepSeek 内部研究过程。
2. 从固定 query 的注意力输出误差推导丢弃质量 ε 和上界 2Mε；多头共享选择集合推出平均注意力分布的 top-k。
3. 昂贵教师不能每次全量算；引入独立低维索引器。推导线性多头打分的退化，解释 ReLU 的非线性、查询相关权重、共享 key、无 value 路径。
4. 解释 softmax 的排序不变性、KL 梯度 r-p，以及 KL 对 top-k 保留质量差距的上界；明确该全局上界不适用于仅选中集合的 KL。
5. MLA 中的两路张量、64×128 索引器、top-2048、RoPE、Hadamard、FP8、缓存不减少。
6. 稠密热身与稀疏继续训练，官方数字，双目标与 detach；配可运行的 CPU 小模型教学脚本。
7. 工程：长序列的分块 teacher、重算归一化、top-k/掩码边界、稀疏反向、checkpoint/优化器分组；区分建议与已公开事实。
8. 计算/显存/访存分开计量；MQA absorbed 576/512 与 MHA expanded 192/128 不混用。说明适用边界与核函数版本。

## 配图

- `src/components/dsa/DsaFlow.astro`：索引器与主 MLA 的张量分工；用 FlowDiagram。
- `src/components/dsa/AttentionMassDemo.astro`：交互选择 k 与排序，显示保留质量、实际输出误差和理论界；明确合成数据。
- `src/components/dsa/DsaTraining.astro`：稠密热身、稀疏训练的目标和梯度边界；用 FlowDiagram。

## 配套代码

`public/files/dsa/dsa_training_demo.py`：微型 causal attention LM，先准备 dense checkpoint，再 warm-up indexer，再 sparse adaptation；梯度隔离/因果性/概率归一化自检。教学实现保留稠密矩阵，不声称复现大模型性能或生产加速。

## 范围与验证

- 通用 KV/MLA 背景链接已有文章；不展开 V4/Kimi 和 RL 全流程。
- 来源和推导见 `refs/dsa.md`；不沿用旧笔记中把 MHA 运算维度当 MQA decode 的 FLOPs。
- 用词扫描、Node 24 类型检查和构建、站点 smoke、图浅/深色与手机截图、教学脚本运行与数学自检。

## 完成记录

- 正文逐节复核：公式 (1)–(4) 对齐论文；其余字母编号为补充推导；数值对回固定版本配置；外链论文章节锚点与站内链接已检查。
- 用词扫描剩余 1 条“退化成”提示，是去掉 ReLU 后多头打分可合为单点积的严格数学含义，保留。
- `npm run check`：0 errors、0 warnings；6 条 hints 均来自已有其他组件。
- `npm run build`：通过，97 个页面；新文章与下载脚本均生成。
- `npm run test:smoke`：桌面、390px、320px 的全站布局、主题、导航、搜索与图表回归通过。
- 专项浏览器验证：新文章公式无 KaTeX 错误；两种排序 × k=1…8 的交互、误差界、SVG 放大、客户端导航通过；390/320px 无页面横向溢出。
- 浅/深色、1440/390/320px 的三组配图均截图；已人工检查，截图位于本地 `/tmp/dsa-blog-shots/`。
- CPU 教学脚本完整运行通过；因果边界、教师归一化、梯度隔离、k=N 等价性、warm-up 主干不变均通过。运行环境为临时 venv，不改博客依赖。
- 1000 组随机概率与 value 的数值检查通过：输出误差界、top-k 保留质量差距界与 Pinsker 界。
- 实作过程中修正：交互示例静态默认值；公式在 JS 字符串内的反斜杠转义；手机控件排布。没有更改既有文章或共享图元。
