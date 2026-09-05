# Kimi K3 模型结构连载：写作大纲

> 目标读者：手边打开着 Kimi K3 技术报告（arXiv 2607.24653）§2 "Model Architecture"，想把每一个模块看懂的人。
> 每篇文章对应论文的一个小节，配一张「你现在在这里」的总图高亮，加该模块自己的方块图。
> 参考资料精读笔记在 `refs/`（Kimi Linear / Attention Residuals / LatentMoE），真实超参在 `refs/k3-config.json`。

## 0. 结论：要连载，7 篇 + 1 篇可选

论文 §2 只有 8 页，但每个模块背后都压着一篇独立论文（KDA ← Kimi Linear 2510.26692；AttnRes ← 2603.15031；LatentMoE ← 2601.18089），加上 K3 自己在每个模块上的改动（下界衰减、满秩门、RMSNorm、SiTU-GLU、Quantile Balancing）。塞进一篇会超过两万字，而且读者无法按需跳读。按论文「三个维度」的叙事切：**序列（token mixing）→ 深度（layer mixing）→ 宽度（channel mixing）**，再加输入端的视觉和几个零件。

| # | 标题（暂定） | 对应论文 | 主要图 | 预估篇幅 |
|---|---|---|---|---|
| 0 | 总览：一张图看懂 Kimi K3 | §2 开头、Table 1 | **总架构方块图**（后续每篇复用并高亮） | 3–4k 字 |
| 1 | 序列维度（上）：从线性注意力到 KDA | §2.1.1 前半；Kimi Linear §2–4 | 谱系图（LA → DeltaNet → GDN → KDA）、状态更新 Eq.1 图解、KDA cell 方块图 | 5–6k |
| 2 | 序列维度（下）：chunkwise 并行与 K3 的下界衰减 | §2.1.1 后半（Eq.3–6, Fig.3）；Kimi Linear §3 + App.B/C | 双通道图（inter-chunk 状态流 + intra-chunk 掩码矩阵）、对角 tile 图 | 4–5k |
| 3 | 序列维度：Gated MLA 与 3:1 混合 | §2.1.2, §2.1 | MLA 压缩/解压方块图（带 K3 真实维度）、93 层排布条形图、KV cache 账 | 3–4k |
| 4 | 深度维度：Attention Residuals | §2.2；AttnRes 全文 | 三联图（普通残差 / Full / Block）、AttnRes 算子内部、K3 的 9 个 source | 4–5k |
| 5 | 宽度维度：Stable LatentMoE | §2.3 + App.B/C/D；LatentMoE 全文；DeepSeek-V3 aux-loss-free | 宽度收窄图（7168 → 3584 → 3072 → 3584 → 7168）、SiTU-GLU 曲线、QB 二分图 | 5–6k |
| 6 | 输入端与零件：MoonViT-V2、投影、MTP、Per-Head Muon | §2.4, §2.5, §4.1.4（MTP/EAGLE-3） | 视觉通路方块图、MTP draft 取特征位置图 | 3k |
| 7（可选） | 结构决定系统：为 K3 的结构写的内核与缓存 | §5.1, §5.4.1, §5.4.2 | KCP 前缀扫描图、512-token 哈希块 vs KDA checkpoint 图 | 4k |

第 7 篇是桥，接你专栏后面「并行 / 性能优化」两章，可以最后再决定写不写。

## 1. 每篇要讲什么（要点清单）

### 0 总览
- 三句话：2.8T / 104B 激活 / 1M 上下文；三个维度各一个新模块；对 K2 的 2.5× scaling efficiency。
- Table 1（K2 vs K3）翻成中文表，补 config.json 里论文没写的：`head_dim=128`、`q_lora_rank=1536`、`kv_lora_rank=512`、`qk_rope_head_dim=64`（保留但不加旋转）、`routed_expert_hidden_size=3584`、`moe_intermediate_size=3072`、第 0 层 dense FFN `intermediate_size=33792`、词表 163840。
- 93 层怎么排：23 个 [KDA, KDA, KDA, MLA] + 末尾额外 1 层 MLA = 69 + 24。full_attn_layers = 4, 8, …, 92, 93。
- AttnRes 分块：`attn_res_block_size=12`，边界在第 0, 12, …, 84 层 → 7 个满块 + 1 个 9 层的尾块，加 embedding 共 9 个 source。
- 一张总图 + 阅读地图（每个方块链接到对应篇）。
- 明确本连载**不讲**：数据、训练 recipe、post-training、评测。

### 1 KDA（上）：递推形式
- 谱系（Kimi Linear Table 7 的「在线学习」视角）：
  - LA：$S_t = S_{t-1} + k_t v_t^\top$，只加不擦。
  - DeltaNet：对 $\tfrac12\|S^\top k_t - v_t\|^2$ 做一步 SGD → $(I-\beta_t k_t k_t^\top)S_{t-1} + \beta_t k_t v_t^\top$，「先擦后写」。
  - GDN：标量遗忘 $\alpha_t$。
  - KDA：逐通道遗忘 $\mathrm{Diag}(\alpha_t)$，类比 RoPE 逐维频率。
- Eq.1 图解：decay → Householder 擦除 → rank-1 写入，三步顺序。
- 参数化（Eq.2）：q/k/v 各自 Linear → ShortConv(4) → Swish，q/k 再 L2Norm；$\beta$ 每头一个 sigmoid 标量；$\alpha$ 低秩（7168 → 128 → 96×128）+ 每头 bias；$A_h$ 每头 log-scale。
- 输出：逐头 RMSNorm → sigmoid 门 → $W_o$。**K3 改动一：门从低秩改满秩**（`use_full_rank_gate=True`，`g_proj: 7168 → 12288`）。
- 每层状态大小：96 头 × 128 × 128，和 MLA 的 per-token cache 对照，留到第 3 篇算账。
- 直觉证据：Kimi Linear 的 Palindrome / MQAR / Stack 合成任务（Mamba2 全挂、KDA 最快收敛）。

### 2 KDA（下）：chunkwise 与下界衰减
- 为什么需要 chunkwise：递推形式无法用 Tensor Core。
- WY 表示 / UT 变换：$U$ 是「减去块内已解释部分」的伪值，$W$ 是配套的衰减键；一次 $C\times C$ 三角求逆。
- Eq.4 两项：inter-chunk $(\Gamma\odot Q)S$ + intra-chunk $\mathrm{Tril}[\dots]\tilde V$。
- 数值问题：$1/\Gamma$ 溢出 → log 空间 + 16-token 二级分块 → 对角 tile 只能逐位置对算，成为瓶颈（Fig.3b）。
- **K3 改动二：下界衰减** $g = g_{\min}\,\sigma(e^{A_h} z)$，$g_{\min}=-5$；16 步累计 log-decay ∈ (−80, 0)，$e^{80}$ 在 BF16 范围内 → 对角 tile 也能上 Tensor Core。画 Fig.3a 的两条曲线。
- 与 GDN / Mamba-2 的 −softplus 映射对比；提一句 RWKV-7 / Griffin / HGRN2 的下界门是先例。
- Kimi Linear 的 FLOPs 公式和 DPLR 内核 2× 的原因（$a=b=k$ 少两张二级矩阵）。

### 3 Gated MLA 与混合
- MLA 复习（DeepSeek-V2）：$c_t = W_c x_t$ 只缓存 latent，上投影重建 K/V。K3 维度：q 经 1536 低秩，kv latent 512，qk_nope 128，v 128；`kv_a_proj_with_mqa` 输出 512+64，那 64 维是 rope 位但 **NoPE，不旋转**，作为所有头共享的 MQA 键分量。
- NoPE 的理由：KDA 的 $\prod \mathrm{Diag}(\alpha_j)$ 就是可学习的位置编码（Kimi Linear §6.1 与 RoPE 的形式对照）；长上下文扩展不需要动 RoPE base / YaRN（K3 §3.4 直接外推到 1M）。
- 满秩输出门 Eq.7（引 Qwen 的 gated attention：非线性、稀疏、去 attention sink）。
- 训练时 attention 输出保持 FP32（Qiu & Yao 2026 的 flash attention 偏差舍入）——一句话带过，属于训练细节。
- 3:1 的证据：Kimi Linear Table 1 消融（0:1 / 1:1 / 3:1 / 7:1 / 15:1）。
- 账：每 token 每 MLA 层 KV = 576 维 × 24 层；69 层 KDA 状态固定 96×128×128。1M 上下文时两者数量级对比。

### 4 Attention Residuals
- 动机三条：残差 = 深度上的 RNN；PreNorm 下 $\|h_l\|$ 随深度 $O(L)$ 增长、早层被稀释；无法按内容取回。
- Full AttnRes Eq.8–9：每个 **sublayer** 一个 $w_l\in\mathbb R^d$（不是每 token 投影），key 做 RMSNorm、value 不做，softmax 含 embedding；零初始化 = 均值残差。
- Block AttnRes Eq.10：块内求和，块间 attention，当前块的 partial sum 也是 source。
- K3 接线（以 HF 代码为准）：每个 decoder layer 有两次 AttnRes（`self_attention_res_proj` / `mlp_res_proj`），`_apply_attn_res` 里 score = RMSNorm(v)·(norm.weight ⊙ w)；block 边界 `layer_idx % 12 == 0` 时把 prefix_sum 推进 block_residual；最后 `output_attn_res` 再聚合一次。
- 推理两阶段 + online softmax 合并（Alg.1）。
- 结果：scaling law 1.25×、GPQA +7.5、输出幅度锯齿图、深度注意力热图（embedding 持续被关注 = 深度上的 attention sink）。
- 与 mHC / DenseFormer / LAuReL 的区别一表。

### 5 Stable LatentMoE
- LatentMoE 原论文的论证链：expert 处在 weight-loading-bound → 通信 ∝ $K d$ → 只能砍 $d$ → 砍成 $\ell = d/\alpha$ 后把省下来的用于 $\alpha\times$ 专家数和 top-k；$\binom{\alpha N}{\alpha K} \ge \binom{N}{K}^\alpha$。推荐 $\alpha=4$，K3 用 2。
- K3 前向 Eq.11 与维度：$x$(7168) → $W^\downarrow$(3584) → 16 个 routed expert（3584 → 3072 → 3584，SiTU-GLU）加权和 → **RMSNorm** → $W^\uparrow$(7168)；并联 2 个全宽 shared expert（代码里合成一个 intermediate 6144 的 MLP）。
- 原论文 routed 路径没有任何归一化，「四连矩阵乘」在 2.8T 尺度炸激活 → K3 的三个 "Stable"：RMSNorm、SiTU-GLU、QB。
- SiTU-GLU Eq.12：$\beta_1\tanh(\cdot/\beta_1)\sigma(\cdot)$ 门 × $\beta_2\tanh(\cdot/\beta_2)$ 上支，$\beta_1=4,\beta_2=25$，输出 $\le 100$；一阶等价 SwiGLU；比硬 clamp 好在梯度不断。画 Fig.4 曲线。
- 路由 Eq.13：sigmoid 打分 + 偏置 top-k + 归一化（`topk_method=noaux_tc`，`e_score_correction_bias` 即冻结的 QB 偏置）。
- Quantile Balancing：从「平衡指派」LP 出发（App.C）→ 对偶 → 坐标下降的闭式解是分位数 → 只保留 expert 侧 $\beta$（= $-b$），token 侧 $\alpha$ 用 top-(k+1) 的第 k+1 名当截止；DeepSeek-V3 的 sign 更新 = 同一对偶目标上的 SignSGD。画 Fig.5 的 8 token / 4 expert 二分图。
- 直方图估计（App.D）：一次 all-reduce $n\times B$ 个整数，误差 ≤ bin 宽。
- 引苏剑林《MoE 环游记 6》（spaces.ac.cn/archives/11619）作为中文读者入口。

### 6 输入端与零件
- MoonViT-V2：27 层、0.4B、patch 14、hidden 1024、12 头、无 bias、RMSNorm；从零用 NTP 训练而非 SigLIP 初始化（Fig.6 梯度范数）。
- 通路：patch embed（Conv 14×14）→ 2D 可学习位置嵌入 + 时间 sincos → 27 层（2D RoPE）→ 2×2 pixel-shuffle + 时间池化（`sd2_tpool`，token ÷4）→ `PatchMergerMLPV2`（4096 → 4096 → 7168，GELU，RMSNorm）→ 进主干。
- 论文说时空注意力是分解的（帧内空间 + 帧间时间），HF 代码里是对 t·h·w 个 token 的联合注意力；写的时候标注这个出入。
- MTP 层：结构镜像主干 block，预训练 1 层；post-training 微调成 EAGLE-3 draft，输入取第 1、4、最后 AttnRes 块的输出拼接，$W_{E3}$ 初始化为 $[0\ 0\ I]$。HF config `num_nextn_predict_layers=0`，权重可能没放出。
- Per-Head Muon：Newton–Schulz 按头分块正交化；只是优化器，一小节。
- 第 0 层 dense FFN、词表、tie 关系等收尾。

### 7（可选）结构决定系统
- FlashKDA：chunk 内并行 vs chunk 间递推的重叠。
- KCP Eq.17：把每段效果分解为「累积转移 $M$」和「零初始状态 $\tilde S$」两个可本地算的量，一次 all-gather + 前缀扫描；为什么普通线性注意力直接求和就够而 KDA 不够（delta rule 的 $M_t$ 依赖 token）。
- KDA-aware prefix cache（Fig.12）：MLA 用 512-token 哈希块，KDA 只在稀疏边界存 checkpoint，命中点 = 两者都满足的最长边界。
- MTP 投机解码下 KDA 状态回滚：只缓存投影输入、片上重放（ReplaySSM）。
- Block AttnRes 的 SP 化、LatentMoE 的 down-proj + router 融合 GEMM。

## 2. 图的方案

论文 Fig.2 是一张「总图 + 三个局部放大」，我们要做的就是它的可交互中文版。

1. **总架构图**（第 0 篇主图，每篇复用）：自定义 SVG 做成 Astro 组件 `K3ArchDiagram`，数据驱动（块列表 + 连线），支持 `highlight="kda"` 之类的 prop 把当前篇讲的方块点亮、其余淡化；每个方块可点击跳转到对应文章。Mermaid 画不出带维度标注的嵌套方块和高亮，所以这张用手写 SVG。
2. **模块方块图**（KDA cell、MLA、AttnRes 算子、LatentMoE 宽度收窄、视觉通路）：优先 mermaid（构建期渲染、自动适配深浅主题、改起来快）；需要画矩阵/张量形状的（Eq.1 的三步、chunkwise 双通道、QB 二分图、AttnRes 深度混合矩阵）用内联 SVG。
3. **曲线图**（下界衰减两条曲线、SiTU-GLU vs SwiGLU）：小的交互组件（滑块调 $A$、$\beta$），参考现有 `SoftmaxDemo.astro`。

## 3. 建议的写作顺序与专栏挂载

- 顺序：0（骨架 + 总图组件）→ 1 → 2 → 3 → 4 → 5 → 6 →（7）。总图先做，因为每篇都用。
- 挂载：新建独立专栏 `src/content/series/kimi-k3.md`，parts = 总览 / 序列 / 深度 / 宽度 / 输入端与零件 / 系统。原因：这 7 篇是强顺序阅读，独立专栏页能放总图当封面；现有 `llm` 专栏的「架构」章只在开篇里放一个链接指过来（schema 里一篇文章只能属于一个 series）。

## 4. 写的时候要小心的点

- 论文里的 "layer" 在 AttnRes 论文里指 sublayer，在 K3 论文和 config 里指 decoder layer；K3 的 "12-layer block" = 24 个 sublayer。
- Kimi Linear 论文没写显式的衰减函数和 16-token 子块宽度，都来自 GDN/Mamba-2 惯例和 FLA 内核；K3 论文写了。
- "6.3× decode" 是批量放大后的数字，batch=1 时约 2.2×，上限由 3:1 决定。
- LatentMoE 原文 95B 模型的 expert 是无门 Squared-ReLU 两矩阵，K3 是带门三矩阵，比参数量时别混。
- QB 的偏置推理时冻结，`e_score_correction_bias` 就是它。

## 5. 资料

- K3 报告：https://arxiv.org/abs/2607.24653 ；权重与代码：https://huggingface.co/moonshotai/Kimi-K3
- Kimi Linear：https://arxiv.org/abs/2510.26692 ；Attention Residuals：https://arxiv.org/abs/2603.15031 （代码 github.com/MoonshotAI/Attention-Residuals，分支 master）；LatentMoE：https://arxiv.org/abs/2601.18089
- DeepSeek-V2（MLA）、DeepSeekMoE、DeepSeek-V3（aux-loss-free）；Gated Attention（Qwen，2505.06708）；Gated DeltaNet；苏剑林 MoE 环游记 6：https://spaces.ac.cn/archives/11619
- FlashKDA：https://github.com/MoonshotAI/FlashKDA ；KCP：flash-linear-attention PR #691
