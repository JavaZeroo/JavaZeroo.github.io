# DeepSeek-V4.1 模型结构连载：写作大纲

> 目标读者：手边打开着 DeepSeek-V4.1-Flash 技术报告（arXiv 2609.19969）§2 "Architecture"，读过或愿意回查本站 DeepSeek-V4 连载的人。
> 写法与 Kimi K3、DeepSeek-V4 连载相同：每篇对应论文一个小节，开头放总图并高亮本篇模块，公式从问题推出来，数字对回 config。
> **只写 V4.1 相对 V4 的改动**。V4 连载讲过的（压缩算子的推导、lightning indexer 打分公式、部分 RoPE 与输出反旋转、attention sink、mHC 的双随机约束、sqrt-softplus、Muon）一律引用，不重写。
>
> 本连载自己的精读笔记在 `drafts/deepseek-v41/refs/`：`background.md`（YOCO、四篇层维复用前作、V4-Flash 3514 B 的复算）、`engram.md`、`dspark.md`。
> 论文与代码仍在 `drafts/deepseek-v4/refs/`（没有搬家，文件名带 `v41-` 前缀，含后补的 `v41-inference-engram.py`、`v41-inference-vision.py`、`v41-inference-image_processor.py`）：
> 论文 `v41-paper.txt`；`v41-config.json` / `v41-inference-config.json`；官方推理实现 `v41-inference-model.py` / `v41-inference-kernel.py` / `v41-inference-convert.py`；
> 权重清单 `v41-index.json`；Figure 1b `v41-kv-cache.png`；CSA2 与 CED 的精读笔记 `attn-csa2.md`（含 890 B/token 的复算）。

## 0. 结论：9 篇，按论文 §2 的顺序切

| # | 文件 | 标题 | 对应论文 | 主要图 | 状态 |
|---|---|---|---|---|---|
| 0 | `deepseek-v41-00-overview` | 总览：一张图看懂 DeepSeek-V4.1-Flash | §1、§2.1、§4.2.1 | `V41ArchDiagram`、`V41LayerStrip` | **已写（2026-10-02）** |
| 1 | `deepseek-v41-01-ced` | 序列（上）：CED，decoder 的全局 KV 由 encoder 末态投影 | §2.2、§3.2.2 的 Decoder SWA Bounded Replay | prefill / decode 两条路径对比图；YOCO → CED 改了什么 | **已写（2026-10-02）** |
| 2 | `deepseek-v41-02-csa2` | 序列（中）：CSA2 的三种模式 | §2.3、§2.3.1、§3.1.2 | 复用 `dsattn/Csa2Cell`；共享池读写时序图（谁写、谁读、何时覆盖） | **已写（2026-10-02）** |
| 3 | `deepseek-v41-03-indexer-fp4` | 序列（下）：Hierarchical Sparse Indexer 与 FP4 main KV | §2.3.2、§2.4.4 | 候选池两级选择图；FP4 位布局图；890 B/token 的账 | **已写（2026-10-02）** |
| 4 | `deepseek-v41-04-single-pass-mhc` | 深度：Single-Pass mHC | §2.4.1 | 三个 kernel 的读写量对比（(4n+4)d → (3n+2)d → (2n+2)d）；系数错位一拍的时序图 | **已写（2026-10-02）** |
| 5 | `deepseek-v41-05-engram` | 记忆：Engram | §2.4.2、§3.1.3、Engram 论文 2601.07372 | n-gram 哈希查表流程；门控公式图 | **已写（2026-10-02）** |
| 6 | `deepseek-v41-06-dspark` | 解码：DSpark | §2.4.3、DSpark 论文 2607.05147 | 草稿块一次前向出 5 个位置；Markov head 的串行修正；置信度调度 | **已写（2026-10-02）** |
| 7 | `deepseek-v41-07-vision-optim` | 输入端与优化器 | §2.1.1、§2.5、§4.2.2 | ViT → pixel-unshuffle → projector 通路；Sinkhorn-balanced update 的行列归一化交互 | **已写（2026-10-02）** |
| 8 | `deepseek-v41-08-systems` | 系统：persistent KV cache 与 SWA Bounded Replay | §3.2、§3.1.2（共享状态的训练支持） | V4 与 V4.1 的缓存分层对比；Encoder / Decoder 两种回放 | **已写（2026-10-02）** |

系列文件 `src/content/series/deepseek-v41.md`，parts：总览 / 序列 / 深度 / 记忆与解码 / 输入端与优化器 / 系统。
图组件放 `src/components/v41/`。CSA2 的三列算子图和层排布图在 `src/components/dsattn/` 里已有（`Csa2Cell`、`CedStrip`，属于独立篇 `deepseek-attention-structures`），第 2 篇直接复用 `Csa2Cell`。

## 1. 每篇要点

### 0 总览（已写）
- 三个成本：HBM 里的全局 KV（890 B/token）、SSD 上的 persistent KV（V4-Flash 的 1/8）、prefill 计算（8B 激活）。
- 总图：4 条残差流 + 两个共享池 + encoder/decoder 分界 + Engram + DSpark + 视觉通路。
- 层排布：`compress_ratios` 43 项 = 40 + 3 个 DSpark 层；`kv_source_layer_ids`、`index_source_layer_ids`。
- 参数账：552B backbone、196B Engram、DSpark ≈ 14B 不计入 backbone（我算的，论文没给分项）；8B / 16B 的来历。
- V4-Flash → V4.1-Flash 对照表。

### 1 CED
- 问题：agent 场景 prefill 多，V4 每个 prompt token 要过全部 L 层。
- YOCO（Sun 2024）：上半层共享下半层的 KV cache，$W_K$、$W_V$ 全体上半层共用一份，上半层没有自注意力。CED 相对 YOCO：式 (1) 写成逐层的 $W_l^{KV}$、$W_l^Z$，但 V4.1-Flash 的 decoder 只有第 20 层一个 Full 层，实际只有一组投影；SWA 仍逐层自产（「增加了 local KV 的计算深度」）；encoder 自己也有全局 KV。出处见 `refs/background.md` A 部分。
- 式 (1) 在代码里没有专门模块：层 20 的 `Compressor` 输入就是 $H_{20}$。和 CSA2 组合后 decoder 只有层 20 一个 Full 层，所以式 (1) 里「逐层投影」实际只剩一份。这一点要讲清楚：论文 §2.2 是不带 CSA2 的一般形式。
- 复杂度：$O(NL) \to O(NL/2 + n_{\text{win}} L/2)$。8B / 16B 的账放这里细算。
- Decoder SWA Bounded Replay：精确重建要回放 $n_{\text{win}} \times L/2 = 2560$ 个 token，改成只回放 128 个，SWA 截断到回放段内。post-training 里模拟同样的回放。
- 误解：CED 没有 cross-attention，encoder 不是双向的。

### 2 CSA2 三种模式
- 三个相乘的维度：条目大小（GQA、MLA）× 序列（m 个 token 一条）× 层（跨层复用）。前人工作各占一格：IndexCache（只复用 top-k）、YOIO（全网共享一次路由）、HySparse（稀疏层复用稠密层 KV）。
- 压缩算子的简化：不重叠、无位置偏置，m=1 时退化成 `norm(wkv(x))`、没有 `wgate`。引用 V4 第 1 篇的式子，只写差别。
- indexer K 由 main KV latent 投影（`wk` 512 → 128），去掉 Hadamard 旋转。
- 三种模式的参数清单用 checkpoint 权重名佐证（attn-csa2.md §3.5）。
- 「缓存共享」和「索引复用」是两件事：Reindex 只共享缓存。
- 训练支持（§3.1.2）：shadow indexer、pipeline payload、micro-batch 级的共享状态生命周期。一段带过，细节放第 8 篇。
- 层排布为什么是 6 层一组 / 4 层一组：论文没给理由，不猜。

### 3 Hierarchical Sparse Indexer 与 FP4
- 问题：Reindex 层仍要对全部可见条目打分，1M 上下文 decoder m=1 就是 1M 条。
- 候选池：层 20 全量打分 → 块得分 = 块内最大值 → top-2048 块 × 8 = 16384 个位置。代码细节：最新一块强制入池；mask 形式存 `shared_attn.candidates`。
- 每 query 打分量：1M → 16384，Figure 2 的「4K → 1M decode FLOPs 只涨 1/4」。
- FP4：E2M1 + 每 16 通道一个 E4M3 scale；幅度上界的论证（RMSNorm 权重 ≤ 1 → L2 ≤ √512 ≈ 22.6；448 × 6 = 2688）。RoPE 之后量化。SWA KV 留在 FP8。
- 890 B/token：288 + 68 = 356 B/条，× 2.5 条/token。和 V4-Flash 的 3514 对比，分项列出哪一步省了多少（层维 / 精度 / 去 HCA）。V4-Flash 的分项要回 V4 论文核对。

### 4 Single-Pass mHC
- 复习 V4 第 4 篇的式子和图。
- 访存量的下界 $(2n+2)d$ 怎么来的；V4 的三个 kernel 读写各多少，合计 $(4n+4)d$。
- 把 norm 权重折进投影、RMS 除法挪到投影之后 → 残差更新和系数预测共用一次遍历，$(3n+2)d$。
- 剩下的依赖：$A_l$ 要等整条 $X_l$ 过完才有。解法：用 $A_{l-1}$（式 6）。代码：`Block.forward` 返回 `ffn_pre` 给下一层，第 0 层用 one-hot `make_identity_pre_mix`，最后的输出头用最后一层 FFN 产生的系数，所以 V4 的 `hc_head` 没有了。
- Mega-mHC kernel。

### 5 Engram
- 先 `/research-topic`：Engram 论文 2601.07372、HF repo 的 `inference/engram.py`（本地没存）。
- 已从 model.py 确认的：两个模块在层 1、14 的 block 入口；n-gram 阶 {2,3,4} × 8 头 × 256 维 = 6144 维 → `wkv` 投到 5 × 5120（4 条流各一个 key + 一个共享 value）；门 = sigmoid(sign·sqrt(|归一化点积|))；图像 token 不参与。表 384M 行 × 256 维 × 2 = 196.6B。
- V4.1 的两处改动：去掉短因果卷积；Sinkhorn-balanced update。

### 6 DSpark
- 先 `/research-topic`：DSpark 论文 2607.05147。
- 已从 model.py 确认的：3 个 block（层 40–42，纯 SWA，128 选 3 的 MoE），读主干第 37、38、39 层注意力输入的 4 流均值拼接 → `main_proj`；一次前向出 5 个位置（噪声 token 占位）；Markov head rank 256 逐位置加偏置；confidence head。参考实现只有前向，没有调度循环。
- 训练：预训练不带 MTP，之后单独一个阶段冻结主干训 DSpark。

### 7 输入端与优化器
- DeepSeek-ViT：2D-RoPE、线性 patch embedding（为了能用 Muon）、RMSNorm、SwiGLU；3×3 pixel-unshuffle；1344² → 96×96 patch → 32×32 = 1024 token。两阶段训练（SigLIP 对比 47B 对 + 接 4B MoE 做自回归 236B token）。
- 分模态的无辅助损失负载均衡：`bias` 与 `bias_vl` 两套。
- head-wise Muon：Q、K 权重按头拆开各自正交化。和 K3 的 Per-Head Muon 对照（K3 第 6 篇已讲）。
- Sinkhorn-balanced update（Alg. 1、式 7）：从「Adam 的状态太占显存」出发；行 = token，列 = 特征；K=11、τ=1e-3、γ=0.18。

### 8 系统
- V4 的 persistent KV 策略（V4 第 7 篇）→ V4.1：SWA KV 移到 10% host DRAM 的内存池，TTL 分钟级。
- Encoder SWA Bounded Replay 与 Decoder SWA Bounded Replay 各解决什么。
- kernel 数：Reuse 层 prefill 15 个、decode 11 个。EPD 分离部署。
- CSA2 的训练支持三件事。

## 2. 图的方案
- `v41/V41ArchDiagram`（已建）：`highlight` 接受 `vision | embed | swa | csa2 | pool | ced | moe | mhc | engram | dspark | output`，`LINKS` 已全部填上。
- `v41/V41LayerStrip`（已建）：43 格，模式着色，Engram 与 DSpark 读取层的标记。
- 各篇的图（都已建）：01 `CedVsYoco`、`PrefillPath`；02 `Compress2`、`PoolTimeline`（另复用 `dsattn/Csa2Cell`、`v4/CompressCell`）；03 `CandidatePool`、`Fp4Format`、`KvBytesBars`；04 `MhcPasses`、`MhcShift`（另复用 `v4/MhcCell`）；05 `EngramHash`、`EngramGate`、`GateCurve`；06 `DsparkRound`、`MarkovFix`、`SchedulerDemo`（交互）；07 `VisionPath41`、`HeadwiseSplit`、`SinkhornBalance`（交互）；08 `CacheTiers`、`ReplayPath`。

## 3. 小心的点
- 层号一律 0 起（config 与代码）。论文 §2.2 的 $l > L/2$ 是 1 起。
- `compress_ratios` 43 项，最后 3 项属于 DSpark 层。
- 890 B/token 只是全局 KV；SWA 每层 128 条另算，和长度无关。
- 552B 不含 Engram 的 196B，论文两数并列。DSpark 是否在 552B 里论文没说；按 config 手算 backbone 正好 552B，DSpark 另有约 14B。
- V4.1 没有 hash routing，config 里没有 `num_hash_layers`，`Gate` 里也没有查表分支。
- 参考实现的 cache 是 BF16 buffer 加「量化后立刻反量化」，FP4 是部署时的存储格式。
- 论文里 Engram 的引用有 2026b（arXiv）和 2026c（ACL）两条，是同一篇。

## 4. 资料
- V4.1 报告 https://arxiv.org/abs/2609.19969 ；权重与 `inference/` https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
- YOCO（NeurIPS 2024）；Engram https://arxiv.org/abs/2601.07372 ；DSpark https://arxiv.org/abs/2607.05147
- V4 报告 https://arxiv.org/abs/2606.19348 与本站 DeepSeek-V4 连载

## 5. 写完之后留下的事

- 投机解码的接受规则与无损性证明目前放在第 6 篇的折叠块里。它是背景知识，`llm` 专栏还没有对应的文章；以后补了就把折叠块换成链接。
- Sinkhorn-balanced update 放在第 7 篇（第 5 篇只留一句指过去）。
- V4 连载第 6 篇的 MTP 一节补了一句：DSpark 论文 §5.4 说 V4 预览版上线时用 MTP-1 做草稿，两周后换成 DSpark。
- 没有出处、文中已标明的量：V4-Flash main KV 的第 584 个字节；552B 是否含 DSpark 与 ViT；1/4 的 decode FLOPs 增量（Figure 2 的加权口径 BF16 1 / FP8 0.5 / FP4 0.25）没有自己复算。
