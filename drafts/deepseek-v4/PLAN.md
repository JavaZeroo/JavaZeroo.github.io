# DeepSeek-V4 模型结构连载：写作大纲

> 目标读者：手边打开着 DeepSeek-V4 技术报告（arXiv 2606.19348）§2 "Architecture"，想把每个模块看懂的人。
> 写法与 Kimi K3 连载相同：每篇对应论文一个小节，配一张「你现在在这里」的总图高亮，加该模块自己的方块图；
> 公式从动机/目标函数推出来，谱系用流程图，每节先交代为什么讲。
> 论文原文（含公式）在 `refs/v4-paper.txt`，mHC 论文 `refs/mhc-paper.txt`，V3.2 论文 `refs/dsv32-paper.txt`；
> 真实超参 `refs/v4-pro-config.json` / `refs/v4-flash-config.json`；官方推理实现 `refs/official-model.py` / `official-kernel.py`；
> transformers 实现 `refs/modeling_deepseek_v4.py`。精读笔记 `refs/notes-v4.md`。

## 0. 结论：8 篇，和 K3 一样按「序列 → 深度 → 宽度 → 优化器 → 系统」切

论文 §2 只有 9 页，三个升级各压着一篇独立论文（CSA ← DeepSeek-V3.2 的 DSA 2512.02556；mHC ← 2512.24880；Muon ← Moonlight 2502.16982），
加上 V4 自己在 MoE 上的小改动（sqrt-softplus 打分、hash routing、序列级平衡损失、SwiGLU clamp）和 §4.2.3 的两个稳定性技巧。

| # | 文件 | 标题 | 对应论文 | 主要图 |
|---|---|---|---|---|
| 0 | `deepseek-v4-00-overview` | 总览：一张图看懂 DeepSeek-V4 | §2 开头、§4.2.1、Fig.2 | 总架构图（4 条残差流 + mHC 算子）、Pro/Flash 层排布条 |
| 1 | `deepseek-v4-01-kv-compression` | 序列（上）：从 MLA 到压缩 KV | §2.3 开头、§2.3.1 前半 Eq.9–12 | 注意力谱系流程图（MHA→MQA/GQA→MLA→DSA→NSA→CSA/HCA）、压缩算子张量图（重叠窗口） |
| 2 | `deepseek-v4-02-lightning-indexer` | 序列（中）：Lightning Indexer 与稀疏选择 | §2.3.1 后半 Eq.13–19、V3.2 §2.1 | indexer 方块图、CSA 整层方块图（Pro 维度） |
| 3 | `deepseek-v4-03-hybrid-attention` | 序列（下）：HCA、混合排布与零件 | §2.3.2–2.3.4 Eq.20–27 | HCA 方块图、滑窗分支、RoPE 反旋转推导、sink、1M 的 KV cache/FLOPs 账 |
| 4 | `deepseek-v4-04-mhc` | 深度：mHC | §2.2 Eq.1–8；mHC 全文 | 残差/HC/mHC 三联图、Sinkhorn 交互、复合映射增益演示 |
| 5 | `deepseek-v4-05-moe` | 宽度：DeepSeekMoE 的四个改动 | §2.1、§4.2.3 SwiGLU clamp | 打分函数曲线交互、hash routing 图、参数账 |
| 6 | `deepseek-v4-06-muon` | 优化器与稳定性：Muon | §2.4 Alg.1 Eq.28、§4.2.3 | Newton–Schulz 多项式交互、Anticipatory Routing 时间线 |
| 7 | `deepseek-v4-07-systems` | 结构决定系统 | §3.4.2、§3.4.3、§3.5、§3.1 一句 | KV cache 布局图、CP 两阶段通信、磁盘缓存三策略 |

## 1. 每篇要点

### 0 总览
- 两个模型：Pro 1.6T/49B 激活、61 层、d=7168；Flash 284B/13B、43 层、d=4096；都 1M 上下文、32T+ token、FP8/FP4。
- 三个升级：mHC（深度）、CSA+HCA 混合注意力（序列）、Muon（优化器）；MoE 与 MTP 沿用 V3 只小改。
- 层排布（config `compress_ratios`，0-indexed）：Pro = [HCA, HCA] + [CSA, HCA]×29 + [CSA]，即偶数层 CSA（30 层）、奇数层 + 第 0 层 HCA（31 层）；Flash = [SWA, SWA] + [CSA, HCA]×20 + [CSA]。**数组有 62/44 项，最后一项 0 是 MTP 层：MTP 只用滑窗注意力**（论文没写）。
- 前 3 层 MoE 用 hash routing（`num_hash_layers=3`），V3 的前 3 层 dense FFN 不存在了。
- mHC：`hc_mult=4`，4 条残差流，每个子层前后各一次读/写/混。
- Pro/Flash 维度表 + config 字段表；V3 → V3.2 → V4 对照表。
- 参数账（refs/notes-v4.md 里算过）：Pro 总 1.60T（routed expert 1.57T），激活 50B；Flash 291B/14B。

### 1 KV 压缩
- 谱系流程图（可点击）：MHA → MQA/GQA（砍头）→ MLA（砍到 latent，缓存 576 数）→ DSA（V3.2，砍长度：只看 top-k）→ NSA（2502.11089，压缩分支的先例）→ CSA/HCA（压缩 + 稀疏 / 重压缩）。每条边标「砍了哪个维度」。
- 动机：1M 上下文下 KV cache 与注意力 FLOPs 随 n 线性/平方增长；三个可砍的量：头数、头维、条目数。V4 把头维反而放大到 512（单头 KV），砍的是条目数。
- 从平均池化推出压缩算子：把 m 个 token 压成一个条目，最简单是均值；改成加权 softmax(Z + B)，Z 是逐通道的可学习权重（Eq.10）、B 是块内位置偏置（哪个位置更重要）；⊙ 表示逐通道各自一套权重（每个通道自己选 token）。
- 重叠窗口（Eq.11–12）：两套 (C^a, Z^a)、(C^b, Z^b)，条目 i 由块 i 的 b 系列和块 i−1 的 a 系列共 2m 个 token 做一次 softmax。为什么要重叠：块边界信息不断裂、每个 token 参与两个条目。代码里 `wkv` 输出 2×512，前半是 a（给下一个条目），后半是 b（给当前）。
- Shared-KV MQA：K = V = 同一条 512 维向量，所有 128 个 query 头共享（`num_key_value_heads=1`）。为什么可以 K=V：MLA 的 MQA 模式里 latent 已经既当 K 又当 V。
- grouped output projection：128×512 = 65536 维输出，直接投影到 7168 要 470M 参数/层；分 16 组，每组 4096 → 1024，再 16384 → 7168（`o_groups=16`、`o_lora_rank=1024`），117M+67M。

### 2 Lightning Indexer
- DSA 复习（V3.2 §2.1）：indexer 公式、ReLU 的原因（吞吐）、FP8/FP4；warmup 用主注意力分布做 KL 蒸馏，稀疏阶段只在选中集合上蒸馏，indexer 输入 detach。
- V4 的变化：索引对象从 token 变成压缩块（K^IComp 由同样的压缩算子产生，`index_head_dim=128`，自己的一套 compressor），query 从共享的 c^Q（q_lora_rank 1536）出来；top-k 从 V3.2 的 2048 token 降到 1024 个块（Flash 512），每块 4 token，覆盖 4096 token。
- 代码细节：indexer 的 q 和 K 都做 Hadamard 旋转再 FP4 量化（`rotate_activation`），权重 w 乘 `softmax_scale * n_heads^-0.5`。
- 训练：先 dense 1T token 再在 64K 长度引入稀疏，两阶段。
- CSA 整层方块图（Pro 维度）。

### 3 HCA 与零件
- HCA（Eq.20–23）：m'=128，不重叠，无 indexer，全看。为什么 HCA 不用 indexer：n/128 个条目在 1M 下只有 8192 个，密集算得起。
- 排布：Pro 第 0、1 层 HCA，Flash 第 0、1 层纯滑窗；之后 CSA/HCA 交错；最后一层 CSA。
- 滑窗分支：n_win=128 个未压缩 KV，理由是因果性（自己所在块看不到）+ 局部性；代码里滑窗 KV 和压缩 KV 拼在同一个 sparse_attn 里，topk_idxs 前 128 项是窗口。
- Q/KV RMSNorm；partial RoPE 64 维；**K=V 导致输出带绝对位置 → 对输出做 −i 的反旋转**，推导：Σ softmax · R_j v_j，乘 R_{-i} 得 Σ · R_{j−i} v_j。压缩条目的位置取块首（代码 `freqs_cis[:cutoff:ratio]`），theta 160000 + YaRN（factor 16，原始 64K）；滑窗层用 theta 10000 无 YaRN。
- attention sink（Eq.27）：从 softmax 的「必须分完 1」推出加一个空 slot；每头一个可学习 logit；与 gpt-oss 同款。kernel 里在 online-softmax 结束后把 exp(sink − max) 加进分母。
- 效率：KV 存储 rope 64 维 BF16 + 448 维 FP8 = 576 B/条目；1M 下 Pro 约 5 KB/token ≈ 4.9 GiB，是 GQA8-BF16（244 GiB）的 2%，V3.2（约 46 GiB）的 ~10%；FLOPs 27%。核心注意力 FLOPs：每 token 每层 128 头 × (128 + 1024) 条目 × 512 × 2（QK 与 PV）。

### 4 mHC
- 残差是深度上的恒等映射（He 2016b）；HC 把残差流拓宽成 n 条（Eq.1），三个映射 A/B/C；Table 1：B（res）贡献最大。
- 为什么不稳：展开 L 层得复合映射 ∏B（mHC Eq.4），无约束时增益到 3000×；用行和/列和最大值度量前向/反向增益。
- 从「想要什么性质」推出双随机：要保均值 → 行和为 1；反向保均值 → 列和为 1；非负防抵消；三条性质：谱范数 ≤ 1、乘法封闭、Birkhoff 多面体 = 置换的凸组合（证明 ‖B‖₂ ≤ 1：‖B‖₂² ≤ ‖B‖₁‖B‖∞ = 1）。
- 参数化：flatten 4×d → RMSNorm → 一个 [24, 4d] 的线性（4 pre + 4 post + 16 res）→ 每部分 scale α + 静态偏置 → sigmoid / 2·sigmoid / Sinkhorn(20)。代码：`hc_attn_fn [24, 28672]`，`hc_attn_scale [3]`，`hc_attn_base [24]`；kernel 里先 softmax 行再列，共 20 次。
- 接线：每个 decoder layer 两套（attn / ffn），最后 `hc_head` 只有 pre（sigmoid + eps）把 4 流压成 1 再 RMSNorm、LM head；MTP 块自己一套 head。embedding 复制 4 份进入。
- 和 AttnRes 的关系：mHC = 深度上的线性 RNN（状态 4×d），AttnRes = 深度上的 softmax 注意力；K3 第 4 篇的对照表可引用。
- 系统：6.7% 开销；重算块大小 L_r* ≈ sqrt(nL/(n+2))；DualPipe 调度。

### 5 MoE
- 沿用 DeepSeekMoE：细粒度 + shared expert；Pro 384 选 6 + 1 shared（V3 256 选 8）；Flash 256 选 6。
- 打分函数 sigmoid → sqrt(softplus)：从「打分要在 top-k 选择后重新归一化，还要乘 route_scale」出发，sigmoid 饱和后梯度消失、上界 1 把相对差异压扁；sqrt(softplus) 负半轴 ≈ e^{x/2}（像 sigmoid 一样压小分数），正半轴 ≈ √x 无界但缓慢；画曲线 + 导数。论文没给理由，标注为推测。
- aux-loss-free 偏置 + 极小的序列级平衡损失（权重 1e-4，偏置更新速度 0.001）；`routed_scaling_factor` 2.5 / 1.5。
- hash routing 前 3 层（Roller 2021）：token id → 固定 6 个 expert（`tid2eid [129280, 6]`），权重仍由 gate 打分；为什么在浅层可以：浅层隐状态 ≈ embedding，路由本来就近似只看 token；好处是无需平衡、决定性；代替了 V3 的 dense 前 3 层。
- 去掉节点限制路由（V3 的 M=4 节点）；EP 并行策略重设（§3.1 一句）。
- SwiGLU clamp（§4.2.3）：up 截到 [−10, 10]，gate 上限 10（下不截，因为 silu 负半轴自带有界）；与 K3 SiTU-GLU 对照（硬 vs 软）。
- FP4 expert（E2M1，1×32 块 UE8M0 scale）、其余 FP8；QAT 在 post-training 做。
- 参数账：Pro expert 3×7168×3072 = 66M，384 个 = 25.4B/层 ×61 = 1.55T。

### 6 Muon 与稳定性
- 从「把梯度矩阵正交化」的动机推：Adam 逐元素、Muon 逐矩阵；最速下降在谱范数下的解是 UVᵀ；NS 迭代求 UVᵀ 不做 SVD。
- 奇异值上的多项式 p(σ) = aσ + bσ³ + cσ⁵：Jordan 的 (3.4445, −4.7750, 2.0315) 在 0 附近斜率 3.44 快速抬高小奇异值但 p(1) = 0.70 不固定 1；(2, −1.5, 0.5) 满足 p(1)=1、p'(1)=0（二次收敛到 1）但 p'(0)=2 慢。前 8 步用前者、后 2 步用后者 = hybrid。交互图画两条曲线和迭代 k 次后的复合函数。
- Moonlight 的两项：weight decay、RMS 匹配 0.18（Alg.1 第 7 行 √max(n,m)·γ）；Nesterov。
- 哪些参数用 AdamW：embedding、head、mHC 的静态偏置和 α、RMSNorm 权重（β1 0.9、β2 0.95、ε 1e-20、wd 0.1）。
- 为什么不用 QK-Clip：Q/KV 都做了 RMSNorm，logit 自然有界；对照 K2 的 MuonClip。
- 训练稳定性：loss spike ↔ MoE 离群值 ↔ 路由恶性循环；Anticipatory Routing（用 θ_{t−Δt} 算路由索引、只在 spike 后自动开启、20% 额外开销）；SwiGLU clamp（放第 5 篇讲，这里引用）。
- MTP：和 V3 一样 1 层，loss 权重 0.3 → 0.1；结构 = 完整 Block（含 mHC）+ e_proj/h_proj + 自己的 hc_head；注意力是纯滑窗。
- 训练配方数字：Flash 32T / Pro 33T token；batch 75.5M / 94.4M；lr 2.7e-4 / 2.0e-4；4K → 16K → 64K → 1M；dense 1T 后引入稀疏。

### 7 系统
- KV cache 两类：经典 cache（CSA/HCA 压缩条目，块 = lcm(4,128)=128 个原 token → 32 个 CSA 条目 + 1 个 HCA 条目）与状态 cache（滑窗 128 条 + 未满块的尾 token，按序列固定大小分配，像 SSM 状态）。为什么 PagedAttention 不够。
- CP 两阶段：rank i 发最后 m 个未压缩 KV 给 i+1；本地压成 s/m+1 条（含 padding）；all-gather + select-and-pad。
- 磁盘缓存：压缩条目全存；滑窗 KV 三策略（全存 / 每 p 个 token 存一次 / 不存，重算最后 n_win·L 个 token，因为第 l 层滑窗 KV 只依赖上一层最近 n_win 个 token）。
- mHC：融合 kernel、重算、DualPipe；Muon 的分布式实现一句；EP 细粒度重叠一句；TileLang 开源 mega-kernel 一句。

## 2. 图的方案
- `V4ArchDiagram`：4 条残差流竖线，每个子层前一个「读」（A）、后一个「写 + 混」（C, B）算子；子层四种：CSA / HCA / SWA、MoE / hash-MoE；顶部 embedding 复制 4 份，底部 hc_head 压成 1 份 → RMSNorm → LM head，MTP 一行。`highlight` prop + LINKS。
- `V4LayerStrip`：`model="pro"|"flash"`，61/43 格，CSA/HCA/SWA 三色，前 3 层 hash MoE 标记，MTP 格单独。
- `AttnLineage`：可点击谱系流程图。
- `CompressCell`：重叠窗口张量图（m=4，两个条目，2m 个格子）。
- `CsaCell` / `HcaCell`：整层方块图，Pro 维度。
- `KvCacheBars`：1M 下 GQA8 / V3.2 / V4-Pro / V4-Flash 的 KV cache 柱状。
- `MhcCell`：4 流一层的读/写/混；`SinkhornDemo`：随机矩阵 → 迭代到双随机，显示行/列和；`GainDemo`：L 层复合映射增益 HC vs mHC。
- `ScoreDemo`：sigmoid vs sqrt(softplus)；`NewtonSchulzDemo`：两组系数的多项式与 k 步复合。
- 复用 `src/components/k3/SvgTex.astro` 与 `k3/tex.ts`。

## 3. 小心的点
- 论文的层号从 1 数（"first two layers"），config 与代码 0-indexed；本连载正文用 config 的 0-indexed 并注明。
- `compress_ratios` 多出的最后一项属于 MTP 层。
- 论文 §2.3.1 的 "$d_g < c\,n_h/g$" 那句在 arXiv HTML 里被截断；以代码 `wo_a [16 × 4096 → 1024]`、`wo_b [16384 → 7168]` 为准。
- indexer 的 K 在官方代码里做 Hadamard 旋转 + FP4 模拟量化；主注意力 KV 的非 rope 维 FP8（64 维一组 UE8M0 scale）。
- 论文说"first two layers use HCA/SWA"，Flash 的 config 前两层是 0（纯滑窗），Pro 是 128（HCA）。
- HF 模型卡与论文都说 shared expert 1 个；`n_shared_experts=1`。
- attention sink 的 exp 加在分母，不参与分子。
- Muon 只对"logically independent weight"逐个正交化：grouped output projection 的 16 组、expert 各自算独立矩阵（论文没细说，只说 logically independent）。

## 4. 资料
- V4 报告 https://arxiv.org/abs/2606.19348 ；权重 https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro （inference/ 目录是官方实现）
- DeepSeek-V3.2 / DSA https://arxiv.org/abs/2512.02556 ；mHC https://arxiv.org/abs/2512.24880 ；Hyper-Connections https://arxiv.org/abs/2409.19606
- NSA https://arxiv.org/abs/2502.11089 ；Moonlight/Muon https://arxiv.org/abs/2502.16982 ；Kimi K2 MuonClip https://arxiv.org/abs/2507.20534
- Hash Layers https://arxiv.org/abs/2106.04426 ；gpt-oss（attention sink、SwiGLU clamp）https://arxiv.org/abs/2508.10925
- Kimi K3 连载（本站）里的 AttnRes 一篇可与 mHC 对照
