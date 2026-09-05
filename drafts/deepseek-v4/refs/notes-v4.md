# DeepSeek-V4 精读笔记（数字与代码事实）

来源：论文 §2/§3.4–3.5/§4.2；HF config.json（Pro/Flash）；官方 inference/model.py、kernel.py；transformers modeling_deepseek_v4.py。

## 层排布（config.compress_ratios，0-indexed；最后一项属于 MTP 层）
- Pro（61 层）：idx 0,1 = HCA(128)；2..60 偶数 = CSA(4) 共 30 层；3..59 奇数 = HCA 共 29 层 → HCA 31 层；idx 61 = MTP，ratio 0（纯滑窗）。
- Flash（43 层）：idx 0,1 = 0（纯滑窗）；2..42 偶数 = CSA 共 21 层；3..41 奇数 = HCA 共 20 层；idx 43 = MTP，0。
- 前 3 层 MoE 用 hash routing（num_hash_layers=3），其余 topk（noaux_tc）。没有 dense 层。

## 维度
| | Pro | Flash |
|---|---|---|
| d | 7168 | 4096 |
| 层 | 61 | 43 |
| n_h / c | 128 / 512 | 64 / 512 |
| q_lora_rank d_c | 1536 | 1024 |
| KV 条目 | 1 × 512（K=V） | 1 × 512 |
| rope 维 | 64 | 64 |
| o_groups g / o_lora_rank d_g | 16 / 1024 | 8 / 1024 |
| 滑窗 n_win | 128 | 128 |
| CSA m / HCA m' | 4 / 128 | 4 / 128 |
| indexer 头 / 维 / top-k | 64 / 128 / 1024 | 64 / 128 / 512 |
| routed / 激活 / shared | 384 / 6 / 1 | 256 / 6 / 1 |
| expert 中间维 | 3072 | 2048 |
| routed_scaling_factor | 2.5 | 1.5 |
| scoring_func | sqrtsoftplus | sqrtsoftplus |
| swiglu_limit | 10 | 10 |
| hc_mult / sinkhorn iters | 4 / 20 | 4 / 20 |
| vocab | 129280 | 129280 |
| rope_theta（滑窗层）/ compress_rope_theta | 10000 / 160000 | 同 |
| YaRN | factor 16，original 65536，beta 32/1 | 同 |
| MTP | 1 | 1 |
| 总/激活 | 1.6T / 49B | 284B / 13B |

## 参数账（按 config 手算）
- Pro：expert = 3×7168×3072 = 66.1M；384 个/层 = 25.4B；61 层 = 1.573T。shared 4.1B。注意力每层 300M（q_a 11M、q_b 100.7M、kv 3.7M、o_a 16×4096×1024 = 67M、o_b 117M）+ CSA 附加 31M（compressor 2×7168×1024、indexer q_b 1536×8192、w 7168×64、indexer compressor 2×7168×256）/ HCA 附加 7.3M → 19.8B。emb+head 1.85B。mHC 每层 2×24×28672 = 1.4M。总 ≈ 1.60T；激活 ≈ 50B（expert 7×66M×61 = 28.7B + 注意力 19.8B + emb/head 1.85B）。
- Flash：expert 25.2M，256 个 = 6.44B/层，43 层 = 277B；注意力 107M/层 → 5.2B；总 ≈ 291B；激活 ≈ 14B。

## KV cache（1M token，每条目 64 维 BF16 + 448 维 FP8 = 576 B，忽略 scale）
- Pro：CSA 30 层 × 2^18 条目 × (576 + indexer 128 维 FP8 128 B) = 5.16 GiB；HCA 31 层 × 2^13 × 576 = 0.136 GiB；滑窗 61 × 128 × 576 = 4.4 MiB；合计 ≈ 5.3 GiB ≈ 5.4 KB/token（indexer 用 FP4 则 4.8 GiB）。
- 基线 GQA8-BF16-hd128-61 层：250 KB/token → 244 GiB；V4 ≈ 2%。V3.2（576 维 latent：512 FP8 + 64 BF16 = 640 B，+ indexer 128 B）≈ 46.8 KB/token → 45.8 GiB；V4 ≈ 11%（论文说 10%）。
- Flash：≈ 3.7 GiB。

## 注意力一层的前向（官方 model.py Attention.forward）
1. q：x → wq_a (d→1536) → RMSNorm → wq_b (1536 → 128×512) → 逐头 RMSNorm（无权重，`q *= rsqrt(mean(q²)+eps)`）→ 最后 64 维 RoPE。
2. kv：x → wkv (d→512) → RMSNorm(512) → 最后 64 维 RoPE → 非 rope 维 FP8 量化（64 维一组）。写入滑窗 cache（环形，128 条）。
3. 压缩：Compressor(x) → 每 ratio 个 token 产生一条 512 维条目（softmax 门控加权和，fp32 计算）→ RMSNorm → 块首位置 RoPE → 写入压缩 cache。CSA 的 compressor 输出 2×512（a/b 两系列）并重叠。
4. 索引：CSA 有 Indexer（自己的 compressor，128 维，Hadamard 旋转 + FP4）；q 由 qr（1536 维 latent）经 wq_b 得 64 头×128；score = Σ_h w_h ReLU(q_h·k)，w = weights_proj(x) × (128^-0.5 × 64^-0.5)；top-1024。HCA 直接取全部压缩条目（规则生成索引）。
5. 核心注意力 sparse_attn(q, kv, sink, topk_idxs, scale=512^-0.5)：索引 = [128 个滑窗位置; 压缩条目索引]，K=V=kv 条目，online softmax，最后分母加 exp(sink_h − max)。
6. 输出：最后 64 维用当前位置做反旋转（conj）；分 16 组 einsum 到 16×1024；wo_b → d。
- 滑窗层（ratio 0）：rope_theta 10000，无 YaRN；压缩层用 compress_rope_theta 160000 + YaRN(16, 64K)。

## 压缩算子（Eq.9–12，代码 Compressor）
- ape：[ratio, coff×512] 可学习位置偏置（coff=2 若重叠）。score = wgate(x) + ape；kv = wkv(x)。
- 重叠（ratio 4）：new[:, :, ratio:] = 当前块的 b 半段（后 512 维），new[:, 1:, :ratio] = 上一块的 a 半段（前 512 维）；第 0 个条目的 a 段 kv=0、score=−inf。softmax 在 2m=8 个槽上做，逐通道。
- decode：kv_state/score_state 缓冲 2×ratio 槽；每 ratio 个 token 输出一条。
- 输出 RMSNorm(512) → 块首位置 RoPE。

## mHC（Eq.1–8；代码 Block.hc_pre / hc_post；kernel hc_split_sinkhorn）
- 残差流 x: [b, s, 4, d]。每个子层前：flatten → RMS 归一化（无权重）→ linear hc_fn [24, 4d] → mixes[24]：pre[4] = sigmoid(mixes[:4]·scale0 + base[:4]) + eps；post[4] = 2·sigmoid(mixes[4:8]·scale1 + base[4:8])；comb[4×4] = mixes[8:]·scale2 + base[8:] → softmax 行 + eps → 列归一化 → (行、列归一化)×19 = 20 轮。
- 读：y = Σ_j pre_j x_j（d 维）→ RMSNorm → 子层 F。写+混：x' = post ⊗ F(y) + combᵀ x（transformers 注释：按第一个 hc 轴求和，即 x'_k = Σ_j comb[j,k] x_j）。
- 输出：hc_head = sigmoid(pre) + eps 压成 1 条 → RMSNorm → head。MTP 块：x = e_proj(enorm(e)) 广播到 4 流 + h_proj(hnorm(x))，过一个完整 Block，自己的 hc_head。
- 论文 mHC：HC 复合映射增益最高 3000；mHC 20 轮 Sinkhorn 后复合最高约 1.6；27B 实验 loss −0.021 vs baseline；训练开销 6.7%。重算块 L_r* ≈ sqrt(nL/(n+2))。

## MoE（Gate / Expert / MoE）
- scores = sqrt(softplus(x·Wᵀ))（fp32）；选 top-6 用 scores + bias（bias fp32，推理冻结）；权重 = 原始 scores gather / sum × route_scale。
- hash 层：indices = tid2eid[input_ids]（[vocab, 6] int32，不训练），权重仍由 scores 计算。
- Expert：gate=w1(x), up=w3(x)（fp32）；swiglu_limit>0 时 up∈[−10,10]、gate≤10；silu(gate)·up → w2。FP4 权重（E2M1，1×32 块 E8M0 scale），激活 FP8（1×128）。shared expert 与 routed 同形，BF16/FP8。
- 训练：aux-loss-free 偏置更新速度 0.001；序列级平衡损失权重 1e-4；去掉节点限制路由。

## Muon（Alg.1）
- momentum 0.95，wd 0.1，RMS 匹配 0.18；Nesterov；hybrid NS：8 步 (3.4445, −4.7750, 2.0315) + 2 步 (2, −1.5, 0.5)；O = NS(·)·√max(n,m)·γ。
- AdamW：embedding、head、mHC 静态偏置和 α、RMSNorm 权重；β 0.9/0.95、ε 1e-20、wd 0.1。
- 不用 QK-Clip：q、kv 都过 RMSNorm。

## 训练配方（§4.2.2）
- Flash：32T token；batch 增至 75.5M；lr warmup 2000 步 → 2.7e-4 → cosine 到 2.7e-5；长度 4K→16K→64K→1M；前 1T dense，64K 时引入稀疏（先短暂 warmup indexer）。MTP 权重 0.3 → 0.1。
- Pro：33T；batch 94.4M；lr 2.0e-4 → 2.0e-5；dense 阶段更长。
- §4.2.3：Anticipatory Routing（路由索引用 θ_{t−Δt}，spike 触发，20% 开销）；SwiGLU clamp。

## 系统（§3.4.3、§3.5）
- CP：rank i 发最后 m 个未压缩 KV 给 i+1；每 rank 产出固定 s/m+1 条（含 padding）；all-gather + select-and-pad，padding 放尾部。
- KV cache：经典 cache 块 = lcm(m, m') = 128 原 token → 32 CSA 条目 + 1 HCA 条目；状态 cache 每序列固定块（滑窗段 + 未满块尾 token）。
- 磁盘缓存：压缩条目全存，尾块重算；滑窗 KV 三策略（全存 / 每 p token 存最近 n_win / 不存并重算最后 n_win·L 个 token）；滑窗 KV 体积 ≈ 压缩条目的 8 倍。
