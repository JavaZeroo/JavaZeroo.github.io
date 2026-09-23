# DeepSeek-V3.2(-Exp) 的 DSA（DeepSeek Sparse Attention）一手调研

> 调研日期 2026-09-23。以 HuggingFace 官方 repo `deepseek-ai/DeepSeek-V3.2-Exp` 自带推理代码为准，论文 arXiv:2512.02556 §2.1 为辅。

## 0. 一句话

DSA = 在 V3 的 MLA 骨架上加一个 FP8 的 lightning indexer（64 头 × 128 维、key 单头共享），用 $\mathrm{ReLU}(\mathbf{q}^I\cdot\mathbf{k}^I)$ 加权和给每个 query 对全部前文打分，选出 top-2048 个 token，主注意力只在这 2048 条 KV 上算——**省的是计算，KV cache 一字节没省**（还多了每 token 128 维的索引键缓存）。

## 1. 一手资料清单

| 编号 | 资料 | 本地副本 / URL |
|---|---|---|
| [S1] | HF repo 文件列表（API） | https://huggingface.co/api/models/deepseek-ai/DeepSeek-V3.2-Exp ，lastModified 2025-11-18 |
| [S2] | HF `config.json`（DeepseekV32ForCausalLM, model_type `deepseek_v32`） | `refs/dsv32-hf-config.json`（raw: https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/raw/main/config.json） |
| [S3] | HF `inference/model.py`（922 行，含 `Indexer` / `MLA` 类） | `refs/dsv32-inference-model.py` |
| [S4] | HF `inference/kernel.py`（TileLang 写的 `fp8_index` 打分 kernel） | `refs/dsv32-inference-kernel.py` |
| [S5] | HF `inference/config_671B_v3.2.json`（demo 版参数，与 [S2] 数值一致） | `refs/dsv32-inference-config.json` |
| [S6] | HF `README.md`（含 2025-11-17 indexer RoPE bug 修正公告、FlashMLA/DeepGEMM kernel 指引） | `refs/dsv32-hf-readme.md` |
| [S7] | 论文 arXiv:2512.02556 §2.1 / §2.3 | `refs/dsv32-paper.txt`（本地拷贝，公式编号以它为准） |
| [S8] | GitHub https://github.com/deepseek-ai/DeepSeek-V3.2-Exp —— 其 `inference/` 与 HF repo 的 `inference/` 是同一份（论文脚注 2 直接指 HF 的 inference 目录，[S7]） | — |

注：HF repo **没有** `modeling_deepseek.py`；transformers 侧的类名是 `DeepseekV32ForCausalLM`（model_type `deepseek_v32`，[S1][S2]）。官方自己的推理代码就是 `inference/model.py`，论文明确说「provide an open-source implementation … to specify the details unambiguously」[S7 §2.1 脚注 2]。

## 2. 核心机制与推导

### 2a. 主注意力 = MLA（与 V3/V3.1 同骨架）

相对 V3.1-Terminus 的唯一架构改动就是加 DSA（[S7 §2.1]：「the only architectural modification … is the introduction of DSA through continued training」）。MLA 参数全部不变 [S2]：

- `hidden_size` 7168，`num_attention_heads` 128，`q_lora_rank` 1536，`kv_lora_rank` 512，`qk_nope_head_dim` 128，`qk_rope_head_dim` 64，`v_head_dim` 128，`num_hidden_layers` 61，前 3 层 dense FFN。
- KV cache 仍是每 token 每层 **576 维 latent**（512 维 kv latent + 64 维 rope 键），代码里是 `kv_cache[..., 512]` 和 `pe_cache[..., 64]` 两个 buffer [S3, MLA.__init__, L541-542]。
- decode 走 MLA 的 **MQA 模式**（latent 直接当共享 K/V，不展开）；DSA 就是搭在 MQA 模式上的——「each latent vector … will be shared across all query heads」，因为 kernel 层要求一条 KV 被多个 query 共享才划算 [S7 §2.1 "Instantiate DSA Under MLA"]。prefill 走 MHA 模式（[S3] `MLA.forward` 的 `mask is not None` 分支；[S7] Appendix A）。

### 2b. Lightning Indexer 数据流（推理，逐 token）

打分公式（[S7] 公式 (1)）：

$$I_{t,s}=\sum_{j=1}^{H^{I}} w^{I}_{t,j}\cdot\mathrm{ReLU}\big(\mathbf{q}^{I}_{t,j}\cdot \mathbf{k}^{I}_{s}\big)$$

稀疏注意力（[S7] 公式 (2)）：

$$\mathbf{u}_t=\mathrm{Attn}\Big(\mathbf{h}_t,\ \{\mathbf{c}_s \mid I_{t,s}\in\mathrm{Top\text{-}k}(I_{t,:})\}\Big)$$

对照代码 [S3, `Indexer` 类 L435-487] 的逐步张量流（`x` = hidden_states `[b, L, 7168]`，`qr` = MLA 的 query latent，见下）：

1. **q^I 的来源：复用 MLA 的 q 低秩 latent。** `qr = q_norm(wq_a(x))`（7168 → 1536，RMSNorm）在 `MLA.forward` L560 算好，同时喂给主注意力的 `wq_b` 和 indexer。indexer 自己的 `wq_b: 1536 → 64×128 = 8192`，得 `q [b, L, 64, 128]`。
2. **K^I 的来源：直接从 hidden_states。** `wk: 7168 → 128`（**单头，所有 64 个 query 头共享同一份 key**），接 `k_norm = LayerNorm(128)`，得 `k [b, L, 128]`。
3. **RoPE。** q 和 k 都按 `[64 rope 位, 64 nope 位]` 切分，前 64 维加 RoPE，**非交错（non-interleaved）布局**——和 MLA 的交错布局不同，官方 2025-11-17 专门修过这个 bug [S6; S3 L462-471 注释]。θ=10000，YaRN factor 40（[S2] `rope_scaling`，original 4096 → 163840）。
4. **Hadamard 旋转 + FP8 量化。** `rotate_activation`（hadamard_transform）后 `act_quant`（block_size 128，e4m3，scale ue8m0）。k 写入 FP8 的 `k_cache [b, seq, 128]` + `k_scale_cache [b, seq, 1]` [S3 L453-454, L472-477]。**indexer 的 key cache 是 FP8 的，每 token 128 字节 + 4 字节 scale。**
5. **逐头权重 w。** `weights_proj: 7168 → 64`（**fp32 存储**），乘 $64^{-1/2}$；再和 q 的反量化 scale、$128^{-1/2}$（`softmax_scale`）融合成 kernel 的 `q_s` 入参 [S3 L478-479]。
6. **打分 kernel。** `fp8_index`（TileLang）：fp8 q @ fp8 k → fp32 logits → **`max(logit, 0)` 即 ReLU** → 乘逐头权重 → 对 64 头求和 → 乘 k 的 dequant scale，输出 `index_score [b, L_q, L_kv]` fp32 [S4 L254-274 docstring 与 L240-249]。
7. **因果约束 + top-k。** prefill 时把 causal mask（上三角 -inf）加进 `index_score` 再 topk [S3 L481-483]，未来 token 得分 -inf 必然落选；decode 时 cache 只到 `end_pos`，天然因果。`topk(min(2048, end_pos))`——**序列短于 2048 时等价于稠密**。
8. **稀疏主注意力。** 参考实现里是「算全量分数再置 -inf」：`index_mask = full(-inf).scatter_(-1, topk_indices, 0)` 加到 MLA scores 上再 softmax [S3 L584-588（prefill）/ L600-604（decode）]。语义等价于 gather top-2048 条 latent KV 做注意力；生产环境用 FlashMLA 的稀疏 kernel 真 gather [S6]。

训练两阶段（[S7 §2.1.1]，简记）：

- **Dense warm-up**：冻结主模型、保持稠密注意力，只训 indexer。目标 = 主注意力全头分数求和、L1 归一化后的分布 $p_{t,:}$ 与 $\mathrm{Softmax}(I_{t,:})$ 的 KL（公式 (3)）。lr 1e-3，1000 步 × 16 条 128K = 2.1B token。
- **Sparse training**：开 top-k，全部参数一起训；indexer 的 KL 只在选中集合 $\mathcal{S}_t$ 上算（公式 (4)）；**indexer 输入 detach**，indexer 只收 KL 梯度，主模型只收 LM 梯度。lr 7.3e-6，k=2048，15000 步 × 480 条 128K = 943.7B token。post-training 同样用稀疏注意力 [S7 §3 开头]。

## 3. 代码落点

### [S3] `inference/model.py` — `Indexer.__init__`（L435-454，节选）

```python
class Indexer(torch.nn.Module):
    def __init__(self, args: ModelArgs):
        self.n_heads: int = args.index_n_heads        # 64
        self.head_dim: int = args.index_head_dim      # 128
        self.rope_head_dim: int = args.qk_rope_head_dim  # 64
        self.index_topk: int = args.index_topk        # 2048
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim)  # 1536→8192
        self.wk = Linear(self.dim, self.head_dim)     # 7168→128，单头共享 key
        self.k_norm = LayerNorm(self.head_dim)
        # weights_proj in the checkpoint is stored in bf16, while the parameters here are stored in fp32
        self.weights_proj = Linear(self.dim, self.n_heads, dtype=torch.float32)  # 7168→64
        self.softmax_scale = self.head_dim ** -0.5
        self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len,
            self.head_dim, dtype=torch.float8_e4m3fn), persistent=False)  # FP8 索引键缓存
```

### [S3] `Indexer.forward` 打分与 top-k（L466-487，节选）

```python
k = self.wk(x); k = self.k_norm(k)
k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
# rope in indexer is not interleaved
k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, False).squeeze(2)
q = rotate_activation(q); k = rotate_activation(k)          # Hadamard
q_fp8, q_scale = act_quant(q, block_size, self.scale_fmt)
k_fp8, k_scale = act_quant(k, block_size, self.scale_fmt)
self.k_cache[:bsz, start_pos:end_pos] = k_fp8               # 写 FP8 索引缓存
weights = self.weights_proj(x.float()) * self.n_heads ** -0.5
weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
index_score = fp8_index(q_fp8.contiguous(), weights,
                        self.k_cache[:bsz, :end_pos].contiguous(), ...)
if mask is not None:
    index_score += mask                                     # 因果 mask 先于 top-k
topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]
```

### [S4] `inference/kernel.py` — `fp8_index` docstring（L260-273）与 ReLU 核心行（L240-241）

```python
"""
fp8 q @ fp8 k -> fp32 logits
relu(fp32 logits) * q_s (weights) -> fp32 logits
fp32 logits -> fp32 logits_sum
fp32 logits_sum * k_s (e8m0) -> fp32 index_score
"""
# kernel 内：
logits[i3_n, i_h] = T.max(logits[i3_n, i_h], 0) * q_s_frag[i_h]   # ReLU × 逐头权重
```

### [S3] `MLA.forward` 里 indexer 的两个调用点（L582-588 prefill / L599-604 decode，节选）

```python
# indexer
topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask)   # qr 来自 L560 的 q_norm(wq_a(x))
index_mask = torch.full((bsz, seqlen, seqlen), float("-inf"), device=x.device).scatter_(-1, topk_indices, 0)
index_mask += mask
scores += index_mask.unsqueeze(2)
scores = scores.softmax(dim=-1)
```

## 4. 关键数字

| 数字 | 条件 | 出处 |
|---|---|---|
| indexer 头数 64、头维 128、topk 2048 | `index_n_heads` / `index_head_dim` / `index_topk` | [S2][S5] |
| indexer q 投影 1536→8192（`wq_b`），与 MLA 共享 `wq_a`(7168→1536)+`q_norm` | 论文只写「derived from h_t」，代码明确复用 qr | [S3] L445, L560, L583 |
| indexer k 投影 7168→128，单头共享，LayerNorm | `wk` / `k_norm` | [S3] L446-447 |
| indexer 逐头权重投影 7168→64，fp32 | `weights_proj` | [S3] L448-449 |
| indexer q/k 前 64 维加 RoPE，非交错；scale $128^{-1/2}$；权重再乘 $64^{-1/2}$ | | [S3] L462-471, L450, L478 |
| indexer key cache：FP8 e4m3，每 token 128 B + 4 B scale（block 128） | `k_cache` / `k_scale_cache` | [S3] L453-454 |
| MLA KV cache：每 token 每层 576 维（512 latent + 64 rope），与 V3 相同 | `kv_lora_rank` 512 + `qk_rope_head_dim` 64 | [S2]; [S3] L541-542 |
| 主注意力 128 头，qk_head_dim 192，v_head_dim 128 | | [S2] |
| 主注意力复杂度 O(L²)→O(Lk)，indexer 仍 O(L²) 但常数小 | 论文原话 | [S7 §2.3] |
| 每层每 token 核心注意力 FLOPs @ L=1M：dense ≈ 2·128·(192+128)·1M ≈ **82 GFLOP**；DSA 稀疏 ≈ 2·128·320·2048 ≈ **168 MFLOP**（约 1/488） | decode、MQA 模式，乘加计 2 FLOP | 按 [S2] 数字推算 |
| indexer 打分 FLOPs @ L=1M：2·64·128·1M ≈ **16.4 GFLOP**/层/token，FP8；投影约 +28 MFLOP | | 按 [S2][S3] 推算 |
| 61 层合计 @ 1M：V3 dense 注意力 ≈ 5.0 TFLOP/token；V3.2 ≈ 1.01 TFLOP/token（其中 indexer 占 ~1.0 TFLOP，FP8） | | 按 [S2] 推算 |
| KV cache @ 1M：V3 与 V3.2 相同（576 维/层/token；bf16 约 70 KB/token/61 层 ≈ 68 GiB；V3.2 另加 indexer 键约 8 GiB FP8） | DSA 不省缓存 | 按 [S2][S3] 推算 |
| 训练：warm-up lr 1e-3、1000 步、2.1B token；稀疏阶段 lr 7.3e-6、15000 步、943.7B token、k=2048 | 数据分布同 V3.1-Terminus 128K 扩展数据 | [S7 §2.1.1] |
| 起点 checkpoint：V3.1-Terminus（已扩到 128K）；唯一改动 = DSA | | [S7 §2.1] |
| max_position_embeddings 163840；YaRN factor 40、original 4096 | | [S2] |
| 推理成本曲线（prefill/decode）基于 H800、2 USD/GPU·h 实测；短序列 prefill 用 masked MHA 模式模拟 DSA 更划算 | Figure 3 | [S7 §2.3] |
| 生产 kernel：indexer logits（含 paged 版）在 DeepGEMM PR#200，稀疏注意力在 FlashMLA PR#98 | | [S6] |

## 5. 常见误解

1. **「DSA 减少了 KV cache」——错。** 缓存的条目数没变，仍是每 token 一条 576 维 latent；V3.2 还多存一份 indexer 的 128 维 FP8 键。省的是**计算**（每个 query 只对 2048 条做注意力），不是显存 [S2][S3][S7 §2.3]。条目数要压缩得等 V4 的 CSA/HCA。
2. **「indexer 打分用 softmax」——错。** 打分是 $\sum_j w_j\cdot\mathrm{ReLU}(q\cdot k)$，选 ReLU 是为了吞吐（论文原话「for throughput consideration」[S7 §2.1]），kernel 里是 `T.max(logit, 0)` [S4]。softmax 只出现在**训练** indexer 的 KL 损失里（公式 (3)(4)），推理打分无 softmax。
3. **「top-k 是在算完注意力分数后裁剪」——错。** 选择发生在主注意力**之前**，用的是 indexer 独立的分数；没被选中的 token 根本不进主注意力的 softmax（参考实现里是置 -inf 模拟）[S3 L582-604; S7 公式 (2)]。
4. **「indexer 的 q、k 都是独立的小投影」——半错。** k 确实独立（7168→128），但 q 复用 MLA 的 query latent `qr`（共享 `wq_a`+`q_norm`，indexer 只有自己的 `wq_b`）[S3 L445, L560, L583]。
5. **「indexer 也是多头 key」——错。** key 只有一份 128 维，64 个 query 头共享，MQA 式；每 token 索引缓存就 128 字节 [S3 L446, L453]。
6. **「indexer 把复杂度降成 O(L)」——错。** indexer 打分本身仍是 O(L²)（每 token O(L)），只是常数比 MLA 小得多（64×128 FP8 对 128×320 bf16，约 1/5 且精度更低）[S7 §2.3]。
7. **「topk=2048 是硬编码常数」——近似但有边界。** 代码是 `min(index_topk, end_pos)`，序列短于 2048 时等价稠密 [S3 L483]。
8. **「因果性由 top-k 之后处理」——错。** prefill 里 causal mask 先加进 index_score 再 topk，保证选不出未来 token [S3 L481-483]。
9. **「V3.2 是重新预训练的」——错。** 从 V3.1-Terminus 出发 continued training，两阶段共约 946B token，且刻意对齐训练配置做 parity 对照 [S6][S7 §2.1.1, §2.2]。
10. **「indexer 的 RoPE 和 MLA 一样」——错且坑过人。** indexer 用非交错布局，MLA 用交错；官方 demo 代码曾因此有 bug，2025-11-17 修复 [S6]。

## 6. 与本专栏其他文章的接口

- `deepseek-v4-00-overview.mdx`：L69-79 的「V3 → V3.2 → V4 对照表」可直接引用本文第 4 节数字（V3.2 行：MLA+DSA、KV 每 token 每层 576 数、index_topk 2048）。L104 的 `index_n_heads/index_head_dim/index_topk = 64/128/1024` 是 **V4** 的值，注意和 V3.2 的 64/128/**2048** 区分。
- `deepseek-v4-01-kv-compression.mdx` L61-63 已有「DSA」小节，说「砍的是看的条目数，缓存不变」——与本文一致，可回链本文作一手出处。
- `deepseek-v4-02-lightning-indexer.mdx` L31-52 是 DSA 复习段：公式、ReLU、FP8、两阶段训练都已覆盖且与本文一致；L70 已指出「q^I 从 c^Q_t 展开（V3.2 的开源实现里其实已经是这样做的）」——本文 [S3] L445/L560 为此提供直接证据。该篇的 IndexerCell 组件画的是 V4 版本，V3.2 版差异点：k^I 来自 hidden_states（无压缩）、topk 2048 个 **token**（非块）、FP8（非 FP4）、无滑窗拼接。
- `deepseek-v4-03-hybrid-attention.mdx` L116/L118 的 cache 与 FLOPs 账里已用「V3.2 每 token 每层 576+128、1M 时 indexer ≈ 1 TFLOP」，与本文第 4 节吻合，可互为引用。
- 若新开「DeepSeek 系列注意力对比」文，可按 MHA → MQA/GQA → MLA（V2/V3）→ MLA+DSA（V3.2）→ CSA/HCA（V4）的谱系组织，本文承担 V3.2 节点。

## 7. 图的建议

**主图：DSA 单层的双路径小算子张量流图（decode 一步，query 为单 token t，上下文 L 条）。** 建议上下两条泳道，共享输入 `h_t [7168]`。

Indexer 路径（上泳道，节点 → 边）：

1. `h_t [7168]` → `wq_a` → `q_norm` → `qr [1536]`（与下泳道共享的节点，画在中间）
2. `qr [1536]` → `wq_b (1536→8192)` → `q^I [64, 128]` → 前 64 维 `RoPE(非交错)` → `Hadamard` → `act_quant` → `q^I_fp8 [64, 128] + q_scale`
3. `h_t [7168]` → `wk (7168→128)` → `LayerNorm` → 前 64 维 `RoPE(非交错)` → `Hadamard` → `act_quant` → 写入 `k_cache_fp8 [L, 128]（+scale [L,1]）`
4. `h_t [7168]` → `weights_proj (7168→64, fp32)` → `w_t [64]`（×64^-1/2）
5. `q^I_fp8 × k_cache_fp8` → `fp8_index kernel`（ReLU、×w、对 64 头求和）→ `I [L] fp32`
6. `I [L]` → `+causal mask（prefill 时）` → `top-k(2048)` → `indices [2048]`

主注意力路径（下泳道）：

7. `h_t [7168]` → `wkv_a (7168→576)` → 拆 `kv_latent [512]`（RMSNorm）+ `k_pe [64]`（RoPE，交错）→ 写入 `kv_cache [L,512]` / `pe_cache [L,64]`
8. `qr [1536]` → `wq_b (1536→128×192)` → `q [128, 192]`（128 nope + 64 rope）
9. `indices [2048]` → **gather** `kv_cache/pe_cache` → `KV_sel [2048, 576]`（边上标注：参考代码用 -inf mask 模拟，FlashMLA 真 gather）
10. `q [128,192] × KV_sel` → softmax → ×V（latent 当 V，MQA 模式）→ `[128, 128]` → `wo (16384→7168)` → 输出

图注里标三类数字：每个张量的形状、每步 FLOPs 量级（indexer 打分 16.4 GFLOP@1M vs 核心注意力 168 MFLOP）、精度（FP8 / bf16 / fp32）。可用颜色区分「随 L 增长的算子」（indexer 打分）和「只随 k=2048 的算子」（核心注意力）——这是整张图的核心信息。

辅助小图（可选）：x 轴上下文长度 L、y 轴每层 FLOPs，两条线（dense MLA ∝ L 的 82 GFLOP@1M；DSA = 2048 常数项 + indexer 斜率小得多的线性项），交点在 ~10K 附近，解释论文 §2.3 为什么短序列用 masked MHA 模式。

## 8. 待确认 / 没查到的点

1. **transformers 库的 `DeepseekV32ForCausalLM` 建模代码未读。** HF repo 本身不带 `modeling_deepseek.py`（model_type `deepseek_v32` 由 transformers ≥4.56 内置）；若博客要引用 transformers 实现，需另查 `transformers/models/deepseek_v32/modeling_deepseek_v32.py`。本调研以官方 `inference/model.py` 为准。
2. **GitHub repo 未单独 clone 比对。** 论文脚注 2 把「开源实现」指向 HF 的 `inference/` 目录 [S7]，GitHub 仓与 HF 的 inference 文件应一致，但未逐文件 diff。
3. **FP8 下主注意力 kv_cache 的精度记法不唯一。** 参考代码 kv_cache buffer 是 bf16（仅模拟一次 fp8 量化往返，[S3] L569-571），注释说实际部署用 fp8；第 4 节的 cache 账按 bf16/fp8 两种口径都给了，博客引用时需注明口径。
4. **weights 与 q_scale 融合后 kernel 入参 `q_s` 的精确广播形状**（`[b, L, 64, 1]` vs kernel 签名 `(b, m, h)`）在 TileLang `T.copy` 语义下应能 squeeze，但没有跑代码验证。
5. **prefill 的 masked MHA 模式**论文 §2.3 只提了一句，参考代码里就是 `mask is not None` 分支的全量打分 + index_mask，没有独立的短序列优化实现可读。
6. **indexer 在 TP 下的一致性**：代码里 `dist.broadcast(topk_indices_, src=0)` + assert 全相等（[S3] L484-486），说明各 rank 独立算 topk 并要求一致；多卡下数值是否严格一致（fp8 打分 + 逐 rank 部分头？实际 wq_b 未做并行切分）没有深究。
