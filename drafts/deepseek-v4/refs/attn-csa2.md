# DeepSeek-V4.1-Flash 注意力结构调研：CSA2 与 CED

## 0. 一句话

CSA2 把 KV cache 压缩从「单层内部」扩展到「跨层」：40 层里只有 4 个 Full 层真正生成全局 KV（main KV + indexer K），4 个 Reindex 层用自己的 indexer Q 重打分共享 K 选新 Top-K，其余 30 个 Reuse 层连索引都直接复用；再叠加 CED（decoder 全局 KV 一律从 encoder 最终隐藏态投出）与 FP4 main KV，全局 KV 降到 **890 B/token**（V4-Flash 的 ~1/4），且这个数可以用 config 精确复算（356 B/条 × 2.5 条/token）。

## 1. 一手资料清单（URL + 读了哪些部分）

| # | 资料 | 读了什么 |
|---|---|---|
| [P1] | 论文 HTML 全文 https://arxiv.org/html/2609.19969v1 （已存 `refs/v41-paper.txt`） | Abstract；§1；§2.1/2.1.1；§2.2 CED（含式 (1)）；§2.3/2.3.1/2.3.2 CSA2 全文；§2.4.1–2.4.4（Single-Pass mHC / Engram / DSpark / FP4 Main KV）；§2.5；§3.1.2 Attention Sharing Training；§3.2/3.2.1/3.2.2（Persistent KV、SWA Bounded Replay）；§4.2.1 Model Setups（层排布唯一权威出处）；§6 |
| [P2] | HF repo 文件列表 https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash | 全量 siblings：无根目录 `modeling_*.py`（transformers 5.6.0 内置 `DeepseekV41ForCausalLM`）；官方参考实现在 `inference/`（model.py / kernel.py / generate.py / convert.py / config.json） |
| [P3] | `config.json`（raw，已存 `refs/v41-config.json`） | text_config 全部字段：`compress_ratios`(43)、`kv_source_layer_ids`、`index_source_layer_ids`、`candidate_*`、量化配置 |
| [P4] | `inference/config.json`（raw，已存 `refs/v41-inference-config.json`） | 与 [P3] 互验；字段名（`kv_source_layers` 等） |
| [P5] | `inference/model.py`（raw 全文 1309 行，已存 `refs/v41-inference-model.py`） | `Compressor` / `Indexer` / `select_candidate_blocks` / `Attention` / `SharedAttentionRuntime` / `Block` / `DSpark*` 全部精读 |
| [P6] | `inference/kernel.py`（raw 全文 591 行，已存 `refs/v41-inference-kernel.py`） | `fp4_act_quant`（E2M1 + E4M3/E8M0 scale）、`act_quant`、`sparse_attn`（gather + online softmax + attn_sink） |
| [P7] | `README.md` + `inference/README.md`（已存 `refs/v41-README.md` 等） | 890 B/token、552B+196B、45T、64K→1M、Table 1（8B/16B） |
| [P8] | `assets/dsv41_kv_cache.png`（Figure 1b，已存 `refs/v41-kv-cache.png`，已读图） | 各代全局 KV/token：V1 389,120 → V3.2 48,068 → V4-Flash 3,514 → V4.1-Flash 890 |
| [P9] | `model.safetensors.index.json`（已存 `refs/v41-index.json`） | 按层权重名核对三模式的参数差异（layer 0/2/3/20/24） |
| [L1] | 本地 `refs/official-model.py`（V4 官方推理代码，827 行） | V4 `Compressor`（overlap+APE）、V4 `Indexer`（自带压缩路径+Hadamard）、V4 `Attention` |
| [L2] | 本地 `refs/v4-flash-config.json` / `official-config.json` / `notes-v4.md` | V4-Flash 层排布（CSA m=4 与 HCA m=128 交替）、V4 的 KV 账 |
| [L3] | `src/content/posts/deepseek-v4-0{0..7}.mdx` 标题与小节结构 | 第 6 节接口 |

## 2. 核心机制与推导

### 2.1 CED（Causal Encoder-Decoder，论文 §2.2）

- 40 层因果 Transformer，下 20 层 = causal encoder，上 20 层 = decoder（§2.1、§4.2.1 [P1]）。灵感来自 YOCO（Sun et al., 2024）。
- **全局注意力**：decoder 层（l > L/2）的 KV 条目不从自己的 H_l 产生，而是从 encoder 最终隐藏态 H_{L/2} 用**逐层投影**产生，论文式 (1)：
  `C_l = H_{L/2} W_l^{KV},  Z_l = H_{L/2} W_l^{Z},  l > L/2`
  C 是 KV 条目、Z 是压缩权重。→ prefill 只需算前半层，**激活参数 prefill 8B / decode 16B**（Abstract、Table 1 [P1][P7]）。
- **SWA**：仍逐层从自己隐藏态产生（这「增加了 local KV 的计算深度」），代价是 prefill 时 decoder 需补算 SWA KV → Decoder SWA Bounded Replay：只对 prompt 最后 `n_win=128` 个 token 跑 decoder（§2.2、§3.2.2 [P1]）。
- 复杂度：O(NL) → O(N·L/2 + n_win·L/2) ≈ O(NL/2)（§2.2）。
- 注意 CED 在代码里**没有专门模块**：decoder 唯一的 KV 源层恰好是第 20 层（0 基），它的输入就是 encoder 末层输出——式 (1) 由该层的 `Compressor` 天然实现（见 §3）。参考实现不含「decoder 只回放 128 token」的 serving 优化，那是部署路径（§3.2.2）。

### 2.2 CSA2 三模式（§2.3、§2.3.1 [P1]）

CSA2 把压缩沿三个**相乘**的维度联合推进：条目大小（GQA/MLA 一路）、序列维（m token → 1 条目）、**层维**（跨层复用）。相对 CSA 的两处简化（§2.3）：① 压缩比 m 时每个条目来自 **m 个不重叠** token（CSA 是 2m 个且相邻条目重叠），并去掉压缩时的绝对位置 embedding（APE）；② **indexer K 由 main KV 条目投影得到**，取代 CSA 从 hidden states 单独压缩一路。

每层静态指派三模式之一；三种模式都保留**自己的 global Q 和自己的 SWA KV**：

- **Full**：自己算 main KV（Compressor）+ indexer Q，从 main KV 投出 indexer K，跑 indexer 产出新 Top-K。职责等同 V4 的一个完整 CSA 层。
- **Reindex**：复用前一层的 main KV **和 indexer K**，只用**自己的 indexer Q** 重新打分、选新 Top-K。选择可逐层变化而缓存保持共享。
- **Reuse**：复用前一层的 main KV 和**针对该 KV 最新算出的 Top-K 索引**，直接做稀疏注意力；无 indexer Q、无打分。

与 CED 组合时：decoder 里的 Full 层从 H_{L/2} 算自己的全局 KV，Reindex/Reuse 不变（§2.3.1 末段）。

实际排布（§4.2.1 + config [P3] 完全吻合）：

| 层（0 基） | 模式 | m | 备注 |
|---|---|---|---|
| 0–1 | 纯 SWA | 0 (ratio=0) | 无 compressor/indexer 权重 [P9] |
| 2–19（encoder 18 层） | 3 组 ×（1 Full + 5 Reuse） | 2 | Full = {2, 8, 14} |
| 20–23（decoder 第 1 组） | Full(20) + 3 Reuse | 1 | 层 20 输入 = H_{L/2} |
| 24–39（decoder 后 4 组） | 4 组 ×（1 Reindex + 3 Reuse） | 1 | Reindex = {24, 28, 32, 36} |
| 40–42 | DSpark 草稿层 | 0 | 纯 SWA，window 128（§2.4.3） |

即 Full=4、Reindex=4、Reuse=30。「绝大多数层是 Reuse Mode，prefill 仅 15 个 kernel、decode 仅 11 个」（§3.2 [P1]）。

### 2.3 Hierarchical Sparse Indexer（§2.3.2 [P1]）

- 只用于 **CED 的 decoder**，只省 decode 时深层 indexer 的**打分算力**（不省 KV 存储）。
- decoder 第一个 Full 层（层 20）对所有因果可见位置打分，除自选 Top-K 外另做 **blockwise candidate selection**：块得分 = 块内最大 index score，选得分最高的 `candidate_topk_blocks=2048` 块 × `candidate_block_size=8` 位置 → **≤16384 个候选位置**组成 candidate pool（§2.3.2、§4.2.1 [P1][P3]）。
- 后续 Reindex 层只在候选池内打分选 Top-K：每个 query 的打分成本从 O(上下文) 变 O(16384) 常数；第一个 Full 层仍全量扫描。
- **training-aware**：post-training 起训练/推理用同一候选域（§2.3.2）。
- 代码细节（[P5] `select_candidate_blocks`）：最新块强制入池（`masked_fill(last, +inf)`），不足 2048 块时丢弃 -inf 溢出项；候选以 bool mask 存 `shared_attn.candidates`，消费层 `masked_fill(~candidates, -inf)` 后再 top-k。

indexer 打分公式（代码 [P5] `Indexer.forward`，与 V3.2/V4 的 lightning indexer 同构）：
`score_t = Σ_h weights_proj(x)_h · ReLU(q_h · k_t)`，q：32 头 × 128 维（FP4），k：共享 indexer K（FP4），`weights_proj` 输出经 `index_head_dim^-0.5 · n_heads^-0.5` 缩放；ReLU 截断负数后按头加权求和；因果掩蔽到「压缩块最后一个 token 可见」粒度。

### 2.4 FP4 main KV 与 890 B/token 的账（§2.4.4、Figure 1b [P1][P8]）

- 格式：**E2M1 + 每 16 通道一个 E4M3 scale**（类 NVFP4 但去掉第二级全局 scale）。上界论证：最大 RMSNorm 权重 ≈1 → 512 维 KV latent 的 L2 ≤ √512 ≈ 22.6，RoPE 保范数，训练实测最大 ~10；E2M1 最大 6 × E4M3 scale 最大 448 = 2688，余量充足。
- RoPE/非 RoPE 部分同一格式；**RoPE 之后再量化**（之前量化精度收益微小且增加 decode 开销）；post-training 引入 QAT；SWA KV 保持 FP8 不动（对量化敏感）。
- indexer Q/K 从 V4 起就是 FP4（QAT），V4.1 把 FP4 扩展到 main KV——目的是**省存储**而非加速 GEMM。
- **890 B/token 精确复算**（用 [P3][P5][P6] 的真实格式）：
  - 每条 main KV 条目：512 × 4 bit = 256 B，+ 512/16 = 32 个 E4M3 scale = 32 B → **288 B**
  - 每条 indexer K 条目：128 × 4 bit = 64 B，+ 128/32 = 4 个 E8M0 scale = 4 B → **68 B**
  - 每个 KV 源层条目合计 **356 B**
  - 条目数/token：encoder 3 个 Full 层 m=2 → 3 × 0.5 = 1.5；decoder 1 个 Full 层 m=1 → 1；共 **2.5 条/token**
  - 356 × 2.5 = **890 B/token**，与论文 Abstract/§6/Figure 1b 完全一致 ✓
- 对比（Figure 1b [P8]）：V4-Flash 3,514（3.9×）、V3.2 48,068（13.7×）、V1 389,120（8.1×；890×437 ≈ 389k ✓「437-fold」）。
- persistent KV（SSD/主机内存）≈ V4 的 1/8：全局 KV ×1/4 × 不再持久化 SWA KV（≈再一半）（§3.2.1）。SWA KV 改放每台机器 10% host DRAM 的分布式内存池（TTL 分钟级），未命中时 Encoder SWA Bounded Replay 只回放 `n_win` 个 token 重建（§3.2.2）。

### 2.5 SWA 与 CSA2 如何并存（代码语义）

每层注意力输出 = 对「本层 SWA 窗口 KV（ring buffer，window=128，FP8）∪ 全局 Top-K 压缩 KV（≤512 条，FP4）」的**一次** `sparse_attn`：两份 KV 沿序列维 cat、两份索引 cat（全局索引加 offset = 窗口 KV 长度），外加每头一个可学 `attn_sink`（[P5] `Attention.forward`、[P6] `sparse_attn`）。全局 Top-K 对当前层不可达的条目记 -1（gather 置零、score 置 -inf）。

## 3. 代码落点

repo：`deepseek-ai/DeepSeek-V4.1-Flash`，文件 `inference/model.py`（下称 model.py，行号为 raw 文件行号）。**注意：参考实现里所有 cache 都是 BF16 buffer，量化是 `inplace` 的「量化→立刻反量化」数值模拟（QAT 语义）；FP4/FP8 是部署存储格式**。类/函数命名直接对应论文概念。

### 3.1 跨层共享的枢纽：`SharedAttentionRuntime`（model.py:1166）

```python
class SharedAttentionRuntime:
    """What attention layers hand down the stack instead of recomputing. ...
    Sources: compress_kv and index_k from kv_source_layers, topk_idxs
    from index_source_layers, candidates from candidate_source_layer."""
    def __init__(self):
        self.compress_kv: torch.Tensor | None = None
        self.index_k: torch.Tensor | None = None
        self.topk_idxs: torch.Tensor | None = None
        self.candidates: torch.Tensor | None = None
shared_attn = SharedAttentionRuntime()
```

三模式在 `Attention.__init__`（model.py:653）落地为两个布尔位：`is_kv_source = layer_id ∈ kv_source_layers`（→ Full：挂 `Compressor`）、`is_index_source = layer_id ∈ index_source_layers`（→ Full/Reindex：挂 `Indexer`）；两者皆非 = Reuse（无任何压缩/索引参数）。层序执行保证「源先写、消费者后读」，无需版本管理。

### 3.2 三种模式的分流：`Attention._compress_kv` / `_compress_topk_idxs`（model.py:722–763）

```python
def _compress_topk_idxs(self, x, qr, latent, start_pos, offset, compress_len):
    if not self.is_index_source:
        return shared_attn.topk_idxs            # Reuse：直接用源层的 Top-K
    ...
    idxs = self.indexer(x, qr, latent, start_pos, offset)   # Full/Reindex：自己打分
    shared_attn.topk_idxs = idxs
    return idxs

def _compress_kv(self, x, qr, start_pos, offset):
    ...
    if self.is_kv_source:                       # Full：自己压 main KV
        latent = self.compressor(x, start_pos)
        shared_attn.compress_kv = self.compress_kv_cache
    # the indexer needs the latent before RoPE, so it runs before the cache is written
    idxs = self._compress_topk_idxs(x, qr, latent, start_pos, offset, compress_len)
    if latent is not None:
        ...
        apply_rotary_emb(latent[..., -self.rope_head_dim:], freqs)
        # Compressed KV uses groups of 16 with E4M3 scales; the indexer uses 32 with E8M0.
        fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
        self.compress_kv_cache[:bsz, start_pos // ratio : ...] = latent
    return shared_attn.compress_kv[:bsz, :compress_len], idxs   # 非源层：读共享 cache
```

### 3.3 压缩算子（简化版 CSA compressor）：`Compressor.forward`（model.py:458）

m=2 的 encoder Full 层：fp32 下 `wkv: 5120→512`、`wgate: 5120→512`，每 2 个 token `softmax(gate) ⊙ kv` 求和 → RMSNorm，返回 **RoPE 前** latent（组 j 占位置 j·m）。m=1（decoder 层 20）退化为 `norm(wkv(x))`，**无 gate**（checkpoint 里层 20 无 `compressor.wgate`，[P9] 佐证）：

```python
if ratio == 1:  # one token per group: nothing to pool, so no gate and no fp32
    return self.norm(self.wkv(x))
...
kv = kv.unflatten(1, (-1, ratio)); score = score.unflatten(1, (-1, ratio))
kv = (kv * score.softmax(dim=2)).sum(dim=2)     # m token 不重叠、无 APE
```

### 3.4 indexer K 复用与两级索引：`Indexer`（model.py:488–580）

```python
self.owns_k = layer_id in args.kv_source_layers   # 只有 Full 层产生 indexer K
...
if self.owns_k and latent is not None:            # indexer K = main KV latent 的投影
    k = self.k_norm(self.wk(latent))              # 512 → 128，bf16
    apply_rotary_emb(k[..., -rd:], freqs)         # compress_rope_theta=160000
    fp4_act_quant(k, fp4_block_size, True)        # FP4/E8M0，block 32
    self.k_cache[...] = k
    shared_attn.index_k = self.k_cache            # 供 Reindex 层共享
q = self.wq_b(qr).unflatten(-1, (n_local_heads, index_head_dim))   # 1280 → 32×128
...
index_score = torch.einsum("bshd,btd->bsht", q, index_k)
index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
...
if self.is_candidate_source:                      # 层 20：建候选池
    shared_attn.candidates = select_candidate_blocks(index_score, ...)
elif self.uses_candidates:                        # 层 24/28/32/36：池内打分
    index_score = index_score.masked_fill(~shared_attn.candidates, -torch.inf)
topk = min(self.index_topk, end_pos // ratio)     # 512
idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
return torch.where(idxs < compress_lens, idxs + offset, -1).int()
```

（V4.1 相对 V4 多了末尾 `.sort(dim=-1)`：Top-K 结果按位置重排。）

### 3.5 checkpoint 权重名佐证三模式（[P9]）

- 层 0（纯 SWA）：只有 `attn.{wq_a,wq_b,wkv,kv_norm,wo_a,wo_b,attn_sink}`，无 compressor/indexer。
- 层 2（Full, m=2）：多 `compressor.{wkv,wgate,norm}` + `indexer.{wq_b,wk,k_norm,weights_proj}`。
- 层 20（Full, m=1）：`compressor.{wkv,norm}`（**无 wgate**）+ 完整 indexer。
- 层 24（Reindex）：只有 `indexer.{wq_b,weights_proj}`——无 `wk`/`k_norm`（K 共享）、无 compressor。
- 层 3（Reuse）：与层 0 完全相同的注意力参数清单（自己的 Q、SWA KV、O），无任何索引/压缩参数。

### 3.6 V4 CSA → V4.1 CSA2 改动对照（[L1] `official-model.py` vs [P5]）

| 维度 | V4 CSA（local `official-model.py`） | V4.1 CSA2（`inference/model.py`） |
|---|---|---|
| 层间关系 | 每个 CSA 层各自压 main KV、各自跑 indexer | 4 Full + 4 Reindex + 30 Reuse 跨层共享（`shared_attn`） |
| 混合架构 | CSA(m=4) 与 HCA(m=128) 交替（V4-Flash config：层 2–41 交替 4/128） | 纯 CSA2，m∈{2(encoder), 1(decoder)} |
| 压缩窗口 | m=4 时每条目来自 2m=8 个**重叠** token + 可学 APE（`self.ape`，`overlap_transform`） | m 个不重叠 token，无 APE |
| indexer K 来源 | indexer 自带一个 Compressor（hidden states → 128 维，Hadamard `rotate_activation` 后 FP4） | `wk` 从 main KV latent 投 512→128，无 Hadamard |
| indexer 头数 | 64（V4-Flash） | 32 |
| main KV 精度 | 非 RoPE 448 维 FP8(block 64) + RoPE 64 维 BF16 | 整 512 维 FP4 E2M1 + E4M3/16（RoPE 后统一量化） |
| SWA KV 精度 | 非 RoPE FP8 + RoPE BF16 | 整 512 维（含 RoPE 尾）FP8(block 32, UE8M0) |
| Q 归一化 | `wq_b` 后逐头 rsqrt（`q *= torch.rsqrt(...)`） | 无 |
| 两级索引 | 无 | candidate pool（层 20 → Reindex 层） |
| cache 布局 | window+compress 同一大 buffer | 独立 `window_kv_cache`（128 槽 ring）+ `compress_kv_cache` |

## 4. 关键数字

| 数字 | 条件/含义 | 出处 |
|---|---|---|
| 552B backbone + 196B Engram | 两数分列，Engram 稀疏查表不计入激活 | §2.1 [P1]，§2.4.2，[P7] |
| 激活 8B / 16B | 每 token：prefill 8B（CED 只跑 encoder 全程）/ decode 16B | Abstract、§2.1、§4.2.1、Table 1 [P1][P7] |
| 40 层 = 20 encoder + 20 decoder；层 0–1 纯 SWA | CED 切分 | §2.1、§4.2.1 [P1]，config `n_layers` [P3] |
| Full {2,8,14,20} / Reindex {24,28,32,36} / Reuse 30 层 | config 字段 `kv_source_layer_ids` / `index_source_layer_ids`；模式排布唯一文字出处 §4.2.1 | [P1][P3][P4]，权重佐证 [P9] |
| m=2（encoder 18 层）/ m=1（decoder 20 层） | `compress_ratios`（43 项 = 40 主干 + 3 DSpark，DSpark 为 0） | [P3][P4] |
| 全局 KV **890 B/token**（HBM 常驻） | 356 B/条（main 288 + indexer K 68）× 2.5 条/token，精确复算一致 | Abstract、§6、Figure 1b [P1][P8]；算式见 §2.4 |
| V4-Flash 3,514 / V3.2 48,068 / V1 389,120 B/token | 3.9× / 13.7× / 8.1×；「≈1/4」「437×」 | Figure 1b 读图 [P8] |
| persistent KV ≈ V4 的 1/8 | 全局 ×1/4 × 不持久化 SWA KV（≈1/2）；SWA 改放 10% host DRAM 池 | §3.2.1 [P1] |
| FP4 = E2M1 + 每 16 通道 E4M3 scale | 无第二级全局 scale；幅度上界 448×6=2688 ≫ 实测 ~10；RoPE 后量化；post-training QAT | §2.4.4 [P1]；代码 `fp4_act_quant(latent,16,…,e4m3fn)` [P5][P6] |
| indexer Q/K：FP4 E2M1 + E8M0 scale/32 | 自 V4 起 QAT | §2.4.4 [P1]；`fp4_act_quant(k, 32, …)` [P5] |
| SWA KV：FP8（整 512 维，含 RoPE 尾，UE8M0/32） | 对量化敏感故不动；每层 ring buffer 128 槽 | §2.4.4 [P1]；`_window_kv` [P5] |
| index_topk=512；index 32 头 × 128 维 | 每个 query 全局读 ≤512 条 | §4.2.1 [P1]；`index_topk` [P3] |
| candidate pool ≤ 2048 块 × 8 = 16384 位置 | 仅 decoder；源层 `candidate_source_layer_id=20` | §2.3.2、§4.2.1 [P1]；[P3][P5] |
| 主注意力：64 头 × 512 维（448 nope + 64 rope）；q_lora 1280；O：8 组 × 1024 LoRA | `num_key_value_heads=1`：K、V 同一 512 维 latent | §4.2.1 [P1]；[P3][P5] |
| SWA window n_win=128，每层都有 | Bounded Replay 只回放 128 token | §2.2、§4.2.1 [P1]；`window_size` [P3] |
| 上下文 1M（1,048,576）；YaRN factor 16、原始 64K | 45T 语料预训练，34T 时扩到 1M；稀疏注意力 64K 从头训 | §1、§4.2 [P1]；[P3][P7] |
| Reuse 层 15 kernels（prefill）/ 11（decode） | kernel 融合后的部署口径 | §3.2 [P1] |
| decode FLOPs 4K→1M 仅 +1/4 | Figure 2 | §1（Figure 2 说明）[P1] |

## 5. 常见误解

1. **「CSA2 是 CSA 的小改版」**。错。CSA2 同时改了三个相乘维度中的「层维」（跨层共享 main KV / indexer K / Top-K，30/40 层为 Reuse），简化了 compressor（去重叠、去 APE）和 indexer（K 从 main KV 投影、去 Hadamard），并且 V4 的 CSA+HCA 混合被**纯 CSA2**取代；再叠加 CED 把 decoder 全局 KV 的来源整个改到 encoder 末层——这是架构级改动（§1、§2.3 [P1]）。
2. **「890 B/token 是全部 KV 开销」**。890 只是**全局 KV**（HBM 常驻、随长度线性增长的部分）。每层还有 SWA ring buffer：128 槽 × 512 维 FP8(528 B) × 40 层 ≈ 2.7 MB/序列，与长度无关所以不计入「per token」（§3.2.1 [P1]、`_window_kv` [P5]）。
3. **「CED 是 encoder-decoder 交叉注意力/encoder 双向」**。不是。全栈都是因果的、单向的；「encoder/decoder」只描述**全局 KV 由谁产生**，没有 cross-attention 模块；SWA 仍然 40 层逐层自产（§2.2 [P1]）。
4. **「Reuse 层 ≈ 没有注意力」**。Reuse 层有自己的 Q（wq_a/wq_b）、自己的 SWA KV（wkv/kv_norm）、自己的 O 投影和 attn_sink（[P9] 权重清单）；它复用的只是 main KV、indexer K 和 Top-K 索引（§2.3.1 [P1]）。
5. **「index 复用 = Top-K 复用」**。两者**解耦**：Reindex 复用 KV+indexer K 但重选 Top-K；Reuse 才连 Top-K 一起复用（§2.3.1「cache sharing and index reuse decoupled」[P1]）。
6. **「FP4 用于所有 cache」**。只有全局 main KV 新上 FP4（E2M1+E4M3/16）；indexer Q/K 从 V4 起已是 FP4（E8M0/32）；SWA KV 保持 FP8（§2.4.4 [P1]；[P5][P6]）。
7. **「prefill 8B / decode 16B 是两个模型或两种权重」**。同一份权重；prefill 只对全序列跑 encoder（decoder 全局 KV 由 H_{L/2} 投影、SWA 只回放 128 token），decode 才过全部 40 层（§2.2、§3.2.2 [P1]）。
8. **「Hierarchical Sparse Indexer 省存储」**。它省的是 decode 时深层 indexer 的**打分算力**（每 query 从 O(N) 变 O(16384) 常数），不省一字节 KV；且只存在于 decoder（§2.3.2 [P1]）。
9. **「参考实现里 cache 真的按 FP4 存」**。`inference/model.py` 的 cache 是 BF16 buffer + inplace 量化-反量化模拟（QAT 语义）；FP4 存储格式见部署侧与 checkpoint 的 expert FP4 存储（[P5][P6] kernel 注释）。

## 6. 与本专栏其他文章的接口

已有文章（`src/content/posts/`）可直接引用、**不要重写**的内容：

- `deepseek-v4-01-kv-compression.mdx`（序列维度·上）：K=V 同一 latent 向量、压缩算子从平均池化的推导、V4 的 overlap/2m/APE 细节、分组输出投影（o_groups）。→ 本文只写 **delta**：去 overlap/APE、m∈{1,2}、FP4。
- `deepseek-v4-02-lightning-indexer.mdx`（序列维度·中）：lightning indexer 打分公式（ReLU + weights_proj）、FP4 QAT、Hadamard 旋转的来源、V4 indexer 自带压缩路径。→ 本文只需引用打分公式；改动是 K 侧来自 main KV 投影、32 头、去 Hadamard、新增 candidate pool 第二级。
- `deepseek-v4-03-hybrid-attention.mdx`（序列维度·下）：HCA、CSA/HCA 排布、SWA 必须存在的论证、部分 RoPE 与**输出反旋转**、attn sink、1M 的账。→ 部分 RoPE/反旋转/sink/SWA 并存机制在 V4.1 完全沿用（代码同构），直接引用；HCA 已删除这一点指向本文。
- `deepseek-v4-04-mhc.mdx`：mHC 基础。→ 本文只需一句 delta：Single-Pass mHC 把输入混合系数移位一拍（`A_{l-1}`，式 (6)，§2.4.1），推理融合为 Mega-mHC，激活访存 (4n+4)d → (2n+2)d。
- `deepseek-v4-07-systems.mdx`：V4 的持久化混合策略、Zero SWA Caching。→ 本文的 SWA Bounded Replay / 1/8 persistent KV 是它的直接续集，引用其对 V4 策略的描述。
- `deepseek-v4-00-overview.mdx`：V4 总览图与「全局压缩 + 局部 SWA」框架。→ CED 可表述为该框架下「全局分支再砍一半深度」。

新文章建议定位为「V4.1 = 层维压缩 + CED + FP4」，与 01/02/03 构成「条目维/序列维/层维」三维度完结篇。

## 7. 图的建议

### 图 A（必须）：CSA2 三模式小算子级张量流图

建议三列并排（Full / Reindex / Reuse），共用左侧输入 `x[B,N,5120]`（mHC 塌缩+norm 后）与共享池。节点与边清单：

**公共节点（每列都有，标注「每层自有」）**：
1. `wq_a: 5120→1280` → `q_norm` → `qr[B,N,1280]`
2. `wq_b: 1280→64×512` → `RoPE(末64维)` → `q[B,N,64,512]`
3. `wkv: 5120→512` → `kv_norm` → `RoPE(末64)` → `FP8(block32)` → `window_kv_cache[B,128,512]`（ring）＋窗口索引 `[B,N,128]`
4. `sparse_attn(q, cat(window_kv, compress_kv), attn_sink[64], cat(128窗, 512全局))` → `o[B,N,64,512]` → `逆RoPE(末64)` → `wo_a 分组einsum(8组, 4096→1024)` → `wo_b: 8192→5120`

**Full 列（层 2/8/14，m=2；层 20，m=1 无 wgate）独有节点**：
5. `compressor.wkv/wgate: 5120→512 (fp32)` → `softmax(gate)⊙kv 按 m=2 求和` → `norm` → `latent[B,N/2,512]`（RoPE 前）
6. `indexer.wk: 512→128` → `k_norm` → `RoPE(θ=160000)` → `FP4/E8M0/32` → `index_k_cache[B,N/2,128]` ──▶ **写共享池 `index_k`**
7. `indexer.wq_b: qr 1280→32×128` → RoPE → FP4；`weights_proj: 5120→32`
8. `score=Σ_h w·ReLU(q·k)` `[B,N,N/2]` → 因果掩码 →（层 20 另有 `select_candidate_blocks`：块8取max、pin末块、top 2048块 → `candidates` bool 池）→ `topk 512` → 按位置 sort → +offset ──▶ **写共享池 `topk_idxs`**
9. `latent` → `RoPE(末64)` → `FP4/E4M3/16` → `compress_kv_cache[B,N/2,512]` ──▶ **写共享池 `compress_kv`**

**Reindex 列（层 24/28/32/36，m=1）**：画节点 7′（自己的 indexer Q + weights_proj），边从共享池 `index_k`（层 20 产）流入打分节点；打分后先经 `candidates` 掩码再 topk 512；写共享池 `topk_idxs`（覆盖）；无节点 5/6/9。

**Reuse 列（30 层）**：无任何 indexer/compressor 节点；两条粗边从共享池 `compress_kv`、`topk_idxs` 直接进入节点 4。图上用颜色区分「每层自有」（Q/SWA/O）vs「跨层共享」（KV/indexer K/top-k/candidates）。

### 图 B（建议）：CED 层排布 + 全局 KV 流向图

纵轴 40 层（0 下 39 上），标出：层 0–1 纯 SWA；encoder 层 2–19（m=2，Full{2,8,14} 三组）；分界线 L/2；decoder 层 20–39（m=1，Full{20}、Reindex{24,28,32,36}、Reuse）。画一条粗箭头：encoder 末层隐藏态 `H_20` → 层 20 的 Compressor（式 (1)），再从层 20 的 `compress_kv/index_k` 向层 21–39 发散（Reindex 层另有 `candidates` 从层 20 出发的虚线）。旁边标注 prefill 时 decoder 只对最后 128 token 跑（SWA Bounded Replay）。

## 8. 待确认/没查到的点

1. **V4-Flash 的 3,514 B/token 组成**：Figure 1b 只给总数；按 V4-Flash config（20 CSA(m=4)+20 HCA(m=128)，main KV FP8 混合精度）估算约 3.3–3.5 KB，量级吻合但分项账未在 V4.1 论文给出，若文章要精确对比需回 V4 论文核对。
2. **890 B 是否含任何 per-sequence 元数据**：本稿按纯数据+scale 复算恰为 890，论文未给分项表；不排除官方另有取整口径。
3. **Engram 196B 与 552B 是否互相包含**：论文 §2.1 表述「552B backbone parameters and 196B Engram parameters」为并列，README 同； activated 8B/16B 显然不含 Engram 查表，但 552B 是否含 Engram 的非表部分未明说。
4. **transformers 内置实现**：repo 无 `modeling_*.py`（`transformers_version: 5.6.0`，`DeepseekV41ForCausalLM`），HF transformers 里的实现本稿未读；若要在文章里给「HF 版」代码引用需另查 transformers 源码。
5. **论文 Figure 3/4/5 的图内细节**（三模式图示、层级索引示意）只有正文描述，HTML 版图片未逐一读图核对；图 A/B 设计基于正文+代码，可能与官方图示布局不同。
6. **`o_lora_rank=1024` / `o_groups=8` 在 V4 是否相同**：V4.1 代码明确（wo_a 分组 LoRA）；与 V4 的异同未逐项核对（本地 `official-model.py` 结构相同，但 rank 值以 V4 config 为准再写进对比表更稳）。
