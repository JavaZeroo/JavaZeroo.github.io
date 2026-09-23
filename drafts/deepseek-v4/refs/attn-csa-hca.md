# DeepSeek-V4 CSA / HCA 小算子级数据流整理（注意力对比文素材）

## 0. 一句话

CSA 与 HCA 共用同一副骨架（低秩 Q、K=V 单头 MQA、128 滑窗、attention sink、输出反旋转、分组输出投影），差别只在压缩率与检索方式：CSA 每 m=4 token 重叠压一条 512 维条目、靠 lightning indexer 选 top-1024 块；HCA 每 m′=128 token 不重叠压一条、不要 indexer、全部条目都看。

## 1. 一手资料清单（文件路径 + 读了哪些部分）

- `drafts/deepseek-v4/refs/v4-paper.txt` — V4 技术报告 arXiv HTML 全文（5078 行）。读了 §2.3 全部（L598-832，公式 Eq.9-27）、§3.4.3 CP（L1137-1152）、§3.5 KV cache（L1172-1264）、§4.2.1 模型配置（L1286-1320）、§2.3.4 效率讨论（L821-832）。注意：arXiv HTML 抽取把分组输出投影的公式截断了（L714、L778 处被图 4 打断），该处维度以代码为准。
- `drafts/deepseek-v4/refs/official-model.py` — 官方 inference/model.py（827 行）。全文读，重点 `Compressor` L279-377、`Indexer` L380-433、`Attention` L436-543、`MTPBlock` L738-766。
- `drafts/deepseek-v4/refs/official-kernel.py` — 官方 TileLang kernel（536 行）。重点 `sparse_attn_kernel` L277-352（sink 在 L345-346）、`act_quant` L40-126、`fp4_act_quant` L128-200。
- `drafts/deepseek-v4/refs/modeling_deepseek_v4.py` — transformers 风格实现（1504 行）。全文读，重点 `DeepseekV4Attention` L746-864、`DeepseekV4CSACompressor` L580-693、`DeepseekV4HCACompressor` L353-434、`DeepseekV4Indexer`/`Scorer` L437-577、`DeepseekV4HCACache`/`CSACache` L162-291、`eager_attention_forward` L708-736。
- `drafts/deepseek-v4/refs/configuration_deepseek_v4.py` — HF config（324 行）。`compress_ratios`→`layer_types` 映射 L28-32、默认排布 L270-275、双 rope（main/compress）与 YaRN `attention_factor=1.0` L294-321。
- `drafts/deepseek-v4/refs/v4-pro-config.json` / `v4-flash-config.json` / `official-config.json` — 全部字段。Pro 的 `compress_ratios` 有 62 项（61 层 + MTP 位的 0）。
- `drafts/deepseek-v4/refs/notes-v4.md` — 既有精读笔记（数字账已抽查核对，一致）。
- `src/content/posts/deepseek-v4-01-kv-compression.mdx`（169 行）、`-02-lightning-indexer.mdx`（148 行）、`-03-hybrid-attention.mdx`（130 行）— 全文读，用于第 6 节接口划分。

## 2. 核心机制与推导

### 2.1 CSA 与 HCA 共用的骨架（Pro 数字，以 `v4-pro-config.json` 为准）

每层输入 `hidden_states` x: [b, n, 7168]（mHC collapse 后的单流）。数据流 9 步，括号内为张量形状与论文公式号：

1. **Q 低秩路径**：`wq_a` 7168→1536 得 c^Q（Eq.13/24，`q_lora_rank=1536`）→ 带权重 RMSNorm(1536) → `wq_b` 1536→65536 → [b,n,128,512]（128 头 × 512 维）→ **逐头无权重 RMSNorm**（`q *= rsqrt(mean(q²)+eps)`，official-model.py L498；§2.3.3 论文只说了「加 RMSNorm」没说无权重）→ 最后 64 维 RoPE（压缩层 θ=160000 + YaRN factor 16 / 原始 64K）。
2. **滑窗 KV 路径**：`wkv` 7168→512（**K=V 同一条**，`num_key_value_heads=1`）→ 带权重 RMSNorm(512) → 最后 64 维按自身位置 RoPE → 前 448 维按 64 维一组模拟 FP8（official-model.py L502-506；§2.3.4）→ 写 128 条环形 cache。
3. **压缩算子**：见 2.2 / 2.4。
4. **检索**：CSA 有 indexer（2.3）；HCA 规则生成全部可见块索引。
5. **索引拼接**：topk_idxs = [128 个滑窗位置; 压缩条目索引 + offset]，一次 `sparse_attn` 算完——**滑窗与压缩条目在同一个 softmax 里**，不是两个分支事后合并（official-model.py L507-515、L528）。
6. **核心注意力（K=V 的 MQA，Eq.19/26）**：q [b,n,128,512] 对 gather 出来的条目（CSA 最多 1152 条；HCA 在 1M 时 8320 条）做注意力，scale = 512^−0.5；**分母加 exp(sink_h − max)**（Eq.27；official-kernel.py L345-346）。K=V ⇒ 同一条 512 维条目既做内积也做加权和。
7. **输出反旋转**：K=V 使 V 带上了绝对位置 R_j，输出最后 64 维乘 R_{−t}（共轭）变回相对位置（§2.3.3；official-model.py L534 `inverse=True`；modeling L859 用 `-sin`）。
8. **分组输出投影**：[b,n,128,512] → reshape [b,n,16,4096] → `wo_a` 每组 4096→1024（einsum `bsgd,grd->bsgr`，official-model.py L537-541）→ flatten 16384 → `wo_b` 16384→7168（Eq. 见 §2.3.1「Grouped Output Projection」段，公式被 HTML 抽取截断；`o_groups=16`、`o_lora_rank=1024`）。
9. 回到 mHC（不在本文范围）。

### 2.2 CSA 压缩算子（Eq.9-12；official-model.py `Compressor`，overlap 分支）

m=4，重叠。对 x（fp32）：

- C^a, C^b = H·W^{aKV}, H·W^{bKV}；Z^a, Z^b = H·W^{aZ}, H·W^{bZ}（Eq.9-10）。代码里一个矩阵出两系列：`wkv`/`wgate` 各 7168→**1024**，前 512 维是「a 系列」、后 512 维是「b 系列」。
- 加块内位置偏置 `ape` [4, 1024]（即论文 B^a, B^b ∈ R^{m×c} 拼接）。
- 重叠布局（`overlap_transform` L307-314）：条目 i 的 2m=8 个槽 = 上一块的 a 系列（前 4 槽）+ 本块的 b 系列（后 4 槽）；i=0 时 a 槽 kv=0、score=−inf（Eq.11 注）。
- **逐通道 softmax**（Eq.11-12）：[b, n/4, 8, 512] 在 dim=2 上做 softmax——512 个通道各自独立一套 8 槽权重，再加权求和得 [b, n/4, 512]。宽 2m=8、步长 m=4，条目数仍是 n/4。
- 后处理：带权重 RMSNorm(512) → **块首位置**的 RoPE（`freqs_cis[::ratio]`，L364）→ 非 rope 448 维 FP8（64 维一组）→ 写压缩 cache。
- decode 增量路径（L343-359）：`kv_state`/`score_state` 缓冲 8 槽，每 4 个 token 输出一条。

命名陷阱：代码的「a 半段」喂给**下一个**条目（对应论文的 C^b 角色），「b 半段」喂当前条目（对应论文 C^a）。论文公式与代码注释的 a/b 是反的，对比文里建议只用「当前块半段 / 前一块半段」表述。

### 2.3 Lightning indexer（仅 CSA；Eq.13-17；official-model.py `Indexer` L380-433）

- key：`Indexer` 内置一个**自己的** Compressor（ratio=4、头维 c^I=128，同样重叠 8 槽、自己的 ape [4,256]），产出 K^IComp [b, n/4, 128]；多一步 Hadamard 旋转 + FP4（L368-370、L414-416，QAT 模拟）。
- query：**与主注意力共享 c^Q**（1536 维 latent，Eq.13 注），`wq_b` 1536→8192 → [b,n,64,128]（64 头），RoPE（同压缩层 θ）+ Hadamard + FP4；**没有**逐头 RMSNorm（主 q 有，这是代码里才看得出的不对称）。
- 打分（Eq.15-16）：w^I = `weights_proj`(x) 7168→64，乘 128^−0.5·64^−0.5 缩放；I_{t,s} = Σ_h w^I_{t,h}·ReLU(q^I_{t,h}·K^IComp_s)，[b,n,n/4]。
- 因果 mask 后 top-1024（`index_topk`，Flash 512）；不足 k 或越界的条目填 −1 哨兵，kernel 里 −1 → kv 取 0、score 置 −inf（official-kernel.py L322-327）。

### 2.4 HCA 压缩算子（Eq.20-23；`Compressor` 非重叠分支 / modeling `DeepseekV4HCACompressor`）

与 CSA 同一类、同一组公式形，三处不同：

- m′=128，**不重叠**（`overlap = (compress_ratio == 4)` 为 False）：`wkv`/`wgate` 各 7168→**512**（单系列），ape [128, 512]，softmax 在 128 个槽上逐通道做（Eq.22-23）。
- **无 indexer**：可见集合 = 全部已闭合条目，索引由规则生成（official-model.py `get_compress_topk_idxs` L268-276；modeling 里只加因果 `block_bias` L426-434）。
- 其余完全相同：RMSNorm → 块首 RoPE → FP8 → cache；核心注意力、sink、反旋转、分组输出与 CSA 共用代码路径。

### 2.5 CSA vs HCA 差异速查

| 维度 | CSA | HCA |
|---|---|---|
| 压缩率 | m=4 | m′=128（=32×） |
| 重叠 | 是，感受野 2m=8，步长 4 | 否 |
| 压缩投影 | 7168→2×512（kv+gate 各一） | 7168→512 |
| 检索 | lightning indexer，top-1024 块 | 无，全看 |
| 每 query 条目数 | ≤1024 + 128 滑窗 | n/128 + 128（1M 时 8320） |
| 层数（Pro 61 层） | 30（偶数 idx 2..60） | 31（idx 0,1 + 奇数 3..59） |
| 额外 cache | indexer 键 n/4 条 × 128 维 | 无 |
| 每 token 每层注意力算术量 | ≈151 MFLOP（与 n 无关） | 1M 时 ≈1.09 GFLOP（随 n 涨） |

出处：`v4-pro-config.json` 的 `compress_ratios`；论文 §4.2.1（L1305-1314）；§2.3.1/§2.3.2。

## 3. 代码落点（文件/类名，摘录 ≤30 行/处）

### 3.1 官方实现：CSA/HCA 判定与检索分叉（official-model.py L466-471、L507-515）

```python
if self.compress_ratio:
    self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
    if self.compress_ratio == 4:
        self.indexer = Indexer(args, self.compress_ratio)
    else:
        self.indexer = None
...
topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)   # 128 个滑窗位置
if self.compress_ratio:
    offset = kv.size(1) if start_pos == 0 else win
    if self.indexer is not None:                                # CSA
        compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
    else:                                                       # HCA：规则生成全部可见块
        compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)
    topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
```

### 3.2 官方 Compressor：重叠池化核心（official-model.py L337-342、L362-367）

```python
kv = kv.unflatten(1, (-1, ratio))                    # [b, n/4, 4, 1024]
score = score.unflatten(1, (-1, ratio)) + self.ape   # + 块内位置偏置 B
if overlap:                                          # ratio==4 → CSA
    kv = self.overlap_transform(kv, 0)               # [b, n/4, 8, 512]：后4槽本块b半段，前4槽上一块a半段
    score = self.overlap_transform(score, float("-inf"))
kv = (kv * score.softmax(dim=2)).sum(dim=2)          # 逐通道 softmax，8 槽加权 → [b, n/4, 512]
...
kv = self.norm(kv.to(dtype))                         # RMSNorm(512)
freqs_cis = self.freqs_cis[:cutoff:ratio]            # 块首位置
apply_rotary_emb(kv[..., -rd:], freqs_cis)           # 条目级 RoPE
```

### 3.3 官方 Indexer：打分 + top-k（official-model.py L411-433 节选）

```python
q = self.wq_b(qr)                                    # 共享的 c^Q(1536) → 64×128
q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
apply_rotary_emb(q[..., -rd:], freqs_cis)
q = rotate_activation(q)                             # Hadamard
fp4_act_quant(q, fp4_block_size, True)               # FP4 QAT 模拟
self.compressor(x, start_pos)                        # 自己的压缩算子 → K^IComp
weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
...
topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
```

### 3.4 kernel：sink 在分母、−1 哨兵（official-kernel.py L322-327、L345-348）

```python
idxs[i] = T.if_then_else(t * block + i < topk, topk_idxs[by, bx, t * block + i], -1)
kv_shared[i, j] = T.if_then_else(idxs[i] != -1, kv[by, idxs[i], j], 0)
acc_s[i, j] = T.if_then_else(idxs[j] != -1, 0, -T.infinity(FP32))
...
for i in T.Parallel(h):
    sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])   # sink 只进分母
for i, j in T.Parallel(h, d):
    acc_o[i, j] /= sum_exp[i]
```

### 3.5 transformers：输出反旋转 + 分组投影（modeling_deepseek_v4.py L859-863）

```python
attn_output = apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)  # R_{-t}
grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)   # [b,n,16,4096]
grouped = self.o_a_proj(grouped).flatten(2)                             # 每组 4096→1024 → 16384
output = self.o_b_proj(grouped)                                         # 16384→7168
```

### 3.6 transformers：HCA 的无 indexer 因果 mask（modeling_deepseek_v4.py L426-434）

```python
# query `t` may only see cache entries at pos `w` t > w * compress_rate
entry_indices = torch.arange(compressed_len, device=compressed_kv.device)
causal_threshold = (position_ids + 1) // self.compress_rate  # [B, S]
block_bias = compressed_kv.new_zeros((batch, 1, seq_len, compressed_len))
block_bias = block_bias.masked_fill(
    entry_indices.view(1, 1, 1, -1) >= causal_threshold.unsqueeze(1).unsqueeze(-1),
    float("-inf"),
)
```

## 4. 关键数字

| 数字 | 条件 | 出处 |
|---|---|---|
| d=7168，61 层，n_h=128，c（头维）=512 | Pro | `v4-pro-config.json`：`hidden_size`/`num_hidden_layers`/`num_attention_heads`/`head_dim` |
| q_lora_rank d_c=1536；o_groups g=16，o_lora_rank d_g=1024 | Pro（Flash：1024 / 8 / 1024） | 同上；论文 §4.2.1 L1312-1313 |
| m=4（CSA），m′=128（HCA），n_win=128 | 两个模型相同 | `compress_ratios`/`sliding_window`；论文 §4.2.1 L1293-1294、L1310-1311 |
| indexer：64 头 × 128 维，top-1024 | Pro；Flash top-512 | `index_*` 字段；论文 L1310（注意 L1293 Flash=512） |
| 层排布：HCA 31 层（idx 0,1+奇数）、CSA 30 层（偶数 2..60）、MTP ratio=0 纯滑窗 | Pro 61 层 + MTP | `v4-pro-config.json` `compress_ratios`（62 项）；论文 L1308-1309；MTP 一项论文未写 |
| 每条 KV 条目 512 维，K=V 共享 | 所有注意力层 | `num_key_value_heads=1`；论文 Eq.19/26；official-model.py L460 |
| 每条目存储 576 B = 64 维 rope×BF16(128 B) + 448 维×FP8(448 B) | 加 scale：448/64=7 组 ×1 B (ue8m0) = 583 B | 论文 §2.3.4 L824；official-model.py L372/L506（`act_quant(…, 64, …)`）；`official-config.json` `scale_fmt=ue8m0` |
| 条目数 @1M：CSA 2^18=262144 条/层；HCA 2^13=8192 条/层 | 差 32× = m′/m | 由 m、m′ 算出；论文 L652、L753 |
| Pro KV cache @1M ≈ 5.3 GiB ≈ 5.4 KB/token | CSA 4.22 GiB + indexer 0.94 GiB(FP8 计) + HCA 0.136 GiB + 滑窗 4.3 MiB | 按上表手算（与 notes-v4.md L40 一致）；论文摘要给「V3.2 的 10%」（L282） |
| 每 token 每层摊到：CSA 144 B(+indexer 32 B)，HCA 4.5 B | 576/4 与 576/128 | 同上 |
| 基线 GQA-8 BF16 hd128×61 层 = 244 KiB/token → 1M 时 244 GiB | V4 ≈ 2% | 论文 §2.3.4 L830 |
| 核心注意力算术量：CSA ≈151 MFLOP/token/层（2·128·1152·512）；HCA @1M ≈1.09 GFLOP（2·128·8320·512）；indexer ≈4.3 GFLOP/token/层（FP4） | Pro | 由 config 数字算出；同 deepseek-v4-02/03 的账 |
| rope：压缩层 θ=160000+YaRN(factor 16, 原始 64K, β 32/1)；纯滑窗层 θ=10000 无 YaRN；YaRN attention_factor 强制 1.0（不乘 mscale） | 全部 | `compress_rope_theta`/`rope_scaling`；official-model.py L473-482；configuration_deepseek_v4.py L296-300（仅代码） |

## 5. 常见误解

1. **「CSA 的 K 和 V 是分开压的两条」**——错。每层每 4 个 token 只产出**一条** 512 维向量，同时当 K 和 V（Eq.19，`num_key_value_heads=1`，official-model.py `wkv` 只出 512 维）。「两条」的错觉来自压缩算子的 a/b 双系列——那是同一个 token 对相邻两个条目的两份贡献，压完仍是一条。
2. **「重叠压缩产生双倍条目」**——错。条目数仍是 n/m；每个条目感受野 2m=8、步长 m=4（Eq.12 注 L650-652）。
3. **「压缩就是 4 个 token 平均池化」**——错。权重是逐通道（512 套）softmax(Z+B) 学出来的，B 是可学习块内位置偏置；且全程 FP32（official-model.py L322、L294-298）。
4. **「indexer 选的是 token」**——错。选的是压缩块：top-1024 块 × m=4 = 覆盖 4096 个原始 token；V3.2 的 top-2048 才是 token。
5. **「HCA 也有 indexer，只是 k 更大」**——错。HCA 没有任何打分器，可见集合由规则生成（全部已闭合条目），modeling 里只有因果 `block_bias`。
6. **「滑窗分支是独立注意力、输出再合并」**——错。128 个滑窗索引和压缩块索引拼成一个数组，进**同一个** softmax（official-model.py L514、L528）。
7. **「RoPE 加在压缩前的 token 上」**——错。C^a/C^b/C 不带位置；压缩、RMSNorm 之后在**条目级**加一次，位置取块首 token 的绝对位置（`freqs_cis[::ratio]`；modeling L667-671）。
8. **「反旋转是可选的数值技巧」**——错。K=V 使 V 携带绝对位置 R_j，不乘 R_{−t} 输出就和绝对位置绑定；这是结构必需的（§2.3.3 L793-797）。
9. **「attention sink 是一个可学习的 key/value 条目」**——错。只是每头一个标量 logit z′_h 加在 softmax **分母**（Eq.27；kernel L345-346），等价于 value 为 0 的虚拟条目。
10. **「`compress_ratios` 有 61 项」**——Pro 的 config 有 **62** 项：第 62 项（0）属于 MTP 层，即 MTP 是纯滑窗注意力（official-model.py L789-791 `MTPBlock(args.n_layers + layer_id)` 取到该项）。论文完全没写 MTP 的注意力类型。
11. **「Flash 与 Pro 排布相同」**——错。Pro 前两层是 HCA，Flash 前两层 ratio=0 纯滑窗（对比两个 config 首两项；论文 §4.2.1 L1291、L1308）。
12. **「代码 a/b 半段对应论文 C^a/C^b」**——正好相反：代码前 512 维（"a"）进下一条目=论文 C^b 的角色。写文章时避免直接对字母。

## 6. 与本专栏其他文章的接口

已发布三篇讲得很细，对比文**不要重复展开**，按下表引用：

- **01（kv-compression）已讲**：MHA→MQA→GQA→MLA→NSA→DSA→CSA/HCA 谱系；K=V 单向量的论证；压缩算子从平均池化的四步推导（含 Eq.9-12/20-23 公式与官方代码摘录）；分组输出投影（含「论文公式被 HTML 截断」的提醒）。→ 对比文只需引用「压缩算子逐通道 softmax + 重叠」结论，直接链 01。
- **02（lightning-indexer）已讲**：DSA 复习、indexer 训练（KL、detach）、V4 块级改动、Hadamard+FP4、缩放系数、一层 CSA 完整维度流水、参数账（CSA 331M / HCA 307M 每层）。→ 对比文把 indexer 当黑盒「打块级分数 → top-1024」，细节链 02。
- **03（hybrid-attention）已讲**：HCA「粗读 vs 精读」动机、层排布（含 MTP 纯滑窗）、滑窗分支必要性、Q/KV RMSNorm 与 Muon 不用 QK-Clip 的关系、部分 RoPE + 反旋转的完整推导、sink 分母形式、1M KV/FLOPs 账。→ 对比文引用反旋转与 sink 的结论即可。
- **对比文应新增的价值**（本稿素材支撑）：① CSA/HCA 并排的小算子级张量流图（第 7 节）；② 2.5 差异速查表 + 第 4 节带出处数字表；③ 第 5 节误解清单（尤其 K=V 一条、a/b 字母反转、MTP ratio=0、Pro/Flash 前两层不同）；④ 论文 vs 代码的边界差异（第 8 节因果 off-by-one）。

## 7. 图的建议

### 图 A：CSA 单层小算子张量流图（Pro 数字）

横向主流水，节点（形状标在边上）：

- 输入 `h [n, 7168]`
- 分支 1（Q）：`wq_a` → `c^Q [n,1536]` → `RMSNorm_w` → 分两叉：
  - `wq_b` → `q [n,128,512]` → `逐头RMSNorm(无权重)` → `RoPE_后64维(θ=160k,YaRN)`
  - （虚线进 indexer 框）`idx wq_b` → `q^I [n,64,128]` → `RoPE` → `Hadamard` → `FP4`
- 分支 2（滑窗 KV）：`wkv` → `kv [n,512]` → `RMSNorm_w` → `RoPE_后64维` → `FP8(448维,组64)` → 「滑窗 cache（环形 128 条）」
- 分支 3（压缩）：`wkv_c/wgate_c` → `C,Z [n,1024]=[Ca|Cb]` → `+B[4,1024]` → `重叠布局 [n/4,8,512]` → `逐通道softmax(8槽,fp32)` → `Σ` → `[n/4,512]` → `RMSNorm_w` → `RoPE(块首位置)` → `FP8` → 「压缩 cache [n/4,512]」
- 分支 4（indexer 框）：`idx 压缩算子(128维版)` → `K^IComp [n/4,128]`（Hadamard+FP4）；与 `q^I`、`w=weights_proj(h)[n,64]×128^-0.5·64^-0.5` 进 `打分 Σ_h w·ReLU(q·k)` → `I [n,n/4]` → `因果mask` → `top-1024`
- 汇合：`索引拼接 [128 滑窗 ; ≤1024 块]` → `sparse_attn(K=V, scale=512^-0.5, sink 在分母)` → `o [n,128,512]` → `反旋转 R_{-t}(后64维)` → `分组 wo_a 16×(4096→1024)` → `[n,16384]` → `wo_b` → `y [n,7168]`

边要点：c^Q → indexer q^I 画共享标记；压缩 cache → sparse_attn 的边标注「gather by topk」；滑窗 cache → sparse_attn 标注「同 softmax」。

### 图 B：HCA 单层小算子张量流图（与图 A 同版式，差异高亮）

结构与图 A 相同，四处差异用红色/删除线：

- 分支 3：`wkv_c/wgate_c` 输出 `[n,512]`（单系列，**无 Ca|Cb**）→ `+B[128,512]` → `softmax(128槽)` → `[n/128,512]`；标「m′=128，不重叠」。
- 分支 4 整个 indexer 框打叉：「无 indexer」。
- 汇合处：索引 = `规则生成的全部可见块 [n/128]` + 128 滑窗；1M 时 8192+128=8320 条，标「dense，全看」。
- 条目数对比边注：同一 n 下 CSA cache n/4 条 vs HCA cache n/128 条（32×）。

两图建议画成上下对齐的两栏，共用骨架部分用灰色复用，差异部分上色——读者一眼看到「同构 + 四处差异」。

## 8. 待确认/没查到的点

1. **论文 Eq.16 的因果边界与代码差一个块**：论文写 s < ⌊t/m⌋；代码（official-model.py L425、modeling L569）用 (t+1)//m——当 query 恰好是自己所在块的最后一个 token 时，代码允许看到**包含自己的刚闭合块**，论文公式不允许。03 篇按论文表述写的，对比文若提边界需注明以代码为准。
2. **indexer 键的生产存储精度不确定**：论文 §2.3.4 只说 indexer 内**计算**用 FP4；官方推理代码 cache 是 BF16（QAT 模拟，`Indexer` L399 buffer 为 bf16，L419 注释「could also use fp8」）。按 FP4(68 B)/FP8(128 B)/BF16(256 B) 每条，1M 总量在 4.7–5.9 GiB 间浮动；本稿第 4 节按 FP8 计 5.3 GiB（与 notes-v4.md、03 篇一致）。
3. **分组输出投影公式缺失**：arXiv HTML 在 §2.3.1/§2.3.2 两处都被图截断（L714、L778），g、d_g 的公式形式未能从论文核对，维度以代码（`DeepseekV4GroupedLinear` L294-323、official-model.py L537-542）为准。01 篇已注明这一点。
4. **CSA/HCA 无消融**：论文未解释为何 1:1 交错、为何 Pro 前两层用 HCA 而 Flash 用纯滑窗。03 篇已点明，对比文不要再追问出答案。
5. **wo_a 的生产精度**：official-model.py L539-540 注释说 wo_a 在 checkpoint 里是 FP8、可做 FP8 einsum，但参考实现用 BF16；实际部署精度未确认。
6. **Flash 的 1M 总账（≈3.7 GiB，notes-v4.md L42）未逐项重算**；本文只验算了 Pro。
7. **HF 滑窗 cache 存 127 条**（`HCACache.update` L205 `-sliding_window+1`，当前 token 由 `full` 补上看满 128）与官方环形 128 条的实现差异只是实现划分，语义一致；若写 cache 布局段落可提一句。
