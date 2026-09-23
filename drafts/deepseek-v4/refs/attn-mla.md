# DeepSeek-V2/V3 MLA（Multi-head Latent Attention）一手调研

## 0. 一句话

MLA 把每个 token 的 K/V 压成一个 512 维 latent 向量（外加 64 维带 RoPE 的共享 key）缓存，推理时可以把上投影矩阵「吸收」进 query 和输出投影、直接在 latent 空间做注意力（MQA 形态）；但 HuggingFace 官方 repo 自带的 modeling 代码**没有实现吸收路径**，只有先展开成 128 个头的完整 K/V 再缓存的 naive 写法——两条路径分别对应 GitHub 官方推理代码里的 `attn_impl="absorb"`（默认）和 `"naive"`。

## 1. 一手资料清单

| # | 资料 | 读了哪些部分 |
|---|---|---|
| S1 | [DeepSeek-V3 HF repo modeling_deepseek.py](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/modeling_deepseek.py) | 全文读完。重点：`DeepseekV3Attention.__init__/forward`（行 629–858）、`DeepseekV3FlashAttention2.forward`（行 877–1010）、`DeepseekV3YarnRotaryEmbedding._set_cos_sin_cache`（行 287–329）、`yarn_get_mscale`（行 249–252） |
| S2 | [DeepSeek-V3 HF repo config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json) | 全文：61 层、128 头、q_lora_rank 1536、kv_lora_rank 512、qk_nope 128、qk_rope 64、v 128、YaRN（factor 40、mscale=mscale_all_dim=1.0、original 4096） |
| S3 | [DeepSeek-V2-Lite HF repo modeling_deepseek.py](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/raw/main/modeling_deepseek.py) 与 [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/raw/main/config.json) | `DeepseekV2Attention`（行 685–916）与 V3 版结构同构；config 确认 `q_lora_rank: null`（Lite 不做 q 压缩）、27 层 16 头、mscale=0.707 |
| S4 | [DeepSeek-V2 完整版 config.json](https://huggingface.co/deepseek-ai/DeepSeek-V2/raw/main/config.json) | 60 层、128 头、q_lora_rank 1536（V2 完整版有 q 压缩，Lite 没有） |
| S5 | [DeepSeek-V2 论文（arXiv 2405.04434v5 HTML）](https://arxiv.org/html/2405.04434v5) | §2.1.2 低秩联合压缩（式 9–13）、§2.1.3 解耦 RoPE（式 14–19）、§2.1.4 + Table 1（KV cache 对比）、Appendix C（完整公式与吸收论证）、Appendix D.2（MLA vs MHA 消融） |
| S6 | [DeepSeek-V3 官方 GitHub 推理代码 inference/model.py](https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3/main/inference/model.py) | 全文。`attn_impl: Literal["naive","absorb"] = "absorb"`（默认 absorb）；`MLA` 类的两条 forward 分支与两套 cache buffer；softmax_scale 的 mscale 调整 |
| S7 | [DeepSeek-V3.2-Exp inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/raw/main/inference/model.py) | 抽查：MLA 只剩 latent cache（`kv_cache` 512 维 + `pe_cache` 64 维），mscale 公式不变；新增 Indexer（与本文无关） |
| S8 | [transformers v4.56.0 modeling_deepseek_v3.py](https://raw.githubusercontent.com/huggingface/transformers/v4.56.0/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py) | 抽查 `yarn_get_mscale`（行 317–320），与 repo 版一致：`0.1 * mscale * math.log(scale) + 1.0`，**没有 softplus** |

## 2. 核心机制与推导（V2 论文公式编号）

低秩联合压缩（§2.1.2，式 9–11）：

$$c_t^{KV} = W^{DKV} h_t \quad(9),\qquad k_t^C = W^{UK} c_t^{KV}\ (10),\qquad v_t^C = W^{UV} c_t^{KV}\ (11)$$

$c_t^{KV} \in \mathbb{R}^{d_c}$（V3 中 $d_c=512$），$W^{UK}, W^{UV} \in \mathbb{R}^{d_h n_h \times d_c}$。推理只需缓存 $c_t^{KV}$（§2.1.2：「KV cache has only $d_c l$ elements」）；且因矩阵乘法结合律，$W^{UK}$ 可吸收进 $W^Q$、$W^{UV}$ 可吸收进 $W^O$（§2.1.2 末段；Appendix C 明确写出吸收论证）。

query 也做低秩压缩（§2.1.2，式 12–13），但目的不是省 KV cache，是省训练激活内存：$c_t^Q = W^{DQ} h_t$（12），$q_t^C = W^{UQ} c_t^Q$（13）。

解耦 RoPE（§2.1.3，式 14–19）：RoPE 若直接加在 $k_t^C$ 上，位置相关矩阵会夹在 $W^Q$ 与 $W^{UK}$ 之间，使吸收不可行（式 10 下面的论证段）。所以单独造一组逐头 query $q_{t,i}^R = \mathrm{RoPE}(W^{QR} c_t^Q)$（14）和**所有头共享**的 key $k_t^R = \mathrm{RoPE}(W^{KR} h_t)$（15），拼接：

$$q_{t,i}=[q_{t,i}^C;q_{t,i}^R]\ (16),\qquad k_{t,i}=[k_{t,i}^C;k_t^R]\ (17)$$

$$o_{t,i}=\sum_j \mathrm{Softmax}_j\!\Big(\frac{q_{t,i}^\top k_{j,i}}{\sqrt{d_h+d_h^R}}\Big) v_{j,i}^C\ (18),\qquad u_t = W^O[o_{t,1};\dots]\ (19)$$

注意式 18 的 scale 是 $\sqrt{d_h + d_h^R}^{-1} = 192^{-1/2}$，按**拼接后**的总头维算——代码里 `self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim`（S1 行 654）正是这个。

总 KV cache：$(d_c + d_h^R)\, l$ 个元素（§2.1.3 末），$k_t^R$ 因为所有头共享只需存一份。

## 3. 代码落点

### 3.1 模块构造（S1，`DeepseekV3Attention.__init__`，行 658–697）

```python
if self.q_lora_rank is None:
    self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)
else:
    self.q_a_proj = nn.Linear(self.hidden_size, config.q_lora_rank, bias=config.attention_bias)
    self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)
    self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

self.kv_a_proj_with_mqa = nn.Linear(
    self.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim, bias=config.attention_bias)
self.kv_a_layernorm = DeepseekV3RMSNorm(config.kv_lora_rank)
self.kv_b_proj = nn.Linear(
    config.kv_lora_rank,
    self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim), bias=False)

self.softmax_scale = self.q_head_dim ** (-0.5)
if self.config.rope_scaling is not None:
    mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
    scaling_factor = self.config.rope_scaling["factor"]
    if mscale_all_dim:
        mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
        self.softmax_scale = self.softmax_scale * mscale * mscale
```

要点：V3 的 $W^Q$ 实际是两段（q_a 7168→1536、RMSNorm、q_b 1536→24576）；`kv_a_proj_with_mqa` 一口气产出 512+64=576 维；`kv_b_proj` 是 512→128×(128+128)=32768，把 K 的 nope 部分和 V 一起展开。**`softmax_scale` 被乘上 mscale 的平方**（mscale 作用于 cos/sin 双侧，见 3.3）。

### 3.2 naive 前向数据流（S1，`forward`，行 768–815；V3 数字）

```python
q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
kv = (self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
      .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
      .transpose(1, 2))
k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
...
q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
query_states[:, :, :, :self.qk_nope_head_dim] = q_nope
query_states[:, :, :, self.qk_nope_head_dim:] = q_pe
key_states[:, :, :, :self.qk_nope_head_dim] = k_nope
key_states[:, :, :, self.qk_nope_head_dim:] = k_pe
key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
```

逐步张量形状（decode 单 token，bsz=1，V3 数字，记号 [B, H, T, D]）：

1. `hidden_states` [1,1,7168] → `q_a_proj` → [1,1,1536] → `q_a_layernorm`（RMSNorm，fp32 内部）→ `q_b_proj` → [1,1,24576]
2. view+transpose → q [1,128,1,192]；split → `q_nope` [1,128,1,128]、`q_pe` [1,128,1,64]
3. `kv_a_proj_with_mqa` → [1,1,576]；split → `compressed_kv` [1,1,512]、`k_pe` [1,1,64] → [1,**1**,1,64]（头维为 1，共享）
4. `kv_a_layernorm`(512) → `kv_b_proj` → [1,1,32768] → [1,128,1,256]；split → `k_nope` [1,128,1,128]、`value_states` [1,128,1,128]
5. RoPE **只加在 64 维上**：`apply_rotary_pos_emb(q_pe, k_pe, cos, sin)`；cos/sin 来自 `DeepseekV3YarnRotaryEmbedding`，维数 `qk_rope_head_dim=64`、theta=10000
6. 拼接 `query_states`/`key_states` [1,128,1,192]；`past_key_value.update` 后 cache 沿 T 维增长 → [1,128,t,192] 与 [1,128,t,128]
7. `QK^T × softmax_scale`（V3 实际值 ≈0.1352，见 §4）→ softmax（fp32 上抛，行 835）→ dropout → ×V → [1,128,1,128] → reshape [1,1,16384] → `o_proj`（16384→7168）

`DeepseekV3FlashAttention2.forward`（行 900–1005）数据流完全一样，只是 V pad 到 192 维喂 flash_attn（行 944：`if self.q_head_dim != self.v_head_dim: value_states = F.pad(...)`），输出再截回 128 维。cache 内容不变——**还是展开后的 K/V**。

### 3.3 YaRN mscale 如何进入 softmax_scale（S1 行 249–252、318–328）

```python
def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0
```

```python
_mscale = float(
    yarn_get_mscale(self.scaling_factor, self.mscale)
    / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim))
self.register_buffer("cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False)
self.register_buffer("sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False)
```

即 RoPE 的 cos/sin 乘 $m_{\text{scale}}/m_{\text{scale\_all\_dim}}$，softmax_scale 乘 $m_{\text{scale\_all\_dim}}^2$；V3 config 里 mscale=mscale_all_dim=1.0，所以 cos/sin 不缩放、softmax 端吃满 $(0.1\ln 40 + 1)^2$。**官方各版代码（S1/S3/S6/S7/S8）都没有 softplus**，softmax_scale 调整就是这一个公式；S6 的等价写法是 `mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0; softmax_scale *= mscale * mscale`（条件 `max_seq_len > original_seq_len`）。

### 3.4 absorb 路径只在 GitHub 官方推理代码里（S6）

```python
attn_impl: Literal["naive", "absorb"] = "absorb"   # 默认 absorb
...
if attn_impl == "naive":
    self.register_buffer("k_cache", torch.zeros(bs, seq, n_local_heads, qk_head_dim), ...)
    self.register_buffer("v_cache", torch.zeros(bs, seq, n_local_heads, v_head_dim), ...)
else:
    self.register_buffer("kv_cache", torch.zeros(bs, seq, kv_lora_rank), ...)   # 512
    self.register_buffer("pe_cache", torch.zeros(bs, seq, qk_rope_head_dim), ...)  # 64
```

absorb 分支前向（S6，对应论文 Appendix C 的吸收论证）：

```python
wkv_b = self.wkv_b.weight ...  # [n_heads, 256, 512]
q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)   # 存归一化后的 latent
self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos])
        + torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
...
x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
```

即：$W^{UK}$ 按头切出来乘进 `q_nope`（每头 query 从 128 维变 512 维），score 直接在 latent 维算；输出先在 latent 维加权求和、再乘 $W^{UV}$ 部分，等价于把 $W^{UV}$ 吸进 $W^O$ 的前一步。S7（V3.2）进一步连 naive 分支都删了，只留 latent cache。

**结论：HF repo 的 modeling_deepseek.py（S1、S3）没有 absorb 分支，ATTENTION_CLASSES 只有 eager 和 flash_attention_2 两种，都是 naive 展开式。** 博客的结构图如果按「官方 HF 代码」画就是 naive 路径；按「官方推理代码/生产框架」画就是 absorb 路径——两条都要画，并标清出处。

## 4. 关键数字

| 数字 | 值 | 条件 | 出处 |
|---|---|---|---|
| 每 token 每层 latent cache | 512 + 64 = **576 元素** | $d_c + d_h^R$；V2/V3 同值 | S5 §2.1.3 末、式 15–17；S2/S4 config |
| V3 全模型 cache（latent 口径） | 576 × 61 = 35,136 元素 = **70,272 B ≈ 68.6 KiB**（BF16） | 61 层，官方推理/生产框架实际存法 | S2（61 层）+ S5；cache buffer 见 S6 |
| V3 HF naive 路径实际 cache | 128×(192+128)×61 = 2,498,560 元素 ≈ **4.77 MiB**（BF16） | HF 参考实现 cache 的是展开后 K/V | S1 行 804–815 |
| naive/latent 体积比 | **71.1×** | 40,960 / 576（每层） | 由 S1 形状算出 |
| MHA 基线（同 128 头、d_h=128） | 2·128·128·61 = 1,998,848 元素 ≈ 3.81 MiB | V2 论文 Table 1 公式 $2n_h d_h l$ | S5 Table 1 |
| latent/MHA 压缩比 | **56.9×**（≈1.76%）；论文原话「等于 GQA 2.25 组」 | $(4d_h + d_h/2) = 4.5\,d_h$ 对 $2n_h d_h$，$n_h$=128 | S5 Table 1 注（$d_c=4d_h$，$d_h^R=d_h/2$） |
| latent/GQA-8 | 28.1%（≈3.56× 省） | 2·8·128=2048 对 576 | 由 S5 Table 1 公式推 |
| V2 论文摘要口径 | 「reduces the KV cache by **93.3%**」 | 对 DeepSeek 67B（MHA 32 头） | S5 摘要 |
| V3 softmax_scale | 192$^{-1/2}$ × (0.1·ln40+1)² = 0.07217 × 1.87385 ≈ **0.1352** | mscale_all_dim=1.0、factor=40 | S1 行 691–697 + S2 |
| V2 / V2-Lite softmax_scale | ≈ **0.1147**（mscale² = 1.5896） | mscale_all_dim=0.707 | S3/S4 config + S1 同段代码 |
| q 侧权重形状 | q_a 7168×1536；q_b 1536×24576 | V3；V2-Lite 是单个 q_proj 2048→3072（q_lora_rank=null） | S1 行 663–669；S3 config |
| kv 侧权重形状 | kv_a 7168×576；kv_b 512×32768；o 16384×7168 | V3 | S1 行 671–688 |
| `num_key_value_heads`=128 | 对 MLA 无意义 | config 遗留字段，代码从不读它做 MLA | S2；S1 里没有任何引用 |

## 5. 常见误解

1. **「MLA 缓存的是展开后的 K/V」。** 只有 HF 参考实现（S1/S3）这么干（每层 40,960 元素）。官方推理代码（S6，默认 absorb）和生产框架缓存的是 576 维 latent（512 维归一化后的 $c^{KV}$ + 64 维 $k^R$），71 倍差距。
2. **「MLA 省 FLOPs」。** 不省。naive 路径反而多一次 kv_b_proj 展开（512→32768）；absorb 路径下每头 QK^T 内积从 192 维变 512+64=576 维，score 计算量更大。省的是**显存占用和 decode 时读 cache 的访存带宽**（decode 是 memory-bound）。论文的卖点也是 KV cache 与吞吐（S5 摘要、Figure 1b）。
3. **「q 的低秩压缩也省 KV cache」。** 不省，论文明说「even if it cannot reduce the KV cache」，目的是省训练激活内存（S5 §2.1.2 式 12 前）。
4. **「MLA = MQA，所以质量跟 MQA 一样差」。** 训练时它是 128 个头的满参数结构，只是推理时（结合律）可折叠成 MQA 形态；V2 论文 Appendix D.2 消融显示 MLA 性能**优于** MHA，且 KV cache 只有 MHA 的 14%（小模型）/4%（大模型）。
5. **「RoPE 加在全部 192 维上」。** 只在 64 维 decoupled rope 分量上（S1 行 802：只对 `q_pe/k_pe` 调 `apply_rotary_pos_emb`）；nope 部分不旋转，这正是吸收可行的前提（S5 §2.1.3）。
6. **「softmax_scale 就是 $1/\sqrt{d_h}$」。** MLA 按拼接后的 $d_h+d_h^R=192$ 算（论文式 18），YaRN 长上下文下还要再乘 $m_{\text{scale\_all\_dim}}^2$（S1 行 691–697）。另外 DeepSeek 各版官方代码里**没有 softplus**，调整式只有 $0.1\,m\ln(\text{factor})+1$。
7. **「k_pe 每个头一份」。** $k^R$ 是所有头共享的一条（S5 式 15、17；S1 行 781 头维为 1），这也是 cache 里它只占 64 维的原因。

## 6. 与本专栏其他文章的接口

已覆盖、不要重复：

- `kimi-k3-03-gated-mla.mdx`：MLA 机制复习（按 **K3 的 96 头**口径走了一遍维度）、「只缓存 latent / MQA 模式」、576 维的来源、RoPE 与低秩吸收的矛盾（第 60 行）、HF 代码 cache 是展开 K/V 的脚注（第 146 行，一句话带过）。本文的差异化：换 **V3/V2 真实维度**（128 头、61 层）、给出 HF naive vs GitHub absorb 两套**官方代码证据**（kimi-k3-03 只有脚注）、softmax_scale 的 mscale² 精确值、71×/56.9×/3.56× 的对比账。
- `deepseek-v4-01-kv-compression.mdx`：谱系段落里已有一段 MLA（第 53–55 行，含 MQA 模式、576 维），以及「V4 沿用 MQA 模式骨架、干脆不要 $W^{UK}/W^{UV}$」（第 79–81 行）。本文是它的展开前传：把「吸收」这一步从一段话展开成带代码行号的完整机制。
- `kimi-k3-01-kda-recurrence.mdx` 等多处引用「MLA 每 token 每层 576 个数」作为对照常量。

建议链接：本文 ↔ deepseek-v4-01（V4 的 CSA/HCA 是 MLA MQA 模式的后继）↔ kimi-k3-03（K3 的 NoPE MLA 变体）。

## 7. 图的建议

**图 A（主图）：MLA 小算子级张量流图（naive 路径，V3 数字，decode 单 token）**

节点（算子为矩形，张量为圆角框，标注形状）：

- 输入：`h_t` [1,7168]
- Q 支路：`q_a_proj`(7168→1536) → `q_a_layernorm`(RMSNorm) → `c_t^Q` [1536] → `q_b_proj`(1536→24576) → `q` [128头,192] → `split` → `q_nope` [128,128] 与 `q_pe` [128,64] →（`q_pe` 经 `RoPE` 节点）→ `concat` → `query_states` [128,192]
- KV 支路：`kv_a_proj_with_mqa`(7168→576) → `split` → 两路：
  - `c_t^{KV}` [512] → `kv_a_layernorm` → `kv_b_proj`(512→32768) → [128头,256] → `split` → `k_nope` [128,128] 与 `v` [128,128]
  - `k_pe` [1,64] → `RoPE` → 广播拼进每个头
- `concat(k_nope, k_pe_rope)` → `key_states` [128,192]
- **KV cache 节点（重点标注）**：HF naive 实际写入 `key_states`/`value_states` 全量（[t,128,192]+[t,128,128]，71×）；虚线框标 absorb 替代方案「只存 c_t^{KV}[512] + k_pe[64]，576 维」
- 注意力核：`QK^T × softmax_scale≈0.1352`（标注 192^{-1/2}×mscale²）→ `+ mask` → `softmax(fp32)` → `PV` → [128,128] → `reshape` [16384] → `o_proj`(16384→7168) → 输出

边：按上面顺序连；RoPE 的 cos/sin 作为侧输入（标 YaRN：factor 40、theta 10000、_mscale=1.0）。颜色约定：参数（Linear 权重）一色、中间张量一色、cache 一色、RoPE 相关一色。

**图 B（对比小图）**：naive vs absorb 双栏。absorb 栏画 S6 的三个 einsum：`q_nope × wkv_b[:, :128]` → [128,512]；`scores = q_nope'·kv_cache + q_pe·pe_cache`；`out = (P·kv_cache) × wkv_b[:, 128:]`。标注 cache 从 [t,128,320] 缩到 [t,576]。

**图 C（可选）**：KV cache 柱状对比 MHA / GQA-8 / GQA-2.25 / MLA-latent / HF-naive（对数轴，每 token 字节数）。

## 8. 待确认/没查到的点

- **softplus 未找到**：任务提到的「softplus 调整 softmax_scale」在 S1/S3/S6/S7/S8 中均不存在，官方一律用 `0.1·m·ln(factor)+1` 再平方。若在某处见过 softplus 形式，大概率是第三方框架（vLLM/SGLang 旧版）或别的模型族，需要另行定位。
- **V2-Lite 的 HF naive cache 体积**与 q_proj 单矩阵路径（q_lora_rank=null）本文只在文字里提了，没单独画形状表；如需 Lite 完整张量流可照 §3.2 套 16 头/2048 维重算。
- **S6 的 `attn_impl` 是模块级全局变量**（代码里 `attn_impl: Literal[...] = "absorb"` 硬编码），README 如何切换没读（未读 S6 repo 的 README/generate.py）。
- V3.2（S7）里 `kv_cache` 存的是**未归一化**的 `kv`（S7 行 572），与 S6 存 `kv_norm(kv)` 不同，推测 V3.2 把 RMSNorm 折进了权重，未深挖。
- transformers 主线（S8）只抽查了 `yarn_get_mscale`；新版 transformers 对 DeepseekV3 的 attention 实现已重构（ALL_ATTENTION_FUNCTIONS），是否与 repo 版逐行等价未核对。
