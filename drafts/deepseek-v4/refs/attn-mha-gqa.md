# DeepSeek 第一代模型的注意力：MHA 与 GQA（deepseek-llm-7B/67B、DeepSeekMoE-16B）

## 0. 一句话

DeepSeek 第一代**没有**任何自研注意力结构：deepseek-llm-7B 是标准 MHA（32/32 头），deepseek-llm-67B 是 GQA（64 个 query 头共享 8 组 KV），DeepSeekMoE-16B 是 MHA（16/16 头）；三者全是 full RoPE（θ=10000），且 deepseek-llm 两个 repo 连模型代码都没有，直接复用 transformers 的 `LlamaForCausalLM`。

## 1. 一手资料清单

| # | 资料 | URL | 读了什么 |
|---|---|---|---|
| S1 | deepseek-llm-7b-base config.json | https://huggingface.co/deepseek-ai/deepseek-llm-7b-base/raw/main/config.json | 全文：`model_type/num_attention_heads/num_key_value_heads/rope_theta/max_position_embeddings` 等 |
| S2 | deepseek-llm-67b-base config.json | https://huggingface.co/deepseek-ai/deepseek-llm-67b-base/raw/main/config.json | 全文，同上 |
| S3 | deepseek-moe-16b-base config.json | https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/raw/main/config.json | 全文，同上 + `auto_map` |
| S4 | HF API 文件列表（3 个 repo） | https://huggingface.co/api/models/deepseek-ai/deepseek-llm-7b-base （另两个同构） | `siblings` 列表：确认 repo 是否自带 modeling 代码 |
| S5 | deepseek-llm-67b-base README | https://huggingface.co/deepseek-ai/deepseek-llm-67b-base/raw/main/README.md | "a 67B parameter model with Grouped-Query Attention" 一句 |
| S6 | transformers v4.33.1 `modeling_llama.py` | https://raw.githubusercontent.com/huggingface/transformers/v4.33.1/src/transformers/models/llama/modeling_llama.py （本地副本 `refs/transformers-4.33.1-modeling_llama.py`） | `LlamaRotaryEmbedding`、`apply_rotary_pos_emb`、`repeat_kv`、`LlamaAttention` 全部 |
| S7 | deepseek-moe-16b-base `modeling_deepseek.py` | https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/raw/main/modeling_deepseek.py （本地副本 `refs/deepseek-moe-modeling_deepseek.py`） | `DeepseekRotaryEmbedding`、`repeat_kv`、`DeepseekAttention`、`DeepseekSdpaAttention` 全部 |
| S8 | deepseek-llm 论文 | arXiv:2401.02955（README 指向的 DeepSeek-LLM repo） | 背景出处（未逐段读，仅用于谱系定位） |
| S9 | GQA 原始论文 | Ainslie et al. 2023，arXiv:2305.13245 | GQA 定义出处（二手定位，见 §8） |

注：v4.33.1 正好是 deepseek-llm config 里 `transformers_version` 写的版本（S1/S2），所以读它就是读「官方推理代码」。

## 2. 核心机制与推导（MHA→GQA）

**MHA（Multi-Head Attention）**。每层：Q、K、V 各有 $n_h$ 个头，头维 $c=d/n_h$。decode 时 KV cache 存全部头的 K、V：每 token 每层缓存 $2\,n_h c$ 个元素（S6 `LlamaAttention` 的 cache 存的是 repeat 之前的 `key_states/value_states`，形状 `[b, n_h, s, c]`）。

**GQA（Grouped-Query Attention）**。K、V 只留 $n_{kv}<n_h$ 组，每组被 $n_{rep}=n_h/n_{kv}$ 个 query 头共享（S9 提出；S5 官方自述 67B 用了它）。动机：decode 是显存带宽瓶颈，KV cache 的大小和每步要读的字节数都正比于 $n_{kv}$，砍 KV 头数几乎不掉质量但把 cache 和访存都除以 $n_{rep}$。

代码里的实现（S6, modeling_llama.py:343-346）：

```
key_states   = repeat_kv(key_states,   num_key_value_groups)  # [b,8,s,c] -> [b,64,s,c]
value_states = repeat_kv(value_states, num_key_value_groups)
attn_weights = matmul(query_states, key_states.transpose(2,3)) / sqrt(c)
```

即 GQA 不改变注意力公式本身 $\mathrm{softmax}(QK^\top/\sqrt c)V$，只是在算分之前把 8 组 KV **逻辑复制**成 64 组（`repeat_kv` 内部是 `expand`，n_rep=1 时直接原样返回，S6:225-234）。cache 里存的仍是 8 组，复制发生在 cache 读出之后。

**谱系定位**：MHA（7B、MoE-16B）→ GQA（67B，$n_{kv}=8$）→ MQA（$n_{kv}=1$ 的极限）→ MLA（V2 起，压维度）。本文给出谱系起点两个真实数据点。

## 3. 代码落点

### 3.1 deepseek-llm-7b / 67b：repo 无代码，复用 transformers 的 Llama

- S4 文件列表：两个 repo 只有 config、tokenizer、权重、README，**没有任何 .py**。
- S1/S2：`"model_type": "llama"`、`"architectures": ["LlamaForCausalLM"]`、`"transformers_version": "4.33.1"`。
- 结论：官方推理代码 = transformers v4.33.1 的 `src/transformers/models/llama/modeling_llama.py`（S6）。

关键行摘录（S6）：

`LlamaAttention.__init__`（237-259 行，节选）：

```python
self.num_heads = config.num_attention_heads
self.head_dim = self.hidden_size // self.num_heads
self.num_key_value_heads = config.num_key_value_heads
self.num_key_value_groups = self.num_heads // self.num_key_value_heads
self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
```

`repeat_kv`（225-234 行，全文）：

```python
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
```

`LlamaAttention.forward` 核心（325-346 行，节选）：

```python
query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
if past_key_value is not None:
    key_states = torch.cat([past_key_value[0], key_states], dim=2)
    value_states = torch.cat([past_key_value[1], value_states], dim=2)
past_key_value = (key_states, value_states) if use_cache else None
key_states = repeat_kv(key_states, self.num_key_value_groups)      # cache 之后才复制
value_states = repeat_kv(value_states, self.num_key_value_groups)
attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
```

RoPE（S6, 92-125、173-188 行）：`inv_freq = 1/(θ^(2i/c))`，θ=`rope_theta`=10000（S1/S2）；`rotate_half` 对整个 head_dim 操作，`q*cos + rotate_half(q)*sin` 覆盖全部 128 维 → **full RoPE，无 partial**。`rope_scaling: null`（S1/S2）→ 走 `_init_rope` 的基础分支（S6:262-268），无任何外推缩放。

### 3.2 deepseek-moe-16b-base：repo 自带 modeling_deepseek.py

- S3：`"model_type": "deepseek"`，`auto_map` 指向 `modeling_deepseek.DeepseekForCausalLM`；S4 确认 repo 含 `modeling_deepseek.py` + `configuration_deepseek.py`。
- S7 中 `DeepseekAttention`（428 行起）类头注释明确写着 `# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->Deepseek`（427 行）——即逐行复制 4.36 时代的 LlamaAttention。
- 与 S6 的唯一实质差异：用 `Cache.update()` 管理 KV cache（551 行）替代手工 `torch.cat`，且投影 bias 由 `attention_bias`（config 中为 false）控制。
- `num_key_value_groups = num_heads // num_key_value_heads`（447 行）；config 中 16/16 → n_rep=1，`repeat_kv` 第一行就 return（421-422 行）→ **MHA**。
- RoPE 同样是 full、θ=10000（S7 `DeepseekRotaryEmbedding` 123-158 行 + S3 `rope_theta: 10000`）。

## 4. 关键数字

| 数字 | 值 | 条件 | 出处 |
|---|---|---|---|
| 7B 层数 / 头数 / KV 头数 | 30 / 32 / 32 | MHA，n_rep=1 | S1 |
| 67B 层数 / 头数 / KV 头数 | **95** / 64 / 8 | GQA，n_rep=8 | S2 |
| MoE-16B 层数 / 头数 / KV 头数 | 28 / 16 / 16 | MHA（28 层中第 1 层 dense，其余 MoE，但注意力全同构） | S3 |
| hidden_size | 4096 / 8192 / 2048 | 7B / 67B / MoE | S1/S2/S3 |
| head_dim c | 128 / 128 / 128 | hidden÷heads | S1/S2/S3 推算 |
| max_position_embeddings | 4096 | 三者相同 | S1/S2/S3 |
| rope_theta / rope 类型 | 10000 / full（整个 head_dim） | `rope_scaling=null` | S1/S2/S3 + S6:92-125 + S7:123-158 |
| dtype | BF16 | torch_dtype | S1/S2/S3 |

**KV cache 账（BF16，2 字节/元素；每 token 每层 = 2(K,V) × n_kv × c × 2B）**：

| 模型 | 每 token 每层 | 每 token 全部层 | 4096 token 上下文 |
|---|---|---|---|
| 7B（MHA，32 KV 头） | 2×32×128×2 = 16 KiB | ×30 = 480 KiB | ≈ 1.88 GiB |
| 67B（GQA，8 KV 头） | 2×8×128×2 = 4 KiB | ×95 = 380 KiB | ≈ 1.48 GiB |
| 67B 假想 MHA（64 KV 头） | 32 KiB | ×95 = 3.04 MiB | ≈ 11.9 GiB |
| MoE-16B（MHA，16 KV 头） | 8 KiB | ×28 = 224 KiB | ≈ 0.88 GiB |

要点：67B 靠 GQA 把每 token KV cache 压到比自己小 10 倍的 7B 还低（380 KiB vs 480 KiB）；若 67B 用 MHA 则是现在的 8 倍。这就是 GQA 的账。

## 5. 常见误解

1. **「DeepSeek 从第一代就用 MLA」**——错。MLA 首现于 DeepSeek-V2；第一代的 7B 是 MHA、67B 是 GQA、MoE-16B 是 MHA（S1/S2/S3 的 `num_key_value_heads` 直接可证）。
2. **「deepseek-llm 有自研 modeling 代码」**——错。两个 repo 无任何 .py（S4），`model_type: llama` 直接复用 transformers 的 `LlamaForCausalLM`（S1/S2 + S6）。博客口径应写「以 transformers v4.33.1 的 LlamaAttention 为官方实现」。
3. **「GQA 是 DeepSeek 提出的」**——错。GQA 来自 Google 2023 年论文（S9），DeepSeek 只是采用并在 README 中明示（S5）。
4. **「67B 是 32 层」**——错，config 写明 95 层（S2）；32 是 7B 的头数，容易串。
5. **「`rope_scaling: null` 表示没用 RoPE」**——错，它只是不做长度外推缩放；标准 RoPE（θ=10000、full head_dim）始终存在（S6 `_init_rope` 分支）。
6. **「GQA 的 KV 复制会占 8 倍显存」**——不会。cache 存的是复制前的 8 组（S6:335-340 在 343-344 之前），`repeat_kv` 只是计算时的逻辑 expand。
7. **「DeepSeekMoE 的注意力也是新的」**——错。`DeepseekAttention` 类注释自认是从 `LlamaAttention` 复制的（S7:427），MoE 的创新全在 FFN 侧。

## 6. 与本专栏其他文章的接口

- `src/content/posts/deepseek-v4-01-kv-compression.mdx` 已建立「MHA → MQA → MLA → DSA → CSA/HCA」的砍量谱系（头数/维度/条目数三轴）。本文给谱系起点补上 DeepSeek 自家真实数据点：**GQA 恰好是 MHA 与 MQA 之间那一格**（67B 的 n_kv=8），是 V1→V2 演进中「先砍头数、再砍维度」的中间站。可在 v4-01 的谱系图上加一个「DeepSeek-LLM 67B (2024.01) = GQA-8」的标注并互链。
- `deepseek-v4-00-overview.mdx` 的谱系表可引用本文 §4 的 KV cache 账：V1 67B 的 380 KiB/token 与 V2+ MLA 的对比，量化「为什么 V2 要发明 MLA」（380 KiB × 4096 token ≈ 1.5 GiB，长上下文下仍不可接受）。
- 本文 §3 的「小算子级张量流」写法与 kimi-k3-03（gated MLA）等篇的逐算子拆解口径一致，可交叉引用作为「标准注意力」基线。

## 7. 图的建议

1. **MHA vs GQA 头共享示意图**（横排）：67B 真实数字——64 个 Q 头分 8 组，每组 8 个 Q 头共享 1 组 K/V；旁边放 7B 的 32↔32 一一对应做对照。颜色编码「同一组」。表达「GQA 只砍 K/V 头，Q 头不变」。
2. **小算子级张量流图**（本篇核心图，以 67B 为例，b=batch, s=新 token 数, t=总长度=s+past）：
   - 节点 `hidden_states [b,s,8192]` → 三条边：
     - `q_proj (8192→8192)` → `Q [b,s,8192]` → `view+transpose` → `Q [b,64,s,128]`
     - `k_proj (8192→1024)` → `K [b,s,1024]` → `view+transpose` → `K [b,8,s,128]`
     - `v_proj (8192→1024)` → `V [b,s,1024]` → `view+transpose` → `V [b,8,s,128]`
   - `cos/sin [b,1,s,128] (θ=10000, full)` 与 `K`、`Q` 汇入 `RoPE (apply_rotary_pos_emb)` 节点 → `Q' [b,64,s,128]`、`K' [b,8,s,128]`
   - `KV cache [b,8,t,128]` 节点与 `K'`、`V` 汇入 `concat/cache update` → `K_all, V_all [b,8,t,128]`（此形状即写入 cache 的形状，加粗标注）
   - `repeat_kv ×8` 节点 → `K̂, V̂ [b,64,t,128]`
   - `Q'·K̂ᵀ / √128` → `scores [b,64,s,t]` → `+causal mask` → `softmax(fp32)` → `P [b,64,s,t]`
   - `P·V̂` → `ctx [b,64,s,128]` → `transpose+reshape` → `[b,s,8192]` → `o_proj (8192→8192)` → `out [b,s,8192]`
   - 边上标注字节数可选；关键设计点：把 `KV cache` 节点画在 `repeat_kv` **之前**，一眼看出 cache 只存 8 头。
3. **KV cache 对比柱状图**：每 token KV cache 字节数——7B MHA 480 KiB、67B GQA 380 KiB、67B 假想 MHA 3.04 MiB（虚色），可加 MoE-16B 224 KiB。表达「GQA 让 67B 的 cache 比 7B 还小」。

## 8. 待确认/没查到的点

- GQA 原始论文（arXiv:2305.13245）和 DeepSeek-LLM 论文（arXiv:2401.02955）本次只定位未逐段精读；论文中对 67B 选 GQA 的消融细节（若有）值得补读。
- transformers 4.33.1 之后 `LlamaAttention` 已多次重构（4.36 起 Cache 类、4.38+ eager/sdpa 拆分），若博客要引用「现在的 transformers 长什么样」需另查当前版本；本文锁定 config 声明的 4.33.1。
- deepseek-llm 训练时是否用了与 HF 实现逐位等价的自家代码（官方 GitHub 训练仓库未查）；以 HF repo 为口径时此点不影响结论。
- 67B 的 `pretraining_tp=1`，故正文忽略 TP 分支（S6:303-318）；未验证历史 checkpoint 是否曾用 tp>1 训练。
