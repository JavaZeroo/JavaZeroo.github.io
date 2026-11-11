# Kimi K3.1 Pro：从推理实现确认 Gated DSA

## 0. 一句话：这篇要解决什么问题

K3.1 Pro 保留 K3 的 69 KDA + 24 MLA 层级混合骨架，把后者变成带输出门的 DSA：每层独立用 32×128 的索引器选最多 2048 个历史 token，再用原来的 96 头 MLA 对选中集合做 softmax 注意力。

## 1. 一手资料清单（URL + 版本/commit + 读了哪些部分）

核对日期：2026-11-12。全部可复现 URL 与已读文件 SHA256 见 `k31-sources.json`，HF 配置快照见 `k31-config.json`。不下载模型权重，不声称跑过模型。

- HF revision：`d9e99b0792a5e93b78ea9bea62a68167e36ddd9e`。[config](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e/config.json) 全文；[README](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e/README.md) Model Summary。该 revision 有 configuration、tokenizer、processor，无 modeling 文件；auto_map 只有 AutoConfig。
- vLLM commit：`62529d6a2d22586c72c19aaf993840f4e064a680`。`vllm/models/kimi_k3/nvidia/dsa.py` 的 `KimiK3DSAAttention`、`KimiK3DSAIndexer`、`KimiK3DSAIndexerKNorm`；`nvidia/model.py` 层分派、共享工作缓冲、权重映射；`nvidia/ops/dsa_indexer.py` 量化、归一化、656-byte main cache；`common/dsa_cache.py` 132/68-byte index cache；`vllm/transformers_utils/configs/kimi_linear.py` 配置；`vllm/model_executor/layers/sparse_attn_indexer.py` logits 与 top-k 调用。
- 同一 vLLM commit 的 [csrc/fused_mqa_topk/common.cuh](https://github.com/vllm-project/vllm/blob/62529d6a2d22586c72c19aaf993840f4e064a680/csrc/fused_mqa_topk/common.cuh#L106)：`reduce_heads` 注释和实现确认 ReLU 点积的加权头求和；`dense_scan.cuh` 将 key 的量化 scale 乘回。
- SGLang commit：`b174f4bb4fe0f316cf625aa5441f0ea5bd65cd84`。`python/sglang/srt/configs/kimi_linear.py::_init_dsa`；`models/kimi_k3.py::KimiK3MLAAttention`（继承 `DeepseekV2AttentionMLA`、输出门）与 `_check_dsa_indexer_loaded`；`models/deepseek_v2.py` 构造 indexer；`layers/attention/dsa/dsa_indexer.py::Indexer` 的投影、head gate、NoPE、跳过短上下文打分。
- [Moonshot tech blog](https://www.kimi.com/en/blog/kimi-k3-1-pro)：官方称 K3.1 Pro 是 K3 的 post-trained version。它不提供结构公式、索引器训练配方或消融；不能当成完整架构论文。

## 2. 核心机制与推导

### 2.1 结构差异

与本仓库 `k3-config.json` 比较：`vision_config` 完全一致；text config 结构新增 `dsa_config`，其余差异是 `auto_map`、`quantization_config`、`transformers_version`。层号从 1 开始：`dsa_layers == full_attn_layers == [4,8,...,92,93]`。KDA 层补集为 69 层。不能把旧字段 full_attn_layers 解读成仍然执行 dense attention。

### 2.2 索引器（按实数计算还原，不含量化/舍入）

令 x 为本层 attention 的 7168 维输入，q_c=RMSNorm(W_q_a x)，维度 1536。

1. QI=W_Iq q_c，reshape 成 32×128；主 query 是另一套 W_q_b，从同一个 q_c 展开成 96×192。
2. KI=LayerNorm(W_Ik x)，128 维，key 在 32 头之间共享。LayerNorm 带 weight 和 bias，FP32 计算再 cast 回输入 dtype；不是 RMSNorm。
3. a=W_Iw x，32 个有符号实数，无 sigmoid/softmax。SGLang 会保留 BF16 head weight 中间舍入；vLLM 融合路径把 1/sqrt(32)、1/sqrt(128)、query quant scale 合到 weights。
4. I(t,s)=sum_h a(t,h) ReLU(QI(t,h)·KI(s)) / sqrt(32×128)，因果候选 s≤t。
5. S_t=TopK(I(t,:), min(2048,t))，此处 t 从 1 数；集合供本层所有 96 个主注意力头共用。

常数 1/sqrt(32×128) 不影响精确算术下的排序。实际低精度可能改变临界排序，因此正文伪代码不能称为 bitwise 复现。

### 2.3 稀疏 MLA 和门

KV down-proj 输出 512+64，前 512 RMSNorm 后存 latent c_s，后 64 共享 key r_s 不旋转。主 query 每头 128+64。展开形式 k_s,h=[W_K,h c_s; r_s]，v_s,h=W_V,h c_s。对选中集合做 softmax(q·k/sqrt(192))，加权 v，再拼成 96×128=12288；乘 sigmoid(W_g x) 后才过 W_o。MLA 吸收计算只改计算顺序，逻辑 KV 仍是 576 个数。

NoPE 两条证据：`mla_use_nope=true` 与 `index_rope_dim=0`。qk_rope_head_dim=64 是保留的张量分量，不等于实际做 RoPE。SGLang `_init_dsa` 还设 `index_apply_hadamard=false`；不要照抄 DeepSeek 默认的旋转处理。

### 2.4 代价（补：从维度计算）

indexer 一个 query 扫 t 条 128 维 key，32 头点积是约 2×32×128×t FLOPs；主 attention 读取上限 k=2048，核心计算随 k 而非 t 增长。prefill 索引器仍含全体因果 query-key 对，O(n²)，核心 sparse attention O(nk)。不能说整个模型线性或 512 倍端到端提速。

1M=1048576，k/n=1/512=0.1953125%，仅是末端 query 的候选保留比例。

## 3. 代码落点

### vLLM 分派与命名

`vllm/transformers_utils/configs/kimi_linear.py::is_dsa_layer`：

```python
return (
    self.dsa_config is not None
    and (layer_idx + 1) in self.dsa_config.dsa_layers
)
```

`nvidia/model.py::KimiDecoderLayer`：先检查 is_kda_layer，否则检查 is_dsa_layer，再决定 KimiK3DSAAttention 或 dense MLA。`KimiK3DSAAttention` 传 `MLAModules(indexer=..., is_sparse=True, topk_indices_buffer=...)`。

checkpoint → vLLM：wq_b → dsa_indexer.wq_index；wk 与 weights_proj 合并到 dsa_indexer.wkv_index；k_norm → dsa_indexer.k_index_norm。融合没有改变模型数学结构。

model.py 1025 行起的 model-level buffer 复用存储；dsa.py 604 行起拒绝 index_topk_freq != 1。每个 DSA 层自己的参数、索引 KV、当前 top-k 结果；共享临时 buffer 不代表跨层复用 top-k。

### SGLang 交叉核验

`_init_dsa` 明确拒绝 freq != 1、rope_dim != 0、norm != layernorm、dsa_layers != full_attn_layers。`KimiK3MLAAttention` 调用父类并 `skip_rope=True`；父类构造通用 Indexer；输出门在 o_proj 之前。`_check_dsa_indexer_loaded` 检查每层 indexer 参数均被加载。

`Indexer._should_skip_logits_computation` 在某些 prefill / decode 路径上确认 kv_len≤2048 时可以跳过评分、保留全部有效位置。这是短上下文等价优化，不是模型没有 DSA。

## 4. 关键数字

| 数字 | 条件 | 出处 |
|---|---|---|
| 93 层、69 KDA、24 DSA、96 主头 | K3.1 Pro 发布配置 | HF config |
| 32 索引头、128 维、top-2048 | 覆盖 vLLM dataclass 默认值 16/64 | HF dsa_config |
| 7168/3584/3072、896 选 16+2 shared | LatentMoE | HF config |
| 7,438,592 参数/索引器，24 层 178,526,208 | 1536×4096 + 7168×128 + 7168×32 + 2×128；新增子模块计数，非实际权重差分 | config + indexer constructor |
| 576+128 个逻辑缓存数/token/DSA层 | 不含量化比例和页面元数据 | dsa.py |
| 656 B main + 132 B index = 788 B/token/层 | vLLM NVIDIA fp8_ds_mla + 默认 FP8 index；不是每 token 的总模型显存 | dsa_indexer.py + dsa_cache.py |
| 24 层、1M token，18.46875 GiB | 上述 cache payload 合计，未计 TP 副本、padding、KDA、权重、工作区等 | 补：788×24×1048576 / 2³⁰ |
| 68 B index | 显式请求 MXFP4 indexer 且后端支持 | dsa_cache.py；不是 model card MXFP4 权重自动意味着此格式 |
| 27 GiB → 33 GiB | 假设全部逻辑 cache 数都用 BF16，K3 main 与 K3.1 main+index；隔离稀疏结构本身 | 补：576×2×24×2²⁰ 与 704×2×24×2²⁰ |
| 414 MiB KDA recurrent state | 69×96×128×128，假设 FP32，一条请求，不计卷积及 checkpoint | 补：维度计算 |

## 5. 常见误解与澄清

- 所有层改成 DSA：错误，只有 24 个原 MLA 位置，69 KDA 未改。
- DSA 去掉 MLA：错误，DSA 在 MLA 前增加筛选；完整名称 Gated Sparse MLA 很贴切。
- top-2048 代表缓存只留 2048：错误，每次 query 可以选择不同历史位置。
- index_topk_freq=1 意味着重用前一层：错误，两引擎均解释为逐层独立。
- 原始常量默认 16头×64维：不是该公开 checkpoint，用 HF 的32×128覆盖。
- qk_rope_head_dim=64 代表有位置旋转：错误，实际开关是 NoPE。
- 2.8T 权重全是 FP4、KV 也是 FP4：错误，quantization_config.ignore 排除 attention/shared/router 等；运行时 cache 是另一套配置。
- 引用 DeepSeek 的两阶段 KL 训练作为 K3.1 配方：无证据。

## 6. 与本专栏其他文章的接口

KDA 递推/并行：K3 第1/2篇；MLA低秩及 NoPE：第3篇；AttnRes：第4篇；LatentMoE：第5篇；视觉：第6篇。新文作为第11篇，part「K3.1 更新」。DSA 与压缩块差异链接 DeepSeek V4 第2篇；跨层 reuse 对照 V4.1 第2篇。不把已有通用推导重写一遍。

## 7. 图的建议

1. K3.1 专用主干图：23组 KDA/KDA/KDA/Gated DSA + 末层 DSA，标出 FFN 与 AttnRes。不照搬含 MTP 训练头的 K3 图当当前公开 checkpoint。
2. DSA 双分支图：normalized query latent 分叉、index Q/K/weights、TopK 向 sparse MLA 提供位置；两份 cache、输出门在 W_o 前。

## 8. 待确认 / 没查到的点

截至核对日期未找到 K3.1 完整架构论文；官方 tech blog 存在。训练中何时加入 DSA、是否 KL 蒸馏、损失系数、warmup 步数、层级消融不能由 serving 代码确认。`num_nextn_predict_layers=0` 只证明此 checkpoint 主干配置未挂 next-n 层，不能证明训练从未使用 MTP。未运行巨型权重，无法从源码审阅给出准确吞吐/端到端时延。
