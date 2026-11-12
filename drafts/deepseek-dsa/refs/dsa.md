# DeepSeek DSA 一手调研与补充推导

调研日期：2026-11-12。

## 0. 一句话：这篇要解决什么问题

从“在有限计算预算下保留多少注意力质量”推导 token 选择、便宜的 lightning indexer、教师分布蒸馏和两阶段继续训练。推导属于解释，不代表论文公开了作者真实的发现顺序。

## 1. 一手资料清单

- S1：DeepSeek 发布公告，2025-09-29，https://www.deepseek.com/en/news/v3-2-exp/ 。确认首次发布版本与日期。
- S2：DeepSeek-V3.2 报告，arXiv:2512.02556v1（2025-12-02），https://arxiv.org/html/2512.02556v1 ，精读 §2.1–2.3、§3 开头和 Appendix A。式 (1)–(4) 分别为索引打分、稀疏注意力、全局 KL、选中集合 KL。
- S3：官方 V3.2-Exp 推理代码，commit `87e509a2e5a100d221c97df52c6e8be7835f0057`（2025-11-18），https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/87e509a2e5a100d221c97df52c6e8be7835f0057/inference/model.py ，读 `rotate_activation`、`Indexer`、`MLA`。HF 镜像当日 revision `194c67e12b1b0d6df0ef373ddcf215bc84027409`。
- S4：同 commit 的 `inference/config_671B_v3.2.json`，https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/87e509a2e5a100d221c97df52c6e8be7835f0057/inference/config_671B_v3.2.json 。
- S5：同 commit 的 `inference/kernel.py`，`fp8_index_kernel`、`fp8_index`，https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/87e509a2e5a100d221c97df52c6e8be7835f0057/inference/kernel.py 。
- S6：官方 README 同 commit，RoPE 修正公告、核函数入口，https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/87e509a2e5a100d221c97df52c6e8be7835f0057/README.md 。
- S7：FlashMLA 官方 FP8 sparse decode 技术文，2025-09-29；固定旧版 commit `ba89a3466e9470ad08ab39738d4e7bb66989e1e7`，https://github.com/deepseek-ai/FlashMLA/blob/ba89a3466e9470ad08ab39738d4e7bb66989e1e7/docs/20250929-hopper-fp8-sparse-deep-dive.md 。精读缓存格式、MQA 维度与共享反量化。
- S8：vLLM 团队首发实现记录，https://vllm-project.github.io/2025/09/29/deepseek-v3-2.html ，重点 indexer cache、分页、批处理 mask、logits materialization；仅作为 2025-09 历史实现说明，不当作当前安装指南。
- S9：FlashAttention 原论文，https://arxiv.org/abs/2205.14135v2 ，确认 dense FlashAttention 是 exact/IO-aware，与近似选择互补。
- S10：FlashMLA 当前 README，https://github.com/deepseek-ai/FlashMLA ，2026-11-12 查到 2026-09-30 breaking change：主分支不再支持 V3/V3.2/V4.0/Hopper，指向 S7 commit 使用旧格式。文章固定旧版引用。

## 2. 核心机制与推导

### 2.1 输出误差与共享 top-k（补充推导）

对固定 query/head，a_s≥0 且 Σa_s=1，u=Σa_s v_s。选非空 S，ε=Σ_{s∉S}a_s，稀疏重归一化输出 u_S=Σ_{s∈S}a_s v_s/(1-ε)。则 u=(1-ε)u_S+εu_drop。如果每个 ||v_s||≤M，由三角不等式得 ||u-u_S||≤2Mε。ε=0 时误差为零，无需定义 u_drop。

多头共享 S，p_s=(1/H)Σ_h a_{h,s}，平均丢弃质量为 1-Σ_{s∈S}p_s。固定 |S|=k，最小化该量的精确解是 top-k(p)。只优化这一代理目标，不保证模型最终损失最优；不同头的 value 范数和输出投影未被等权教师编码。

依据 S2 的全头聚合教师构造和 S3 的共享 top-k，以上是额外数学解释，不是论文定理。

### 2.2 ReLU 与索引器（S2/S3；非线性解释为补充）

官方 I_ts=Σ_j w_tj ReLU(q_tj·k_s)，64 query heads ×128 dims，单份 key，无 value 路径。若去掉 ReLU，对固定 t 有 Σ_j w_tj(q_tj·k_s)=(Σ_j w_tj q_tj)·k_s，多头退化成一个点积。ReLU 让不同 key 激活不同头，形成分段线性打分。官方只明确说 ReLU 出于吞吐考虑；不能声称数学唯一最优。权重来自线性层，可负；I 也可负，非概率。

### 2.3 KL 与排序质量（补充推导）

r=softmax(I)，top-k(r)=top-k(I)。KL(p||r) 的 logit 梯度是 r-p。令 S*=top-k(p)，S_r=top-k(r)，利用 r(S*)≤r(S_r)，得 p(S*)-p(S_r)≤||p-r||_1≤sqrt(2 KL(p||r))（自然对数、Pinsker）。仅对同一完整因果支持集的分布成立，不可套到阶段 B 的条件分布 KL；后者不观察被排除的候选。

### 2.4 训练（S2）

从已扩展 128K 的 V3.1-Terminus base 继续训练。阶段 A 保持 dense/freeze 主干，只优化 indexer 的全局 KL。阶段 B 开启 top-k，主干按 LM loss 优化，indexer 按所选集合上的 KL 优化。indexer 输入 detach；教师分布也 stop-gradient 才满足论文“主干仅受 LM loss”的约束。共享 query latent 同样要在 indexer 分支 detach。

S2 只给了目标和阶段参数，没公开完整训练器、分块 teacher kernel、优化器细节。正文的 CPU 教学脚本与大规模实现建议必须明确标为自行实现。

## 3. 代码落点

- S3 `Indexer.__init__/forward`：q 从主干 `qr`（1536）投影到 64×128；k 从 hidden（7168）投影到128，LayerNorm；前64维 RoPE，非交错；Hadamard 归一化旋转后 FP8 量化；`weights_proj` 无概率归一化。
- S3 `MLA.forward`：`qr=q_norm(wq_a(x))`；prefill 展开192维 K/128维 V；decode 吸收后 QK 为512+64，AV 为512。两分支的演示实现都先算全量 score 再 mask，不构成真正的稀疏加速基准。
- S5 `fp8_index_kernel`：FP8 GEMM → FP32 ReLU → 逐头权重 → 头求和 → key scale。注意 FP8 指乘法输入/缓存，不是所有中间值。
- S3 top-k：先因果 mask；每行合法位置不足 k 时，批量 `topk(min(k,end_pos))` 可能包含 -inf 占位，主注意力还要保留 causal mask。不能把“先 mask”写成“返回数组绝不会含未来下标”。
- S6 README：2025-11-17 修复 indexer RoPE 布局；MLA 与 indexer 的 RoPE interleaving 不同。

## 4. 关键数字

| 数字 | 条件 | 出处 |
|---|---|---|
| H_I=64、d_I=128、k=2048 | 每层 indexer | S4 |
| hidden=7168、query latent=1536、KV latent=512、RoPE=64、主头128 | V3.2-Exp | S4 |
| 热身1000步 ×16×131072 = 2,097,152,000 token；lr 1e-3 | dense 只训索引器 | S2 §2.1.1 |
| 稀疏15000步 ×480×131072 = 943,718,400,000 token；lr 7.3e-6 | LM 与 KL 分别优化 | S2 §2.1.1 |
| 64倍 | 128Ki token 的最后一行，131072/2048，仅主注意力条目比 | S4，推算 |
| 约32.25倍 | 全因果前向配对数：N(N+1)/2 ÷ [k(k+1)/2+(N-k)k]，N=131072 | 推算 |
| 约36.51 GFLOP dense vs 0.570 GFLOP sparse main，另2.147 GFLOP indexer | 单层/单decode query、N=131072、MQA吸收形式、乘加2 FLOPs，省略投影/选择/量化 | S3/S4/S7，推算 |
| indexer key 128B +4B scale =132B | 参考缓存格式，每层每token | S3 |
| 主缓存656B，额外索引132B，合788B，增加约20.1% | S7 FP8主缓存格式，排除page对齐/MTP/复制 | S3/S7 |

## 5. 常见误解与澄清

- DSA 全部线性：错误；indexer 全上下文扫描仍有 O(N²) 项。
- 改 mask 就能省算：参考实现方便看语义；生产 kernel 必须不访问未选位置。
- DSA 省总 KV：选择不等于淘汰，后续 query 可能选回旧位置；FP8 压缩是另一技术。
- top-k 可以由 LM 梯度直接训练：离散下标不传这种梯度；该方案用独立 KL。
- 教师是把 raw logits 加起来：目标要非负归一化，应由每头 softmax 概率聚合。
- 稀疏 KL 很低意味着全局召回高：错误；被漏掉的候选不在该损失支持集内。
- 旧 refs 把 decode 写成 192+128 FLOPs：那是 expanded MHA 数字；本文用576+512计算 absorbed MQA decode。

## 6. 与本专栏其他文章的接口

- `/2026/10/22/llm-attention-kv-cache/`：缓存、prefill/decode、通用注意力基础。
- `/2026/09/23/deepseek-attention-structures/`：MLA 的吸收形式，读者可回顾。
- `/2026/09/05/deepseek-v4-02-lightning-indexer/`：后续把索引对象改成压缩条目，本篇保持 V3.2 的逐 token 选择。

## 7. 图的建议

1. 主干/索引器两路张量图，标各自缓存和共享选择集合。
2. 合成注意力的质量/误差交互：top-k 与漏掉最大项对照，k=1…8；不冒充模型实测。
3. 两阶段训练图，stop-gradient 和优化器边界分明。

## 8. 待确认 / 没查到的点

- 官方没有公开完整 DSA 训练器、教师概率的具体分块实现、辅助损失权重和完整优化器 schedule；不补造。
- 不用固定误差界推导真实模型的最优 k=2048；它仍需质量/延迟/吞吐实验选定。
- 官方未给“ReLU 比所有其他激活更优”的系统消融；只可解释非线性作用与吞吐动机。
- 当前库接口已演进；正文固定 V3.2 时代代码与 compatible kernel revision，不提供未经运行的最新部署命令。

## 9. MLA 到 DSA 的教学顺序（2026-11-12）

- 先展示 dense MLA 的全部候选访问，再引入便宜预测器；不能先用完整主 attention 排序后把已付出的计算成本算作节省。
- qI 的输入是现成 query latent；w 的输入为当前 hidden；key 的输入为各候选 hidden。官方 Indexer 的 wq_b 是自己的参数，不是直接复用主 MLA 的 query 展开权重。
- 教师 p、预测 r 都沿当前 query 的历史位置分布；主 LM 输出才沿词表分布。参数共享带来跨输入的选择能力，不学习固定 token 位置排名。
- 本次统一合成例子：主头 [0.05,0.8,0.1,0.05] 与 [0.05,0.6,0.3,0.05]，教师均值 [0.05,0.7,0.2,0.05]；索引 logits 从全零开始。r2−p2=−0.45，梯度下降抬高 I2。
- 交互直接更新示意 logits，不能称为真实 indexer 网络训练，也不能把对应 teacher/指标当模型实测。
