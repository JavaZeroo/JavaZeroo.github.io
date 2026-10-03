# V4.1 背景资料：YOCO、层维复用前作、V4-Flash 3514 B/token 复算

## 0. 一句话：这份笔记要解决什么问题

给 V4.1 连载第 1–3 篇提供三块原材料：CED 的出发点 YOCO 到底长什么样、CED 改了什么；§2.3 点名的四篇层维复用前作各复用了什么；Figure 1b 里 V4-Flash 的 3514 B/token 能否用公开代码逐项复算出来。

## 1. 一手资料清单

| 代号 | 资料 | 读了哪些部分 |
|---|---|---|
| [Y] | YOCO，arXiv 2405.05254，https://arxiv.org/abs/2405.05254 与 https://arxiv.org/html/2405.05254（NeurIPS 2024） | Abstract、§1、§2 全部（2.1/2.2/2.3，式 1–3、复杂度两张小表）、§3.1 gRet、§3.2 SWA（式 8）、§4.1–4.4、§5。附录 A–E 没读 |
| [IC] | IndexCache，arXiv 2603.12201（html） | Abstract、§2.2、§3 Overview、§3.1、§3.2（式 1）、§4.2 |
| [YI] | YOIO，arXiv 2606.06467（html）。论文里方法名叫 CLSA | Abstract、§2 全部（式 1–8、Table 1 复杂度）、§3.4 |
| [HS] | HySparse，arXiv 2602.03560（html） | Abstract、§2.4、§3.1–3.3（式 1–9、Algorithm 1）、§4.4 消融 |
| [CLA] | Brandon et al.，Cross-Layer Attention，arXiv 2405.12981（html，NeurIPS 2024）。V4.1 参考文献没给 arXiv 号，按题目找到 | Abstract、§2.1–2.3、§3.3 开头、§4 |
| [P] | V4.1 论文，本地 `drafts/deepseek-v4/refs/v41-paper.txt` | §1、§2.1、§2.2（223–237 行）、§2.3/2.3.1/2.3.2、§2.4.4、§3.2.1、§3.2.2、§4.2.1、References |
| [F] | Figure 1b，本地 `drafts/deepseek-v4/refs/v41-kv-cache.png` | 整图：V1 389,120 / V3.2 48,068 / V4-Flash 3,514 / V4.1-Flash 890 |
| [C4] | V4 官方推理代码，本地 `refs/official-model.py`、`refs/official-kernel.py` | `Attention.__init__/forward`、`Compressor.forward`、`Indexer.__init__/forward`、`ModelArgs`、`Transformer.__init__`、`act_quant`、`fp4_act_quant` |
| [CF] | `refs/v4-flash-config.json` | `compress_ratios`、`head_dim`、`qk_rope_head_dim`、`index_head_dim` |
| [C32] | V3.2 官方推理代码，本地 `refs/dsv32-inference-model.py`、`dsv32-inference-config.json` | `Indexer.__init__/forward`、`MLA.forward` 的 cache 写入段 |
| [P4] | V4 论文，本地 `refs/v4-paper.txt` | 823–830 行（KV 存储格式）、1820–1836 行（FP4 QAT） |
| [N] | 本地 `refs/notes-v4.md`、`refs/attn-csa2.md` §2.4 | KV cache 段；890 的算式 |

## 2. 核心机制

### A. YOCO

#### A.1 结构（[Y] §2，Figure 2）

- 共 L 层，前 L/2 层是 self-decoder，后 L/2 层是 cross-decoder。`X^l = Self-Decoder(X^{l-1})`，l ∈ [1, L/2]；`X^l = Cross-Decoder(X^{l-1}, K̂, V̂)`，l ∈ [L/2+1, L]。（§2 首段）
- 两部分的块布局相同（注意力 + FFN 交替，pre-RMSNorm、SwiGLU、GQA），**差别只在注意力模块**。（§2 第二段）
- self-decoder（§2.1，式 1）：`Y^l = ESA(LN(X^l)) + X^l`，`X^{l+1} = SwiGLU(LN(Y^l)) + Y^l`。ESA 是「efficient self-attention」，带因果掩码，要求推理时只占 O(1) 的 cache。论文给了两种选择：
  - gated retention（gRet，§3.1，式 4–7）：实验默认用它。递归形式 `S_n = γ_n S_{n-1} + K_n^T V_n`，推理时只存状态 S。
  - sliding-window attention（§3.2，式 8）：窗口 C，mask `B_ij = 0 if i−C < j ≤ i else −∞`。§4.2 的 YOCO_SWA 窗口是 1024。
- cross-decoder（§2.2，式 2–3）：
  - 全局 KV：`K̂ = LN(X^{L/2}) W_K`，`V̂ = LN(X^{L/2}) W_V`，`W_K, W_V ∈ R^{d×d}`。（式 2）
  - 层内：`Q̂^l = LN(X^l) W_Q^l`；`Y^l = Attention(Q̂^l, K̂, V̂) + X^l`；`X^{l+1} = SwiGLU(LN(Y^l)) + Y^l`。（式 3）
  - `Attention` 是标准多头注意力，带因果掩码，可配 GQA。

#### A.2 三个要点的直接答案

1. **投影权重是共享的一份，不是逐层一份。** 式 2 的 `W_K`、`W_V` 没有层上标；式 3 的 `W_Q^l` 有层上标。原文：「The KV caches K̂, V̂ are reused by all the L/2 cross-decoder modules」。所以 YOCO 是：一份 KV、一份 KV 投影、每层一份 Q 投影。
2. **cross-decoder 层内没有自注意力。** 式 3 每层只有一个注意力子层，它的 K、V 是 K̂、V̂，不由本层 `X^l` 产生。本层隐状态只通过 Q 进入注意力。也就是说上半层的任何 token 都不产生新的 KV，局部信息也只能从 K̂/V̂ 里读。
3. **KV 的来源是 `X^{L/2}`，即 self-decoder 最后一层的输出**，先过 LN 再投影。

#### A.3 prefill early-exit（[Y] §2.3）

- 论证：cross-decoder 只读 self-decoder 的输出，prompt token 在 cross-decoder 里的计算结果不被后面的 token 用到（后面的 token 只需要 K̂、V̂）。所以 prefill 时算完 self-decoder 就可以停，「without changing the final output」（Abstract）。这是**精确等价**，不是近似。
- 复杂度表（§2.3，注意力部分）：prefill 时间 Transformer `O(L N² D)`，YOCO `O(L N D)`。两个来源：只跑一半层；self-decoder 的注意力本身是线性的。
- 补：YOCO 的表把 L/2 和窗口常数吞掉了。[YI] Table 1 写得更细：YOCO (Dense) prefill 是 `O((L/2) W₁ N D)`，W₁ 是 self-decoder 的窗口。
- 补：最后一个 prompt token 仍要过 cross-decoder 才能出第一个输出 token，原文没单独说这一点，从式 3 可得。

#### A.4 KV 内存的账（[Y] §2.3）

- KV cache 内存：Transformer `O(L N D)`，YOCO `O((N + L) D)`。
- 展开：cache 条数是 `O(N + C·L)`，N 是全局 KV 一份，C 是 self-decoder 每层的常数（窗口大小），L 是层数。N ≫ C·L 时约等于 O(N)，即「only cache once」。
- 脚注 1 自己说明：「once」只指全局 KV，self-decoder 仍有常数大小的 cache。
- 相对 Transformer 约省 L 倍（Transformer 存 N×L 份）。

#### A.5 主要实验结论（数字见 §4 表）

- 3B 模型训到 1T / 1.6T token，下游平均分与 StableLM-3B-4E1T、OpenLLaMA-3B-v2 相当或略高（[Y] §4.1 Table 4）。
- 160M–13B 的 scaling 曲线与 Llama 式 Transformer 相当；YOCO_gRet 优于 Transformer 和 YOCO_SWA（§4.2 Figure 3）。
- 上下文扩到 1M，needle 检索接近满分（§4.3 Figure 4）。
- 推理：内存、prefill 延迟、吞吐都有数量级改善（§4.4）。

#### A.6 CED 对比 YOCO，逐条

「明说」= V4.1 论文原文有这句话；「推断」= 我把两篇论文的公式摆在一起得出，V4.1 论文没有逐条写。

| # | 项目 | YOCO [Y] | CED [P] | 性质 |
|---|---|---|---|---|
| 1 | 上半层有没有自己的局部 KV | 没有。cross-decoder 只有 cross-attention（式 3） | 有。SWA 在所有层逐层从本层 `H_l` 产生 local KV（§2.2 第三段） | CED 一侧明说；与 YOCO 的差别是推断，但 §2.2 说这「increases the computational depth of local KV generation」，等于承认这是相对 YOCO 的改动 |
| 2 | prefill 提前退出是否精确 | 精确，输出不变（Abstract、§2.3） | 不精确。decoder 的 SWA KV 需要回放；精确回放要 `n_win × L/2` 个 token，实际用 Decoder SWA Bounded Replay 只回放最后 `n_win` 个，「not mathematically equivalent」（§2.2、§3.2.2） | CED 一侧明说；差别是推断（由 #1 直接导致） |
| 3 | prefill 复杂度 | `O(LND)` 对 `O(LN²D)`（§2.3） | `O(NL) → O(NL/2 + n_win × L/2) ≈ O(NL/2)`（§2.2 末段）。多出的 `n_win × L/2` 一项正是回放 | 两边都明说 |
| 4 | 全局 KV 的投影权重 | 一份 `W_K, W_V`，全体 cross-decoder 层共享（式 2） | 式 (1) 写的是逐层：`C_l = H_{L/2} W_l^{KV}`，`Z_l = H_{L/2} W_l^Z`，「layer-dependent projection weights」（§2.2） | 两边都明说。见下方注意事项 |
| 5 | 全局 KV 的条目形态 | 每 token 一条，K 和 V 分开，标准多头（式 2–3） | 投影出的是 C（KV 条目）和 Z（压缩权重），再走 CSA2 的 compressor；K=V 共用一条 512 维 latent（§2.2 式 1、§2.3、§2.4.4） | CED 一侧明说；差别是推断 |
| 6 | 上半层怎么读全局 KV | dense 全注意力读整份 K̂/V̂（式 3） | 稀疏：indexer 选 Top-K 512 条，再和本层 SWA KV 一起做注意力（§2.3、§4.2.1） | CED 一侧明说；差别是推断 |
| 7 | 下半层是什么 | 只有高效自注意力（gRet 或 SWA），没有全局注意力（§2.1、§3） | causal encoder 自己也有全局注意力：18 层 CSA2（m=2），有自己的 main KV（3 个 Full 层）（§4.2.1） | CED 一侧明说；差别是推断 |
| 8 | 全局 KV 一共几份 | 1 份（上半层共用） | 4 个源层：encoder 3 个 Full 层（m=2）+ decoder 1 个 Full 层（m=1），合 2.5 条/token（§4.2.1；`attn-csa2.md` §2.4） | 推断（按 §4.2.1 数出来） |
| 9 | 论文自述的改进方向 | — | 「enhance both the overall KV cache capacity and the computational depth of KV generation」（§2.2 首段） | 明说。capacity 对应 #7/#8（encoder 也有全局 KV），depth 对应 #1（local KV 逐层产生） |
| 10 | YOCO 被怎么概括 | — | 「allowing the upper half of the layers to directly share the KV cache generated by the lower half」（§2.2） | 明说。这句概括与 [Y] 式 2 一致 |

注意事项（写正文时别写错）：

- **「逐层投影权重」在实际配置下只有一份在用。** 式 (1) 对所有 `l > L/2` 定义了 `W_l^{KV}`，但 §2.3.1 末段说「the decoder layer assigned to Full Mode computes its own global KV from the hidden state of the (L/2)-th layer … The Reindex and Reuse Modes are unchanged」，而 §4.2.1 的 decoder 只有第一组的第一层是 Full，其余 4 个 Reindex + 15 个 Reuse 都不产生 KV。所以 V4.1-Flash 的 decoder 实际只有一组 `W^{KV}/W^Z`（就是第 20 层的 compressor），效果上和 YOCO「一份共享 KV」相同。式 (1) 是一般形式。（推断，依据 §2.3.1 + §4.2.1；`attn-csa2.md` 第 34 行从代码得出同一结论。）`drafts/deepseek-v41/PLAN.md` 第 39 行写的「每个 decoder 层有自己的投影权重」需要按这一条收紧。
- CED 没有 cross-attention 模块，全栈因果；「encoder / decoder」只说全局 KV 由谁产生。YOCO 的 cross-decoder 也是因果的（式 3 后一句）。两者都不是 T5 式 encoder-decoder。
- YOCO 的 self-decoder 若选 SWA，结构上就是「下半层 SWA + 上半层全局」；CED 是「每层 SWA + 每层（除前两层）全局」。

### B. 四篇层维复用前作

**IndexCache（Bai et al., 2026，arXiv 2603.12201）[IC]**
复用的是 Top-K 索引。在 DSA（V3.2 的稀疏注意力）上把层分成 F（Full，保留 indexer）和 S（Shared，没有 indexer，直接用最近一个 F 层的索引集）两种（§3 Overview）。选哪些层当 F 有两种办法：training-free 的贪心搜索，按校准集 LM loss 逐个去掉 indexer（§3.1）；training-aware 的多层蒸馏，让一个 indexer 去拟合它所服务的各层注意力分布的平均（§3.2 式 1）。30B DSA 模型上去掉 75% 的 indexer，200K 上下文 prefill 1.82×、decode 1.48×（Abstract、§4.2）。与 CSA2 的差别：每层的 KV 仍各存各的，不省 main KV（V4.1 §2.3 原话「index reuse alone saves no main KV storage」）；CSA2 的 Reuse Mode 相当于 S 层，但 CSA2 另有 Reindex Mode（共享 KV、重打分）和 KV 共享，且序列维有压缩（m=2/1）。

**YOIO（Sun et al., 2026b，arXiv 2606.06467，方法名 CLSA）[YI]**
复用的是 KV 加路由索引。建在 YOCO 上：self-decoder 不变，产生一份共享 KV；在共享隐状态 H 上加一个**单头** indexer，`Q_idx = H W_idx^Q`，`K_idx = H W_idx^K`，`S_t = TopK(Q_idx K_idx^T)`（§2.1 式 2–3）；全体 cross-decoder 层用同一份索引做稀疏 cross-attention（式 4）。索引的 query 来自共享的 H 而不是各层自己的状态，所以整个上半层只算一次 top-k。用全部 decoder 层、全部头的平均注意力做蒸馏目标（§2.2 式 5–6）。128K 上下文 decode 吞吐约 7.6×、整体约 17.1×（§3.4）。与 CSA2 的差别：全网一份路由，层与层不能选不同条目（V4.1 §2.3 原话「network-wide routing sharing limits performance」）；CSA2 的 decoder 每 4 层 Reindex 一次，用本层的 indexer Q 重选；YOIO 下半层无全局注意力，序列维不压缩。

**HySparse（Gao et al., 2026，arXiv 2602.03560）[HS]**
复用的是 KV 加块级索引，来源是 full attention 层。结构是「1 个 full attention 层 + N 个稀疏层」重复（§3.1）。full 层在 FlashAttention kernel 里顺手输出块级最大注意力分数（式 3、Algorithm 1），取 TopK 块（默认 k=1024、块大小 64）给后面 N 个稀疏层用；稀疏层的块稀疏分支直接读 full 层的 KV，另有一条 SWA 分支（窗口 128）保留自己的 KV，两支用 sigmoid 门相加（§3.3 式 4–9）。消融显示 SWA 分支不能共享 KV，共享后各项掉 4–7 分（§4.4 Table 4）。80B MoE、49 层里只有 5 层 full attention，KV 存储降近 10×（Abstract）。与 CSA2 的差别：没有 indexer，选择靠真实注意力分数（oracle）；仍保留 dense full attention 层（V4.1 §2.3 原话「hybrid designs still retain full attention layers」），那几层的计算仍是 O(N²)；序列维不压缩。相同点：SWA 分支各层自留 KV，这一点 CSA2 也一样（V4.1 §2.3.1「each layer … computes its own query and SWA KV」）。

**Cross-Layer Attention（Brandon et al., 2024，arXiv 2405.12981）[CLA]**
复用的是 KV。只在一部分层算 K/V 投影，其余层直接用前面层的 KV 激活；「sharing factor」是每份 KV 被几层共用，CLA2 即相邻两层共用（§2.2）。KV cache 按 sharing factor 缩小，参数和 FLOPs 略降；核心注意力的访存不变，因为每层仍要整份读一遍（§2.3）。1B/3B 从头训练，MQA-CLA2 在 KV 减半的情况下 perplexity 变化不到 1%，CLA3 以上的折中略差于 CLA2（§4）。与 CSA2 的差别：是 dense 注意力上的纯 KV 共享，没有稀疏选择，也没有序列维压缩；共享因子 2，CSA2 的一个 Full 层带 5 层（encoder）或 19 层（decoder）。V4.1 §2.3 只把它当「some layers reuse the caches」的出处引用。

四篇都读到了，没有取不到的。

### C. V4-Flash 3514 B/token 复算

#### C.1 有出处的事实

| 事实 | 出处 |
|---|---|
| 层排布：idx 0、1 为 0（纯滑窗）；idx 2–42 偶数为 4，共 21 层 CSA；奇数为 128，共 20 层 HCA；idx 43 是 MTP 层，0 | [CF] `compress_ratios`；[N] |
| main KV 条目 512 维，K=V 共用一条；末 64 维是 RoPE 维 | [CF] `head_dim` 512、`qk_rope_head_dim` 64；[C4] `Attention.__init__`（`self.wkv = Linear(dim, head_dim)`） |
| 存储格式：RoPE 维 BF16，其余维 FP8 | [P4] 824 行「BF16 precision is used for the RoPE dimensions, while FP8 precision is applied to the remaining dimensions」 |
| FP8 的 scale 粒度：每 64 维一个，只量化非 RoPE 的 448 维 | [C4] `Compressor.forward`：`act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)`；`Attention.forward` 滑窗 KV 同一句 |
| scale 的类型：1 字节 E8M0（2 的幂） | [C4] `ModelArgs.scale_dtype = "fp8"` → `Transformer.__init__` 里 `scale_dtype = torch.float8_e8m0fnu`，`scale_fmt = "ue8m0"` |
| indexer K：128 维，Hadamard 旋转后 FP4，每 32 维一个 E8M0 scale | [CF] `index_head_dim` 128；[C4] `Compressor.forward` 的 `rotate` 分支 `fp4_act_quant(kv, fp4_block_size, True)`，`fp4_block_size = 32`；`official-kernel.py` `fp4_act_quant` 的 scale 张量 dtype `float8_e8m0fnu` |
| indexer QK「cached, loaded, and multiplied entirely in FP4」 | [P4] 1823 行 |
| 只有 CSA 层有 indexer；HCA 层不带 | [C4] `Attention.forward`：`self.indexer is not None` 时才走 indexer，否则 `get_compress_topk_idxs` |
| 参考实现里 cache 张量实际是 BF16，量化只做模拟 | [C4] 两处注释「kv could also use fp8 format, though current implementation uses bf16」。字节数是按部署格式算的，不是这份代码的实际占用 |

由此得到每条目的字节数：

- main KV 负载：448 × 1 + 64 × 2 = **576 B**
- main KV scale（按代码）：448 / 64 = 7 个 × 1 B = 7 B → **583 B**
- indexer K：128 × 4 bit = 64 B，加 128 / 32 = 4 个 × 1 B → **68 B**

#### C.2 公式

`全局 KV / token = 21 × (1/4) × (main + indexer) + 20 × (1/128) × main = 5.40625 × main + 5.25 × indexer`

#### C.3 试过的组合

| main (B) | indexer (B) | 结果 | 与 3514 的差 | 假设 |
|---|---|---|---|---|
| 576 | 0 | 3114.00 | −400.00 | 不算 indexer |
| 576 | 64 | 3450.00 | −64.00 | 都不算 scale（[N] 的旧算法） |
| 576 | 68 | 3471.00 | −43.00 | 只算 indexer 的 scale |
| 583 | 64 | 3487.84 | −26.16 | 只算 main 的 scale |
| **583** | **68** | **3508.84** | **−5.16** | **完全按代码：main 7 个 scale，indexer 4 个 scale** |
| **584** | **68** | **3514.25** | **+0.25** | **main 多 1 字节（8 个 scale 或对齐），四舍五入正好 3514** |
| 584 | 64 | 3493.25 | −20.75 | |
| 590 | 64 | 3525.69 | +11.69 | main scale 每 32 维一个（14 个） |
| 590 | 68 | 3546.69 | +32.69 | |
| 604 | 68 | 3622.38 | +108.38 | main scale 用 fp32（7 × 4 B） |
| 576 | 128 | 3786.00 | +272.00 | indexer K 用 FP8 |
| 520 | 68 | 3168.25 | −345.75 | main 全 FP8（512 + 8） |
| 520 | 130 | 3493.75 | −20.25 | main 全 FP8，indexer FP8 + 2 scale |
| 1024 | 256 | 6880.00 | +3366.00 | 全 BF16（参考实现的实际占用） |

#### C.4 结论

- **能精确对上的只有 main = 584 B、indexer = 68 B**：21 × 0.25 × (584 + 68) + 20 × (1/128) × 584 = 3423 + 91.25 = 3514.25，取整 3514。固定 indexer = 68 时，main 的整数解唯一：(3513.5 − 357) / 5.40625 = 583.86，(3514.5 − 357) / 5.40625 = 584.05，区间内只有 584。
- **584 里有 583 是有出处的**（576 负载 + 7 个 E8M0 scale）。**多出的 1 字节没有出处**，两种解释都只是猜测：
  1. scale 按 512 / 64 = 8 个槽存，RoPE 那一块也占一个 scale 位（尽管 RoPE 维是 BF16、不需要 scale）；
  2. 583 字节向上对齐到 8 字节边界得 584。
  代码里 `act_quant(kv[..., :-rd], 64, …)` 明确只对 448 维产生 7 个 scale，两种解释都无法从 [C4] 或 [P4] 证实。
- 完全按代码的组合（583 + 68）得 3508.84，差 5.16 B，相对误差 0.15%。
- 68 这一项与 V4.1 用的 indexer K 字节数相同（`attn-csa2.md` §2.4），两代模型的 indexer K 格式一致，这一项可信。

#### C.5 旁证：同一张图的另外两个数

用来确认 Figure 1b 的口径是「负载 + scale，按部署格式」：

- **V3.2 = 48,068，精确对上。** 61 层 × 788 B。788 = MLA latent 512 维 FP8（512 B）+ 4 个 fp32 scale（block 128，16 B）+ RoPE 64 维 BF16（128 B）+ indexer K 128 维 FP8（128 B）+ 1 个 fp32 scale（4 B）。出处：[C32] `MLA.forward` 的 `act_quant(kv, block_size, …)`（`block_size = 128`）与 `pe_cache`；`Indexer.__init__` 的 `k_cache` dtype `float8_e4m3fn`、`k_scale_cache` 形状 `head_dim // block_size`、dtype `float32`。这说明图里的数**确实把 scale 算进去了**，而且用的是代码里 scale 的实际类型。若 scale 按 1 字节算，得 61 × 773 = 47,153，对不上。
- **V1 = 389,120。** 95 层 × 4096 B，4096 = GQA 8 个 KV 头 × 128 维 × (K+V) 2 × BF16 2 B。这是 DeepSeek LLM 67B 的配置（95 层、8 个 KV 头），**层数与头数凭记忆，没有回原论文核对**。
- 倍率自洽：389120 / 48068 = 8.10，48068 / 3514 = 13.68，3514 / 890 = 3.95，与图上 8.1× / 13.7× / 3.9× 一致。

V3.2 能按代码精确复算而 V4-Flash 差 1 字节/条，说明 V4 的部署格式和参考实现在 main KV 的 scale 布局上有一处小出入，不是口径不同。

## 3. 代码落点

- V4 main KV 量化：`refs/official-model.py` `Compressor.forward` 末段（约 368–376 行）。`self.rotate` 为真（indexer 的 compressor）走 `rotate_activation` + `fp4_act_quant(kv, 32)`；否则 `act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)`。
- V4 滑窗 KV 量化：同文件 `Attention.forward` 约 506 行，同一句 `act_quant`，注释「rope dims stay bf16 for positional precision」。
- scale 类型：同文件 `ModelArgs`（`scale_dtype: "fp8"`）与 `Transformer.__init__`（约 776–778 行）。
- `refs/official-kernel.py`：`act_quant`（105 行起，scale 张量 `N // block_size` 个）；`fp4_act_quant`（186 行起，默认 block 32，scale dtype `float8_e8m0fnu`）。
- V3.2：`refs/dsv32-inference-model.py` 453–454 行（`k_cache`、`k_scale_cache`），570–573 行（latent 量化与 `pe_cache`）。

## 4. 关键数字

| 数字 | 条件 | 出处 |
|---|---|---|
| YOCO KV 内存 `O((N+L)D)` 对 `O(LND)` | 复杂度 | [Y] §2.3 |
| YOCO prefill `O(LND)` 对 `O(LN²D)` | 注意力部分复杂度 | [Y] §2.3 |
| KV cache 内存约省 80× | 65B 模型 | [Y] §1 |
| 整体推理内存 12.4 GB，Transformer 是它的 9.4× | 3B，1M 上下文，H100 | [Y] §4.4 |
| 32K 时内存约省 2× | 3B | [Y] §4.4 |
| prefill 180 s → 不到 6 s | 512K 上下文，对比带 Flash-Decoding 和 kernel fusion 的 Transformer | [Y] §2.3、§4.4 Figure 8 |
| prefill 加速 71.8×（1M）、2.87×（32K） | 3B | [Y] §1、§4.4 |
| 吞吐 4.5 → 43.1 token/s，9.6× | 512K | [Y] §4.4 Figure 9 |
| 1 GB 显存可服务 128K token，Transformer+GQA 只能 1.6K | 65B 规模 | [Y] §4.4 Figure 7 |
| YOCO-3B 配置：26 层，d=3072，24 个 query 头，8 个 KV 头，head dim 128，非 embedding 参数 2.8B | §4.1 实验 | [Y] §4.1 |
| YOCO-3B 平均分 0.634（1T）/ 0.636（1.6T）/ 0.645（1M 版） | LM Eval Harness 八项 zero-shot | [Y] Table 4 |
| 多针检索 0.98 / 0.98 / 0.84 / 0.56（N=1/2/4/8） | 128K，YOCO-3B-1M | [Y] Table 5 |
| CED prefill `O(NL) → O(NL/2 + n_win × L/2)` | N ≫ n_win | [P] §2.2 |
| IndexCache：去掉 75% indexer，prefill 1.82×、decode 1.48× | 30B DSA，200K，H100，SGLang | [IC] Abstract、§4.2 |
| YOIO：decode 约 7.6×、整体约 17.1× | 128K，B200，vLLM，对比 Transformer | [YI] §3.4 |
| HySparse：49 层只留 5 层 full attention，KV 存储降近 10× | 80B MoE | [HS] Abstract |
| HySparse：SWA 分支共享 KV 后 MMLU 58.4 → 52.8 | 7B dense 消融 | [HS] Table 4 |
| CLA2：KV 减半，perplexity 变化 < 1% | 1B/3B 从头训，MQA | [CLA] §4 |
| V4-Flash 全局 KV 3,514 B/token | Figure 1b | [F] |
| 3514.25 = 21/4 × (584+68) + 20/128 × 584 | 584 中 1 字节无出处 | 本笔记 C.3 |
| 3508.84 = 21/4 × (583+68) + 20/128 × 583 | 完全按代码 | 本笔记 C.3 |
| 48,068 = 61 × 788 | V3.2，fp32 scale | [F]、[C32] |

## 5. 常见误解与澄清

1. 「YOCO 的每个 cross-decoder 层有自己的 KV 投影」。不对，`W_K, W_V` 只有一份（[Y] 式 2），逐层的只有 `W_Q^l`（式 3）。
2. 「YOCO 的上半层也有自注意力」。不对，cross-decoder 每层只有一个 cross-attention（[Y] 式 3）。
3. 「CED 的 prefill 提前退出和 YOCO 一样是无损的」。不对。YOCO 无损；CED 因为 decoder 有逐层 SWA KV，要么回放 `n_win × L/2` 个 token，要么用 Bounded Replay 近似（[P] §3.2.2「not mathematically equivalent」）。
4. 「CED 每个 decoder 层各投影一份全局 KV」。式 (1) 这么写，但 V4.1-Flash 的 decoder 只有 1 个 Full 层在产生 KV（[P] §2.3.1 末段 + §4.2.1）。
5. 「YOCO 只缓存一次，没有别的 cache」。脚注 1 说了 self-decoder 还有常数大小的 cache（[Y] §1 脚注）。
6. 「YOIO 就是 YOCO」。两篇不同的论文，同一组作者。YOIO 在 YOCO 上加了共享的稀疏路由（[YI] §2）。
7. 「3514 可以完全按公开代码算出来」。差 1 字节/条，按代码是 3508.84（本笔记 C.4）。

## 6. 与本专栏其他文章的接口

- `deepseek-v41-00-overview.mdx` 已经用了 3514 和 890 两个数，并给了 890 的算式；3514 没有分项。第 3 篇若要列 3514 的分项，按 C.4 写：583 有出处，584 才能对上，差的 1 字节标为未确认。
- `deepseek-v4-07-systems.mdx` 第 61 行写滑窗 KV「每条 576 字节」，用的是不含 scale 的口径；与本笔记的 583/584 不冲突，但口径不同，引用时说明。
- `deepseek-v4-01-kv-compression.mdx`、`-02-lightning-indexer.mdx`、`-03-hybrid-attention.mdx` 已讲 CSA / HCA / indexer，3514 的分项只需引用层排布和压缩率，不重讲机制。
- 现有文章里没有讲过 YOCO，第 1 篇（CED）要自己讲清 A.1–A.4。
- B 部分四篇放第 2 篇（CSA2 跨层复用）的「前作」一节，每篇一两句即可。

## 7. 图的建议

1. YOCO 与 CED 并排的层栈图：左边 YOCO（下半 ESA、上半只有 cross-attention、一份 K̂/V̂ 从中间引出）；右边 CED（每层都有 SWA 小方块、encoder 有 3 个全局 KV 源、decoder 的全局 KV 从 `H_{L/2}` 引出）。要表达差别 #1、#7、#8。
2. prefill 路径对比：YOCO 的 prompt token 到 L/2 层就停；CED 的 prompt token 到 L/2 层停，但最后 `n_win` 个 token 继续走完 decoder。要表达差别 #2、#3。
3. 3514 与 890 的分项堆叠条：V4-Flash 分成 CSA main、CSA indexer、HCA main 三段（3066 / 357 / 91.25，按 584+68），V4.1 分成 encoder、decoder 两段。未确认的 1 字节在图注里说明。

## 8. 待确认 / 没查到的点

- V4-Flash main KV 条目的第 584 个字节是什么（第 8 个 scale 槽，还是对齐填充）。V4 论文与官方代码都没有写。
- V1 = 389,120 的拆法（95 层、GQA 8 个 KV 头、head dim 128、BF16）凭记忆，没有回 DeepSeek LLM 论文核对。
- YOCO 附录 A（chunk parallelism）、B（gRet 等价性证明）、C–E（超参）没读。
- CED 相对 YOCO 的差别里，标「推断」的各条在 V4.1 论文中没有逐条对照的原文，只有 §2.2 首段一句概括（「KV cache capacity」与「computational depth of KV generation」）。
- [CLA] 的 arXiv 号 2405.12981 是按题目匹配的，V4.1 参考文献只给了 NeurIPS 37 的页码。
