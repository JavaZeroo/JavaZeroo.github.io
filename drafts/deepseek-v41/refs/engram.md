# Engram：DeepSeek-V4.1-Flash 的条件记忆模块（参考笔记）

标「补」的是我自己补的推导或手算，不是资料原话。出处缩写见第 1 节（[E] Engram 论文，[D] Engram 官方 demo，[P] V4.1 论文，[M] model.py，[H] engram.py，[C] config）。

## 0. 一句话：这篇要解决什么问题

Transformer 没有「查表」这个原语，认出「Diana, Princess of Wales」这类固定搭配要耗掉前几层的注意力和 FFN；Engram 用以当前 token 结尾的 2/3/4-gram 做哈希，$O(1)$ 地从一张 196B 参数的表里取出向量，经过一个由残差流决定的门加回残差流。这篇要讲清：哈希到底怎么算、门的公式、V4.1 相对原版改了什么、196B 参数怎么训练和存放。

## 1. 一手资料清单

| 缩写 | 资料 | 版本 | 读了哪些 |
|---|---|---|---|
| [E] | Engram 论文 "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models"，https://arxiv.org/abs/2601.07372 | arXiv **v2**（取的是 `arxiv.org/html/2601.07372`，页面内标 v2） | 全文：§1–§8、附录 A（Table 5）、附录 C（Table 6）、Table 1/2/4。Figure 3/5/6 的曲线数值只读了正文里报出来的数，没有读图 |
| [D] | 官方仓库 https://github.com/deepseek-ai/Engram | commit `fb7f84a21f91223715394a33a1dc24bbfb7f788e` | `engram_demo_v1.py` 全文、`README.md`。仓库只有这一个 demo 文件，自述「demo purpose only」，Attention/MoE/mHC 都是 mock |
| [P] | V4.1 论文本地文本 `drafts/deepseek-v4/refs/v41-paper.txt` | 本地存档 | §2.4.2（305–309 行）、§2.5（325–389 行，含 Algorithm 1 和式 7）、§3.1.3（439–441 行）、§4.2.1–4.2.2（502–508 行）。全文 grep `engram`，§3.2 推理系统一节没有再提 Engram |
| [M] | V4.1 推理代码 `drafts/deepseek-v4/refs/v41-inference-model.py` | 本地存档 | `ParallelEngramEmbedding`（296–325）、`Engram`（328–365）、`Block.__init__`（930–932）、`Transformer.__init__/forward`（1197–1272）、`Linear`（210–） |
| [H] | `inference/engram.py`，https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/inference/engram.py | 2026-10-02 取，HF repo sha `2cba9e42aa026125f3ed06c6d98c1db82f7ca027`；已存到 `drafts/deepseek-v4/refs/v41-inference-engram.py`（185 行） | 全文 |
| [C] | `drafts/deepseek-v4/refs/v41-config.json`、`v41-inference-config.json` | 本地存档 | `engram_*` 字段 |
| [I] | `drafts/deepseek-v4/refs/v41-index.json`（safetensors 权重索引）、`v41-inference-convert.py` | 本地存档 | `engram` 相关的 12 个 key；convert 的分片逻辑（135–141 行） |

## 2. 核心机制与推导

### 2.1 动机：条件记忆是和 MoE 并列的另一条稀疏轴

- [E §1]：语言建模有两类子任务，组合推理和知识检索。命名实体、套话这类文本是「local, static, highly stereotyped」的，天然适合查表；标准 Transformer 没有查表原语，只能「simulate retrieval through computation」。例子是 [E Table 3]（转引自 Ghandeharioun et al. 2024）：模型在第 1–2 层只认出 "Wales"，第 4–5 层认出 "Princess of Wales"，第 6 层才认出 "Diana, Princess of Wales"。
- MoE 是 conditional computation（按 token 稀疏激活参数去算），Engram 是 conditional memory（按 token 稀疏查表取静态向量）。[E §1]
- 模块分两个阶段：retrieval（§2.2）和 fusion（§2.3）。只放在少数几层，输入 embedding 和输出头不动。[E §2.1, Figure 1 caption]

### 2.2 Tokenizer compression

- [E §2.2]：子词 tokenizer 追求无损还原，语义相同的词会有不同 id（`Apple` 与 `␣apple`）。预先算一个满射 $\mathcal{P}: V \to V'$，把按「NFKC、小写等」规范化后文本相同的 token 并成一个 id。128k 词表压缩 **23%**（附录 C 给的精确值 23.43%）。之后 n-gram 取在压缩 id 上：$x'_t = \mathcal{P}(x_t)$，$g_{t,n} = (x'_{t-n+1}, \dots, x'_t)$。
- 规范化的具体步骤论文没列全，代码里有 [D `CompressedTokenizer`] [H `build_compressed_token_map`]，两份实现的 normalizer 序列逐项相同：
  1. `NFKC` → 2. `NFD` → 3. `StripAccents` → 4. `Lowercase` → 5. 把 `[ \t\r\n]+` 换成一个空格 → 6. 如果整个 token 就是一个空格，先换成私用区字符 `` 占位 → 7. `Strip` 去首尾空白 → 8. 把占位符换回空格。
  - 第 6–8 步的作用（[H] 注释）：纯空白 token 经过 Strip 会变成空串，和别的 token 并到一起，所以先保护起来。结果是所有纯空白 token（`\t`、`\n`、`␣␣`、`\n\n`…）并成同一个 `' '`，[E Table 6] 里它是合并数第一的类，163 个 token 并成 1 个。
  - 逐 token `decode([id])`；如果解出来含 `�`（不完整的 UTF-8 字节 token），不做规范化，用 token 的原始字符串当 key。规范化后为空串的，用原文本当 key。
  - 新 id 按首次出现的顺序从 0 编号，压缩后词表大小 = 不同 key 的个数。
- [E Table 6] 前 5 个合并类：`' '`（163）、`'a'`（54，含 `A`、`␣a`、`á`、`ä`、`ą`…）、`'o'`（40）、`'e'`（35）、`'i'`（30）。
- V4.1：压缩后大小写死在 config 里，`engram_compressed_vocab_size = 99092` [C]；[H `NgramHashState.__init__`] 启动时重新建表并 `assert vocab_size == args.engram_compressed_vocab_size`，注释说明原因：每个哈希乘子都由这个数推出，对不上就等于把整张表重新哈希了一遍。
  - 补：99092 / 129280 = 76.65%，压缩 23.35%，和 [E] 的 23.43%（V3 tokenizer）接近但不相同。[H] 用的是 `len(tokenizer)`（含 added tokens）而不是 config 的 `vocab_size = 129280`，V4.1 tokenizer 的 `len` 我没实测，见第 8 节。
  - V4.1 与 demo 的一个小差别：[H] 用 `tokenizer.backend_tokenizer.decode`（Rust 后端，注释说和训练时一致，没有 `clean_up_tokenization_spaces`），[D] 用 `tokenizer.decode`。

### 2.3 Multi-head hashing：哈希函数的具体形式

论文只给了形式 [E 式 1–2]：

$$z_{t,n,k} \triangleq \varphi_{n,k}(g_{t,n}),\qquad \mathbf{e}_{t,n,k} = \mathbf{E}_{n,k}[z_{t,n,k}]$$

$$\mathbf{e}_t \triangleq \big\Vert_{n=2}^{N} \big\Vert_{k=1}^{K} \mathbf{e}_{t,n,k}$$

每个阶 $n$ 有 $K$ 个哈希头，每个头有自己的表 $\mathbf{E}_{n,k}$，大小是素数 $M_{n,k}$；$\varphi_{n,k}$ 是「lightweight multiplicative-XOR hash」。多头的目的是缓解碰撞（引 Tito Svenstrup et al. 2017）。

具体形式只在代码里。[D `NgramHashMapping._get_ngram_hashes`] 与 [H `NgramHashState.forward`] 算的是同一个函数。记压缩 id 为 $x'_t$，第 $\ell$ 个 Engram 层有一组乘子 $m^{(\ell)}_0, \dots, m^{(\ell)}_{N-1}$（$N$ = 最大阶，V4.1 为 4）：

$$\text{mix}^{(\ell)}_{t,n} = \bigoplus_{i=0}^{n-1} \big(x'_{t-i} \cdot m^{(\ell)}_i\big),\qquad z^{(\ell)}_{t,n,k} = \text{mix}^{(\ell)}_{t,n} \bmod p^{(\ell)}_{n,k},\qquad \text{row} = z^{(\ell)}_{t,n,k} + \text{offset}^{(\ell)}_{n,k}$$

$\oplus$ 是按位异或，运算在 int64 上做。几点从代码读出的事实：

1. **乘子** [H `compute_hash_multipliers`]：每层一个独立的 numpy RNG，种子 `10007 * layer_id`（V4.1 是 10007 和 140098）；抽 `max_ngram_size` 个 $[0, B)$ 内的整数 $r$，乘子 $= 2r+1$（保证是奇数）。上界 $B = \lfloor \lfloor (2^{63}-1)/V' \rfloor / 2 \rfloor$，$V'$ 是压缩词表大小。这样 $x' \cdot m < V' \cdot 2B \le 2^{63}-1$，乘积不会溢出 int64。
   - 补：$V' = 99092$ 时 $B = 46539438283891 \approx 2^{45.4}$。所以 99092 进了乘子的上界，这就是 assert 的原因。
   - [D] 里种子是 `seed + 10007 * layer_id`，默认 `seed = 0`，和 [H] 一致。
2. **同一阶的 8 个头共用同一个 mix，只是模不同的素数**。所谓「8 个哈希函数」是同一个 64 位混合值对 8 个互不相同的素数取模。[H 183 行 `rolling.unsqueeze(-1) % self.primes[:, i - 1]`]
3. **阶之间是滚动的**：2-gram 的 mix 再异或上 $x'_{t-2} m_2$ 就是 3-gram 的 mix，再异或 $x'_{t-3} m_3$ 是 4-gram 的。[H 177–183 行]
4. **顺序敏感**：每个回看距离 $i$ 的乘子不同，所以 (A,B) 和 (B,A) 的哈希不同。（补）
5. **两层的哈希不同**：乘子按层的种子独立抽，素数也不重复（下一条），同一个 n-gram 在第 1 层和第 14 层落到不相关的行。[H `compute_hash_multipliers` docstring]
6. **素数怎么选** [H `EngramLayout.from_args` + `find_next_prime`]：对每层、每个阶，从 `engram_vocab_size - 1 = 15999999` 开始往上找「下一个还没被用过的素数」，连找 8 个；`seen` 集合跨阶、跨层共用，所以 48 个素数互不相同，依次递增。我照此重算的结果（补，手写 Miller–Rabin）：
   - 第 1 层：2-gram `16000057, 16000079, 16000081, 16000097, 16000121, 16000129, 16000133, 16000183`；3-gram `16000189 … 16000321`；4-gram `16000339 … 16000463`。24 个之和 = **384006168**。
   - 第 14 层：2-gram `16000477 … 16000627`；3-gram `16000667 … 16000769`；4-gram `16000781 … 16000889`。24 个之和 = **384016682**。
   - 两个和恰好等于 config 的 `engram_num_embeddings` [C]。所以表的行数不是独立的超参，是 24 个素数之和；第 14 层比第 1 层多 10514 行，只是因为素数被第 1 层用掉了要往后找。
7. **一层一张物理表**：24 个 (阶, 头) 各占表里连续的一段，`offsets` 是素数的前缀和 [H 149 行]，行号加上 offset 后去同一张 `ParallelEngramEmbedding` 里取。[D `MultiHeadEmbedding`] 是同样的做法。论文记号里的 24 张表 $\mathbf{E}_{n,k}$ 在实现上是一张表的 24 个不相交区间。
8. **序列开头的 pad** [H 171–174 行]：位置 $t$ 往回看 `shift` 步，如果 $t < \text{shift}$，那一格填 `pad_id`。`pad_id` 是原始 token id 2（`engram_pad_token_id` / `engram_pad_id`）经过压缩表之后的 id [H 147 行]。所以第 0 个 token 的 2/3/4-gram 是 (x₀, pad)、(x₀, pad, pad)、(x₀, pad, pad, pad)，照常查表。[D] 用 `np.pad` 实现，行为相同。
9. **图像 token**（V4.1 新增，[D] 没有）：[M 1249–1252 行] `image_mask = token_types >= 0`，`engram_mask = ~image_mask`。[H 165–174 行] 图像位置的压缩 id 记成 `DEAD = -1`；回看时一旦碰到 DEAD，`blocked` 置真并保持（`blocked = blocked | …` 是累积的），这一格和更远的格全部填 pad。所以 n-gram 不会跨过图像 span：图像后面第一个文本 token 的 2-gram 是 (x, pad)，等同于序列开头。图像 token 自己的位置算出来全是 pad，但 [M 363–364 行] 把它的门强制成 0，不写入残差流。
10. **decode 时的状态** [H 155–157, 167, 172 行]：`cache` 是 `[max_batch_size, max_seq_len]` 的 int64 buffer，存每个位置的压缩 id（图像位置存 −1）。每次 forward 把新来的 token 写到 `start_pos` 处，再用 `gather` 取回看的 3 个位置。所以跨 prefill/decode 需要的状态只是前 3 个 token 的压缩 id（以及它们是不是图像）；实现上存了整条序列。哈希只依赖 token id，不依赖任何隐状态。

V4.1 的数：阶 {2,3,4}，每阶 8 头，每头 256 维 → 每阶 2048 维（[P §2.4.2] 「a total embedding dimension of 2048 per order」），$d_\text{mem} = 3 \times 8 \times 256 = 6144$ [M 344–345 行]。每头约 16M 行，「table sizes chosen to be distinct primes」[P §2.4.2]。

### 2.4 Context-aware gating

[E §2.3]：查到的 $\mathbf{e}_t$ 是与上下文无关的先验，会有哈希碰撞和一词多义带来的噪声。用当前隐状态 $\mathbf{h}_t$（已经过前面的注意力，带全局上下文）当 query，$\mathbf{e}_t$ 投出 key 和 value：

$$\mathbf{k}_t = \mathbf{W}_K \mathbf{e}_t,\qquad \mathbf{v}_t = \mathbf{W}_V \mathbf{e}_t \tag{E.3}$$

$$\alpha_t = \sigma\!\left(\frac{\text{RMSNorm}(\mathbf{h}_t)^\top\, \text{RMSNorm}(\mathbf{k}_t)}{\sqrt{d}}\right) \tag{E.4}$$

$$\tilde{\mathbf{v}}_t = \alpha_t \cdot \mathbf{v}_t$$

论文的解释：查到的记忆和当前上下文矛盾时 $\alpha_t \to 0$，把噪声压掉。RMSNorm 是为了梯度稳定（引 Dehghani et al. 2023）。

**代码比式 (E.4) 多一步带符号的平方根**。[D `Engram.forward`]：

```python
gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(backbone_config.hidden_size)
gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
gate = gate.sigmoid().unsqueeze(-1)
```

即 $\alpha = \sigma\big(\text{sign}(s)\sqrt{\max(|s|, 10^{-6})}\big)$，$s$ 是式 (E.4) 里 sigmoid 的自变量。这一步论文正文没写，官方 demo 和 V4.1 推理代码 [M 361–362 行，注释 "signed sqrt before the sigmoid, matching the training kernel"] 都有。所以它**不是 V4.1 的改动**，原版就有。

补：$s$ 的量级。RMSNorm 增益为 1 时，$\text{RMSNorm}(\mathbf{h}) = \mathbf{h}/\text{rms}(\mathbf{h})$，范数为 $\sqrt{d}$，所以

$$s = \frac{\sqrt{d}\cdot\sqrt{d}\cdot\cos\theta}{\sqrt{d}} = \sqrt{d}\,\cos\theta$$

$d = 5120$ 时 $s \in [-71.6, 71.6]$。两个无关的随机向量 $\cos\theta \sim \mathcal{N}(0, 1/d)$，$s \sim \mathcal{N}(0,1)$，这是除以 $\sqrt{d}$ 的用意。带符号平方根把自变量的范围压到 $[-8.46, 8.46]$，$|s| > 1$ 时变小、$|s| < 1$ 时变大。$s = 0$ 时 clamp 让自变量为 $+10^{-3}$，门 ≈ 0.5。论文没有解释为什么加这一步。

### 2.5 短因果卷积（原版有，V4.1 去掉）

[E §2.3 式 5]：为了「expand the receptive field and enhance the model's non-linearity」，对门控后的值序列 $\tilde{\mathbf{V}} \in \mathbb{R}^{T \times d}$ 做一个短的 depthwise 因果卷积，核大小 $w = 4$，dilation $\delta$ = 最大 n-gram 阶，SiLU 激活：

$$\mathbf{Y} = \text{SiLU}\big(\text{Conv1D}(\text{RMSNorm}(\tilde{\mathbf{V}}))\big) + \tilde{\mathbf{V}} \tag{E.5}$$

然后 $\mathbf{H}^{(\ell)} \leftarrow \mathbf{H}^{(\ell)} + \mathbf{Y}$，再过本层的 Attention 和 MoE。

- [D `ShortConv`]：`nn.Conv1d(groups=total_channels, kernel_size=4, dilation=max_ngram_size, padding=(k-1)*dilation, bias=False)` 后截掉右边多出来的部分（因果）；4 条 mHC 流各有自己的 RMSNorm，通道拼成 `hidden_size * hc_mult` 一起卷。
- 卷积参数初始化为零，训练开始时 $\mathbf{Y} = \tilde{\mathbf{V}}$（[E §4.1]「to strictly preserve the identity mapping at the start of training」；注意保持恒等的是卷积这一支，不是整个 Engram）。
- 消融 [E §6.2]：「removing the lightweight depthwise convolution only marginally degrades performance」。
- V4.1 [P §2.4.2]：「we omit the short causal convolution because its performance gains do not justify the added complexity in our inference stack」。[M `Engram.forward`] 里确实没有卷积，输出就是 `h + gate * value`。
- 补：去掉卷积后，位置 $t$ 的 Engram 输出只依赖 $x_{t-3..t}$ 和 $\mathbf{h}_t$，不依赖相邻位置的 Engram 输出。原版 dilation 3、核 4 的卷积要回看 $t-3, t-6, t-9$ 三个位置门控后的值，decode 时就得缓存它们；去掉后 Engram 在 decode 时除了 3 个 token id 之外没有状态。这是我对「added complexity in our inference stack」的解读，论文没展开。

### 2.6 Multi-branch integration：和 mHC 的结合

[E §2.4]：主干用多分支残差（mHC，$M = 4$）。参数共享策略：**一张表和一个 $\mathbf{W}_V$ 在 $M$ 条分支间共用，$M$ 个不同的 $\mathbf{W}_K^{(m)}$ 让每条分支有自己的门**：

$$\alpha_t^{(m)} = \sigma\!\left(\frac{\text{RMSNorm}(\mathbf{h}_t^{(m)})^\top\, \text{RMSNorm}(\mathbf{W}_K^{(m)} \mathbf{e}_t)}{\sqrt{d}}\right),\qquad \mathbf{u}_t^{(m)} = \alpha_t^{(m)} \cdot (\mathbf{W}_V \mathbf{e}_t) \tag{E.6}$$

论文特意说明这样设计可以把 1 个 $\mathbf{W}_V$ 和 $M$ 个 $\mathbf{W}_K^{(m)}$「fused into a single dense FP8 matrix multiplication」。

所以「每条流一个 key、value 共用」是**原版 Engram 就有的设计**，V4.1 是沿用（[P §2.4.2] 列出沿用的四项里有 multi-branch integration）。V4.1 代码里这个融合后的矩阵就是 `wkv`：`Linear(6144, 5120 × 5)`，输出切成 `[4 × 5120, 5120]` [M 345, 353–355 行]。[D] 里还是分开的 `key_projs`（4 个 `nn.Linear`）和 `value_proj`。

消融 [E §6.2]：「w/o multi branch」的做法是保留 mHC 主干，但只对 pre-mapping $\mathcal{H}^{pre}$ 之后的单个隐状态做一次 Engram 融合；这是回退最大的三项之一。

门的可解释性 [E §6.5 脚注]：Engram-27B 每个 token 有 2 层 × 4 流 = 8 个门，「not every branch encodes interpretable activation patterns」，Figure 7 是挑出来的。门在多 token 实体和套话的**最后一个 token** 上亮（"Alexander the Great"、"the Milky Way"、"By the way"、"Princess of Wales"、「四大发明」「张仲景」），因为查的是以 $t$ 结尾的后缀 n-gram。

### 2.7 V4.1 的门：公式与代码逐项对应

[M `Engram.forward` 350–365 行]。记第 $m$ 条流在位置 $t$ 的向量为 $\mathbf{h}^{(m)} \in \mathbb{R}^d$，$d = 5120$，$\mathbf{k}^{(m)}$ 是 `wkv` 输出的第 $m$ 段，$\mathbf{v}$ 是最后一段，$\mathbf{w}^{(m)} = \texttt{q\_weight}^{(m)} \odot \texttt{k\_weight}^{(m)}$：

$$s^{(m)} = \frac{\sum_j h^{(m)}_j\, w^{(m)}_j\, k^{(m)}_j}{\sqrt{\text{mean}(\mathbf{h}^{(m)2}) + \epsilon}\;\sqrt{\text{mean}(\mathbf{k}^{(m)2}) + \epsilon}\;\sqrt{d}}$$

$$\alpha^{(m)} = \sigma\Big(\text{sign}(s^{(m)})\sqrt{\max(|s^{(m)}|, 10^{-6})}\Big),\qquad \mathbf{h}^{(m)} \leftarrow \mathbf{h}^{(m)} + \alpha^{(m)}\, \mathbf{v}$$

- 这和式 (E.6) 是同一个式子：`q_weight`、`k_weight` 是 query 和 key 两个 RMSNorm 的逐元素增益（对应 [D] 的 `norm2`、`norm1`，每条流各一份），因为只以乘积出现，代码把它们先乘起来（[M 356 行] 注释 "only ever used as a product"）。两个增益形状都是 `[hc_mult, dim]`，初值 1。
- 归一化是每个 (token, 流) 在 `dim` 上各自做，不是 4 条流联合做（[M 358 行] 注释）。
- $\epsilon$ = `norm_eps`；门在 float32 里算，结果转回输入 dtype。
- 图像 token：`gate.masked_fill(~token_mask, 0)`。
- `wkv` 没有 bias（[M `Linear`] 默认 `bias=False`；[I] 里只有 `wkv.weight` 和 `wkv.scale`），FP8 权重。

### 2.8 V4.1 里 Engram 接在哪

[M `Transformer.forward` 1261–1267 行]：

```python
for i, layer in enumerate(self.layers):
    if layer.engram is not None:
        h = layer.engram(h, engram_hashes[:, :, layer.engram.layer_hash_index, :], engram_mask)
    if i in self.target_layer_ids:
        main_hiddens.append(h.mean(dim=2))
    h, pre_mix = layer(h, start_pos, pre_mix, image_mask)
```

- 哈希在 forward 一开始、embedding 查表之前就对两层一起算好（1252 行），形状 `[B, L, 2, 24]`。
- Engram 直接改 4 条残差流 $X_l$ 本身（`h` 的形状 `[B, L, 4, 5120]`），在 block 的读算子之前，不经过 mHC 的写权重 $B$、$C$。
- 位置：block 1 和 block 14 的入口（0 起）。第 1 层入口时，残差流已经过了第 0 层的一轮注意力 + MoE，4 条流也已经被第 0 层的 mHC 写算子写成不同的值（embedding 之后 4 条流是同一个向量的 4 份拷贝，1258 行）。
- 两层都在 encoder 里（0–19 层），所以 prefill 和 decode 都会跑两个 Engram。（补：依据 [P §4.2.1] encoder 20 层 + decoder 20 层）
- 第 0、1 层是纯滑窗注意力 [P §4.2.1]。

### 2.9 放在哪两层、为什么

原版的依据 [E §6.2]（12 层 3B MoE 主干，1.6B Engram，100B token）：
- 单模块扫层 1–12：**Layer 2 最好**（Val Loss 1.770），比 Layer 1 好，越往深越差。
- 两个相反的力：放早能在主干花深度之前就卸掉局部模式的重建；但早期隐状态还没聚合足够的全局上下文，门不准，而且 mHC 的几条分支还没分化（「the parallel branches lack the representational divergence required for fine-grained modulation」）。「one round of attention is already sufficient」。
- 同样 1.6B 拆成两个较小的模块放 Layer 2 和 6 更好（1.768）。分层放还有系统上的好处（配合存储层级）。
- 系统约束 [E §2.5]：推理时表放主机内存，索引在 forward 之前就知道，可以异步预取；模块前面的层的计算时间用来盖住传输延迟。放得越深，可用来藏延迟的窗口越长；但建模上偏好早。两者要同时满足。
- 27B/40B 模型放在 [2, 15]（30 层）[E Table 5]。

V4.1 [P §2.4.2]：「The modules are placed at layers 1 and 14 (zero-indexed) to balance memory usage across training pipeline stages」；推理时「prefetching for the first module overlapping computation in the first Transformer block」。
- 论文给的理由是训练时各 pipeline stage 的显存均衡，不是建模效果。
- 补：[E] 的「Layer 2」应是 1 起的编号（[E §6.4] 说把 Engram 插进「the second Transformer block」，预取与「the first block」的计算重叠；扫层范围写的是 1 到 12）。那么 V4.1 的 0 起 1、14 就是 1 起的 2、15，和 Engram-27B 的 [2, 15] 是同一对位置。[D] 的默认值是 `layer_ids = [1, 15]`（0 起），和这个解读差一层，所以只能说「大概率相同」，见第 8 节。

### 2.10 Sparsity Allocation：参数怎么在 MoE 和 Engram 之间分

[E §3.1]：
- $P_\text{tot}$：总可训练参数，不含词表 embedding 和 LM head。$P_\text{act}$：每 token 激活的参数，决定训练 FLOPs。$P_\text{sparse} \triangleq P_\text{tot} - P_\text{act}$：不激活的「免费」参数预算。
- 分配比 $\rho \in [0,1]$：

$$P^{(\text{sparse})}_\text{MoE} = \rho\, P_\text{sparse},\qquad P_\text{Engram} = (1-\rho)\, P_\text{sparse} \tag{E.7}$$

  $\rho = 1$ 是纯 MoE；$\rho < 1$ 减少 routed expert 数，把省下的参数给 Engram 的表。每 token 只取常数个槽，所以加大表不增加 FLOPs。
- 实验：两个算力档，$P_\text{tot}/P_\text{act} \approx 10$ 固定。$C = 2\times10^{20}$ FLOPs：$P_\text{tot} \approx 5.7$B，$P_\text{act} = 568$M，纯 MoE 106 个 expert。$C = 6\times10^{20}$：$P_\text{tot} \approx 9.9$B，$P_\text{act} = 993$M，99 个 expert。
- 结论（Figure 3 左）：验证 loss 对 $\rho$ 是 **U 形**。
  - 把 20%–25% 的稀疏预算给 Engram 最好，最优点 $\rho \approx 75\%$–$80\%$，两个档位置稳定。
  - 10B 档：loss 从 1.7248（$\rho = 100\%$）降到 1.7109（$\rho \approx 80\%$），$\Delta = 0.0139$。
  - $\rho \approx 40\%$ 时（46 / 43 个 expert）还能和纯 MoE 持平。
  - 两端的解释：$\rho \to 100\%$ 没有专门的静态记忆，只能靠深度和计算重建；$\rho \to 0\%$ 失去条件计算能力，「memory cannot replace computation」。

[E §3.2] 无限内存档：固定 3B MoE 主干（$P_\text{act} = 568$M，100B token），槽数从 $2.58\times10^5$ 扫到 $1.0\times10^7$（最多加约 13B 参数）。验证 loss 对槽数是对数线性（幂律），且比 OverEncoding（把 n-gram embedding 和词表 embedding 取平均）的斜率更好。

补：套到 V4.1 上。主干 552B，Engram 196B [P §2.4.2, §4.2.1]，激活 8B（prefill）/ 16B（decode）。按式 (E.7)，$P_\text{sparse} \approx 552 + 196 - 16 = 732$B，Engram 占 $196/732 \approx 26.8\%$，$\rho \approx 73\%$。这和 Engram-27B 的 $\rho = 74.3\%$、U 形最优区间 75%–80% 基本一致。注意口径不严格：[E] 的 $P_\text{tot}$ 不含词表 embedding 和 LM head，V4.1 的 552B 是否含它们论文没说；V4.1 论文也没有说 196B 是按这条规律定的。写正文时只能说「数量上吻合」。

### 2.11 Sinkhorn-balanced update（V4.1 的第二处改动）

动机 [P §2.5]：「applying Adam to the newly introduced Engram parameters substantially increases the optimizer-state memory footprint」。于是 Engram 表、token embedding、预测头三类大矩阵改用「动量 + Sinkhorn balancing」，只需要一个动量 buffer（和 Muon 一样），并且「empirically outperforming Adam」。Sinkhorn balancing 此前在 SinkGD（Scetbon et al. 2025）里用于线性层权重，这里推广到这几类大矩阵。

原版 Engram 用的是 Adam：[E §4.1, Table 5] 表参数用 Adam，学习率 ×5，weight decay 0；主干用 Muon。

各类参数用什么 [P §2.5 Basic Configurations, §4.2.2]：
- Engram 表、token embedding、预测头：Sinkhorn-balanced update，Nesterov 动量，**不加 weight decay**。
- Engram 的投影层（即 `wkv`）：Muon。
- 归一化层权重和其它非矩阵参数：AdamW。（补：`q_weight`、`k_weight` 是 RMSNorm 增益，应归这一类；论文没点名。）
- Engram 的学习率 ×5，沿用 [E]。

Algorithm 1 [P §2.5]。$W_t \in \mathbb{R}^{m \times n}$，$m$ 是大的那一维（行数：词表大小或表的行数），$n$ 是隐藏维；$K$ 为奇数：

1. $M_t \leftarrow \beta M_{t-1} + (1-\beta) G_t$
2. $\hat{G}_t \leftarrow \beta M_t + (1-\beta) G_t$（Nesterov）
3. $\rho_i \leftarrow \lVert \hat{G}_{t,i,:} \rVert_2$，$\bar\rho \leftarrow \frac{1}{m}\sum_i \rho_i$
4. $U^{(0)} \leftarrow \hat{G}_t$；若 $\rho_i \le \tau\bar\rho$ 则 $U^{(0)}_{i,:} \leftarrow 0$（屏蔽近零行）
5. 对 $k = 1, \dots, K$：$k$ 为奇数时每行除以自己的 $\ell_2$ 范数加 $\varepsilon$；$k$ 为偶数时每列除以自己的 $\ell_2$ 范数加 $\varepsilon$
6. $\Delta_t \leftarrow \sqrt{n}\, U^{(K)}$
7. $\tilde\eta_t \leftarrow \gamma \eta_t$
8. $W_{t+1} \leftarrow W_t - \tilde\eta_t \Delta_t$

论文的解释 [P 式 7]：Sinkhorn balancing 找对角缩放 $D_r$、$D_c$ 使

$$\Delta_t = \sqrt{n}\, U^{(K)} = \sqrt{n}\, D_r \hat{G}_t D_c,\qquad \frac{1}{n}\sum_{j=1}^{n} (\Delta_t)_{ij}^2 \approx 1,\qquad \frac{1}{m}\sum_{i=1}^{m} (\Delta_t)_{ij}^2 \approx 1$$

即更新矩阵的行 RMS 和列 RMS 都约等于 1。一行对应一个 token 或一个 n-gram 身份，一列对应一个隐藏特征；Sinkhorn 利用的是这种 token–特征结构。$\sqrt{n}$ 把「行 $\ell_2$ 范数为 1」换算成「行 RMS 为 1」。$\gamma$ 把更新幅度对齐到 Adam，取 **0.18**（接近 Moonlight 的 0.2）。

超参 [P §4.2.2]：动量系数和学习率修正因子与 Muon 相同（动量 0.95，0.18），$K = 11$，$\tau = 10^{-3}$，$\varepsilon = 10^{-20}$。

补的推导：
- $K$ 取奇数，最后一步是行归一化，所以未屏蔽的行 $\ell_2$ 范数精确为 1，列只是近似。行范数 1 ⇒ $\sum_j U_{ij}^2 = 1$ ⇒ 乘 $\sqrt{n}$ 后 $\frac1n\sum_j \Delta_{ij}^2 = 1$。
- 行、列为什么能同时为 1：$\Delta$ 的全部元素平方和 $= m \cdot n$（$m$ 行，每行平方和 $n$）。若各列均衡，每列平方和 $= m$，列 RMS $= \sqrt{m/m} = 1$。两个条件相容。
- 与 Muon 的类比（[P]「same workflow as Muon, with Sinkhorn balancing taking the place of Newton–Schulz orthogonalization」）：Muon 把更新的奇异值全拉到 1，Sinkhorn 把更新的行、列范数全拉到 1。前者要做 $m \times n$ 矩阵的矩阵乘，对 3.84 亿行的表不现实；后者只要行、列的范数。
- 与 Adam 的对比：Adam 给每个元素一个二阶矩，更新的每个元素幅度约为 1；Sinkhorn 只在行和列两个方向上各做一次归一，效果是每行、每列的平均幅度约为 1，但不存二阶矩。
- 为什么要屏蔽近零行：稀疏查表时，很久没被访问的行动量已衰减到接近 0，行归一化会把这些接近噪声的行放大到单位范数。论文只说「for numerical stability」。
- 省多少状态：Adam 是一阶矩 + 二阶矩两份，动量法是一份，状态减半。196.6B 参数若状态用 fp32，一份是 786 GB。状态精度论文没给，这个 GB 数只是量级。

实现 [P §3.1.3]：Sinkhorn 归一化「maintains row and column scaling vectors across iterations to avoid repeated writes of the full normalized matrix」，即只迭代 $D_r$、$D_c$ 两个向量而不反复重写整张矩阵；行归一化和列统计量的累加融合在一个 kernel 里。

相关工作 [P §2.5 末段]：Adafactor（行列归一方式不同）、Adam-mini（embedding 和预测头用另一种行归一化）。

### 2.12 系统：196B 参数怎么放

训练：
- [E §2.5, Figure 2a]：表按 GPU 分片，前向用 All-to-All 收集被激活的行，反向再把梯度发回去，总容量随 GPU 数线性增长。
- [P §3.1.3]：表按行分给专门的进程组（engram parallel size，组的大小在「每设备显存」和「查表的通信范围」之间取舍）；优化器状态在每个分片的副本之间再分片。索引只依赖输入 token，所以在每个 pipeline stage 开始处理本步的 microbatch 之前，就对整个本地 batch 发起预取。梯度在反向时先缓冲，主干反向结束后送回所属 rank。多模态训练时，预取和梯度传输安排在视觉编码器前向/反向期间。embedding 以 FP8 存取，取回的值和 scale 直接送进后面的 GEMM。RL rollout 期间表常驻 GPU 显存（减轻主机内存压力，避免碎片化导致的 OOM）。

推理：
- [E §2.5, Figure 2b]：表放主机内存，经 PCIe 异步预取，和前面 block 的计算重叠。n-gram 服从 Zipf 分布，可以做多级缓存（热的放 GPU HBM 或主机 DRAM，长尾放 NVMe SSD）——这是论文提出的设想，[E §6.4] 的实验没有实现分级，全部走 PCIe。
- [P §2.4.2]：「deterministic addressing enables embeddings to be prefetched from host memory via background RDMA transfers, with prefetching for the first module overlapping computation in the first Transformer block」。
- 参考推理代码 [M `ParallelEngramEmbedding`] **没有**做主机内存卸载和预取：表按行切成 `world_size` 份，每个 rank 放一份在 GPU 上，查表时不属于自己的行输出 0，再 `dist.all_reduce` 求和。

FP8 存储 [M 296–325 行]：表的权重是 `float8_e4m3fn`，另有 `scale`，形状 `[rows, dim // 32]`，dtype `float8_e8m0fnu`。每行 256 维分成 8 个 32 维块，每块一个 scale；查到行后乘 scale 反量化，转 bf16。和 `Linear` 的 32×32 块 scale 不同，表是按行、每 32 维一个 scale（这样一行可以独立取出）。[P §2.4.2]「Both the embedding tables and the key/value projections use FP8 precision」。

## 3. 代码落点

### 3.1 V4.1 推理代码（HF `deepseek-ai/DeepSeek-V4.1-Flash`，sha `2cba9e4`，`inference/`）

| 文件 | 符号 | 作用 |
|---|---|---|
| `engram.py` | `find_next_prime` | 找大于 start 且没用过的最小素数 |
| `engram.py` | `build_compressed_token_map` | tokenizer compression，返回 (lookup, 压缩词表大小) |
| `engram.py` | `compute_hash_multipliers` | 每层 4 个奇数乘子，种子 `10007 * layer_id` |
| `engram.py` | `EngramLayout.from_args` | 算出 `primes[layer][阶][头]` |
| `engram.py` | `NgramHashState` | 压缩 id 缓存 + 哈希，输出 `[B, L, 2, 24]` 行号 |
| `model.py` | `ParallelEngramEmbedding` | 按行分片的 FP8 表，查表反量化 |
| `model.py` | `Engram` | `wkv` 投影 + 门 + 写入 4 条流 |
| `model.py` | `Block.__init__` 930–932 | 只有 `layer_id in engram_layout.layer_ids` 的 block 挂 Engram |
| `model.py` | `Transformer.forward` 1249–1263 | 算哈希、调用 |
| `convert.py` 135–141 | — | 表按行切分片，最后一片不足的行用 0 补，scale 用 1 补 |

哈希主体（[H] 169–184 行）：

```python
positions = torch.arange(start_pos, start_pos + seqlen, device=input_ids.device).expand(batch, seqlen)
tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
for shift in range(self.layout.max_ngram_size):
    source = self.cache[:batch].gather(1, (positions - shift).clamp_min(0))
    blocked = blocked | (positions < shift) | (source == self.DEAD)
    tokens.append(torch.where(blocked, self.pad_id, source))
tokens = torch.stack(tokens, dim=-1)  # [B, L, max_ngram_size]

products = tokens.unsqueeze(2) * self.multipliers  # [B, L, n_engram_layers, max_ngram_size]
rolling, hashes = products[..., 0], []
for i in range(1, self.layout.max_ngram_size):
    rolling = torch.bitwise_xor(rolling, products[..., i])
    hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
return torch.cat(hashes, dim=-1) + self.offsets
```

解释：`tokens[..., i]` 是往回看 i 步的压缩 id（越界或遇到图像后一律 pad）；`products[..., i]` 是它乘第 i 个乘子；`rolling` 异或到第 i 项就是 (i+1)-gram 的 mix；对该阶的 8 个素数取模得 8 个桶号；最后加 offset 变成整张表里的行号。输出最后一维 24 列的顺序是 2-gram 的 8 头、3-gram 的 8 头、4-gram 的 8 头。

门（[M] 353–365 行）：

```python
kv = self.wkv(self.embed(hash_ids).flatten(-2))
key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
key = key.float().unflatten(-1, (self.hc_mult, self.dim))
weight = self.q_weight.float() * self.k_weight.float()  # only ever used as a product
h, eps = x.float(), self.eps
# normalized per (token, hc copy) over `dim`, NOT jointly over the copies
rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
# signed sqrt before the sigmoid, matching the training kernel
gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
if token_mask is not None:
    gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)
```

形状：`embed(hash_ids)` 是 `[B, L, 24, 256]`，flatten 成 6144；`kv` 是 `[B, L, 25600]`；`key` 是 `[B, L, 4, 5120]`，`value` 是 `[B, L, 5120]`；`gate` 是 `[B, L, 4]`。

查表（[M] 312–325 行）：`F.embedding` 取 FP8 行和 scale，`values.unflatten(-1, (-1, 32)) * scales.unsqueeze(-1)` 反量化；`world_size > 1` 时 `dist.all_reduce`。

权重 key（[I]）：每个 Engram 层 6 个张量：`layers.{1,14}.engram.embed.weight`、`.embed.scale`、`.q_weight`、`.k_weight`、`.wkv.weight`、`.wkv.scale`。两张表分别在 `model-00047-of-00048.safetensors` 和 `model-00048-of-00048.safetensors`。

### 3.2 官方 demo（`deepseek-ai/Engram`，commit `fb7f84a`，`engram_demo_v1.py`）

| 类 | 对应 |
|---|---|
| `CompressedTokenizer` | tokenizer compression，normalizer 序列与 [H] 相同 |
| `NgramHashMapping` | 乘子、素数、`_get_ngram_hashes`；与 [H] 同一个哈希 |
| `MultiHeadEmbedding` | 一张 `nn.Embedding` + offsets |
| `ShortConv` | 短因果卷积（V4.1 没有） |
| `Engram` | `key_projs`（4 个）+ `value_proj` + `norm1`/`norm2` + 门 + `short_conv` |

demo 的默认 config 不是论文里任何一个模型的配置：`engram_vocab_size = [129280*5, 129280*5]`（每个阶可以给不同的桶大小），`max_ngram_size = 3`，`n_embed_per_ngram = 512`，`n_head_per_ngram = 8`（每头 64 维），`layer_ids = [1, 15]`，`pad_id = 2`，`kernel_size = 4`，`hidden_size = 1024`，`hc_mult = 4`。

demo 的输出：`value = gates * value_proj(embeddings).unsqueeze(2)`；`output = value + short_conv(value)`；block 里 `hidden_states = engram(...) + hidden_states`，然后才是 attn 和 moe。

### 3.3 V4.1 相对原版 Engram 的差别（汇总）

| 项 | 原版 [E]/[D] | V4.1 [P]/[M]/[H] | 性质 |
|---|---|---|---|
| 短因果卷积 | 有，核 4，dilation = 最大阶，SiLU，零初始化 | 去掉 | 论文明说的改动 1 |
| 表的优化器 | Adam，lr ×5，wd 0 | 动量 + Sinkhorn balancing，lr ×5，无 wd | 论文明说的改动 2 |
| 每条流一个 key、共享 value 和表 | 有（式 E.6） | 沿用，融合成一个 `wkv` | **不是改动** |
| sigmoid 前的带符号平方根 | demo 里有，论文式子里没写 | 有 | **不是改动** |
| n-gram 阶 | {2,3}（27B/40B）；消融里 4-gram 在 1.6B 预算下略差 | {2,3,4} | 配置不同 |
| 头数 | 8 | 8 | 相同 |
| 每头维度 / $d_\text{mem}$ | 80 / 1280（补，见第 4 节） | 256 / 6144 | 配置不同 |
| 每头桶数 | 2,262,400（27B）/ 7,239,680（40B） | 约 16M | 配置不同 |
| 位置 | [2, 15] / 30 层 | [1, 14]（0 起）/ 40 层 | 大概率同一对位置 |
| 图像 token | 无（纯文本模型） | 不参与 n-gram，门置 0 | V4.1 新增 |
| 投影层优化器 | Muon（主干优化器） | Muon | 相同 |
| 推理时取表 | 主机内存 + PCIe 预取 | 主机内存 + 后台 RDMA 预取 | 传输方式不同 |

## 4. 关键数字

### 4.1 V4.1

| 数字 | 条件 | 出处 |
|---|---|---|
| 196B Engram 参数，两个模块平分 | — | [P §2.4.2] |
| 阶 {2,3,4}，8 头，每阶 2048 维 | 每头 256 维 | [P §2.4.2]，[C] `engram_head_dim = 256` |
| 每头约 16M 行，大小为互不相同的素数 | `engram_vocab_size = 16000000` | [P §2.4.2]，[C]，[H `from_args`] |
| 384,006,168 行 / 384,016,682 行 | 第 1 / 14 层 | [C]；等于 24 个素数之和（补，重算验证） |
| 768,022,850 × 256 = 196.61B | 两张表合计 | 补 |
| `wkv`：6144 × 25600 = 157.3M 参数，两个共 0.31B | 无 bias，FP8 | [M 345 行]，补 |
| 压缩词表 99092 | 原词表 129280，压缩 23.35%（补） | [C]，[H] |
| 乘子上界 $B$ = 46,539,438,283,891 | $\lfloor\lfloor(2^{63}-1)/99092\rfloor/2\rfloor$ | [H `compute_hash_multipliers`]，补 |
| 每 token 每层取 24 行 = 6144 字节 FP8 + 192 字节 scale | 两层合计约 12.7 KB | [M 309–310 行]，补 |
| 表的存储约 196.6 GB + scale 6.1 GB | FP8 一字节一个参数；scale 每行 8 字节 | 补 |
| 位置：第 1、14 层（0 起） | 理由：均衡各 pipeline stage 的显存 | [P §2.4.2]，[C] |
| Sinkhorn：$K = 11$，$\tau = 10^{-3}$，$\varepsilon = 10^{-20}$，$\gamma = 0.18$，动量 0.95 | 预训练 | [P §2.5, §4.2.2] |
| Engram 学习率 ×5 | 沿用 [E] | [P §4.2.2] |
| 基础学习率 $2.6\times10^{-4}$，batch 100.6M token，共 45T token | 预训练 | [P §4.2.2] |
| 主干 552B，激活 8B（prefill）/ 16B（decode） | Engram 另计 | [P §4.2.1] |

### 4.2 Engram 论文

| 数字 | 条件 | 出处 |
|---|---|---|
| 词表压缩 23%（23.43%） | 128k tokenizer（DeepSeek-V3） | [E §2.2, Table 6] |
| 卷积核 4，dilation = 最大 n-gram 阶 | — | [E §2.3] |
| U 形最优 $\rho \approx 75\%$–$80\%$ | $P_\text{tot}$ 5.7B / 9.9B，$P_\text{tot}/P_\text{act} \approx 10$ | [E §3.1] |
| loss 1.7248 → 1.7109 | 10B 档，$\rho$ 100% → 约 80% | [E §3.1] |
| $\rho \approx 40\%$ 仍持平纯 MoE | 46 / 43 个 expert | [E §3.1] |
| 槽数 $2.58\times10^5$ → $1.0\times10^7$，loss 对数线性 | 3B MoE 主干，100B token | [E §3.2] |
| Engram-27B：expert 72 → 55，Engram 5.7B，$\rho = 74.3\%$ | 总参数 26.7B，激活 3.8B，262B token，30 层，$d = 2560$ | [E §4.1, Table 1, Table 5] |
| Engram-40B：Engram 18.5B，总 39.5B | 同上 | [E §4.1] |
| 27B 配置：层 [2,15]，n-gram [2,3]，8 头，dim 1280，vocab 2,262,400；40B 的 vocab 7,239,680 | lr ×5，wd 0，Adam，卷积零初始化 | [E Table 5] |
| 每头 80 维 | 补：2 层 × 2 阶 × 8 头 × 2,262,400 × 80 = 5.79B ≈ 5.7B；40B 同式得 18.53B ≈ 18.5B。所以 1280 是拼接后的 $d_\text{mem}$ | 补 |
| 验证 loss：Dense-4B 1.768，MoE-27B 1.634，Engram-27B 1.622，Engram-40B 1.610 | 262B token | [E Table 1] |
| MMLU 57.4 → 60.4（+3.0），CMMLU 57.9 → 61.9（+4.0），MMLU-Pro 28.3 → 30.1 | MoE-27B → Engram-27B，5-shot | [E Table 1] |
| BBH 50.9 → 55.9（+5.0），ARC-Challenge 70.1 → 73.8（+3.7），DROP 55.7 → 59.0（+3.3） | 同上 | [E Table 1] |
| HumanEval 37.8 → 40.8（+3.0），MATH 28.3 → 30.7（+2.4），GSM8K 58.4 → 60.6（+2.2），MBPP 46.6 → 48.2 | 同上 | [E Table 1] |
| TriviaQA 48.8 → 50.7，PopQA 19.2 → 19.4 | 同上；事实问答的提升反而小 | [E Table 1] |
| RULER 32k：Multi-Query NIAH 84.2 → 97.0，Variable Tracking 77.0 → 87.2 | Engram-27B 46k 步（与 MoE-27B 50k 步预训练 loss 相同，1.63） | [E Table 2] |
| 同上，50k 步：MQ 97.0，VT 89.0 | iso-FLOPs | [E Table 2] |
| 41k 步（82% 预训练算力）：LongPPL 持平，MQ 89.5，VT 83.2 | — | [E Table 2] |
| 消融基线 Val Loss 1.808；参考配置 1.768（$\Delta = 0.04$） | 12 层 3B MoE（激活 0.56B），100B token，1.6B Engram，{2,3}-gram，层 2 和 6 | [E §6.2] |
| 单模块 Layer 2 最好，1.770 | 同上 | [E §6.2] |
| 回退最大的三项：分支专属融合、context-aware gating、tokenizer compression | 具体数值只在 Figure 5 里，正文没报 | [E §6.2] |
| 去卷积「only marginally degrades」；加 4-gram「slightly suboptimal」 | 1.6B 固定预算 | [E §6.2] |
| 推理时屏蔽 Engram 输出：事实知识保留 29%–44%（TriviaQA 29%），阅读理解保留 81%–93%（C3 93%） | 事后屏蔽，有训练推理不一致 | [E §6.3] |
| CKA：Engram-27B 第 5 层对齐 MoE 基线约第 12 层 | Few-NERD，实体最后一个 token，top-5 软对齐 | [E §6.1.2] |
| 吞吐：4B-Dense 9031.62 → 8858.28 tok/s（−1.9%，补算）；8B-Dense 6315.52 → 6140.02（−2.8%） | 100B Engram 全放主机 DRAM，插在第 2 个 block；H800；512 条序列，长度 Uniform(100, 1024)；基于 nano-vLLM | [E Table 4] |

## 5. 常见误解与澄清

1. **「V4.1 把 key 改成了 4 条流各一个」**：不是 V4.1 的改动。$M$ 个 $\mathbf{W}_K^{(m)}$ + 共享 $\mathbf{W}_V$ + 共享表是原版 [E §2.4 式 6] 的设计，原版默认主干就是 mHC（$M = 4$）。V4.1 论文明说的改动只有两处：去卷积、换优化器。
2. **「门是 sigmoid(归一化点积)」**：论文式 (E.4) 是这么写的，但官方 demo 和 V4.1 推理代码在 sigmoid 之前都多一步 $\text{sign}(s)\sqrt{|s|}$。照论文公式实现会和权重对不上。
3. **「8 个头是 8 个独立的哈希函数」**：同一阶的 8 个头共用同一个 64 位混合值，只是对 8 个不同的素数取模 [H 183 行]。独立性来自模数互素，而不是来自 8 套乘子。
4. **「每个 (阶, 头) 一张表，共 24 张 / 48 张」**：论文记号是 $\mathbf{E}_{n,k}$，实现上每层是一张 3.84 亿行的表，24 个头各占一段 [H `offsets`]。全模型两张物理表。
5. **「表的行数 384006168 是个调出来的超参」**：它是从 16000000 往上数的 24 个素数之和；真正的超参是 `engram_vocab_size = 16000000`。
6. **「Engram 是 RAG / 外部知识库 / 可编辑记忆」**：是参数化的 embedding 表，随模型端到端训练，key 是 token n-gram 的哈希，推理时不可见也不可单独编辑。[E §7] 把自己归在 parametric memory 一侧，与 REALM/RETRO 等 non-parametric 方法区分。
7. **「Engram 替换了输入 embedding」**：[E Figure 1 caption]「leaving the standard input embedding and un-embedding module intact」。它插在中间层（V4.1 是第 1、14 层入口），不在第 0 层。[E §7] 把「不放在第 0 层」当作和 OverEncoding、SCONE 的区别之一，理由是放深一点才能让取表和计算重叠。
8. **「加记忆只帮知识题」**：[E Table 1] 提升最大的是 BBH（+5.0）、CMMLU（+4.0）、ARC-Challenge（+3.7）；TriviaQA 只 +1.9，PopQA +0.2。论文的解释是省下了前几层的深度（LogitLens、CKA）。但 [E §6.3] 屏蔽 Engram 后事实问答掉得最多（只剩 29%–44%），两个现象要分开讲：前者是「加了之后谁涨得多」，后者是「拿掉之后谁掉得多」。
9. **摘要的 MMLU +3.4 与正文不一致**：[E 摘要、§1] 写 +3.4，[E §4.2] 写 +3.0，Table 1 是 57.4 → 60.4 = +3.0。引用时用 Table 1。另外 [E §1] 的「Variable Tracking 89.0 vs 77.0」是 50k 步的数，[E §5.2] 的「87.2 vs 77.0」是 46k 步（iso-loss）的数，两个都对，条件不同。
10. **「196B 参数每个 token 都要算」**：每 token 每层只取 24 行，共 6144 个数；计算量在 `wkv` 那一次 6144 → 25600 的矩阵乘。所以 196B 不计入「激活参数」。
11. **「参考推理代码演示了主机内存预取」**：没有。[M `ParallelEngramEmbedding`] 把表按行分片放在各 rank 的 GPU 上，`all_reduce` 汇总。RDMA 预取只在论文里。
12. **「4-gram 被原论文否定了」**：原文是「在 1.6B 固定预算下略差，可能是稀释了 2/3-gram 的容量，不排除在更大的记忆规模下有益」[E §6.2]。V4.1 在 196B 规模用了 {2,3,4}，没有给消融。
13. **V4.1 论文里 Engram 有两条引用**（Cheng et al. 2026b 是 arXiv，2026c 是 ACL），是同一篇（PLAN.md 已记）。

## 6. 与本专栏其他文章的接口

grep 范围：`src/content/posts/{llm-,kimi-k3-,deepseek-}*.mdx`。

已讲过，只需引用：
- **Engram 的一段话概述、config 表、参数账**：`deepseek-v41-00-overview.mdx`（第 24、39、90、183–185、201、245–248 行）。开篇已经讲了「24 个行号 → 6144 维 → 5 个 5120 维向量 → 每条流一个门」的流程和 196.6B 的算法。第 5 篇不要重复这段叙述，要往下讲哈希的具体形式、门为什么这么设计、训练和存放。注意开篇第 201 行写的是「过 sigmoid 得到门」，没提带符号平方根，第 5 篇补上即可，不算冲突。
- **mHC 的 4 条残差流、读/写算子、Sinkhorn-Knopp 投影到双随机矩阵**：`deepseek-v4-04-mhc.mdx`（Sinkhorn 出现 13 次）。Single-Pass mHC 是 V4.1 第 4 篇。第 5 篇只需说「Engram 直接写 4 条流，在读算子之前」并链接。
  - 提醒：mHC 的 Sinkhorn 是把 $4\times4$ 矩阵投影成双随机矩阵（行和、列和为 1）；优化器里的 Sinkhorn balancing 是把更新矩阵的行、列 **RMS** 配平到 1。同名不同用，正文要用一句话分开，避免读者混淆。
- **Muon、Newton–Schulz、Nesterov 动量、0.18 的 RMS 缩放**：`deepseek-v4-06-muon.mdx`（Muon 13 次，Newton 5 次）。第 5 篇讲 Sinkhorn-balanced update 时以「把 Newton–Schulz 换成 Sinkhorn」为入口，Muon 本体只引用。
- **MoE、routed expert、激活参数**：`deepseek-v4-05-moe.mdx`。Sparsity Allocation 一节引用它来解释 $P_\text{act}$。
- **RMSNorm**：多篇已用，不需要再定义。
- **FP8**：`deepseek-v4-05-moe.mdx`、`deepseek-v4-02-lightning-indexer.mdx`、`deepseek-attention-structures.mdx` 提到过，但没有讲过「每行每 32 维一个 scale」这种表的量化布局；第 5 篇可以用两三句讲。
- **All-to-All、EP 通信**：`llm-parallel-overview.mdx`、`kimi-k3-08-parallel-moonep.mdx`、`deepseek-v4-07-systems.mdx`。Engram 训练时的 All-to-All 只引用。

没有任何文章讲过（grep 0 命中），需要在第 5 篇首次讲：
- n-gram 哈希 embedding、tokenizer compression、multiplicative-XOR 哈希、素数取模。
- Sparsity Allocation / U 形曲线。
- LogitLens、CKA（如果正文要提「等效加深」的证据，要一句话解释这两个工具）。
- RDMA 预取（grep 0 命中；「预取」在 K3 的 serving 文章里是别的含义）。
- Sinkhorn-balanced update（优化器）。PLAN 里把它列在第 5 篇的「两处改动」里；它同时作用于 token embedding 和预测头，属于模型特有的训练细节，放在本篇讲即可，不需要拆到 llm 专栏。

留给别的文章：
- Single-Pass mHC 的细节 → V4.1 第 4 篇。
- DSpark 读取第 37–39 层 4 流均值 → 第 6 篇（[M 1265–1266 行] 读取发生在 Engram 之后、block 之前，但 DSpark 的目标层不是 Engram 层，互不影响）。
- 视觉通路 → 对应篇；本篇只需「图像 token 不参与 n-gram，门为 0」。

## 7. 图的建议

1. **n-gram 哈希查表流程图（主图）**。一行 token（含一个大小写/空格变体的例子），先过压缩表变成压缩 id；以当前位置结尾画出 2/3/4-gram 的三个括号；每个 id 乘各自的乘子，逐级异或得到三个 mix（画成滚动的：2-gram 的结果接着异或成 3-gram）；每个 mix 对 8 个素数取模得 8 个桶号；24 个桶号落在一张长条表的 24 个不相交区段里；取出 24 × 256 拼成 6144。要表达的机制：同阶 8 头共用一个 mix、只是模数不同；一层只有一张物理表；序列开头用 pad 补。
2. **门控公式图**。左边 6144 维向量过 `wkv` 分成 4 个 key + 1 个 value；右边 4 条残差流；每条流和自己的 key 做归一化点积 → 带符号平方根 → sigmoid → 标量门；同一个 value 乘 4 个不同的门，加回 4 条流。可以配一条小曲线：$\sigma(\text{sign}(s)\sqrt{|s|})$ 对比 $\sigma(s)$，横轴 $s \in [-10, 10]$，表达平方根把饱和推迟了。原版多出来的卷积支路用虚线或灰色画出并标「V4.1 去掉」。
3. **（可选）Sinkhorn-balanced update 对照图**。同一个更新矩阵的热力图三联：原始动量（少数行很大、多数行近零）→ 屏蔽近零行 → 行列交替归一后（行、列 RMS 都约为 1）。旁边并排 Muon 的流程条，只把「Newton–Schulz」那一格换成「Sinkhorn × 11」。也可以改画 U 形曲线示意（横轴 $\rho$，标出 75%–80% 最优区和 V4.1 的约 73%），但 Figure 3 的具体点我没读到数值，只能画示意，不能画成数据图。

`V41ArchDiagram` 的 `highlight="engram"` 和 `V41LayerStrip` 的 Engram 标记已建（PLAN 第 91–92 行），开头可直接用。

## 8. 待确认 / 没查到的点

1. **99092 没有实测复现**。本机没有 `numpy`/`tokenizers`/`transformers`，没跑 `build_compressed_token_map`。99092 来自 config 加上 [H] 里的 assert；「V4.1 tokenizer 的 `len(tokenizer)` 是多少、是否仍是 129280」没确认。
2. **乘子的具体数值没算**。需要 numpy 的 `default_rng(10007)` 和 `default_rng(140098)`；本机没有 numpy。上界 $B$ 是手算的。素数表和两个行数之和是用自写的 Miller–Rabin 重算的，和 config 对上了。
3. **「Layer 2」的编号基准**。[E] 正文没明说 1 起还是 0 起；我按 §6.4「second Transformer block」推断是 1 起，于是 V4.1 的 [1, 14]（0 起）= 原版 [2, 15]。但 [D] 默认 `layer_ids = [1, 15]`（0 起）与此差一层。正文若要说「位置和原版相同」，要加「大概率」。
4. **消融的具体数值**。[E Figure 5] 里去掉各组件后的 Val Loss 只在图上，正文只报了基线 1.808、参考 1.768、单层最优 1.770。Figure 3 的各点、Figure 6 的各 benchmark 保留率同理，我没有读图。
5. **带符号平方根的动机**。论文没提这一步，代码没解释。第 2.4 节的量级分析是我补的。
6. **V4.1 为什么加 4-gram、为什么每头 256 维、为什么 16M 桶**：[P] 没有给消融或理由。
7. **196B 是否按 U 形规律定的**：[P] 没说。第 2.10 节的 $\rho \approx 73\%$ 是我按式 (E.7) 套的，口径不严格。
8. **Sinkhorn-balanced update 相对 Adam 的效果数字**：[P] 只说「empirically outperforming Adam」，没有数。优化器状态的精度、省了多少 GB 也没有给。
9. **`q_weight`/`k_weight` 用什么优化器**：按 [P §2.5]「normalization-layer weights → AdamW」推断，论文没点名。
10. **V4.1 推理时的预取实现**：只有 [P §2.4.2] 一句话（主机内存、后台 RDMA、与第一个 block 重叠）。第 14 层那个模块的预取时机、是否做了 [E §2.5] 设想的多级缓存，都没有说；参考代码里没有这部分。
11. **训练 kernel**：[M] 注释提到「matching the training kernel」，训练代码未公开，门在训练时的实现细节（如 FP8 路径）无从核对。
12. **arXiv 版本**：读的是 v2。v1 与 v2 的差别没有比对；[P] 引用的 ACL 版本（2026c）没有读，可能与 arXiv v2 有出入。
