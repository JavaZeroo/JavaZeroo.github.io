# DSpark：半自回归草稿 + 置信度调度的投机解码（DeepSeek-V4.1 连载第 6 篇参考笔记）

## 0. 一句话：这篇要解决什么问题

主干每跑一次前向只出 1 个 token。DSpark 是挂在主干后面的三层草稿模块：一次前向给出 5 个位置的 logits，再用一个很小的串行头（Markov head）补上草稿 token 之间的依赖，然后由 confidence head + 调度器决定这 5 个里送几个给主干验证。文章要讲清三件事：一次前向出 5 个位置为什么会坏（位置之间独立）、Markov head 怎么修、验证长度为什么要按系统负载动态选，以及这一切为什么仍然是无损的。

## 1. 一手资料清单（URL + 版本/commit + 读了哪些部分）

| 编号 | 资料 | 版本 | 读了哪些 |
|---|---|---|---|
| [D] | DSpark 论文 Cheng et al., "DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"，https://arxiv.org/abs/2607.05147 （读的是 https://arxiv.org/html/2607.05147 ） | v1，2026-07-06 提交，只有 v1 | 全文：摘要、§1–§7、Algorithm 1、Table 1、Appendix A。**Figure 1–8 只读了图注和正文里对图的描述**，没看图本身（HTML 转文本） |
| [P] | DeepSeek-V4.1 报告，本地 `drafts/deepseek-v4/refs/v41-paper.txt`（arXiv 2609.19969） | 本地文本 | 全文 grep "DSpark / MTP / speculative / draft"：只有 3 处正文（184 行 §1 概述、209 行 §2.1、311–315 行 §2.4.3）+ 839 行参考文献。**V4.1 报告没有给 DSpark 的任何实验数字** |
| [C] | V4.1 官方推理代码，本地 `drafts/deepseek-v4/refs/v41-inference-model.py`（HF `deepseek-ai/DeepSeek-V4.1-Flash` 的 `inference/model.py`） | HF repo sha `dba1be0a40aa45a94ad051997016db3960a90277`（`v41-repo-api.json`，lastModified 2026-09-10） | `ModelArgs` 129–148 行；`get_dspark_topk_idxs` 1021 行；`DSparkAttention` 1032 行；`DSparkMarkovHead` 1077 行；`DSparkConfidenceHead` 1089 行；`DSparkBlock` 1100–1156 行；`Transformer.__init__` 1206–1213 行、`forward` 1238–1272 行、`forward_spec` 1274–1282 行；`sample` 1285 行；`__main__` 1295–1309 行；`Block.forward` 973–995 行；`ParallelHead` 997 行；`Attention._window_kv` 700–721 行（对照） |
| [CF] | `drafts/deepseek-v4/refs/v41-config.json`（HF 格式）与 `v41-inference-config.json`（推理格式） | 同上 | dspark_* 字段 |
| [CV] | `drafts/deepseek-v4/refs/v41-inference-convert.py`、`v41-index.json` | 同上 | convert.py 68–75、108–109 行；index 里 `mtp.*` 的 key 名 |
| [L] | Leviathan et al. 2023, "Fast Inference from Transformers via Speculative Decoding"，https://arxiv.org/abs/2211.17192 （读的 https://arxiv.org/html/2211.17192 ） | arXiv HTML 当前版 | §2.1–2.3、Algorithm 1、§3.1–3.3、§3.5 末尾、§3.6、Appendix A.1 |
| [V3] | DeepSeek-V3 技术报告，https://arxiv.org/abs/2412.19437 （读的 https://arxiv.org/html/2412.19437 ） | arXiv HTML 当前版 | §2.2（式 21–25）、§4.2 超参里 MTP 两句、§5.4.3 |

## 2. 核心机制与推导

### 2.1 背景：投机解码的无损验证规则 [L]

记号（Leviathan）：目标模型 $M_p$ 的分布 $p(x)$，草稿模型 $M_q$ 的分布 $q(x)$，一次猜 $\gamma$ 个。

**单 token 的 speculative sampling**（[L] §2.3）：先采 $x\sim q$。若 $q(x)\le p(x)$ 直接留下；若 $q(x)>p(x)$，以概率 $1-p(x)/q(x)$ 拒绝，拒绝后从修正分布重采：

$$p'(x)=\mathrm{norm}\big(\max(0,\ p(x)-q(x))\big)$$

合起来就是「以概率 $\min(1, p(x)/q(x))$ 接受」。

**多 token**（[L] Algorithm 1）：草稿自回归地采 $x_1..x_\gamma$，目标模型一次并行算出 $p_1..p_{\gamma+1}$（$\gamma+1$ 个前缀），取 $r_i\sim U(0,1)$，

$$n=\min\big(\{i-1 \mid 1\le i\le\gamma,\ r_i>p_i(x_i)/q_i(x_i)\}\cup\{\gamma\}\big)$$

接受前 $n$ 个；若 $n<\gamma$，第 $n+1$ 个 token 从 $\mathrm{norm}(\max(0,p_{n+1}-q_{n+1}))$ 采，若全部接受则从 $p_{\gamma+1}$ 直接采。每轮至少出 1 个、至多出 $\gamma+1$ 个 token（[L] §2.1）。这个「目标模型补的那一个」在 DSpark 论文里叫 **bonus token / anchor token**（[D] §2.2 脚注 1：两个词混用，指上一轮目标模型最后生成的那个 token）。

**无损性证明**（[L] Appendix A.1）：设接受概率为 $\beta$。

- $P(\text{接受},\ x=x')=q(x')\min(1,p(x')/q(x'))=\min(q(x'),p(x'))$
- 修正分布的归一化常数是 $\sum_{x}(p(x)-\min(p(x),q(x)))=1-\beta$，所以 $P(\text{拒绝},\ x=x')=(1-\beta)\,p'(x')=p(x')-\min(q(x'),p(x'))$
- 相加得 $P(x=x')=p(x')$。对任意 $p,q$ 成立。

**接受率 = 两个分布的重叠**（[L] Lemma 3.3、Theorem 3.5）：

$$\beta=\mathbb{E}_{x\sim q}\min\!\Big(1,\frac{p(x)}{q(x)}\Big)=\sum_x\min(p(x),q(x))=1-\tfrac12\lVert p-q\rVert_1$$

最后一个等号：$\min(p,q)=\frac{p+q-|p-q|}{2}$，对 $x$ 求和（[L] Lemma 3.3 的证明里就是这一步；[L] 把 $\tfrac12\lVert p-q\rVert_1$ 记作 $D_{LK}$）。DSpark 的式 (8) 用的就是这个式子。

**期望 token 数与加速比**（[L] §3.1 式 1、Theorem 3.8）：假设各位置接受率 i.i.d.、均值 $\alpha$，一轮产出的 token 数是截断几何分布，

$$\mathbb{E}[\#\text{tokens}]=\frac{1-\alpha^{\gamma+1}}{1-\alpha},\qquad \text{walltime 提升}=\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}$$

$c$ 是草稿模型单步耗时与目标模型单步耗时之比。分母里的 $\gamma c$ 就是「自回归草稿的成本随 $\gamma$ 线性涨」。

**[L] 自己留的两个口子**，DSpark 正好各接一个：

- §3.5 末尾：如果能预测 $\beta$ 并据此逐轮改变 $\gamma$，还能再快；假设有 oracle，期望 token 数上界是 $1/(1-\alpha)$，walltime 提升比固定 $\gamma$ 最多再高约 60%，「留给未来工作」。→ DSpark 的 confidence head + 调度器。
- §3.6 末尾：草稿模型也可以是非自回归模型，「那就不用 Algorithm 1 里的自回归循环，只调一次」。→ 并行草稿。
- §3.3/§3.4 明说 walltime 分析**假设有足够算力并行跑 $\gamma+1$ 个目标模型前缀而不增加耗时**，总运算量是增加的。→ 高并发下这个假设不成立，是 DSpark 调度器的出发点。

### 2.2 背景：DeepSeek-V3 的 MTP [V3]

- 结构（§2.2，式 21–23）：$D$ 个**串行**模块预测 $D$ 个额外 token。第 $k$ 个模块 = 共享 embedding + 共享输出头 + 一个 Transformer block $\mathrm{TRM}_k$ + 投影 $M_k\in\mathbb{R}^{d\times 2d}$：

  $$\mathbf{h}'^{k}_i=M_k\big[\mathrm{RMSNorm}(\mathbf{h}^{k-1}_i);\ \mathrm{RMSNorm}(\mathrm{Emb}(t_{i+k}))\big],\quad \mathbf{h}^k_{1:T-k}=\mathrm{TRM}_k(\mathbf{h}'^k_{1:T-k}),\quad P^k_{i+k+1}=\mathrm{OutHead}(\mathbf{h}^k_i)$$

  $k=1$ 时 $\mathbf{h}^0$ 是主模型输出。「保持完整因果链」：每一深度都要拿到上一个真实 token 的 embedding，所以多深度只能串行（每多猜一个 token 多跑一个 block）。
- 目标（式 24–25）：每个深度一个交叉熵，$\mathcal{L}_{\text{MTP}}=\frac{\lambda}{D}\sum_k\mathcal{L}^k_{\text{MTP}}$，**和主干一起预训练**。$D=1$；$\lambda=0.3$（前 10T token）→ 0.1（后 4.8T）（§4.2）。
- 定位：「主要目的是提升主模型性能，推理时可以直接丢掉 MTP 模块；也可以改用于投机解码」（§2.2 "MTP in Inference"）。
- 数字（§5.4.3）：第二个 token 的接受率 85%–90%，TPS 1.8 倍。

### 2.3 三种草稿结构的账 [D] §2

DSpark 用一个式子组织全文（[D] 式 1）：

$$L=\frac{T_{\text{draft}}+T_{\text{verify}}}{\tau}$$

$\tau$ 是每轮接受的 token 数。三个杠杆：草稿更快（$T_{\text{draft}}$）、草稿更准（$\tau$）、验证更省（$T_{\text{verify}}$）。

| | 自回归草稿（小模型、EAGLE、MTP） | 并行草稿（Medusa、DFlash） | DSpark |
|---|---|---|---|
| $T_{\text{draft}}$ | $\propto\gamma$，所以只能用小 $\gamma$ 和浅网络（[D] §2.2） | 一次前向，几乎与 $\gamma$ 无关，可以用更深的网络和更大的块（如 $\gamma=16$） | 一次前向 + 一个很轻的串行头 |
| 位置间依赖 | 有 | **没有**：每个位置对前面所有可能的 token 求边缘，不是以实际采到的那个为条件 | 一阶（Markov）或 RNN |
| 验证长度 | 固定 | 固定 | 按置信度 × 负载动态选 |

**DFlash 的结构**（DSpark 的并行主干直接沿用，[D] §2.2 式 2–3）：

- 取目标模型若干层 $\{l_1..l_m\}$ 的 hidden 拼接后投影：$H_{\text{ctx}}=\mathrm{RMSNorm}(W_c[H^{(l_1)};\dots;H^{(l_m)}])$，$W_c\in\mathbb{R}^{d\times md}$。
- 「KV injection」：$H_{\text{ctx}}$ 在草稿的**每一层**都和草稿块自己的表示沿序列维拼成 K、V：$K_i=[W_i^KH_{\text{ctx}};\ W_i^KH_d]$，$V_i$ 同理。即上下文只当 KV，不当 query，不过草稿层的 FFN。
- 块内所有位置**双向**互相可见，并能看到注入的上下文。
- 共享目标模型的 embedding 和 LM head（都冻结）。
- 输入是 anchor token 的 embedding + $\gamma$ 个 mask token embedding，一次前向出所有 mask 位置的 logits。

### 2.4 半自回归生成 [D] §3.1

**问题：多峰碰撞。** [D] §3.1 的例子：上下文允许 "of course" 和 "no problem" 两种续写时，并行草稿可能采出 "of problem" / "no course"。

（补，把例子算出来）设真实联合分布是 $P(\text{of course})=P(\text{no problem})=0.5$。并行草稿只能输出两个位置各自的边缘：位置 1 是 $\{\text{of}:0.5,\text{no}:0.5\}$，位置 2 是 $\{\text{course}:0.5,\text{problem}:0.5\}$，独立采样有一半概率采到不通的组合。位置 2 的草稿分布 $q_2=(0.5,0.5)$，而目标模型在已知位置 1 = "of" 时 $p_2=(1,0)$，接受率 $\sum\min(p,q)=0.5$。位置 1 自己没问题，坏的是后面的位置，而且越往后前缀可能性越多、边缘越平，这就是论文说的 suffix decay（Figure 2：DFlash 的条件接受率 Code 上从 0.87 掉到 0.78，Chat 上从 0.72 掉到 0.63；自回归的 Eagle3 反而从 0.53 升到 0.74）。

**并行阶段。** 主干（DFlash）一次前向给出 hidden $h_1..h_\gamma$ 和 base logits $U_1..U_\gamma$。对 DFlash 只改了一处：不再是「anchor + $\gamma$ 个 mask，只预测 mask 位置」，而是**把 anchor 位置本身当作第一个预测位置**，$\gamma$ 个输入（anchor + $\gamma-1$ 个 mask）出 $\gamma$ 个草稿 logits，省一个位置的计算，质量相近。

**串行阶段。** 给 base logits 加一个依赖前缀的转移偏置 $B_k$，按自回归分解定义块分布（[D] 式 4）：

$$P(X\mid x_0)=\prod_{k=1}^{\gamma}p_k(x_k\mid x_0,x_{<k}),\qquad p_k(v\mid x_0,x_{<k})=\frac{\exp\big(U_k(v)+B_k(x_0,x_{<k},v)\big)}{\sum_{u\in\mathcal V}\exp\big(U_k(u)+B_k(x_0,x_{<k},u)\big)}$$

$x_0$ 是 anchor。推理时从左到右按 $p_k$ 采样。论文强调**不是全局归一化的能量模型**，每个位置是一次普通 softmax。要求 $T_{\text{sequential}}\ll T_{\text{parallel}}$。

**Markov head**（[D] 式 5）：让 $B_k$ 只依赖前一个 token，$B(x_{k-1},x_k)$。完整形式是 $V\times V$ 矩阵，做低秩分解 $B=W_1W_2$，$W_1\in\mathbb{R}^{V\times r}$，$W_2\in\mathbb{R}^{r\times V}$：

$$B(x_{k-1},\cdot)=W_1[x_{k-1}]\,W_2\in\mathbb{R}^{V}$$

$W_1$ 是一张 embedding 表（查表），$W_2$ 是一个 logit 投影。默认 $r=256$。例子收尾：位置 1 采到 "of" 后，Markov head 在位置 2 抬高 "course"、压低 "problem"。

（补）几点从式子能直接读出的东西：

- $B$ 与位置 $k$ 无关，所有位置共用同一对 $W_1,W_2$；与上下文无关的那部分「bigram 知识」放在 $B$ 里，与上下文有关的放在 $U_k$ 里。
- 第一个位置 $k=1$ 的前驱是 anchor $x_0$，同样加偏置（代码确认，见 3.4）。
- 规模：$V=129280$、$r=256$ 时，完整 $V\times V$ 是 $1.67\times10^{10}$ 项；低秩后 $2Vr=6.6\times10^{7}$ 项。每步计算是一次查表 + 一次 $r\times V$ 的矩阵向量乘（$3.3\times10^{7}$ 次乘加），而 LM head 一个位置是 $d\times V=5120\times129280=6.6\times10^{8}$，Markov head 约为它的 1/20。
- 串行的只有「查表 → 加偏置 → softmax → 采样」这条链，三层 Transformer 不重跑。

**RNN head**（[D] 式 6，备选，V4.1 没用）：维护状态 $s_k\in\mathbb{R}^r$，$z_k=[s_{k-1};\,W_1[x_{k-1}];\,h_k]\in\mathbb{R}^{2r+d}$，

$$s_k=\sigma(W_gz_k)\odot s_{k-1}+(1-\sigma(W_gz_k))\odot\tanh(W_cz_k),\qquad B_k(x_{<k},\cdot)=W_2^\top\tanh(W_oz_k)$$

$s_0=0$。论文结论：RNN head 只比 Markov head 多一点点，主要在更长的块上；实现更复杂、部署性质更差，默认用 Markov（[D] §4.3.2）。

**为什么不用别的结构化输出层**（[D] §6 末段）：投机解码的拒绝采样需要草稿在每个 token 上的**精确概率** $q(x_k)$。CRF-NAT 是全局归一化（配分函数），拿不到逐 token 概率；CTC-drafter 要对对齐路径求边缘，只能做 greedy 验证。DSpark 的修正是局部的，每个 token 的概率仍是一次 softmax。

### 2.5 confidence head [D] §3.2.1

对每个草稿位置 $k$ 输出标量 $c_k\in(0,1)$，含义是**条件概率**：在块内前面的 token 都被接受的前提下，位置 $k$ 的草稿 token 通过验证的概率。

$$c_k=\sigma\big(w^\top[h_k;\ W_1[x_{k-1}]]\big)\qquad\text{（式 7）}$$

$h_k$ 是主干 hidden，$W_1[x_{k-1}]$ 是前一个草稿 token 的 Markov embedding。监督信号是解析的单步接受率（式 8，即 2.1 的 $\beta$）：

$$c_k^*=1-\tfrac12\lVert p_k^d-p_k^t\rVert_1$$

注意 $c_k$ 的输入里有 $x_{k-1}$ 但没有 $x_k$：它预测的是「这个位置的草稿分布与目标分布有多重合」，是对 $x_k$ 求了期望的量，不是「$x_k$ 这个具体 token 会不会被接受」。这一点对无损性很关键（见 2.7）。

**前缀存活概率**：验证从左到右，第一个被拒绝的位置之后全部作废，所以位置 $j$ 的 token 最终被接受的概率是连乘

$$a_j=\prod_{i\le j}c_i$$

**Sequential Temperature Scaling (STS)**：调度器要用 $a_j$ 的**绝对数值**算期望接受数，只排对序不够；神经网络的置信度通常偏高。做法：在留出的验证集上，从左到右逐个位置 $k\in\{1..\gamma\}$ 做一维网格搜索，找一个温度标量使累积乘积 $\prod_{i\le k}c_i$ 的 ECE 最小，前面已校准的位置固定。温度缩放保序。效果（[D] §4.3.3，Qwen3-4B）：原始 ROC-AUC 0.81–0.90，ECE 3%–8%；STS 后平均 ECE 约 1%。

### 2.6 调度器：用吞吐曲线选验证长度 [D] §3.2.2

**设定。** 一个 batch 有 $R$ 个活跃请求。请求 $r$ 的置信度序列 $c_{r,1..\gamma}$，验证长度 $\ell_r\in\{0..\gamma\}$，$a_{r,j}=\prod_{i\le j}c_{r,i}$。

- 一次验证送进目标模型的 token 总数：$B=\sum_{r=1}^R(1+\ell_r)$（每个请求的 anchor 占 1 个）。
- 期望接受的 token 数：$\tau=\sum_{r=1}^R\big(1+\sum_{j=1}^{\ell_r}a_{r,j}\big)$。

  （补，这个式子的来历）设请求 $r$ 接受的草稿数为 $N_r$，$\mathbb{E}[N_r]=\sum_{j\ge1}P(N_r\ge j)=\sum_{j=1}^{\ell_r}a_{r,j}$；再加上每轮必出的 1 个 bonus token。
- $\mathrm{SPS}(B)$：引擎在前向 batch 大小为 $B$ 时每秒能跑多少步（steps per second）。**引擎初始化时 profile 一次**，存成一张成本表。假设吞吐主要由 $B$ 决定（[D] 脚注 2 给了理由：实际上下文长度远低于 1M，对 V4 这类结构 decode 延迟影响小；prefill/decode 分离部署下负载均衡会把各 rank 的请求数和上下文总长摊平）。

**目标函数**：最大化期望的全系统 token 吞吐

$$\Theta=\tau\cdot\mathrm{SPS}(B)\qquad[\text{token/step}\times\text{step/s}=\text{token/s}]$$

**贪心解。** $a_{r,j}$ 对 $j$ 单调不增，把 $\ell_r$ 从 $j-1$ 延长到 $j$ 的边际收益正好是 $a_{r,j}$。所以把所有请求的所有 $(r,j)$ 放进一个池子按 $a_{r,j}$ 降序排，排序自动满足「同一请求内前缀先于后缀」。给定 $B$，最优分配就是取池子里最大的若干个；剩下只需沿这条贪心路径扫一遍 $B$。

**Algorithm 1**（[D]）：

```
输入: 请求 r=1..R, 每个请求的 c_{r,1..γ}, profile 好的 SPS(B)
1  a_{r,j} ← ∏_{i≤j} c_{r,i}
2  E ← {(r,j) | a_{r,j} > 0}，按 a_{r,j} 降序
3  ℓ_r ← 0;  B ← R;  τ* ← R
4  Θ_best ← R·SPS(R);  ℓ*_r ← 0
5  for (r,j) in E:
6      ℓ_r ← j;  B ← B+1;  τ* ← τ* + a_{r,j}
7      Θ ← τ*·SPS(B)
8      if Θ > Θ_best:  Θ_best ← Θ;  ℓ*_r ← ℓ_r
9      else: break
10 return ℓ*
```

（补，把第 8 行的条件解出来，得到「负载相关的阈值」）多收一个存活概率为 $a$ 的 token 值得，当且仅当

$$(\tau+a)\,\mathrm{SPS}(B+1)>\tau\,\mathrm{SPS}(B)\iff a>\tau\Big(\frac{\mathrm{SPS}(B)}{\mathrm{SPS}(B+1)}-1\Big)$$

右边是「batch 多一个 token 让所有已收 token 慢下来的比例 × 已有的期望产出」。负载轻时 SPS 曲线几乎是平的，右边接近 0，几乎任何 $a>0$ 的 token 都值得验；负载重时 SPS 随 $B$ 掉得快，阈值变高，只有高置信的前缀留下。静态阈值相当于把右边固定成常数，这就是论文说它在高并发下次优的原因（[D] §3.2.2 第一段、§4.3.3）。

### 2.7 是否无损：non-anticipating 条件 [D] §3.2.2、Appendix A

验证/接受规则本身就是 2.1 的标准规则（[D] §2.1：以 $\min(1,p_k^t(x_k)/p_k^d(x_k))$ 接受，从左到右，首次拒绝后全部丢弃；[D] §1：接受规则「精确保持目标分布」）。新增的问题只在**截断**：调度器决定 $\ell_r$ 时不能偷看将被它决定去留的 token。

- **要求**（non-anticipating）：「第 $k$ 个草稿 token 是否送验」这件事必须由采样 $x_{r,k}$ **之前**就可见的信息决定，不能依赖 $x_{r,k}$ 的取值。
- **隐患**：$c_{k+1}$ 的输入含 $W_1[x_k]$，所以 $a_{k+1}$ 依赖 $x_k$ 的取值。如果调度器先算完所有 $\Theta$ 再回头取全局最大（没有 break），$x_k$ 的取值就通过 $a_{k+1}$ 影响了「$x_k$ 是否被送验」。
- **Appendix A 的反例**（数字我逐个核过）：$R=1$，$\gamma=2$，$a_1=0.8$，$\mathrm{SPS}(1)=1.0,\ \mathrm{SPS}(2)=0.5,\ \mathrm{SPS}(3)=0.45$。$\Theta_0=1.0$，$\Theta_1=1.8\times0.5=0.9$。
  - 若 $x_1$ 使 $c_2=0.9$：$\Theta_2=(1+0.8+0.72)\times0.45=1.134$，全局最大，$\ell=2$，$x_1$ 被送验。
  - 若 $x_1$ 使 $c_2=0$：$\Theta_2=0.81$，最大仍是 $\Theta_0$，$\ell=0$，$x_1$ 不送验，目标模型自己从 $p_t$ 采。
  - 词表 $\{A,B\}$，$p_t=(0.7,0.3)$，$p_d=(0.5,0.5)$（$\sum\min=0.8$，与 $a_1$ 一致）。设 $x_1=A$ 走第一种、$x_1=B$ 走第二种：$x_1=A$ 时接受概率 $\min(1,0.7/0.5)=1$，输出 A；$x_1=B$ 时目标模型重采。$\Pr(Y=A)=0.5\times1+0.5\times0.7=0.85\ne0.7$。有偏。
- **解法（Algorithm 1 的 break）**：吞吐一下降立刻停。上例里 $\Theta_1<\Theta_0$，直接返回 $\ell=0$，根本不会去算依赖 $x_1$ 的 $c_2$。截断决定只依赖到当前步为止的前缀。代价：早停拿到全局最优**当且仅当 $\Theta$ 单峰**，这隐含假设 SPS 曲线平滑下降。

（补）一个值得在正文点明的细节：$c_k$ 依赖 $x_{k-1}$ 而不依赖 $x_k$，所以「是否收第 $k$ 个」只用到 $x_{<k}$，这是合法的；非法的是用 $c_{k+1}$ 来决定第 $k$ 个的去留。

**生产环境的改动**（[D] §5.2）：

1. 真实 $\mathrm{SPS}(B)$ 是离散的、锯齿状阶梯下降，不单峰，早停会卡在局部。
2. 逐步动态决定草稿 token 数与 CUDA graph replay、Zero-Overhead Scheduling (ZOS) 冲突：ZOS 要求当前步结束前就知道下一步的 batch 大小。
3. 做法：调度器**异步**。用**两步之前**的 confidence head 输出来估计本步的验证容量，只用它决定截断长度（batch 容量上限 $K$）；本步的候选 token 仍按**当前**的累积置信度排序。于是变成一个动态 top-$K$ 选择。
4. 在这个异步管线上**去掉 break，做无约束全局搜索**。按 Appendix A 这本该破坏无损性，但全局搜索只看两步之前的历史预测，截断长度只依赖两步之前可得的信息，与当前 token $x_{r,k}$ 的取值隔离，「异步设计形成了因果屏障」，仍保持目标分布。

（我的疑问，放第 8 节）第 3 点里「候选按当前的累积置信度排序」而当前的 $a_{r,j}$ 依赖本轮已采的 $x_{r,<j}$；论文只论证了「截断长度 $K$」不依赖当前 token，对「top-$K$ 在请求之间怎么分」是否满足 non-anticipating 没有展开证明。正文若写到这里，照论文原话转述，不要替它补证明。

### 2.8 训练 [D] §3.3、§5.1；[P] §2.4.3

- 从每条目标序列随机采多个 anchor 位置，各取 $\gamma$-token 块。目标模型冻结；草稿共享其 embedding 和 LM head 且冻结；只更新并行主干、串行头、confidence head。
- 位置权重 $w_k=\exp(-(k-1)/\gamma)$（沿用 DFlash），前面的位置权重大，因为前缀验证下它们对期望接受长度贡献更大。
- 三项损失：

  $$\mathcal{L}_{\text{ce}}=-\sum_{k=1}^{\gamma}w_k\log p_k^d(x_k^*)\quad(9)\qquad \mathcal{L}_{\text{tv}}=\sum_{k=1}^{\gamma}w_k\lVert p_k^d-p_k^t\rVert_1\quad(10)$$

  $$\mathcal{L}_{\text{conf}}=-\sum_{k=1}^{\gamma}w_k\big[c_k^*\log c_k+(1-c_k^*)\log(1-c_k)\big]\quad(11)$$

  $$\mathcal{L}=\alpha_{\text{ce}}\mathcal{L}_{\text{ce}}+\alpha_{\text{tv}}\mathcal{L}_{\text{tv}}+\alpha_{\text{conf}}\mathcal{L}_{\text{conf}},\qquad \alpha_{\text{ce}}=0.1,\ \alpha_{\text{tv}}=0.9,\ \alpha_{\text{conf}}=1.0\quad(12)$$

  $x_k^*$ 是 ground-truth token。$\mathcal{L}_{\text{tv}}$ 直接是接受率的代理（单步接受率 $=1-\tfrac12\lVert p^d-p^t\rVert_1$），所以主项是 TV 而不是 CE/KL。$\mathcal{L}_{\text{conf}}$ 是对软标签 $c_k^*$ 的二元交叉熵。
- 「噪声 token」：论文叫 **mask token**，代码/config 叫 `noise_token`（`dspark_noise_token_id = 128799`）。论文没有描述任何加噪/去噪过程，它就是占位符。**不要写成扩散式去噪**。
- 训练的系统优化（[D] §5.1，HAI-LLM 框架）：(a) 不在 worker 间传目标模型的全词表 logits（$V\approx10^5$），只传 LM head 之前的 hidden，LM head 投影在草稿侧本地做，且只对采到的位置做，每 token 通信量 $O(d)$；(b) anchor-bounded sequence packing：固定数量的 anchor，把孤立的预测块打包成稠密 batch，用 token 级 attention 索引而不是 2D mask 维持因果性。
- V4.1 的训练阶段（[P] §2.1 209 行、§2.4.3 315 行）：主干预训练**不带 MTP**；预训练结束后单独一个阶段，冻结主干只训 DSpark；后训练阶段 DSpark 跟着主干继续训，但 **DSpark 目标的梯度不回传到主干**，让 DSpark 跟上不断变化的策略，从而既加速在线服务，也加速 RL 和 OPD 的 rollout 生成。

### 2.9 DSpark 与 MTP、与「小模型当 draft」各差在哪（汇总，条条有出处）

| | 小模型当 draft（[L]） | DeepSeek-V3 MTP（[V3] §2.2） | DSpark（[D]、[P]、[C]） |
|---|---|---|---|
| 草稿从哪来 | 独立的小模型，自己从头读整个前缀（[L] §3.6：通常比目标小两个数量级） | 主干最后的 hidden + 下一个 token 的 embedding，过 1 个 block | 主干第 37–39 层入口的 hidden 做 KV 上下文，3 个 block |
| 多个 token 怎么出 | 自回归跑 $\gamma$ 次小模型，成本 $\gamma c$ | $D$ 个模块串行，猜几个就要几个 block；V3 实际 $D=1$ | 一次前向 5 个位置 + Markov head 串行采样 |
| 块内依赖 | 完整 | 完整（因果链） | 一阶 Markov（低秩 bigram 偏置） |
| 训练 | 与目标模型无关 | **与主干联合预训练**，交叉熵，主要目的是提升主模型（「推理时可丢弃」） | 主干预训练后单独阶段，主干冻结；后训练阶段继续训但梯度不进主干；损失以 TV 为主 |
| 验证长度 | 固定 $\gamma$ | 固定（MTP-1 每请求验 2 个 token，[D] §5.4） | confidence head + 调度器按负载动态选 |
| 生产状况 | — | V4-preview 发布时的生产配置是 MTP-1，**两周后被 DSpark 取代**；MTP-3/5 这种静态多 token 草稿在高并发下严格降低总吞吐，所以一直只用 MTP-1（[D] §5.4） | V4-Flash/Pro（preview）与 V4.1-Flash |

共同点：三者的接受规则都是 2.1 的标准规则，都共享/可共享 embedding 与 LM head（MTP、DSpark）。

## 3. 代码落点

仓库：HF `deepseek-ai/DeepSeek-V4.1-Flash`，sha `dba1be0a`，`inference/model.py`（本地 `drafts/deepseek-v4/refs/v41-inference-model.py`，下称 model.py）。**参考实现只有前向**：`ModelArgs` 注释原文「Only the forward pass is implemented here -- nothing calls forward_spec, so these are read but the speculative-decoding loop itself is out of scope for this repo」（129–130 行）；`inference/README.md` 也写「Generation itself is plain autoregressive sampling」。没有验证、接受、sigmoid/STS 校准、调度器的任何代码。

### 3.1 config（[CF]）

| 推理 config 字段 | HF config 字段 | 值 | 含义 |
|---|---|---|---|
| `n_mtp_layers` | `num_nextn_predict_layers` | 3 | DSpark 层数，字段名沿用 MTP |
| `dspark_block_size` | 同名 | 5 | 草稿输入位置数 = 草稿 token 数 |
| `dspark_noise_token_id` | 同名 | 128799 | 占位 token |
| `dspark_target_layer_ids` | 同名 | [37, 38, 39] | 读主干哪几层 |
| `dspark_markov_rank` | 同名 | 256 | $r$ |
| `dspark_n_routed_experts` | 同名 | 128 | 主干是 384 |
| `dspark_n_activated_experts` | **`dspark_num_experts_per_tok`** | 3 | 主干是 6。两个 config 字段名不同 |
| `compress_ratios[40:43]` | 同名 | 0, 0, 0 | 三层都是纯滑窗 |
| `window_size` | `sliding_window` | 128 | |

权重存在 checkpoint 的 `mtp.*` 命名空间下（`DSparkBlock` docstring；`v41-index.json` 里有 `mtp.0.main_proj.weight`、`mtp.0.main_norm.weight`、`mtp.2.confidence_head.proj.weight` 等）。`mtp.*.embed.weight` / `head.weight` 在转换时被跳过，与主干绑定（convert.py 108–109 行注释「an MTP layer ties its token embedding and output head to the backbone's」）。

### 3.2 主干侧：取哪几层的什么（model.py `Transformer.forward`，1255–1264 行）

```python
for i, layer in enumerate(self.layers):
    if layer.engram is not None:
        h = layer.engram(h, engram_hashes[:, :, layer.engram.layer_hash_index, :], engram_mask)
    # the MTP head reads the attention input of its target layers, not their output
    if i in self.target_layer_ids:
        main_hiddens.append(h.mean(dim=2))
    h, pre_mix = layer(h, start_pos, pre_mix, image_mask)
...
main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
return output_ids, logits, main_hidden
```

- 取的是第 37、38、39 层 **block 入口**处 4 条残差流（`[b,s,4,d]`）的**算术平均**，`dim=2` 是 hc 维。等价于第 36、37、38 层的输出。**第 39 层（最后一层）的输出没有被读**。
- 是直接平均，不是 `hc_pre(x, pre_mix)` 那个加权读，也没过 `attn_norm`。代码注释叫它「attention input」，严格说是「block 入口的 4 流均值」。
- 拼成 $3\times5120=15360$ 维。

### 3.3 `forward_spec` 与 `forward_embed`（1274–1282、1128–1135 行）

```python
def forward_spec(self, input_ids, main_hidden, start_pos=0):
    h, main_x = self.mtp[0].forward_embed(main_hidden, input_ids)
    pre_mix = make_identity_pre_mix(h, self.hc_mult)
    for layer in self.mtp:
        h, pre_mix = layer(h, start_pos, pre_mix, main_x)
    if start_pos == 0:
        return None
    return self.mtp[-1].forward_head(h, pre_mix, input_ids)

def forward_embed(self, main_hidden, input_ids):
    main_x = self.main_norm(self.main_proj(main_hidden))
    draft_input_ids = input_ids.new_full([input_ids.size(0), self.block_size], self.noise_token_id)
    draft_input_ids[:, 0] = input_ids
    x = self.embed(draft_input_ids)
    x = x.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
    return x, main_x
```

- `main_x = RMSNorm(main_proj(main_hidden))`，`main_proj: Linear(15360 → 5120)`。对应 [D] 式 2 的 $H_{\text{ctx}}$。只有第 0 个 DSpark block 有 `main_proj`/`main_norm`，**同一个 `main_x` 传给三层**，每层用自己的 `wkv` 投成 KV（式 3）。
- 草稿输入 = `[刚采样的 token, noise, noise, noise, noise]`，共 5 个位置，过**主干的** embedding，复制成 4 条流。对应 [D] §3.1「anchor + $\gamma-1$ 个 mask」。
- 三层自己有完整的 mHC（index 里有 `mtp.N.hc_attn_fn` 等），第一层的 `pre_mix` 是 one-hot（只读第 0 条流），和主干第 0 层一样。
- `start_pos == 0`（prefill）时只把 `main_x` 写进各层的窗口 KV cache，返回 `None`，不出草稿（`DSparkBlock.forward` 1122–1126 行）。
- 调用方式见 `__main__`（1301–1309 行）：`output_ids, logits, main_hidden = model(x[:, i:i+1], i)` 之后 `model.forward_spec(output_ids, main_hidden, i)`，返回 `(output_ids, logits, confidence)`。

### 3.4 `DSparkAttention`（1032–1074 行）与 `get_dspark_topk_idxs`（1021–1029 行）

```python
main_kv = self.kv_norm(self.wkv(main_x))            # 上下文只做 KV
apply_rotary_emb(main_kv[..., -rd:], main_freqs_cis) # 位置 start_pos
...
freqs_cis = self.freqs_cis[start_pos + seqlen : start_pos + seqlen + block_size]
q  = wq_b(q_norm(wq_a(x)))                           # 草稿 5 个位置做 query
kv = self.kv_norm(self.wkv(x))                       # 草稿自己也做 KV，同一个 wkv
topk_idxs = get_dspark_topk_idxs(win, bsz, block_size, start_pos)
self.window_kv_cache[:bsz, start_pos % win] = main_kv.squeeze(1)
kv = torch.cat([self.window_kv_cache[:bsz], kv], dim=1)
o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
```

```python
matrix = torch.cat([torch.arange(min(window_size, start_pos + 1)),
                    window_size + torch.arange(block_size)])
return matrix.int().view(1, 1, -1).expand(bsz, block_size, -1).contiguous()
```

- KV 集合 = 环形缓冲里最近 ≤128 个位置的 `main_kv`（槽位 `start_pos % win`，和主干滑窗层 `_window_kv` 同一种写法）+ 5 个草稿位置自己的 kv，共最多 133 条。
- **5 个草稿位置用的是同一份索引**：每个位置都能看到全部 5 个草稿位置（包括自己右边的）。块内是双向注意力，没有因果 mask，和 [D] §2.2「bidirectionally」一致。这也说明 5 个位置的 hidden 里没有任何「已采样 token」的信息，依赖只能靠后面的 Markov head 补。
- 位置编码：上下文在 `start_pos`，草稿 5 个位置在 `start_pos+1 .. start_pos+5`。anchor 是序列第 `start_pos+1` 个 token，第 $i$ 个草稿位置（0 起）的 logits 预测的是第 `start_pos+2+i` 个 token。
- anchor 这个 token 本身**没有**对应的 `main_x`（主干还没跑过它），它只以 embedding 的形式作为草稿的第 0 个输入位置。
- `assert self.compress_ratio == 0`：纯滑窗，没有压缩分支和 indexer。KV 同样过 FP8 `act_quant`。
- MoE：`get_moe_config`（143–148 行）对 `layer_id >= n_layers` 返回 `(128, 3)`；`DSparkBlock.forward` 传 `image_mask=None`（注释「drafts are text: no VL bias」）。

### 3.5 `DSparkMarkovHead`、`DSparkConfidenceHead`、`forward_head`（1077–1098、1137–1156 行）

```python
class DSparkMarkovHead(nn.Module):
    def __init__(self, vocab_size, dspark_markov_rank):
        self.embed = ParallelEmbedding(vocab_size, dspark_markov_rank)   # W1: V×r
        self.head  = ParallelHead(vocab_size, dspark_markov_rank)        # W2: r→V
    def forward(self, token_ids):
        embed = self.embed(token_ids)
        logits = self.head(embed, full_logits=True)
        return logits, embed

class DSparkConfidenceHead(nn.Module):
    def __init__(self, input_dim):
        # proj in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for fp32 confidence score.
        self.proj = Linear(input_dim, 1, dtype=torch.float32)
    def forward(self, hidden, markov_embed):
        hidden = torch.cat([hidden, markov_embed], dim=-1)
        return self.proj(hidden.float()).squeeze(-1)
```

```python
def forward_head(self, x, pre_mix, input_ids):
    x = self.hc_pre(x, pre_mix)
    logits = self.head(self.norm(x), full_logits=True)       # 主干的 LM head，5 个位置
    output_ids = input_ids.new_empty(input_ids.size(0), self.block_size + 1)
    output_ids[:, 0] = input_ids
    markov_embeds = []
    for i in range(self.block_size):
        logits_bias, markov_embed = self.markov_head(output_ids[:, i])
        logits[:, i].add_(logits_bias)
        markov_embeds.append(markov_embed)
        output_ids[:, i + 1] = sample(logits[:, i], self.temperature)
    markov_embed = torch.stack(markov_embeds, dim=1)
    confidence = self.confidence_head(x, markov_embed)
    return output_ids, logits, confidence
```

逐条对论文：

- `logits[:, i] += W2(W1[output_ids[:, i]])` 就是式 (5)，`output_ids[:, i]` 是位置 $i$ 的前驱：$i=0$ 时是 anchor，之后是刚采出的草稿 token。**第 0 个位置也加偏置**。
- 循环 5 次，每次一次查表 + 一次 256→129280 的线性 + 一次采样；三层 Transformer 和 5120→129280 的 LM head 都在循环外，只算一次。
- 返回的 `logits` 是加完偏置的，也就是验证时要用的草稿分布 $p^d_k$（softmax 前）。`output_ids` 有 6 个：anchor + 5 个草稿。
- 采样用 `sample(logits, temperature)`（Gumbel-max 的等价写法，1285 行），与主干同一个温度。
- 输出 norm：最后一个 DSpark block 有**自己的** `self.norm`（1116 行），LM head 权重与主干共用（1212 行 `self.mtp[-1].head = self.head`）。V4.1 的输出端没有 `hc_head`，用最后一层 FFN 给出的 `pre_mix` 做 `hc_pre`（Single-Pass mHC，连载第 4 篇）。
- confidence head：输入 = `hc_pre` 之后、**`norm` 之前**的 `x`（5120 维）拼前驱 token 的 Markov embedding（256 维）= 5376 维 → 1，无 bias（index 里只有 `proj.weight`），fp32。**代码返回的是 logit，没有过 sigmoid**，也没有 STS 的温度；式 (7) 的 $\sigma$ 和校准都在仓库之外。
- confidence 的输入只含 $x_{k-1}$ 不含 $x_k$，与式 (7) 一致。

### 3.6 参数量（补，按 config 手算，论文没给）

- 每个 expert：$3\times5120\times2304=35.4$M；每层 128 routed + 1 shared = 129 个 → 4.57B。
- 每层注意力：`wq_a` 5120×1280 + `wq_b` 1280×32768 + `wkv` 5120×512 + `wo_a` 8×1024×4096 + `wo_b` 8192×5120 ≈ 0.13B。
- 三层 ≈ 14.1B；`main_proj` 15360×5120 = 78.6M；Markov head 2×129280×256 = 66.2M；confidence head 5376。
- 合计 ≈ 14.2B，与 PLAN 里「DSpark ≈ 14B」一致。每个草稿位置激活的参数约 (3+1)×35.4M×3 + 0.13B×3 ≈ 0.8B。

### 3.7 对任务书里「已确认事实」的核对

全部成立，补充和需要改口的地方：

1. 「读主干第 37、38、39 层入口处 4 条残差流的均值」✔。是算术平均，不是 mHC 的加权读；等价于 36–38 层的输出，最后一层输出不读。
2. 「草稿输入 = 刚采样的 token + 4 个噪声 token」✔。输出是 **5 个草稿 token**（anchor 位置本身就是第一个预测位置），不是 4 个。
3. 「KV = 最近 128 个 main_x 的滑窗 + 5 个草稿位置自己」✔。补：块内双向；anchor 没有 main_x；三层共用同一个 `main_x`，各自过自己的 `wkv`。
4. 「输出头与主干共用 LM head」✔。embedding 也共用；但最终 RMSNorm 是 DSpark 自己的。
5. 「Markov head = rank 256 的 embed × head，给位置 i 的 logits 加由前一个草稿 token 决定的偏置，逐位置串行采样」✔。位置 0 的「前一个 token」是 anchor。
6. 「confidence head 输入是 hidden 拼 markov embed」✔。hidden 是 norm 之前的；输出是 logit；位置 $i$ 拼的是前驱 token 的 embedding。
7. 字段名：HF config 是 `dspark_num_experts_per_tok`，推理 config 是 `dspark_n_activated_experts`。

## 4. 关键数字

| 数字 | 条件 | 出处 |
|---|---|---|
| 3 个 block、滑窗 128、一次前向 5 个草稿位置 | V4.1-Flash | [P] §2.4.3；[CF] |
| 3 层 MoE + mHC、滑窗 128、最大块 $\gamma=5$、Markov head | V4-Flash / V4-Pro (preview) 上的部署配置，与 V4.1 相同 | [D] §5.1 |
| $r=256$ | Markov head 默认秩 | [D] §3.1；[CF] |
| $\alpha_{\text{ce}}=0.1,\ \alpha_{\text{tv}}=0.9,\ \alpha_{\text{conf}}=1.0$；$w_k=e^{-(k-1)/\gamma}$ | 默认损失权重（离线实验；V4.1 是否相同未说明） | [D] §3.3 |
| 接受长度 $\tau$ 的宏平均：DSpark 比 Eagle3 高 30.9% / 26.7% / 30.0%，比 DFlash 高 16.3% / 18.4% / 18.3% | 目标 Qwen3-4B / 8B / 14B；9 个 benchmark；标准投机采样，温度 1.0；块大小 7；Eagle3 1 层、DFlash 与 DSpark 5 层；链式草稿；**关闭置信度调度**；$\tau$ 含 bonus token | [D] §4.1–4.2、Table 1（4B 的 30.9% 我按表里 9 个数复算过：4.727 / 3.611） |
| Table 1 单项，Qwen3-4B：GSM8K 5.14 / 5.40 / 6.11；MBPP 3.69 / 4.40 / 5.13；MT-Bench 2.39 / 3.07 / 3.64（Eagle3 / DFlash / DSpark） | 同上 | [D] Table 1 |
| Gemma4-12B 上 Eagle3 反而高于 DFlash（如 GSM8K 5.87 vs 5.45），DSpark 仍最高（6.05） | 同上 | [D] Table 1 |
| 领域差异：DSpark 在 Qwen3-4B 上 math 5.57、code 5.12、chat 3.49 | 三个 benchmark 的平均 | [D] §4.2 |
| 位置 1 的条件接受率：DFlash 0.88 vs Eagle3 0.81（Math），0.72 vs 0.53（Chat）；DSpark 在 Math 上从 0.93 起 | Qwen3-4B，逐位置条件接受率（分母只计前面全被接受的样本） | [D] §4.3.1、Figure 2 |
| 后缀衰减：DFlash 0.87→0.78（Code）、0.72→0.63（Chat）；Eagle3 0.53→0.74（Chat） | 同上，位置 1→7 | [D] §4.3.1 |
| 2 层 DSpark 超过 5 层 DFlash；1→2 层的边际收益最大 | 块大小 7，Qwen3-4B，三个领域都成立 | [D] §4.3.2、Figure 3 |
| DSpark 相对 DFlash 的提升随块变长而扩大：$\gamma=7$ 时 math/code/chat +16% / +15% / +18%；$\gamma=15$ 时 +30% / +26% / +22% | 5 层，草稿长度（$\gamma$+1 个 anchor）∈{4,8,12,16} | [D] §4.3.2、Figure 4 |
| 串行头的延迟开销：草稿长度 4→16，整轮延迟比 DFlash 多 0.2%–1.3% | 一轮 = 一次目标验证 + 并行草稿前向 + 串行采样循环；batch 128；上下文 {512,1024,2048,4096} 取平均 | [D] §4.3.2、Figure 4 右 |
| 静态阈值扫描：阈值升高后整体接受率 Chat 45.7%→95.7%，Math 76.9%→92.5%，Code 67.6%→92.0% | Qwen3-4B，离线，阈值 0 = 固定长度验证 | [D] §4.3.3、Figure 5 |
| confidence head：ROC-AUC 0.81–0.90；原始 ECE 3%–8%，STS 后约 1% | Qwen3-4B，Figure 6 是 Alpaca | [D] §4.3.3 |
| 线上：同等总吞吐下每用户生成速度 +60%–85%（V4-Flash）、+57%–78%（V4-Pro） | 真实用户流量，DSpark-5 对 MTP-1 | [D] 摘要、§5.4 |
| 线上：SLA 80 tok/s/user 时总吞吐 +51%（Flash）；SLA 35 tok/s/user 时 +52%（Pro） | 同上 | [D] §5.4 |
| 线上：SLA 120 tok/s/user 时名义 +661%（Flash）；SLA 50 时名义 +406%（Pro） | MTP-1 在此 SLA 下只能维持极小并发；**论文自己说这是可行域被扩展的证据，不是有代表性的倍数** | [D] §5.4 |
| 每请求验证预算：MTP-1 固定 2 个 token → DSpark 约 4–6 个；并发升高后平滑下降 | 中等并发（Flash < 200、Pro < 150 并发请求） | [D] §5.4、Figure 8 |
| MTP-1 是 V4-preview 发布时的生产配置，两周后被 DSpark 取代 | | [D] §5.4 |
| V3 MTP：$D=1$；$\lambda=0.3$（前 10T）→ 0.1（后 4.8T）；第二个 token 接受率 85%–90%；TPS 1.8× | DeepSeek-V3 | [V3] §4.2、§5.4.3 |
| [L]：小模型 draft 的 $c<0.05$；bigram 模型当 draft $\alpha\approx0.2$，$\gamma=3$ 时 1.25× | T5-XXL 11B，英德翻译 | [L] §3.3、§3.6 |
| [L]：有 oracle 选 $\gamma$ 时 walltime 提升最多再高约 60% | 理论上界，假设算力无限 | [L] §3.5 |
| DSpark 参数量 ≈ 14.2B；Markov head 66M | 按 config 手算（补） | [CF]、[C]，见 3.6 |

## 5. 常见误解与澄清

1. **「DSpark 就是 MTP 加深到 3 层 / 猜 5 个」。** 不是。MTP 每多猜一个 token 就多串一个 block，且与主干联合预训练、首要目的是提升主模型（[V3] §2.2）。DSpark 三层只跑一次出 5 个位置，主干冻结后单独训（[P] §2.4.3）。checkpoint 命名空间仍叫 `mtp.*`、config 字段仍叫 `num_nextn_predict_layers`，容易看错。
2. **「一次前向出 5 个 token，完全并行」。** logits 的主体是并行的，采样是串行的 5 步，每步要查表加偏置（`forward_head` 的 for 循环）。论文叫 semi-autoregressive。
3. **「Markov head 是一张 $V\times V$ 的 bigram 表」。** 是秩 256 的分解，两张 $V\times256$ 的表（[D] 式 5，[C]）。
4. **「噪声 token = 扩散/去噪」。** 论文叫 mask token，只是占位输入，一次前向，没有迭代去噪（[D] §2.2、§3.1）。
5. **「confidence 就是草稿自己的最大概率 / 草稿 token 的概率」。** 是单独一个线性头，监督信号是 $1-\tfrac12\lVert p^d-p^t\rVert_1$，即草稿分布与目标分布的重合度（[D] 式 7–8）。输入不含当前草稿 token。
6. **「$c_k$ 是第 $k$ 个 token 被接受的概率」。** 是条件概率（前面都接受的前提下）；真正的存活概率是连乘 $a_k=\prod_{i\le k}c_i$（[D] §3.2.1；[P] §2.4.3 的原话也是 conditional acceptance probabilities → prefix survival probabilities）。
7. **「按置信度截断会改变输出分布 / 是有损加速」。** 接受规则仍是标准拒绝采样；截断只要满足 non-anticipating 就无损，Algorithm 1 的 break 就是为此（[D] §3.2.2、Appendix A）。反过来，「随便怎么截都无损」也不对：Appendix A 的反例里输出分布从 (0.7, 0.3) 变成 (0.85, 0.15)。
8. **「验证得越多越快」。** [L] 的 walltime 分析假设并行验证不花额外时间；高并发下多验的 token 占 batch 容量。DeepSeek 自己的生产经验是 MTP-3/5 在高并发下严格降低总吞吐，所以此前只用 MTP-1（[D] §5.4）。
9. **「DSpark 加速 661%」。** 那是 MTP-1 基线在 120 tok/s/user 的 SLA 下几乎撑不住时的名义比值，论文明确说不应当作代表性加速比。可引用的是同吞吐下每用户速度 +60%–85%（[D] §5.4）。
10. **「自回归草稿一定比并行草稿准」。** 同延迟预算下并行草稿能用更深的网络，第一个位置更准，而第一个位置杠杆最大（它被拒整块作废）；Table 1 里 DFlash 在 Qwen3 上普遍高于 Eagle3（[D] §4.3.1）。但 Gemma4-12B 上 Eagle3 高于 DFlash，不要说成「总是」。
11. **「接受长度 5.x 意味着猜了 5 个都对」。** $\tau$ 含目标模型给的 bonus token（[D] 脚注 4），且离线实验的块大小是 7，不是 V4.1 的 5。
12. **「DSpark 论文的加速数字是 V4.1 的」。** 离线数字是 Qwen3/Gemma4，线上数字是 V4-Flash/Pro preview。V4.1 报告没有给 DSpark 的数字。
13. **「官方仓库能跑投机解码」。** 只有 `forward_spec` 前向，没有任何地方调用它；验证、调度、校准都没有（[C] 129–130 行注释）。
14. **「DSpark 是独立的小模型」。** 它不自己读上下文：上下文信息全部来自主干第 37–39 层的 hidden，embedding 和 LM head 也是主干的。

## 6. 与本专栏其他文章的接口

grep `src/content/posts/` 中 "MTP / 投机解码 / speculative / DSpark" 的结果：

**已讲过，只需引用：**

- `deepseek-v4-06-muon.mdx` 的「## MTP」节（112–121 行）：V4 的 MTP 是一个完整 decoder block、输入是 `h_proj` + `e_proj`、只有滑窗分支、有自己的 `hc_head`、损失权重 0.3→0.1。**对照点**：V4.1 没有 MTP，DSpark 占了原来 MTP 的位置（`mtp.*`、`compress_ratios` 末尾）。该文 121 行写「论文没有说 MTP 在部署时是否用于投机解码」，现在 DSpark 论文 §5.4 给了答案（MTP-1 是 V4-preview 的生产草稿，两周后被 DSpark 取代）。第 6 篇可以一句话更正，V4 那篇要不要回头改由主代理定。
- `deepseek-v4-03-hybrid-attention.mdx` 43 行、`deepseek-v4-00-overview.mdx` 65 行：MTP 层是纯滑窗。DSpark 三层同理，可直接类比。
- `kimi-k3-06-vision-and-parts.mdx` 的「## MTP 层与 EAGLE-3 draft」（70–81 行）：K3 把 MTP 层微调成 EAGLE-3 风格的自回归 draft，读低/中/高三层特征拼接，展开 7 步训练，LK 损失；已写出「接受率 $\sum_x\min(p(x),q(x))$」。**对照点**：都读目标模型多层特征、都直接优化接受率（K3 用 LK 损失即接受率的负对数，DSpark 用 TV 损失，两者优化的是同一个量 $\sum\min(p,q)=1-\tfrac12\lVert p-q\rVert_1$）；差别是 K3 自回归 7 步，DSpark 一次前向 + Markov head，且 DSpark 多了调度。Eagle3 正是 DSpark 论文的自回归基线。
- `kimi-k3-07-systems.mdx` 35–37 行：投机解码被拒后的状态回滚（KDA 的递推状态）。V4.1 是注意力 + KV cache，没有这个问题，最多一句带过。
- `deepseek-v41-00-overview.mdx` 40、92、153、186–189、225、249、294 行：DSpark 的一段概述、config 表、参数账（每层约 4.7B、合计 ≈ 14B）。第 6 篇展开这些，数字保持一致。注意 225 行说「主干第 37、38、39 层的注意力输入」，与本笔记 3.2 的说法（block 入口的 4 流均值）是同一件事，正文可以说得更准。
- mHC、Single-Pass mHC 的读写、`pre_mix`：连载第 4 篇（V4 第 4 篇 `deepseek-v4-04-mhc.mdx` 195 行有 V4 的 `hc_head`）。DSpark 层自带 mHC，只引用。
- MoE（128 选 3 只是规模不同）：V4 第 5 篇 / V4.1 对应篇。

**没人讲过（缺口）：**

- 投机解码的无损验证规则与证明、$(1-\alpha^{\gamma+1})/(1-\alpha)$。`llm-*.mdx` 六篇里没有任何投机解码内容（grep 无命中）。按 post-style 的边界规则，这属于背景知识，应放「大模型笔记」专栏并从连载链接过去；现在专栏里没有这篇。第 6 篇要么用一小节自带最短版本（规则 + 接受率 = 重合度两条，本笔记 2.1），要么先补一篇 llm 专栏的投机解码文章。由主代理决定。
- Markov head、confidence head、调度器、non-anticipating：全新，归第 6 篇。

**留给别的文章：**

- 变长验证前缀在推理内核里的支持（flatten + marker tensor，只需改 index-attention 与 compress 两个 kernel，[D] §5.3）、ZOS、CUDA graph：更适合连载第 8 篇「系统」，第 6 篇点到为止。
- DSpark 加速 RL / OPD rollout（[P] §2.4.3 末句）：若后训练有单独篇目，放那里。

## 7. 图的建议

1. **一轮解码的时间线（主图）。** 横轴是 token 位置。上排主干：处理位置 $t$，输出 anchor（位置 $t+1$），同时在第 37/38/39 层入口引出三条线汇入 `main_proj`。下排 DSpark：输入 `[anchor, ⌀, ⌀, ⌀, ⌀]`，KV 来自左边 128 格的 `main_x` 窗口 + 自己 5 格（块内双向），三层一次前向出 5 列 $U_k$；再画一条从左到右的细链：$x_{k-1}$ → 查 $W_1$ → 过 $W_2$ → 加到 $U_k$ → 采 $x_k$。最后调度器在 5 个草稿上切一刀，主干并行验证前缀，标出「接受 / 拒绝 / bonus」。对应 [D] Figure 1，但把 V4.1 的具体数字（37–39、128、5、256）标上。要让读者一眼看出「粗的并行、细的串行」。
2. **"of course / no problem" 的两格对比。** 左：纯并行草稿，两个位置各自的边缘分布，独立采样得到四种组合，其中两种不通，位置 2 的接受率 0.5。右：加 Markov head 后，位置 1 采到 "of"，位置 2 的 logits 被偏置成 "course" 占主导。可以做成能点位置 1 的 token 的小交互。表达的机制：位置间独立 → 后缀接受率衰减 → 一阶偏置修掉最常见的碰撞。旁边可配 [D] Figure 2 的三条曲线示意（Eagle3 起点低但上升，DFlash 起点高但下降，DSpark 起点高且平）。
3. **调度器：同一组置信度，两种负载，两个切点。** 上半：2–3 个请求各 5 个草稿的 $a_{r,j}$ 柱子（code 类请求衰减慢，chat 类衰减快），合并后按降序排成一条。下半：$\Theta(B)=\tau(B)\cdot\mathrm{SPS}(B)$ 随收入 token 数变化的曲线，画两条 SPS（轻载几乎平、重载下降快），峰值位置不同，对应的切点投回上半的柱子。适合做成拖动「并发数」的交互。表达的机制：阈值不是常数，$a>\tau(\mathrm{SPS}(B)/\mathrm{SPS}(B+1)-1)$。SPS 的具体数值论文没给，图上只能用示意数字并注明。

（可选的第 4 张小图：Appendix A 的反例，一棵两分支的树，标出 0.85 ≠ 0.7。篇幅紧就用文字。）

## 8. 待确认 / 没查到的点

1. **V4.1 上 DSpark 的效果数字完全没有。** [P] 只有 §2.4.3 两段描述。所有数字来自 [D]（Qwen3/Gemma4 离线、V4-preview 线上）。
2. **[D] 的 Figure 1–8 我没有看到图**，只读了图注和正文。逐位置接受率曲线的完整数值、Figure 7/8 的坐标读数、SPS 曲线的形状都没有。正文若要引图里的点，需要打开 PDF 核对（scratchpad 里有 PDF，但本机没有 pdftotext / PyMuPDF）。
3. **V4.1 的 DSpark 训练超参**：损失权重是否仍是 0.1/0.9/1.0、训练数据量、每条序列采多少 anchor、训练块大小，[P] 和 [D] §5.1 都没说。[D] §3.3 的权重是离线实验的默认值。
4. **STS 的温度**不在公开 checkpoint 里（`mtp.2.confidence_head` 只有 `proj.weight`），具体数值和它在推理时加在哪未知。代码返回原始 logit。
5. **SPS(B) 成本表**的实际数值、锯齿的来源细节（论文只引了一篇文献）未给。
6. **异步调度的无损性论证只有一段文字**（[D] §5.2）：「截断长度只依赖两步之前的信息」。top-$K$ 在请求间的分配用的是当前累积置信度，这部分是否满足 non-anticipating 论文没有形式化证明。正文照原话转述，不要加强。
7. **为什么读 37、38、39 层入口而不读最后一层的输出**：论文和代码都没解释。[D] §4.1 只说「所有草稿用相同的目标模型特征层」，没列层号。不要猜原因；如果正文想提，只陈述事实。
8. **多 token 被接受后 DSpark 的窗口 KV cache 怎么补**：`DSparkAttention` 在 decode 时 `main_kv.squeeze(1)`，一次只写一个位置的 `main_x`。一轮接受 $n$ 个 token 后需要这 $n$ 个位置的 `main_x`（验证那次前向能给出），参考实现没有这段逻辑。属于「调度循环不在仓库里」的一部分。
9. **`dspark_noise_token_id = 128799` 在 tokenizer 里是哪个 token**（专用 mask token 还是借用的保留位）没查，本地没有 tokenizer.json。
10. **DeepSpec 仓库与 DSpark checkpoint 的 URL**：论文正文说开源了，但我读到的 HTML 文本里没有链接，没去找仓库，也就没有对照 DeepSpec 的训练代码。式 (9)–(12) 只有论文出处。
11. **草稿采样温度**：参考实现用全局 `args.temperature`，生产里按请求的采样参数（top-p 等）怎么处理草稿分布未知。[L] §2.2 的说法是把各种采样方式都化成对调整后分布的标准采样。
12. **RNN head 的 $W_2^\top$ 记号**：式 (6) 里 $B_k=W_2^\top\tanh(W_oz_k)$ 与式 (5) 的 $W_1[x]W_2$ 行列约定不完全一致，照抄了原文。V4.1 没用 RNN head，正文大概用不到。
13. Leviathan 论文读的是 arXiv HTML 当前版本，没有核对版本号；Theorem 3.5 的证明在 HTML 里排版破损，式子按 Lemma 3.3 + 正文重建，结论 $\beta=\sum\min(p,q)$ 无疑问。
