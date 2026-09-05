# Kimi Linear: An Expressive, Efficient Attention Architecture (arXiv 2510.26692v2) -- technical summary

Source: arXiv HTML v2 (1 Nov 2025), Kimi Team. Code: `fla/ops/kda` in flash-linear-attention; checkpoints `moonshotai/Kimi-Linear-48B-A3B-Instruct`. Notation follows the paper: $\mathbf{S}_t\in\mathbb{R}^{d_k\times d_v}$ is the per-head matrix state, $\mathbf{M}$/$\mathbf{M}^-$ are lower-triangular masks with/without diagonal (`Tril`/`StrictTril`), $\odot$ is elementwise product.

Headline claims: a 48B-total / 3B-active MoE with a 3:1 layerwise hybrid of Kimi Delta Attention (KDA) and NoPE-MLA beats a full-MLA model trained with the identical 1.4T-token recipe on short-context, long-context (128k) and RL evals, cuts KV cache by up to 75%, and reaches up to 6.3x decode TPOT at 1M context. KDA itself is Gated DeltaNet with a channel-wise (diagonal) forget gate plus a bespoke chunkwise kernel that is ~2x faster than the general DPLR kernel.

---

## 1. Lineage: linear attention -> DeltaNet -> Gated DeltaNet -> KDA

The paper frames the whole family as **online learning of a fast-weight associative memory**: each step is a gradient step of $\mathbf{S}$ on a per-token loss (Table 7 of the paper lays this out for LA, RetNet, Mamba2, GLA, HGRN2, Longhorn, Comba, RWKV7, GDN, KDA).

**Linear attention (Katharopoulos 2020).** Accumulate key-value outer products, read out with the query:
$$\mathbf{S}_t=\mathbf{S}_{t-1}+\bm{k}_t\bm{v}_t^\top,\qquad \bm{o}_t=\mathbf{S}_t^\top\bm{q}_t .$$
This is gradient descent on the *unbounded correlation* objective $\mathcal{L}_t(\mathbf{S})=-\langle\mathbf{S}^\top\bm{k}_t,\bm{v}_t\rangle$: it only ever reinforces, has no criterion for erasing, and the state grows without bound, causing interference over long contexts.

**DeltaNet (Schlag 2021; Yang 2024 for the parallel form).** Reinterpret the update as online gradient descent on a *reconstruction (key-value regression)* loss
$$\mathcal{L}_t(\mathbf{S})=\tfrac12\|\mathbf{S}^\top\bm{k}_t-\bm{v}_t\|^2 ,$$
and take one SGD step with (data-dependent) learning rate $\beta_t$:
$$\mathbf{S}_t=\mathbf{S}_{t-1}-\beta_t\nabla_{\mathbf{S}}\mathcal{L}_t(\mathbf{S}_{t-1})=(\mathbf{I}-\beta_t\bm{k}_t\bm{k}_t^\top)\mathbf{S}_{t-1}+\beta_t\bm{k}_t\bm{v}_t^\top .$$
This is the classical delta rule: the memory *corrects itself toward* the mapping $\bm{k}_t\mapsto\bm{v}_t$ (it first erases whatever value was stored under $\bm{k}_t$, then writes the new one), instead of blindly adding. The transition $(\mathbf{I}-\beta_t\bm{k}_t\bm{k}_t^\top)$ is a rank-1 generalized Householder transform, which is what makes chunkwise parallelization via the WY representation possible. What it still lacks: nothing is ever forgotten unless a matching key overwrites it.

**Gated DeltaNet (Yang et al. 2024).** Add a data-dependent *scalar* forget gate $\alpha_t\in[0,1]$ per head:
$$\mathbf{S}_t=\alpha_t(\mathbf{I}-\beta_t\bm{k}_t\bm{k}_t^\top)\mathbf{S}_{t-1}+\beta_t\bm{k}_t\bm{v}_t^\top .$$
$\alpha_t$ acts as weight decay on the fast weights (data-dependent $L_2$ regularization), controlling memory lifespan and mitigating interference while keeping the parallelizable structure. The paper also observes GDN can be read as a *multiplicative positional encoding* with a learnable, data-dependent transition matrix (relaxing RoPE's orthogonality constraint) -- this observation motivates the NoPE design in section 4 below. Limitation: GDN (like Mamba2) uses one scalar per head, so all $d_k$ channels of the state decay at the same rate.

**KDA (this paper).** Replace the scalar gate by a diagonal, channel-wise gate $\operatorname{Diag}(\bm{\alpha}_t)$, $\bm{\alpha}_t\in[0,1]^{d_k}$, as in GLA, but combined with the delta rule. In the online-learning view (Table 7) both GDN and KDA are "SGD on the *decayed* state": $\mathbf{S}_t=\tilde{\mathbf{S}}_{t-1}-\nabla_{\tilde{\mathbf{S}}_{t-1}}\tfrac{\beta_t}{2}\|\tilde{\mathbf{S}}_{t-1}^\top\bm{k}_t-\bm{v}_t\|^2$ where $\tilde{\mathbf{S}}_{t-1}$ is $\mathbf{S}_{t-1}$ decayed by a scalar (GDN) or a per-channel vector (KDA). The motivation the paper gives: RoPE's strength is per-dimension frequency diversity (a non-uniform Fourier transform along the feature axis); a scalar decay has no such per-dimension diversity, so a channel-wise gate is the natural analogue when the recurrent layer is asked to carry positional information.

Blog-friendly one-liner for each step: LA = "add"; DeltaNet = "erase-then-write under this key"; GDN = "erase-then-write, and let everything fade at one rate"; KDA = "erase-then-write, and let each feature channel fade at its own rate".

---

## 2. The KDA recurrence and its neural parameterization

**State update (Eq. 1):**
$$\mathbf{S}_t=\left(\mathbf{I}-\beta_t\bm{k}_t\bm{k}_t^\top\right)\operatorname{Diag}(\bm{\alpha}_t)\,\mathbf{S}_{t-1}+\beta_t\bm{k}_t\bm{v}_t^\top\in\mathbb{R}^{d_k\times d_v},\qquad \bm{o}_t=\mathbf{S}_t^\top\bm{q}_t\in\mathbb{R}^{d_v}.$$
Order matters: decay first, then the Householder correction, then the write. Expanding gives the DPLR form
$$\mathbf{S}_t=\left(\operatorname{Diag}(\bm{\alpha}_t)-\beta_t\bm{k}_t\bm{k}_t^\top\operatorname{Diag}(\bm{\alpha}_t)\right)\mathbf{S}_{t-1}+\beta_t\bm{k}_t\bm{v}_t^\top,$$
i.e. the general DPLR transition $\mathbf{D}-\bm{a}_t\bm{b}_t^\top$ with $\mathbf{D}=\operatorname{Diag}(\bm{\alpha}_t)$, $\bm{a}_t=\beta_t\bm{k}_t$, $\bm{b}_t=\bm{k}_t\odot\bm{\alpha}_t$. Tying both low-rank vectors to $\bm{k}_t$ is what makes the fast kernel possible (section 3).

Versus GDN: $\alpha_t\mathbf{I}\to\operatorname{Diag}(\bm{\alpha}_t)$. Each of the $d_k=128$ key-channels (rows of $\mathbf{S}$) has its own forgetting rate. $\beta_t\in[0,1]$ is still a scalar per head and plays the role of the write strength / SGD learning rate.

**Per-head input projections (section 4, $d_k=d_v=128$ throughout):**
$$\bm{q}^h_t,\bm{k}^h_t=\operatorname{L2Norm}\big(\operatorname{Swish}(\operatorname{ShortConv}(\mathbf{W}^h_{q/k}\bm{x}_t))\big)\in\mathbb{R}^{d_k}$$
$$\bm{v}^h_t=\operatorname{Swish}(\operatorname{ShortConv}(\mathbf{W}^h_v\bm{x}_t))\in\mathbb{R}^{d_v}$$
$$\bm{\alpha}^h_t=f(\mathbf{W}^\uparrow_\alpha\mathbf{W}^\downarrow_\alpha\bm{x}_t)\in[0,1]^{d_k}$$
$$\beta^h_t=\operatorname{Sigmoid}(\mathbf{W}^h_\beta\bm{x}_t)\in[0,1]$$

- ShortConv: depthwise causal conv with small kernel (the paper cites kernel size 4) followed by Swish, following GDN. Ablation (Table 1): removing it hurts (val PPL 5.70 vs 5.65).
- L2Norm on $\bm{q},\bm{k}$: keeps $\|\bm{k}_t\|=1$ so the eigenvalues of $\mathbf{I}-\beta_t\bm{k}_t\bm{k}_t^\top$ lie in $[1-\beta_t,1]\subset[0,1]$ ("eigenvalue stability", citing DeltaNet).
- Forget gate $\bm{\alpha}$: a **low-rank** projection $\mathbf{W}^\downarrow_\alpha,\mathbf{W}^\uparrow_\alpha$ with rank equal to the head dimension (128), then a decay function $f(\cdot)$ "similar to those used in GDN and Mamba". The paper does not write $f$ out. The GDN/Mamba2 convention it refers to is $\bm{\alpha}_t=\exp\!\big(-A_h\cdot\operatorname{softplus}(\cdot)\big)$ with a learnable positive per-head scale $A_h$ stored in log-space ($A_h=\exp(A^{\log}_h)$), so that $\log\bm{\alpha}_t=-A_h\,\operatorname{softplus}(\cdot)\le 0$ is what the kernel actually consumes (the pseudocode takes `g` = $\log\bm{\alpha}$ and cumsums it). Verify the exact form against the released `fla/ops/kda` and HF modeling code before stating it as fact in the blog.
- $\beta$: plain sigmoid of a linear projection, scalar per head.

**Output gate and norm (Eq. 10):**
$$\bm{o}_t=\mathbf{W}_o\Big(\operatorname{Sigmoid}\big(\mathbf{W}^\uparrow_g\mathbf{W}^\downarrow_g\bm{x}_t\big)\odot\operatorname{RMSNorm}\big(\operatorname{KDA}(\bm{q}_t,\bm{k}_t,\bm{v}_t,\bm{\alpha}_t,\beta_t)\big)\Big),\quad \mathbf{W}_o\in\mathbb{R}^{d\times d}.$$
Head-wise RMSNorm on the recurrence output, then a data-dependent sigmoid gate (citing the "attention sink / gated attention" paper, ref [79]), then output projection. See section 5 for the gate's parameterization.

---

## 3. Chunkwise parallel algorithm (section 3.1 + Appendix B/C)

Split the sequence into chunks of length $C$ ($C=64$ in the FLOPs analysis and pseudocode). Within chunk $t$, index $r\in[1,C]$; $\mathbf{S}_{[t]}:=\mathbf{S}^0_{[t]}$ is the state entering the chunk. Cumulative decays: $\bm{\gamma}^{i\to j}_{[t]}=\prod_{k=i}^{j}\bm{\alpha}^k_{[t]}$ (elementwise), $\bm{\gamma}^r_{[t]}:=\bm{\gamma}^{1\to r}_{[t]}$, and $\bm{\Gamma}^{1\to C}_{[t]}\in\mathbb{R}^{C\times d_k}$ stacks $\bm{\gamma}^1,\dots,\bm{\gamma}^C$ as rows.

**Partial unroll (Eq. 2).** Within a chunk,
$$\mathbf{S}^r_{[t]}=\underbrace{\prod_{i=1}^{r}\big(\mathbf{I}-\beta^i\bm{k}^i\bm{k}^{i\top}\big)\operatorname{Diag}(\bm{\alpha}^i)}_{\mathbf{P}^r_{[t]}}\mathbf{S}^0_{[t]}+\underbrace{\sum_{i=1}^{r}\Big(\prod_{j=i+1}^{r}\big(\mathbf{I}-\beta^j\bm{k}^j\bm{k}^{j\top}\big)\operatorname{Diag}(\bm{\alpha}^j)\Big)\beta^i\bm{k}^i\bm{v}^{i\top}}_{\mathbf{H}^r_{[t]}} .$$
$\mathbf{P}$ = how the incoming state is transformed; $\mathbf{H}$ = contribution of writes inside the chunk.

**WY representation (Eq. 3-5; Propositions 1-2 in Appendix B, proved by induction).** A product of $r$ gated Householder matrices collapses to "diagonal minus a sum of rank-1 terms":
$$\mathbf{P}^r_{[t]}=\operatorname{Diag}(\bm{\gamma}^r_{[t]})-\sum_{i=1}^{r}\operatorname{Diag}(\bm{\gamma}^{i\to r}_{[t]})\bm{k}^i_{[t]}\bm{w}^{i\top}_{[t]},\qquad
\mathbf{H}^r_{[t]}=\sum_{i=1}^{r}\operatorname{Diag}(\bm{\gamma}^{i\to r}_{[t]})\bm{k}^i_{[t]}\bm{u}^{i\top}_{[t]},$$
with auxiliary vectors defined by the triangular recurrences
$$\bm{w}^r_{[t]}=\beta^r\Big(\operatorname{Diag}(\bm{\gamma}^r)\bm{k}^r-\sum_{i=1}^{r-1}\bm{w}^i\big(\bm{k}^{i\top}\operatorname{Diag}(\bm{\gamma}^{i\to r})\bm{k}^r\big)\Big),\qquad
\bm{u}^r_{[t]}=\beta^r\Big(\bm{v}^r-\sum_{i=1}^{r-1}\bm{u}^i\big(\bm{k}^{i\top}\operatorname{Diag}(\bm{\gamma}^{i\to r})\bm{k}^r\big)\Big).$$
Interpretation: $\bm{u}^r$ is the "pseudo-value" -- the value $\bm{v}^r$ minus what earlier keys in the same chunk already explain (the delta-rule correction, restricted to the chunk); $\bm{w}^r$ is the matching decayed-key that tells you how to apply that correction to the *incoming* state. The paper follows Comba's formulation of $\mathbf{P}$ to avoid an additional matrix inversion later.

**UT transform (Eq. 6-7).** The two recurrences above are solved for the whole chunk at once via a single $C\times C$ triangular inverse:
$$\mathbf{M}_{[t]}=\Big(\mathbf{I}+\operatorname{StrictTril}\Big(\operatorname{Diag}(\beta_{[t]})\big(\bm{\Gamma}^{1\to C}_{[t]}\odot\mathbf{K}_{[t]}\big)\Big(\frac{\mathbf{K}_{[t]}}{\bm{\Gamma}^{1\to C}_{[t]}}\Big)^{\!\top}\Big)\Big)^{-1}\operatorname{Diag}(\beta_{[t]})$$
$$\mathbf{W}_{[t]}=\mathbf{M}_{[t]}\big(\bm{\Gamma}^{1\to C}_{[t]}\odot\mathbf{K}_{[t]}\big),\qquad \mathbf{U}_{[t]}=\mathbf{M}_{[t]}\mathbf{V}_{[t]} .$$
The inverse of the unit lower-triangular matrix is done by forward substitution (the row-wise loop in the pseudocode). This converts non-matmul, sequential work into matmuls -- "crucial for hardware utilization".

**Inter-chunk state recurrence (Eq. 8).** One state update per chunk:
$$\mathbf{S}_{[t+1]}=\operatorname{Diag}(\bm{\gamma}^C_{[t]})\mathbf{S}_{[t]}+\big(\bm{\Gamma}^{i\to C}_{[t]}\odot\mathbf{K}_{[t]}\big)^{\top}\big(\mathbf{U}_{[t]}-\mathbf{W}_{[t]}\mathbf{S}_{[t]}\big).$$
Decay the whole state by the chunk's total per-channel decay, then add keys (each decayed by the remaining distance to chunk end) times the corrected pseudo-values.

**Output (Eq. 9), inter-chunk recurrent + intra-chunk parallel:**
$$\mathbf{O}_{[t]}=\underbrace{\big(\bm{\Gamma}^{1\to C}_{[t]}\odot\mathbf{Q}_{[t]}\big)\mathbf{S}_{[t]}}_{\text{inter chunk}}+\underbrace{\operatorname{Tril}\Big(\big(\bm{\Gamma}^{1\to C}_{[t]}\odot\mathbf{Q}_{[t]}\big)\Big(\frac{\mathbf{K}_{[t]}}{\bm{\Gamma}^{1\to C}_{[t]}}\Big)^{\!\top}\Big)}_{\text{intra chunk}}\underbrace{\big(\mathbf{U}_{[t]}-\mathbf{W}_{[t]}\mathbf{S}_{[t]}\big)}_{\text{pseudo-value}}\in\mathbb{R}^{C\times d_v}.$$
The intra-chunk term is a masked $C\times C$ "attention" matrix where the pairwise decay between positions $i$ and $j$ appears as the ratio $\gamma^i/\gamma^j$ -- the matrix $\mathcal{A}_{[t]}$ with entries $\gamma^i_{[t]}/\gamma^j_{[t]}$ in the notation section.

**Numerical range: log space and secondary tiling.** The ratio $\bm{\Gamma}\odot\mathbf{Q}$ times $\mathbf{K}/\bm{\Gamma}$ is the problem: with fine-grained decay, $1/\gamma$ can overflow/underflow in half precision. GLA's remedy is to work in the log domain ($g=\log\bm{\alpha}$, cumsum, then $\exp(g_i-g_j)$ which is always $\le 1$ for $i\ge j$) and to compute the *diagonal* $C\times C$ blocks by a "secondary chunking" loop in full precision. The Appendix C pseudocode does exactly this: `g = g.cumsum(-2)`, then a per-row loop labelled `# secondary chunking for numerical stability` that computes `A[..., j] = einsum(q_i * (g_i - g_j).exp(), k_j)` in fp32, while the off-diagonal/inter-chunk parts remain bf16 tensor-core matmuls. The paper says the secondary chunking "prevents full utilization of half-precision matrix multiplications and significantly reduces operator speed" -- i.e. the diagonal tiles, which are computed in fp32 with elementwise exp instead of one big bf16 matmul, are the kernel bottleneck. (The 16-token sub-tile width used in the actual Triton kernel is an implementation detail from the FLA GLA/KDA kernels, not stated in the paper.) KDA's specific win over general DPLR: because $\bm{a}=\bm{b}=\bm{k}$, the kernel needs only two such second-level tiled matrices (`Aqk`, `Akk`) instead of four (`Aab`, `Aak`, `Aqb`, `Aqk`), and drops roughly three additional matmuls in the inter-chunk/output stage (DPLR's `o1,o2,o3` and two state updates collapse to one output line and one state update). Result: "operator efficiency improves by roughly 100%"; Figure 2 shows KDA at nearly 2x the DPLR kernel speed for sequence lengths 2k-64k (batch 1, 16 heads).

**FLOPs (Eq. 13-14), per head, $C=64$:**
$$\mathrm{FLOPs}_{\text{KDA}}(T)=6Td_h^2+3TCd_h+TC^2,\qquad \mathrm{FLOPs}_{\text{Attn}}(T)=2T^2d_h .$$
Linear in $T$ versus quadratic. For inference, prefill uses the chunkwise kernel; decode uses the plain recurrent form (Eq. 1) with a fixed $128\times128$ state per head.

---

## 4. The hybrid layout

- **Layerwise, 3:1.** Each block is [token mixer, MoE FFN]; token mixers repeat as KDA, KDA, KDA, MLA. Layerwise (whole layers) rather than headwise (mixing heads inside a layer) was chosen for "infrastructure simplicity and training stability". Backbone follows Moonlight (DeepSeek-V3-style MLA + MoE), first layer is a dense FFN.
- **NoPE on every MLA layer.** All positional information and recency bias is delegated to KDA. Justification (section 6.1): with the delta rule, the readout can be written as
$$\bm{o}_t=\sum_{i=1}^{t}\Big(\bm{q}_t^\top\Big(\prod_{j=i+1}^{t}\mathbf{A}_j(\mathbf{I}-\beta_j\bm{k}_j\bm{k}_j^\top)\Big)\bm{k}_j\Big)\bm{v}_j,$$
which has the same shape as RoPE's $s_{t,i}=\bm{q}_t^\top\big(\prod_{j=i+1}^{t}\mathbf{R}_j\big)\bm{k}_i$ -- a cumulative product of transition matrices between key position $i$ and query position $t$. RoPE's $\mathbf{R}_j$ is a fixed block-diagonal rotation; KDA's $\mathbf{A}_j=\operatorname{Diag}(\bm{\alpha}_j)$ is a data-dependent, learnable, non-orthogonal transition, hence "learnable positional embedding". The per-channel gate mirrors RoPE's per-dimension frequencies. Practical NoPE bonuses called out: MLA with NoPE converts to pure MQA at inference; long-context extension needs no RoPE base tuning / YaRN. Table 5 shows the RoPE variant (Kimi Linear (RoPE)) matches on short context but loses on long context (RULER 78.8 vs 84.3), which the paper attributes to RoPE in the global layer over-emphasizing short-range order and making mid-training context extension less flexible.
- **KV cache and decode.** Only 1 in 4 layers keeps a growing KV cache; the KDA layers hold a fixed $d_k\times d_v$ state per head. Hence "up to 75%" KV-cache reduction. I/O-bound decode time therefore approaches a 3:1 hybrid-efficiency ceiling versus full attention at batch 1 (Figure 7b: ~2.2-2.3x TPOT at 1M, ~1.8x at 512k, batch size 1). The freed memory allows larger batches, giving the headline 6.3x TPOT at 1M (1.84 ms vs 11.48 ms; Figure 1b, 5.7x at 512k, 4.8x at 256k). Prefill (Figure 7a): indistinguishable from GDN-H, 2.3x faster than MLA at 512k and 2.9x at 1M; parity at 4k-16k. RULER-128k point: 3.98x acceleration at 84.3 accuracy (Figure 1a).

---

## 5. Output-gate parameterization

Eq. 10 uses a **low-rank** sigmoid gate $\operatorname{Sigmoid}(\mathbf{W}^\uparrow_g\mathbf{W}^\downarrow_g\bm{x}_t)$, explicitly "similar to the forget gate, to ensure a fair parameter comparison, while maintaining performance comparable to full-rank gating and alleviating the Attention Sink". So in Kimi Linear the low rank is a parameter-budget decision for fair baselines, not a quality preference -- the paper says a full-rank gate performs comparably. (This is the knob K3 reportedly turns back to full rank.) Ablation (Table 1, 16-layer/16-head model, val PPL): sigmoid gate 5.65; no gate 5.67; Swish gate (GDN's default) 5.81 -- Swish is "substantially worse", consistent with ref [79]. Sigmoid gating was therefore used everywhere, including in the GDN-H baseline.

---

## 6. Key ablations and results

**Synthetic tasks (section 5.1; 2 layers, 2 heads, head dim 128, lengths 256-2048, up to 20k steps, LR grid $\{5\text{e-}5,1\text{e-}4,5\text{e-}4,1\text{e-}3\}$).** Three tasks: *Palindrome* (reverse a random token string -- exact copy from compressed state), *MQAR* (multi-query associative recall, known to correlate with LM quality), *Stack* (64 independent LIFO stacks, `<push> id x` / `<pop> id ?`, state tracking). KDA has the highest accuracy at every length on all three and converges markedly faster than GDN on Palindrome and MQAR. Mamba2 (decay only, no delta rule) "fails on all tasks" in this setting -- the cleanest evidence in the paper that the delta rule (erase-then-write) is what buys exact recall/copy/state-tracking, and fine-grained decay is what makes it converge faster.

**Component ablation (Table 1; 653M-active model, 16 heads, 16 layers, same FLOPs; train / val PPL):**

| variant | train | val |
|---|---|---|
| hybrid 3:1 (chosen) | 9.23 | 5.65 |
| 0:1 (pure MLA) | 9.45 | 5.77 |
| 1:1 | 9.29 | 5.66 |
| 7:1 | 9.23 | 5.70 |
| 15:1 | 9.34 | 5.82 |
| w/o output gate | 9.25 | 5.67 |
| w/ Swish output gate | 9.43 | 5.81 |
| w/o ShortConv | 9.29 | 5.70 |

Reading: 7:1 matches training loss but generalizes worse (val is on an out-of-distribution high-quality set); 1:1 matches val but costs inference; pure full attention is worst of the near-neighbours. 3:1 is the quality/throughput sweet spot.

**Scaling law (section 5.3, Figure 5).** Five MoE models (8 of 64 experts active, Muon optimizer, 4k context), 653M-1.7B active params, 38.8B-128B tokens, LR $2.0\text{e-}3\to1.37\text{e-}3$ (Table 2). Fits: MLA $L=2.3092\,C^{-0.0536}$, Kimi Linear $L=2.2879\,C^{-0.0527}$; Kimi Linear is ~1.16x more compute-efficient at compute-optimal training, with hyperparameters tuned only for MLA.

**1.4T-token fair comparison (48B-A3B, MLA vs GDN-H vs Kimi Linear, identical recipe).** Base (Table 3): MMLU 71.6 / 72.2 / 73.8; MMLU-Pro 47.2 / 47.9 / 51.0; BBH 71.6 / 70.6 / 72.9; HellaSwag 81.7 / 82.2 / 82.9; GSM8K 83.7 / 81.7 / 83.9; CRUXEval-O-cot 61.5 / 58.1 / 62.0; EvalPlus 59.5 / 63.1 / 60.2 (GDN-H wins); CMMLU 79.5 / 80.7 / 80.8. SFT (Table 4): MMLU-Pro 65.7 / 64.8 / 67.4; GPQA-Diamond 57.1 / 58.6 / 62.1; AIME 2025 20.6 / 21.1 / 21.3; HMMT 2025 11.3 / 11.3 / 12.5; LiveCodeBench v6 25.1 / 25.4 / 26.0; exceptions MATH500 (GDN-H 83.0 best) and EvalPlus (MLA 62.6 best). Long context at 128k (Table 5, avg over RULER, MRCR, HELMET-ICL, LongBench v2, Frames, RepoQA, Long Code Arena): MLA 52.2, GDN-H 51.2, Kimi Linear (RoPE) 51.8, Kimi Linear 54.5; RULER 81.3 / 80.5 / 78.8 / 84.3; MRCR 22.6 / 23.9 / 22.0 / 29.6; RepoQA 63.0 / 63.0 / 66.5 / 68.5. Notable pattern: GDN-H beats MLA on short context but falls *behind* MLA on long context; Kimi Linear stays on top in both.

**RL (Figure 6).** Math RLVR with identical algorithm/hyperparameters: Kimi Linear's training accuracy rises faster than MLA's and the gap widens; MATH500 and AIME 2025 test curves show the same.

**5.7T checkpoint (Appendix D).** The released model is trained on 5.7T tokens (matching Moonlight), supports 1M context, and scores RULER 95.4 at 128k and 94.8 at 1M. Versus Moonlight-Instruct (16B total, 3B active): GPQA-Diamond 71.7 vs 24.7, MMLU-Pro 72.7 vs 43.8, MATH500 94.6 vs 58.0, LiveCodeBench v6 45.7 vs 11.9, AIME 2025 58.6, HMMT 2025 44.5.

---

## 7. Figures worth redrawing as block diagrams

- **Figure 3 (architecture).** Left: the stack -- repeat [Norm -> KDA -> Norm -> MoE] x3 then [Norm -> MLA -> Norm -> MoE] x1, with the MoE box showing a router, $N_s$ shared experts and $N_r$ routed experts. Right: the KDA cell -- input $\bm{x}_t$ fans out to three Linear -> ShortConv -> Swish branches ($\bm{q}$, $\bm{k}$ get an extra L2Norm; $\bm{v}$ does not), plus low-rank $\bm{\alpha}$ and scalar $\beta$ projections and a low-rank sigmoid output gate; the recurrence box, then RMSNorm, gate multiply, output Linear. This is the diagram to redraw for K3, with the gate marked full-rank.
- **Eq. 1 pictorial (in-paper diagram under Eq. 1).** $[\mathbf{S}_t] = ([\mathbf{I}] - \beta_t\bm{k}_t\times\bm{k}_t^\top)\cdot[\operatorname{Diag}(\bm{\alpha}_t)]\cdot[\mathbf{S}_{t-1}] + \beta_t\bm{k}_t\times\bm{v}_t^\top$ drawn with matrix boxes; good for explaining "decay per row, Householder erase, rank-1 write".
- **Eq. 8/9 pictorials.** Chunk state update as [state] = [diag decay][state] + ($\bm{\Gamma}\odot\mathbf{K}$)$^\top$ (U - W S); output as ($\bm{\Gamma}\odot\mathbf{Q}$)[S] + [masked $C\times C$ block]$\cdot$(U - W S). Redraw as a "two lanes" figure: an inter-chunk lane carrying $\mathbf{S}$ across chunks, and an intra-chunk lane doing a $C\times C$ masked matmul.
- **Figure 1** (Pareto scatter and TPOT-vs-length curve) and **Figure 7** (prefill latency and batch-1 TPOT) are the efficiency plots; the numbers are given above.
- **Table 6 / Table 7** (unified recurrent-vs-parallel forms and update-rule-vs-objective for LA, Mamba2, GLA, DeltaNet, GDN, Comba, RWKV7, KDA) make a good lineage chart: rows = models, columns = "decay type (none / scalar / diagonal)" x "delta rule (no / yes)". KDA is the diagonal-decay + delta-rule cell; GLA is diagonal-decay without delta; GDN is scalar-decay with delta.

---

## 8. Model configurations

- **Main 48B-A3B models (MLA, GDN-H, Kimi Linear, Kimi Linear (RoPE)):** Moonlight-derived MoE with sparsity raised to 32: 256 experts, 8 activated per token including 1 shared expert; 48B total / 3B activated; first layer dense; head dims $d_k=d_v=128$; 3:1 KDA:MLA; MLA layers NoPE. Pretrain: 4,096 context, MuonClip optimizer, WSD schedule, 1.4T tokens from the K2 corpus, LR $1.1\text{e-}3$, global batch 32M tokens, K2's annealing and long-context activation phase. Release: same recipe at 5.7T tokens, 1M context, plus SFT (multi-stage, K2 SFT data + reasoning) and RL (math/code/STEM, PTX loss, K1.5-style algorithm with truncated importance sampling and dynamic KL / mini-batch). The paper does not list depth or hidden size; take those from the HF config (`moonshotai/Kimi-Linear-48B-A3B-Instruct`).
- **Scaling-law models (Table 2):** active params 653M / 878M / 1.1B / 1.4B / 1.7B; heads = layers = 16 / 18 / 20 / 22 / 24; hidden 1216 / 1376 / 1536 / 1632 / 1776; tokens 38.8B / 59.8B / 85.2B / 102.5B / 128.0B; batch 336 / 432 / 512 / 576 / 640 sequences of 4,096; 8 of 64 experts; Muon.
- **Synthetic models:** 2 layers, 2 heads, head dim 128.
- **Kernel benchmark (Figure 2):** batch 1, 16 heads, lengths 2k-64k, chunk size 64.

## Points to be careful about in the blog

- The "6.3x" and "6x" decode numbers are batch-scaled TPOT at 1M (memory freed by the smaller KV cache allows bigger batches); the batch-1 speedup at 1M is ~2.2-2.3x, bounded by the 3:1 ratio. Prefill is 2.9x at 1M.
- "75% KV cache reduction" is exactly the 3:1 ratio (MLA layers still cache; KDA layers keep a $128\times128$ state per head).
- The paper never writes the explicit decay function $f$ or the 16-token sub-tile; both come from the GDN/Mamba2 convention and the FLA kernel respectively.
- The low-rank output gate is justified as parameter-matching for fair comparisons and "comparable to full-rank", which is consistent with a later model switching to full rank without contradicting this paper.
