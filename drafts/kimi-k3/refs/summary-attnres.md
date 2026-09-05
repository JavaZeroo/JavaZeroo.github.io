# Attention Residuals (AttnRes) — technical summary for a K3-architecture blog

Source: Kimi Team, "Attention Residuals", arXiv 2603.15031v1 (16 Mar 2026), 21 pp. Code/README: github.com/MoonshotAI/Attention-Residuals (branch `master`; README contains the pseudocode below, the scaling-law figure and the downstream table; no training code). Note: the arXiv HTML build is broken (empty LaTeXML page), so everything here comes from the PDF.

Conventions in the paper: **each self-attention sublayer and each MLP/MoE sublayer is a separate "layer"** $l\in\{1,\dots,L\}$; a Transformer block therefore contributes two layers, $L_b = L/2$. $h_l\in\mathbb{R}^d$ is the input to layer $l$ for one token; $h_1$ is the token embedding; $f_l$ is the sublayer function.

---

## 1. Motivation

**Standard residual = fixed unit-weight sum.** $h_l = h_{l-1} + f_{l-1}(h_{l-1})$ unrolls to $h_l = h_1 + \sum_{i=1}^{l-1} f_i(h_i)$: every layer receives the *same* uniformly weighted sum of all prior outputs. Gradient: $\partial\mathcal{L}/\partial h_l = \partial\mathcal{L}/\partial h_L \prod_{j=l}^{L-1}(I + \partial f_j/\partial h_j)$, whose identity term is the "gradient highway". Highway nets generalise to $h_l = (1-g_l)\odot h_{l-1} + g_l\odot f_{l-1}(h_{l-1})$; both are instances of $h_l = \alpha_l h_{l-1} + \beta_l f_{l-1}(h_{l-1})$.

**Three problems the paper names (§2.1 "Limitations"):**
1. *No selective access* — attention and MLP sublayers get the identical aggregated state, although they might want different mixtures; only $h_{l-1}$, a single compressed state, is visible.
2. *Irreversible loss* — anything blurred by summation cannot be recovered later.
3. *Output growth / PreNorm dilution* — with PreNorm, $\|h_l\|$ grows as $O(L)$ with depth, so each layer's relative contribution shrinks; deeper layers must learn ever-larger outputs (from fixed-scale normalised inputs) to stay influential, which destabilises training (cites SiameseNorm [27], Xiong et al. [60]). Empirically many layers can be pruned with little loss [11].

**Framing: residual-over-depth is an RNN-over-depth.** The recurrence $h_l = h_{l-1} + f_{l-1}(h_{l-1})$ has the same additive form as linear attention / TTT over time, $S_t = S_{t-1} + k_t v_t^\top$ (§6.1). Gated variants map onto Highway; the delta rule onto DDL; GLA onto MRLA. All remain *recurrences*. The Transformer fixed the sequence-side bottleneck by replacing recurrence with attention; AttnRes does the same for depth:

$$h_l = \alpha_{0\to l}\, h_1 + \sum_{i=1}^{l-1} \alpha_{i\to l}\, f_i(h_i), \qquad \sum_{i=0}^{l-1}\alpha_{i\to l} = 1. \tag{1}$$

Depth is small ($L<1000$), so $O(L^2)$ attention over depth is cheap.

**Structured-matrix view (§6.2).** Write $h_l = \sum_{i<l} M_{i\to l} v_i$ with $v_0=h_1$, $v_i = f_i(h_i)$. Standard residual: $M$ = all-ones lower-triangular (rank-1 semiseparable). Highway: 1-semiseparable with input-dependent gates ($M_{i\to l} = g_{i+1}\prod_{j=i+2}^{l}(1-g_j)$, a stick-breaking attention). (m)HC: $M_{i\to l} = \beta_i^\top A^\times_{i+1\to l}\alpha_l$, $m$-semiseparable — i.e. **depth-wise *linear* attention with matrix-valued state** ($\alpha_l$ = query, $\beta_i$ = key, $A^\times$ = depth-relative positional operator). Full AttnRes: dense rank-$L$ $M$ = **depth-wise *softmax* attention**. Block AttnRes: rank between $N$ and $N+S$.

---

## 2. Full AttnRes — exact definition

Kernel $\phi(q,k) = \exp\!\big(q^\top \mathrm{RMSNorm}(k)\big)$, weights

$$\alpha_{i\to l} = \frac{\phi(q_l,k_i)}{\sum_{j=0}^{l-1}\phi(q_l,k_j)}, \tag{2}$$

$$q_l = w_l, \qquad k_i = v_i = \begin{cases} h_1 & i=0 \\ f_i(h_i) & 1\le i\le l-1\end{cases}, \tag{3}$$

$$h_l = \sum_{i=0}^{l-1}\alpha_{i\to l}\, v_i. \tag{4}$$

Key facts to state precisely in the blog:
- **The pseudo-query $w_l\in\mathbb{R}^d$ is one learned parameter vector per layer (per sublayer), shared by all tokens and all positions.** It is *not* projected from the hidden state. This is a deliberate design choice: because $w_l$ is independent of the forward pass, attention scores for a whole group of layers can be computed in parallel/batched before those layers run (§3.1 "Blockwise optimization", §4.2).
- **The weights are nevertheless input-dependent per token**: keys are $\mathrm{RMSNorm}(v_i)$ of that token's own layer outputs, so $\alpha_{i\to l}$ varies token by token (README: "learned, input-dependent attention over depth"; Fig. 8 heatmaps are "averaged over tokens"). Confirmed.
- **Keys are RMSNormed, values are not.** Only $K=\mathrm{RMSNorm}(V)$; the mixture is over raw $v_i$. Purpose: stop layers with naturally large outputs from dominating the softmax. Each sublayer adds exactly one RMSNorm (for its keys) plus one $w_l$ — "a negligible fraction of the total parameter count". The ordinary PreNorm ($\mathrm{attn\_norm}$, $\mathrm{mlp\_norm}$) is still applied to $h_l$ *after* the AttnRes mixture, before $f_l$.
- **Softmax is jointly normalised over all sources including the embedding** ($i=0$); there is no separate identity path — the "residual" *is* the mixture.
- **Initialisation: all $w_l = 0$** (mandatory). Then $\alpha$ is uniform, and AttnRes starts as an equal-weight *average* of previous outputs (note: an average, not the sum of a standard residual). The authors say this prevents training volatility.
- No multi-head: a single scalar weight per source per token (multihead ablated and found worse, §5.3).
- Cost per token: $O(L^2 d)$ arithmetic, $O(Ld)$ memory to keep all layer outputs. In vanilla training the memory is free (those activations are retained for backprop anyway). Under activation recomputation + pipeline parallelism, the $L$ outputs must be kept alive and shipped across stages: $O(Ld)$ memory *and* communication — this is what motivates Block AttnRes. Batching queries within blocks of $S$ layers cuts per-layer memory I/O from $O(Ld)$ to $O((S+N)d)$ (Appendix B) but cannot cut cross-stage communication.

---

## 3. Block AttnRes (what K3 uses)

Partition the $L$ layers into $N$ blocks $B_1,\dots,B_N$ of $S=L/N$ consecutive layers (if $L\bmod N\ne 0$ the last block holds the remainder and its final partial sum is its representation).

**Intra-block: plain residual sum.**
$$b_n = \sum_{j\in B_n} f_j(h_j), \qquad b_n^{i} = \text{partial sum over the first } i \text{ layers of } B_n,\; b_n = b_n^{S},\; b_n^{0}:=0. \tag{5}$$

**Embedding is block 0:** $b_0 = h_1$, always present as a source.

**Inter-block attention.** For the $i$-th layer of block $n$ the value set is
$$V = \begin{cases}[\,b_0, b_1,\dots,b_{n-1}\,] & i=1 \text{ (first layer of the block)}\\ [\,b_0, b_1,\dots,b_{n-1},\, b_n^{\,i-1}\,] & i\ge 2\end{cases} \tag{6}$$
with keys $\mathrm{RMSNorm}(V)$, the same per-layer pseudo-query $w_l$, and softmax over all entries (Eq. 2). So a layer sees at most $n+1$ sources: embedding, $n-1$ finished blocks, and the running partial sum of its own block. The partial sum plays the role of the ordinary residual stream *inside* a block; the RMSNorm on keys keeps a small partial sum and a big completed block on equal footing. The input to layer 1 of the network is just $b_0 = h_1$.

**Final output**: "the final output layer aggregates all $N$ block representations" (plus $b_0$) with the same attention op before the final norm / LM head.

**Costs**: memory $O(Nd)$ per token instead of $O(Ld)$; compute $O(N^2)$-ish instead of $O(L^2)$; pipeline communication $N$ vectors instead of $L$. $N=L$ recovers Full AttnRes; $N=1$ is a standard residual with the embedding split off as its own source. Depth-mixing matrix rank is between $N$ and $N+S$. Empirically $N\approx 8$ recovers most of the gain ("only eight stored hidden states per token").

**Reference implementation (paper Fig. 2 / README).** Note `block_size` counts sublayers (attn + MLP), so a Transformer layer contributes 2; the boundary test appends the completed partial sum to `blocks` *after* it has been used as the last value in the pre-attention AttnRes call, which is exactly Eq. 6's $i=1$ case.

```python
def block_attn_res(blocks, partial_block, proj, norm):
    V = torch.stack(blocks + [partial_block])          # [N+1, B, T, D]
    K = norm(V)                                        # RMSNorm on keys only
    logits = torch.einsum('d, n b t d -> n b t', proj.weight.squeeze(), K)  # w_l . k_i
    return torch.einsum('n b t, n b t d -> b t d', logits.softmax(0), V)    # softmax over sources

def forward(self, blocks, hidden_states):
    partial_block = hidden_states
    h = block_attn_res(blocks, partial_block, self.attn_res_proj, self.attn_res_norm)
    if self.layer_number % (self.block_size // 2) == 0:   # block boundary
        blocks.append(partial_block); partial_block = None
    attn_out = self.attn(self.attn_norm(h))
    partial_block = partial_block + attn_out if partial_block is not None else attn_out
    h = block_attn_res(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)
    mlp_out = self.mlp(self.mlp_norm(h))
    partial_block = partial_block + mlp_out
    return blocks, partial_block
```
`proj` is an `nn.Linear(d, 1, bias=False)` — its weight *is* $w_l$. Two projections and two RMSNorms per Transformer layer (one pair before attention, one before the MLP).

**Two-phase computation with online softmax (Algorithm 1)** — exact, used at inference for both variants. For block $n$:
- *Phase 1 (parallel)*: $Q = [w_l]_{l\in B_n}\in\mathbb{R}^{S\times d}$, $K=V=[b_0;\dots;b_{n-1}]\in\mathbb{R}^{n\times d}$. One batched matmul yields, for every layer in the block, the unnormalised output $o^{(1)}_l$, running max $m^{(1)}_l$ and sum-exp $\ell^{(1)}_l$. Since the $w_l$ are parameters, this can run before any layer of the block executes, and can overlap with the block's first layer.
- *Phase 2 (sequential)*: for the first layer $h_l = o^{(1)}_l/\ell^{(1)}_l$. For later layers compute the one-key attention against $b_n^{i}$ giving $(o^{(2)}_l, m^{(2)}_l, \ell^{(2)}_l)$ and merge:
$$m_l=\max(m^{(1)}_l,m^{(2)}_l),\qquad h_l = \frac{e^{m^{(1)}_l-m_l}o^{(1)}_l + e^{m^{(2)}_l-m_l}o^{(2)}_l}{e^{m^{(1)}_l-m_l}\ell^{(1)}_l + e^{m^{(2)}_l-m_l}\ell^{(2)}_l},$$
then $b_n^{i}\leftarrow b_n^{i-1}+f_l(h_l)$. The merge is elementwise so it fuses into neighbouring kernels (e.g. the following RMSNorm).

---

## 4. Systems (§4)

**Training / pipeline parallelism — cross-stage caching.** Interleaved 1F1B schedule with $P$ physical stages, $V$ virtual stages per rank, $C=PV$ chunks, each producing on average $N_p$ block vectors of size $d$ per token. Naively re-sending the whole history at every chunk transition costs $\mathrm{Comm}_{\text{naive}} = \sum_{j=1}^{C-1} jN_p d = \tfrac{C(C-1)}{2}N_p d$. Because each rank runs several virtual stages in succession, blocks received earlier stay in local memory; for $v\ge2$ a transition only carries the $\sim PN_p$ blocks produced since the receiver's chunk in the previous virtual stage:
$$\mathrm{Comm}_{\text{cached}} = \underbrace{\tfrac{P(P-1)}{2}N_p d}_{v=1} + \underbrace{(V-1)P^2N_p d}_{v\ge2}.$$
Peak per-transition cost drops from $O(C)$ to $O(P)$ — a $V\times$ improvement — enough to fully overlap with compute in steady-state 1F1B; backward uses the same scheme. Fig. 3 example: $P=4$, $V=2$, each AttnRes block spanning two physical stages; caching removes 6 redundant block transfers in the second virtual stage. Memory: each block is stored once across all virtual stages; per-layer activation footprint is unchanged because activation checkpointing drops the inter-block attention intermediates and the checkpointed layer input is the same size as the $h_l$ it replaces. **Measured: <4% end-to-end training overhead with PP, negligible without.**

**Inference — two-phase strategy + I/O accounting (Table 1, per token per layer, typical $L=128$, $N=8$, $S=16$, $m=4$).** Standard residual: $3d$. mHC ($m$ streams): $(8m+2)d + 2m^2 + 4m \approx 34d$. Full AttnRes two-phase: $(S+N)d = 24d$ (Appendix B derivation: reads $(S+N-2)d$, writes $2d$). Block AttnRes: Phase 1 amortised read $\tfrac{N}{S}d$ + write $d$, Phase 2 read $3d$ + write $d$, total $(\tfrac{N}{S}+5)d = 5.5d$. **Measured: <2% inference latency overhead on typical workloads.** The fixed $N$ also bounds the "depth KV cache".

**Memory-efficient long-context prefill.** Storing block reps costs $N\cdot T\cdot d$ elements: **15 GB for a 128K sequence with 8 blocks** (at the 48B model's $d$). Fix: shard along the sequence across the $P$ tensor-parallel devices; Phase 1 runs on local shards; the Phase 2 online-softmax merge is folded into the existing TP all-reduce path (reduce-scatter, merge locally, all-gather, fused with RMSNorm). Per-device: $N(T/P)d$ → ~1.9 GB; with 16K chunked prefill → <0.3 GB.

---

## 5. Relation to prior work (Table 5, §6.2, §7)

- **Single-state recurrences** (ReZero, LayerScale, Highway, DeepNorm, KEEL, PostNorm/PreNorm variants): layer $l$ only sees $h_{l-1}$; fixed or gated scalars; nothing can be selectively retrieved. AttnRes is "orthogonal" and compatible with any of these norm/gating schemes.
- **Multi-state recurrences** — Hyper-Connections / **mHC (DeepSeek)**, DDL, SiameseNorm: widen the stream to $m$ copies with learned (mHC: doubly-stochastic) mixing matrices; still condition on the predecessor state. Via the structured-matrix analysis they are depth-wise *linear* attention with state expansion $d\to d\times m$; AttnRes is depth-wise *softmax* attention. mHC residual I/O $\approx 34d$ vs $5.5d$ for Block AttnRes; Block AttnRes matches mHC(-lite) loss, Full beats it (Table 2, 4).
- **Cross-layer access with static weights** — DenseNet (concat + conv), ELMo (softmax over learned scalars), **DenseFormer** (learned per-pair scalars fixed after training; ablation: 1.767 = no gain over 1.766 baseline), ANCRe. Key lesson: input-dependence matters.
- **Cross-layer access with dynamic weights** — MUDDFormer (small MLP generates position-dependent weights across four streams; hard to scale), MRLA (elementwise sigmoid gating, separable q·k so closer to linear attention), Value Residual Learning (one earlier layer only), **LAuReL** (low-rank projections over the previous $k$ activations), Dreamer (depth attention + sequence attention + sparse experts).
- **AttnRes's distinguishing combination**: softmax-normalised, per-token input-dependent weights over *all* prior sources incl. embedding, using only one $d$-vector query per layer, no extra projections, plus a block structure and infrastructure that make it a drop-in replacement at 48B scale. Not called out in the paper but relevant to a blog: ResiDual-style dual-stream ideas are represented by SiameseNorm in Table 5 (PreNorm + PostNorm parallel streams).
- It also exposes **depth-wise attention sinks**: some sources (notably the embedding) attract high weight regardless of input, mirroring sequence attention sinks.

---

## 6. Results

**Scaling laws (§5.1, Table 2, Fig. 4).** Five MoE sizes (194M–528M active), three variants each (PreNorm baseline, Full AttnRes, Block AttnRes with $\approx 8$ blocks), 8192-token context, cosine LR, all hyperparameters tuned for the *baseline* (conservative). Fits $L = A\,C^{-\alpha}$ with $C$ in PFLOP/s-days: Baseline $1.891\,C^{-0.057}$; Block AttnRes $1.870\,C^{-0.058}$; Full AttnRes $1.865\,C^{-0.057}$. Same slope, consistent offset. **At 5.6 PFLOP/s-days Block AttnRes reaches 1.692 vs baseline 1.714 — a $1.25\times$ compute advantage.** Full-vs-Block gap shrinks with scale to 0.001 at the largest size.

| Act. params | Tokens | $L_b$ | $H$ | $d_{model}$ | $d_{ff}$ | lr | batch | Baseline | Block | Full | mHC(-lite) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 194M | 38.7B | 12 | 12 | 896 | 400 | 2.99e-3 | 192 | 1.931 | 1.909 | 1.899 | 1.906 |
| 241M | 45.4B | 13 | 13 | 960 | 432 | 2.80e-3 | 256 | 1.895 | 1.875 | 1.874 | 1.869 |
| 296M | 62.1B | 14 | 14 | 1024 | 464 | 2.50e-3 | 320 | 1.829 | 1.809 | 1.804 | 1.807 |
| 436M | 87.9B | 16 | 16 | 1168 | 528 | 2.20e-3 | 384 | 1.766 | 1.746 | 1.737 | 1.747 |
| 528M | 119.0B | 17 | 17 | 1264 | 560 | 2.02e-3 | 432 | 1.719 | 1.693 | 1.692 | 1.694 |

**Ablations (Table 4, on the $L_b=16$ model, i.e. 32 sublayers; baseline 1.766).**
- DenseFormer 1.767; mHC 1.747.
- Full AttnRes **1.737**. Variants: input-dependent query (projected from hidden state, $d\times d$ per layer) **1.731** — better but adds params and forces sequential memory access at decode, so rejected; input-independent mixing (drop q/k, learnable scalars) 1.749; sigmoid instead of softmax 1.741; no RMSNorm on keys 1.743; sliding window over last $W=8$ outputs + embedding 1.764 (barely beats baseline: *distant* selective access matters more than many nearby sources).
- Block AttnRes $S=4$ ($N=8$) **1.746**; multihead ($H=16$ channel groups) 1.752 — worse, "when a layer's output is relevant, it is relevant as a whole"; no RMSNorm on keys 1.750 (norm matters more for blocks because block sums have larger magnitude spread).
- Block size sweep (Fig. 6): $S=1$ 1.737, $S=2$ 1.746, $S=4$ 1.746, $S=8$ 1.748, $S=16$ 1.753, $S=32$ 1.757. Graceful degradation; $N\approx 8$ chosen for infra.

**Architecture sweep (Fig. 7).** 25 configs at fixed $\approx 6.5\times10^{19}$ FLOPs and $\approx 2.3\times10^8$ active params, grid $d_{model}/L_b\in\{15,30,45,60,75\}$, $H/L_b\in\{0.3,\dots,0.7\}$, $d_{ff}/d_{model}\approx0.45$. AttnRes wins every cell by 0.019–0.063. Baseline optimum at $d_{model}/L_b\approx 60$ (1.847); AttnRes optimum shifts to $\approx 45$ (1.802) — **AttnRes prefers deeper, narrower models**, with the caveat that depth costs inference latency.

**Learned attention patterns (Fig. 8, $L_b=16$ model, averaged over tokens).** Heatmaps rows = destination layer 1–16 (attention on the left, MLP on the right), columns = source index 0–32 (Full) or block 0–8 (Block). Findings: (i) *locality preserved* — strongest weight on the immediate predecessor, but off-diagonal concentrations appear (layer 4 attending to early sources; layers 15–16 reaching far back in the block model) = learned skip connections; (ii) *embedding persistence* — source 0 keeps non-trivial weight everywhere, especially for pre-attention inputs; pre-MLP rows are sharply diagonal (local), pre-attention rows are broader — attention routes across layers, MLPs work locally; (iii) Block AttnRes reproduces all of this with sharper, more decisive weights (block compression acts as implicit regularisation).

**Training dynamics at 48B / 1T tokens (Fig. 5).** Validation loss lower throughout, gap widening in the decay phase. Output magnitude per Transformer block: baseline grows monotonically to ~15 at block 27; Block AttnRes shows a bounded *periodic* sawtooth (growth resets at each block boundary because selective aggregation restarts the accumulation). Gradient magnitude: baseline has disproportionately large gradients in the earliest blocks (~3e-5); AttnRes is far flatter — softmax competition for probability mass spreads gradient across depth.

**Downstream (Table 3; Kimi Linear 48B/3B, same recipe).** Baseline → AttnRes: MMLU 73.5→74.6; MMLU-Pro 52.2→52.2; GPQA-Diamond 36.9→**44.4 (+7.5)**; BBH 76.3→78.0; ARC-C 64.6→65.7; HellaSwag 83.2→83.4; TriviaQA 69.9→71.8; GSM8K 81.7→82.4; MGSM 64.9→66.1; MATH 53.5→**57.1 (+3.6)**; CMath 84.7→85.1; HumanEval 59.1→**62.2 (+3.1)**; MBPP 72.0→73.9; CMMLU 82.0→82.9; C-Eval 79.6→82.5. Matches or beats baseline on every task; biggest gains on multi-step reasoning and code, consistent with "later layers selectively retrieve and build on earlier representations".

---

## 7. Figures worth redrawing

- **Fig. 1 (three-panel overview)** — the block-diagram candidate. (a) Standard: Embedding → Attention → MoE → Attention → MoE → Output, a single vertical residual line with "+" junctions. (b) Full AttnRes: same stack, but before *each* sublayer an "$\alpha$" node fans in arrows from the embedding and *every* previous sublayer output; each $\alpha$ node has its own small "$w$" parameter box. (c) Block AttnRes: previous layers collapsed into boxes "Block $n-2$", "Block $n-1$"; the $\alpha$ nodes fan in only from Embedding, the block boxes and the running partial sum of the current block; inset "AttnRes Op ($\alpha$)": $w \to Q$, sources $\to K, V$ (with RMSNorm on K), softmax, weighted sum. For K3 specifically: 54 sublayers, 6 per block, 9 blocks + embedding = 10 sources max.
- **Fig. 9 (depth-mixing matrices, $L=4$, $S=2$)** — four $4\times4$ lower-triangular matrices: all-ones (residual), gate products (Highway), $\beta^\top A^\times\alpha$ (mHC), $\phi(w_l,k_i)$ dense (Full), and Block where rows 3–4 share the entry $\phi(w_l, k_1+k_2)$ for sources 1 and 2. Good for explaining "linear vs softmax attention over depth".
- **Fig. 3 (pipeline caching)** — 4 ranks × 2 virtual stages; boxes list cached blocks `[b0]`, `[b0,b1]`, … and transitions annotated `+[b1,b2]` (incremental) instead of the full history.
- **Fig. 5b/c** (sawtooth output magnitude vs monotone growth; flat vs front-loaded gradients) and **Fig. 8** heatmaps are the most persuasive "why" plots.
- **Algorithm 1** could be drawn as Phase 1 (one batched $S\times n$ score matrix computed up front) feeding Phase 2 (sequential chain with an online-softmax merge box per layer).

---

## 8. Model configs

- **Base architecture (all experiments)**: Kimi Linear — MoE Transformer in the Moonlight / DeepSeek-V3 style; KDA (Kimi Delta Attention) and MLA layers interleaved 3:1, each followed by an MoE FFN; MLA uses NoPE. The *only* change is AttnRes on the residual path (one RMSNorm + one $w_l$ per sublayer, zero-initialised). Depth, widths, routing unchanged.
- **Scaling-law models**: Table above (194M–528M activated, excluding embeddings; $L_b$ Transformer blocks = $2L_b$ sublayers; 8192 context; cosine schedule; Block AttnRes with $\approx 8$ blocks).
- **Ablation model**: the 436M row ($L_b=16$, $H=16$, $d=1168$); Block AttnRes there uses $S=4$ sublayers → $N=8$.
- **Large model**: Kimi Linear 48B total / 3B activated; 27 Transformer blocks = 54 sublayers; 8 of 256 routed experts + 1 shared; **Block AttnRes with 6 sublayers per block → 9 blocks + token embedding = 10 depth-wise sources**. Pre-training: 4096 context, Muon optimiser, WSD schedule, 8M-token global batch; 1T tokens WSD pre-training + ≈400B high-quality mid-training tokens (Moonlight annealing recipe), total 1.4T; then progressive long-context extension to 32K with no YaRN / temperature tricks thanks to NoPE-MLA + KDA.
- Typical numbers quoted for I/O analysis: $L=128$, $N=8$, $S=16$; prefill example: 128K tokens, 8 blocks → 15 GB unsharded.

**One-paragraph takeaway for the blog:** AttnRes treats the residual stream as an RNN over depth and swaps it for softmax attention over depth: each sublayer owns a single learned $d$-vector $w_l$ that scores the RMSNormed outputs of the embedding and all earlier sublayers, per token, and mixes the raw outputs with the softmax weights. Block AttnRes keeps ordinary residual sums inside blocks of $S$ sublayers and only attends over the $N$ finished block sums (plus embedding, plus the current block's running partial sum), which caps memory/communication at $O(Nd)$, makes pipeline-parallel training (<4% overhead) and decoding (<2%, $5.5d$ I/O per layer vs $3d$ for a plain residual) practical, and still delivers a $1.25\times$ effective-compute gain and +7.5 GPQA-Diamond at 48B.
