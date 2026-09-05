# LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts

**Source.** NVIDIA, arXiv 2601.18089 (v1, January 2026). Authors: Venmugil Elango, Nidhi Bhatia, Roger Waleffe, Rasoul Shafipour, ..., Mostofa Patwary, Mohammad Shoeybi, Bita Rouhani. Adopted by Nemotron-3 Super and Ultra. Read from the arXiv HTML; v1 has no appendix.

**One-sentence version.** Shrink the routed-expert input width from $d$ to a latent $\ell = d/\alpha$ with a single shared down-projection, run *all* routed experts (dispatch, compute, weighted combine) in that latent space, project back once with a shared up-projection, and spend the $\alpha\times$ savings in all-to-all bytes and expert-weight bytes on $\alpha\times$ more experts and $\alpha\times$ larger top-k. Shared experts and the router stay in full width $d$.

---

## 1. Framing: "accuracy per FLOP and per parameter", and the design-space exploration

The paper is *not* a broad sweep of MoE hyperparameters in the DeepSeekMoE / "granularity" style. It is a hardware-software co-design argument. The knobs it names are: total routed experts $N$, active experts $K$, hidden width $d$, expert intermediate width $m$, shared experts $S$, and (implicitly) the sparsity $K/N$. Of these, it argues on systems grounds that only $d$ should be reduced, and that $N$ and $K$ should be grown. The empirical exploration then sweeps just the compression ratio $\alpha = d/\ell$ (Fig. 3), with/without expert-count compensation (Fig. 4), and the two variants $\ell\text{-MoE}_{\text{eff}}$ vs $\ell\text{-MoE}_{\text{acc}}$ (Fig. 5, 6, Tables 3, 4).

**Why two metrics.** Accuracy per FLOP captures compute efficiency. Accuracy per parameter is the proxy for memory footprint, HBM bandwidth, routing-induced all-to-all volume, and sharding overhead, which dominate interactive low-latency serving. An architecture can look efficient in aggregate FLOPs and still serve badly.

**Running example** for all systems modelling: Qwen3-235B-A22B ($N=128$, $K=8$, $d=4096$, $m=1536$) on GB200 NVL72, $\mathrm{EP}=64$ (2 experts per GPU), $F = 10$ PFLOPs FP4, $\mathrm{BW}_{\mathrm{HBM}} = 8$ TB/s, $\mathrm{BW}_{\mathrm{NVL}} = 900$ GB/s per direction.

**Design Principle I (memory bandwidth, latency regime).** Compute-bound only if arithmetic intensity $I \ge F/\mathrm{BW}_{\mathrm{HBM}} = 1250$ FLOPs/byte. With $t_{\text{exp}} = t_{\text{total}} K / N$ tokens per expert, FP4 expert cost $C_{\text{exp}} = 2\, t_{\text{exp}} d m$ and traffic $M_{\text{exp}} = d m + t_{\text{exp}}(d+m)$:

$$ I = \frac{2\, t_{\text{exp}}\, d\, m}{d\,m + t_{\text{exp}}(d+m)} \ge 1250 \;\implies\; t_{\text{exp}} \ge 1418. $$

Interactive serving has $t_{\text{exp}}$ of a few hundred, so MoE experts sit in the weight-loading-bound region. Therefore maximise accuracy per parameter.

**Design Principle II (communication, throughput regime).** Per GPU per MoE layer, all-to-all volume $M_{\text{comm}} = 2.5\,(N/\mathrm{EP})\, t_{\text{exp}}\, d$ (0.5 B FP4 dispatch + 2 B BF16 combine) against compute $C_{\text{comp}} = 2 (N/\mathrm{EP})\, t_{\text{exp}}\, d\, m$, giving

$$ \frac{t_{\text{comm}}}{t_{\text{comp}}} = \frac{5\,F}{4\, m\, \mathrm{BW}_{\mathrm{NVL}}} \approx 9 $$

for Qwen3-235B on GB200. Communication is proportional to $t_{\text{total}} K d / \mathrm{EP}$, so it can only be cut via $d$ or $K$; $m$ does not help.

**Design Principle III (nonlinear budget).** Barron (1993): a one-hidden-layer net with $u$ nonlinear units has MSE $\mathcal{O}(1/u)$ independent of input dimension. Per-token nonlinear budget of an MoE layer is $U_{\text{eff}} \propto K\, m$. So do not shrink $K$ or $m$.

**Design Principle IV (feature rank).** Each task has an intrinsic feature rank $r_{\text{eff}}$; reducing $d$ below it collapses quality. This is the lower bound on $\ell$.

**Design Principle V (combinatorial sparsity).** $\binom{N}{K}$ expert combinations per token; scaling both by $\alpha$ grows this super-exponentially:

$$ \binom{\alpha N}{\alpha K} \ge \binom{N}{K}^{\alpha}. $$

**Synthesis.** Bandwidth cost scales with $d, m$; communication with $K, d$; quality needs $K, m$ preserved. So reduce $d \to \ell = d/\alpha$ (with $\ell \ge r_{\text{eff}}$), and since both costs are linear in $K$, raise $K$ by the same $\alpha$ for free. Main empirical conclusions: quality is preserved for $\alpha \le 4$; reducing $d$ *without* raising $N$ hurts; raising both $N$ and $K$ by $\alpha$ beats the baseline at iso-cost.

---

## 2. The LatentMoE layer

Let $x \in \mathbb{R}^d$ be the post-attention residual. Learnable $W_{\downarrow} \in \mathbb{R}^{\ell \times d}$, $W_{\uparrow} \in \mathbb{R}^{d \times \ell}$, each routed expert $E_i(\cdot;\ell)$ with $W^{(i)}_{\text{FC1}}, W^{(i)}_{\text{gate}} \in \mathbb{R}^{m \times \ell}$ and $W^{(i)}_{\text{FC2}} \in \mathbb{R}^{\ell \times m}$ (the gate is absent when the activation is Squared-ReLU, as in the 95B and hybrid models). $N' = \alpha N$ routed experts, $S$ shared experts $E_j(\cdot; d)$ in full width.

**Router on $x$, not on the latent.** $p' = \operatorname{Softmax}(W'_r\, x)$ with $W'_r \in \mathbb{R}^{N' \times d}$; $\mathcal{T}_{K,N'}$ is the top-$K$ index set.

**Efficiency variant** ($K$ unchanged):

$$ \ell\text{-MoE}_{\text{eff}}(x) := W_{\uparrow}\Big(\sum_{i \in \mathcal{T}_{K,N'}} p'_i\, E_i(W_{\downarrow} x;\, \ell)\Big) + \sum_{j=N'+1}^{N'+S} E_j(x;\, d). \tag{1}$$

**Accuracy variant, the recommended default** ($K' = \alpha K$):

$$ \ell\text{-MoE}_{\text{acc}}(x) := W_{\uparrow}\Big(\sum_{i \in \mathcal{T}_{K',N'}} p'_i\, E_i(W_{\downarrow} x;\, \ell)\Big) + \sum_{j=N'+1}^{N'+S} E_j(x;\, d). \tag{2}$$

Notes on the exact structure:

- One $W_{\downarrow}$ and one $W_{\uparrow}$ per layer, shared by all routed experts (not per-expert low-rank factors, which is how it differs from MoLAE).
- The gated sum $\sum p'_i E_i(\cdot)$ is taken *in latent space*; $W_{\uparrow}$ is applied once per token after the all-to-all combine. So combine traffic is $\ell$-wide too.
- Shared experts, router, and the residual add all stay in $\mathbb{R}^d$; the paper says explicitly they "do not significantly contribute to the identified memory and communication bottlenecks."
- **There is no normalization, scaling, or nonlinearity anywhere on the projection path.** The routed path per token is the chain $W_{\uparrow} \cdot W^{(i)}_{\text{FC2}} \cdot \sigma(W^{(i)}_{\text{FC1}} W_{\downarrow} x)$: four matmuls with a single activation in the middle, and $W_{\downarrow}$ / $W_{\uparrow}$ receive gradient from all $K'$ active paths. The paper never discusses conditioning of this chain, initialization of $W_{\downarrow}/W_{\uparrow}$, or any norm between them. The word "stability" appears only once, in the sense that shrinking parameters "can impede training stability" (motivating expert-count compensation, Fig. 4), not in the sense of loss spikes. That is the gap Kimi K3's "Stable LatentMoE" presumably fills.
- Hyperparameters: they reuse the baseline's exactly ("Further hyperparameter tuning might lead to even better accuracy").

**Role of $\alpha$.** It is the single control knob for communication volume, per-expert weight bytes, and per-expert FLOPs. Recommendation: $\alpha = 4$ (validated at 16B and 95B); Fig. 3 shows validation loss is flat for $\alpha \le 4$ and degrades beyond. $\ell = 512$ at $d=2048$, $\ell = 1024$ at $d=4096$. K3's $\ell = 3584 = d/2$ is a more conservative $\alpha = 2$; under the paper's recipe that corresponds to a base of $N=448, K=8$ doubled to $896/16$.

---

## 3. The systems argument and accounting

Per GPU (Table 1; $c \in \{2,3\}$ matrices per expert):

| | comm. volume | weight bytes / expert | expert FLOPs / token | accuracy | speed |
|---|---|---|---|---|---|
| Standard MoE | $(N/\mathrm{EP})\, t_{\text{exp}}\, d$ | $c\, d\, m$ | $2 c K d m$ | – | – |
| $\ell\text{-MoE}_{\text{eff}}$ | $(N/\mathrm{EP})\, t_{\text{exp}}\, \ell$ | $c\, \ell\, m$ | $2cKdm/\alpha$ | $\rightarrow$ | $\uparrow$ |
| $\ell\text{-MoE}_{\text{acc}}$ | $(N/\mathrm{EP})\, t_{\text{exp}}\, d$ | $c\, d\, m$ (aggregate over $K'$) | $2cKdm$ | $\uparrow$ | $\rightarrow$ |

Dispatch and combine both move $\ell$-wide vectors, so all-to-all shrinks by $\alpha$; expert weights are $m \times \ell$, so bytes loaded per active expert shrink by $\alpha$. Neither cost depends on $N$, so multiplying the expert count by $\alpha$ is free at inference and keeps total parameters constant: $\alpha N \cdot c\,(d/\alpha)\, m = c N d m$. In $\ell\text{-MoE}_{\text{acc}}$, $K' = \alpha K$ restores active FLOPs and bytes to baseline exactly. Overhead is the two projections, $2 d \ell$ params and $4 d \ell$ FLOPs per token per layer, i.e. relative to expert FLOPs $2\ell/(cKm)$ (about 6% for the 95B config; the paper's Kimi-K2 projection says native K2 is within ~9% of K2-LatentMoE in throughput).

Table 3 confirms the accounting: 95B baseline 8.47B active, $\ell\text{-MoE}_{\text{acc}}$ 8.44B active / 94.8B total, $\ell\text{-MoE}_{\text{eff}}$ 5.62B active. Active-param drop $= 2 K L\, m\,(d - \ell) - 2 L d \ell \approx 2.9$B, which matches only if the 95B experts are two-matrix Squared-ReLU (no gate).

"Reinvest the savings": Principle V says the $\binom{\alpha N}{\alpha K}$ combinations and the $\alpha\times$ larger per-token nonlinear budget ($K' m = \alpha K m$) buy accuracy; the projection only buys back the cost.

---

## 4. Theory (why latent experts do not lose accuracy)

Only three lightweight arguments, no proofs:

1. **Barron bound.** MSE $\mathcal{O}(1/u)$ with $u = K m$ nonlinear units per token, independent of the input width, so cutting $d \to \ell$ with $K, m$ fixed leaves $U_{\text{eff}}$ unchanged (Principle III). In $\ell\text{-MoE}_{\text{acc}}$ it grows by $\alpha$.
2. **Feature rank.** Loss is only incurred if $\ell < r_{\text{eff}}$; the $\alpha$-sweep estimates $r_{\text{eff}} \le d/4$ for these models (Principle IV).
3. **Combinatorics.** $\binom{\alpha N}{\alpha K} \ge \binom{N}{K}^\alpha$ (Principle V). They also cite DeepSeekMoE for the combinatorial-sparsity view and admit that "larger models are often easier to train and more robust to hyperparameter variations," which is the practical reason to scale $N$ even in the efficiency variant.

---

## 5. Results

**Setups (Table 2).** 16BT-2BA Transformer (27 layers, $d=2048$, $N=64$, $K=6$, $S=2$, $m=1408$, SwiGLU; DeepSeek-V2-Lite hyperparameters) for ablations. 95BT-8BA Transformer (32 layers, $d=4096$, $N=128$, $K=6$, $S=2$, $m=2688$, Squared-ReLU, 32 heads / 8 GQA groups; cosine LR $1.2\times10^{-3} \to 3\times10^{-6}$). Hybrid-73BT-8BA Mamba-Attention (52 layers: 24 Mamba/MoE, 4 attention; same MoE config; WSD LR $8\times10^{-4} \to 8\times10^{-6}$ in the last 15%). Both 8B-active models: seq len 8192, batch 768 (~6M tokens), 8.4B-token warmup, aux-loss coefficient $10^{-4}$ plus DeepSeek aux-loss-free balancing. LatentMoE variants always use $\alpha = 4$ ($N' = 256$ or $512$; $K' = 24$ for the acc variant).

**Ablations (16BT-2BA).** Fig. 3: loss unchanged for $\alpha \le 4$. Fig. 4: $4\times$ compression *without* raising $N$ gives a clearly worse curve; raising $N$ to $4N$ recovers it, "eliminating the need for extensive hyperparameter retuning." Fig. 5 ($\ell=512$): $\ell\text{-MoE}_{\text{eff}}$ tracks baseline; $\ell\text{-MoE}_{\text{acc}}$ is noticeably below it.

**Shared experts.** No ablation. $S=2$ full-width shared experts are simply carried over from the baseline in every configuration.

**95BT-8BA at 300B tokens (Table 3)**, baseline / acc / eff: MMLU-Pro 29.26 / **34.91** / 34.75; MMLU 58.95 / **62.23** / 61.06; Code (HumanEval, HumanEval+, MBPP, MBPP+) 40.33 / **41.50** / 40.68; Math (GSM8K-CoT, MATH-500) 64.39 / **64.88** / 63.61; Commonsense (RACE, ARC-C, HellaSwag, Winogrande) 74.32 / **75.18** / 73.72. So acc: +5.65 MMLU-Pro at identical cost; eff: better on 3 of 5 with 34% fewer active params.

**Hybrid-73BT-8BA at 1T tokens (Table 4)**, baseline / acc / eff: MMLU-Pro 48.30 / **52.87** / 51.29; MMLU 70.10 / **72.11** / 71.34; Code 51.95 / **55.14** / 53.13; Math 78.32 / **80.19** / 77.01; Commonsense 81.73 / **82.10** / 80.78. Active params 8.09B / 8.02B / 5.91B.

**Measured inference (Table 5).** Hybrid-73B, 2x H100, vLLM, FP8 per-tensor. Tokens/s/GPU LatentMoE vs standard at concurrency 1/4/16/64/128: 181.6/206.6, 528.5/509.8, 1130.8/1204.6, 1569.6/1549.3, 1625.8/1725.9. At most 6% slower; they attribute it to unoptimized kernels and propose separate CUDA streams for routed vs shared experts, and small-inner-dimension GEMM kernels since $\ell$ is smaller.

**Projected 1T-scale serving (Fig. 7).** Proprietary simulator, 200K operating points, on Kimi-K2-1T vs a hypothetical Kimi-K2-1T-LatentMoE. Effective Parameter Multiplier $\lambda = N_{\text{eff}}/N_{\text{treat}}$ with $N_{\text{eff}} = f^{-1}(S_{\text{treat}})$ and $f(N) = a \log N + b$ fitted on Qwen3-Dense 0.6B-32B MMLU, gives $\lambda \approx 1.35$. An iso-accuracy standard-MoE baseline "Kimi-K2-1.35T" (61 to 80 layers) is 1.24x-3.46x slower across the throughput-latency frontier, for both decode-heavy (chunked piggybacking) and prefill-heavy (disaggregated) traffic.

**Instability / conditioning.** Nothing beyond the "smaller models are harder to train" remark. No loss-spike reports, no mention of RMSNorm on the latent, no learning-rate or init changes for $W_{\downarrow}, W_{\uparrow}$. The only place normalization appears is in describing the concurrent mHC work.

---

## 6. Relationship to Nemotron-3 and follow-ups

The abstract and introduction state that LatentMoE "has been adopted by the flagship Nemotron-3 Super and Ultra models and scaled to substantially larger regimes, including longer token horizons and larger model sizes (NVIDIA et al., 2025)", with no configuration details given here. The hybrid Mamba-Attention baseline in Table 2 is a Nemotron-style stack. "Multi-Head LatentMoE" is *not* mentioned anywhere in v1. Related work positions LatentMoE against MoLAE (post-training low-rank factoring with grouped projections and FC2-only compression, which forfeits the all-to-all saving) and calls mHC (manifold-constrained hyper-connections) complementary and stackable.

---

## 7. Figures worth redrawing

**Fig. 1(b), the block diagram (with 1(a) as a side-by-side baseline).** Bottom to top: "From previous layer" into Self-Attention with a skip; residual add $\oplus$. From that node three arrows fan out: (i) left, a long skip to the top output $\oplus$; (ii) to the shared expert box SE (full width); (iii) to the Router (bar chart of $N'$ scores) and to a green **Latent down-proj** box. Down-proj feeds **All-to-All dispatch**, which fans to a $2\times 4$ grid of experts E1-E8 (baseline has one row E1-E4), with $K'$ hatched as active (four in 1(b), two in 1(a)). Each active expert's output passes a $\otimes$ node fed by a dashed line from the Router (the gate $p'_i$), then into **All-to-All combine**, then a green **Latent up-proj**, then into the top $\oplus$ together with SE and the residual skip, out to "To next layer". Annotate the vertical edges with widths: $d$ below down-proj and on the SE/router/residual paths, $\ell$ from down-proj through dispatch, experts, gating, combine; $d$ again after up-proj. The two green boxes are the only additions over the standard diagram; the message is "everything between them is $\ell$-wide and there are $\alpha\times$ more of it."

**Fig. 2, roofline** (optional): performance vs arithmetic intensity with the 1250 FLOPs/byte knee and Qwen3 operating points at $t_{\text{exp}}$ of a few hundred sitting on the bandwidth-bound slope.

**Fig. 3/4/5 as one panel:** validation loss vs tokens with curves for baseline, $\alpha=2,4,8$ (flat to 4), "compressed without $N$ scaling" (worse), and $\ell\text{-MoE}_{\text{acc}}$ (best).

**For the K3 blog specifically**, a table mapping the paper's symbols to K3 values is the most useful artifact: $d=7168$, $\ell=3584$, $\alpha=2$, $N'=896$, $K'=16$, $S=2$ full-width shared, router on $x$; plus whatever K3 inserts between the four matmuls, which the original paper leaves as an unnormalized linear chain.
